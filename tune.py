"""
tune_weights.py — Hill-climbing weight tuner for IterMine.

Strategy:
  1. Start from Weights() defaults (or a saved checkpoint).
  2. Run a batch of games to estimate win rate.
  3. Perturb weights randomly within configured bounds.
  4. If win rate improves, keep the new weights.
  5. After N failed improvements, restart from the best known weights
     with a larger perturbation (simulated annealing-lite).
  6. Save best weights to JSON on every improvement.

Usage:
    python tune_weights.py
    python tune_weights.py --checkpoint best_weights.json  # resume
    python tune_weights.py --games 50 --rounds 200        # custom budget
"""

import argparse
import copy
import dataclasses
import json
import math
import random
import threading
import time
from typing import Callable

import requests

from bots.adam.itermine import IterMine, Weights
from bots.sushi_go_client import SushiGoClient

HOST = "localhost"
PORT = 7878
GAMES_URL = "http://localhost:8080/api/games"

# ---------------------------------------------------------------------------
# Weight perturbation config
# Each entry: field_name -> (min_val, max_val, step_sigma)
# step_sigma is the std-dev of gaussian noise applied per perturbation.
# Fields not listed are frozen at their default values.
# ---------------------------------------------------------------------------
WEIGHT_RANGES = {
    # Nigiri — pin to actual point values, small wiggle allowed
    "squid_nigiri": (2.0, 4.0, 0.3),
    "salmon_nigiri": (1.0, 3.0, 0.3),
    "egg_nigiri": (0.5, 2.0, 0.2),
    "wasabi_nigiri_multiplier": (2.0, 4.0, 0.2),
    # Wasabi
    "wasabi_nigiri_in_hand": (3.0, 8.0, 0.5),
    "wasabi_squid_salmon_incoming": (2.0, 6.0, 0.5),
    "wasabi_any_nigiri_incoming": (1.0, 4.0, 0.3),
    "wasabi_blind": (0.3, 3.0, 0.3),
    "wasabi_already_unused": (0.0, 1.0, 0.1),
    # Tempura
    "tempura_completes_pair": (4.0, 6.0, 0.3),
    "tempura_pair_incoming": (2.0, 5.0, 0.4),
    "tempura_speculative": (0.5, 3.0, 0.3),
    "tempura_one_turn_left": (0.0, 2.0, 0.2),
    # Sashimi
    "sashimi_completes_set": (8.0, 12.0, 0.4),
    "sashimi_one_needed_incoming": (4.0, 8.0, 0.5),
    "sashimi_one_needed_blind": (1.0, 4.0, 0.4),
    "sashimi_two_needed_incoming": (2.0, 6.0, 0.5),
    "sashimi_two_needed_speculative": (0.5, 3.0, 0.3),
    "sashimi_fresh_start": (0.0, 2.0, 0.2),
    # Dumplings — tune each marginal independently
    "dumpling_marginals_0": (0.5, 2.0, 0.2),
    "dumpling_marginals_1": (1.0, 3.0, 0.3),
    "dumpling_marginals_2": (2.0, 5.0, 0.4),
    "dumpling_marginals_3": (3.0, 6.0, 0.4),
    "dumpling_marginals_4": (4.0, 8.0, 0.5),
    # Maki
    "maki_lead_per_icon": (1.5, 4.0, 0.3),
    "maki_tie_per_icon": (0.5, 2.5, 0.3),
    "maki_trailing_per_icon": (0.2, 1.5, 0.2),
    "maki_crowded_threshold": (3.0, 8.0, 0.5),
    "maki_crowded_penalty": (-3.0, 0.0, 0.3),
    # Pudding
    "pudding_round1": (0.5, 3.0, 0.3),
    "pudding_round2": (1.0, 4.0, 0.3),
    "pudding_r3_taking_lead": (3.0, 8.0, 0.5),
    "pudding_r3_avoiding_last": (3.0, 7.0, 0.5),
    "pudding_r3_neutral": (1.0, 4.0, 0.3),
    "pudding_race_bonus": (0.0, 3.0, 0.3),
    # Denial
    "denial_weight": (0.1, 1.2, 0.1),
    "denial_sashimi_complete": (5.0, 12.0, 0.6),
    "denial_sashimi_partial": (0.5, 4.0, 0.4),
    "denial_tempura_complete": (2.0, 7.0, 0.5),
    "denial_wasabi_nigiri": (1.0, 5.0, 0.4),
    "denial_maki_per_icon": (0.3, 2.5, 0.3),
    "denial_pudding_r3": (1.0, 6.0, 0.5),
}


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def weights_to_dict(w: Weights) -> dict:
    d = dataclasses.asdict(w)
    # Flatten dumpling_marginals into indexed keys for the tuner
    for i, v in enumerate(d.pop("dumpling_marginals")):
        d[f"dumpling_marginals_{i}"] = v
    return d


def dict_to_weights(d: dict) -> Weights:
    flat = dict(d)
    # Rebuild dumpling_marginals list
    marginals = []
    for i in range(5):
        key = f"dumpling_marginals_{i}"
        marginals.append(flat.pop(key))
    flat["dumpling_marginals"] = marginals
    return Weights(**flat)


def save_weights(w: Weights, path: str, win_rate: float):
    payload = weights_to_dict(w)
    payload["_win_rate"] = win_rate
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  💾  Saved to {path}  (win_rate={win_rate:.3f})")


def load_weights(path: str) -> tuple[Weights, float]:
    with open(path) as f:
        d = json.load(f)
    win_rate = d.pop("_win_rate", 0.0)
    return dict_to_weights(d), win_rate


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------


def perturb(w: Weights, scale: float = 1.0) -> Weights:
    """
    Return a new Weights with each tunable field nudged by gaussian noise
    scaled by `scale` (use scale > 1.0 for large restarts).
    Enforces min/max bounds and the ordering constraint squid > salmon > egg.
    """
    d = weights_to_dict(w)

    for field, (lo, hi, sigma) in WEIGHT_RANGES.items():
        noise = random.gauss(0, sigma * scale)
        d[field] = float(max(lo, min(hi, d[field] + noise)))

    # Enforce nigiri ordering: squid >= salmon >= egg (with small buffer)
    d["salmon_nigiri"] = min(d["salmon_nigiri"], d["squid_nigiri"] - 0.1)
    d["egg_nigiri"] = min(d["egg_nigiri"], d["salmon_nigiri"] - 0.1)
    d["salmon_nigiri"] = max(d["salmon_nigiri"], d["egg_nigiri"] + 0.1)

    # maki_crowded_penalty must be <= 0
    d["maki_crowded_penalty"] = min(0.0, d["maki_crowded_penalty"])

    return dict_to_weights(d)


# ---------------------------------------------------------------------------
# Game runner (adapted from your runner script)
# ---------------------------------------------------------------------------


def create_game(player_count: int) -> str:
    response = requests.post(GAMES_URL, json={"max_players": player_count})
    return response.text


def list_games() -> list[dict]:
    import socket as _socket

    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    sock_file = sock.makefile("r", encoding="utf-8", errors="replace")

    def _send(cmd):
        sock.sendall((cmd + "\n").encode())

    def _recv():
        line = sock_file.readline()
        if not line:
            raise ConnectionError("closed")
        return line.strip()

    def _recv_until(pred):
        while True:
            m = _recv()
            if m and pred(m):
                return m

    _send("GAMES")
    raw = _recv_until(lambda s: s.startswith("GAMES"))
    sock.close()
    return eval(raw[6:])


def newest_game_id() -> str:
    return list_games()[-1]["id"]


def run_single_game(players: list[SushiGoClient]) -> str | None:
    """Run one game, return the winner name or None on error."""
    create_game(len(players))
    time.sleep(0.05)  # brief pause so the server registers the game
    game_id = newest_game_id()

    threads = []
    for i, player in enumerate(players):
        name = player.name if player.name else f"Player{i}"
        t = threading.Thread(target=player.run, args=(game_id, name), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join(timeout=60)

    return players[0].winner  # IterMine is always players[0]


def estimate_win_rate(
    weights: Weights,
    n_games: int,
    opponent_count: int = 2,
) -> float:
    """
    Play `n_games` games of IterMine (with given weights) vs `opponent_count`
    default SushiGoClients. Returns fraction of games IterMine won.
    """
    wins = 0
    errors = 0

    for game_num in range(n_games):
        opponents = [SushiGoClient(HOST, PORT) for _ in range(opponent_count)]
        mine = IterMine(weights=weights)

        players = [mine] + opponents

        try:
            winner = run_single_game(players)
            if winner == mine.name:
                wins += 1
        except Exception as e:
            errors += 1
            print(f"    ⚠️  Game {game_num} error: {e}")

    valid = n_games - errors
    rate = wins / valid if valid > 0 else 0.0
    print(
        f"    → {wins}/{valid} wins  ({rate:.1%})"
        + (f"  [{errors} errors]" if errors else "")
    )
    return rate


# ---------------------------------------------------------------------------
# Hill climber
# ---------------------------------------------------------------------------


def hill_climb(
    n_rounds: int,
    games_per_eval: int,
    checkpoint_path: str,
    start_weights: Weights | None = None,
    start_win_rate: float = 0.0,
    patience: int = 15,
    restart_scale: float = 2.5,
):
    """
    Hill-climbing loop.

    patience: how many failed improvements before a larger random restart.
    restart_scale: perturbation multiplier on restart.
    """
    if start_weights is not None:
        best_w = start_weights
        best_rate = start_win_rate
    else:
        print("Evaluating baseline (default weights)…")
        best_w = Weights()
        best_rate = estimate_win_rate(best_w, games_per_eval)

    current_w = copy.deepcopy(best_w)
    current_rate = best_rate
    save_weights(best_w, checkpoint_path, best_rate)

    no_improve = 0

    for round_num in range(1, n_rounds + 1):
        scale = 1.0 if no_improve < patience else restart_scale
        label = "RESTART" if no_improve >= patience else f"round {round_num}"

        candidate = perturb(current_w, scale=scale)
        print(f"\n[{label}] Evaluating candidate…")
        rate = estimate_win_rate(candidate, games_per_eval)

        if rate > current_rate:
            improvement = rate - current_rate
            print(
                f"  ✅  Improved {current_rate:.3f} → {rate:.3f} (+{improvement:.3f})"
            )
            current_w = candidate
            current_rate = rate
            no_improve = 0

            if rate > best_rate:
                best_w = candidate
                best_rate = rate
                save_weights(best_w, checkpoint_path, best_rate)
        else:
            print(f"  ❌  No improvement ({rate:.3f} ≤ {current_rate:.3f})")
            no_improve += 1

        if no_improve >= patience:
            print(
                f"\n  🔄  {patience} failures — restarting from best weights with scale={restart_scale}"
            )
            current_w = copy.deepcopy(best_w)
            current_rate = best_rate
            no_improve = 0

    print(f"\n🏆  Best win rate: {best_rate:.3f}")
    print(f"    Saved to: {checkpoint_path}")
    return best_w, best_rate


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune IterMine weights via hill climbing"
    )
    parser.add_argument(
        "--checkpoint",
        default="best_weights.json",
        help="Path to save/load weight checkpoint (default: best_weights.json)",
    )
    parser.add_argument(
        "--games", type=int, default=30, help="Games per evaluation batch (default: 30)"
    )
    parser.add_argument(
        "--rounds", type=int, default=100, help="Hill-climbing rounds (default: 100)"
    )
    parser.add_argument(
        "--opponents",
        type=int,
        default=2,
        help="Number of SushiGoClient opponents (default: 2)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Failures before random restart (default: 15)",
    )
    args = parser.parse_args()

    # Try to resume from checkpoint
    start_w = None
    start_rate = 0.0
    try:
        start_w, start_rate = load_weights(args.checkpoint)
        print(f"Resuming from {args.checkpoint}  (recorded win_rate={start_rate:.3f})")
        # Re-evaluate to confirm — variance means stored rate may be optimistic
        print("Re-evaluating checkpoint…")
        start_rate = estimate_win_rate(start_w, args.games, args.opponents)
    except FileNotFoundError:
        print(f"No checkpoint at {args.checkpoint}, starting fresh.")

    hill_climb(
        n_rounds=args.rounds,
        games_per_eval=args.games,
        checkpoint_path=args.checkpoint,
        start_weights=start_w,
        start_win_rate=start_rate,
        patience=args.patience,
    )
