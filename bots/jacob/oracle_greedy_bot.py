#!/usr/bin/env python3
"""
Oracle Greedy Bot - Sushi Go AI

Strategy: Full hand tracking + deck inference + opponent tableau awareness.
Makes the single best greedy decision each turn using a rich scoring heuristic
tuned by player count, opponent tableaux, and known/inferred card positions.

Usage:
    python oracle_greedy_bot.py <host> <port> <game_id> <player_name>

Example:
    python oracle_greedy_bot.py localhost 7878 abc123 OracleBot
"""

import json
import re
import socket
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

# ── Deck composition (standard Sushi Go, no Party/expansions) ───────────────
FULL_DECK = Counter({
    "Tempura":       14,
    "Sashimi":       14,
    "Dumpling":      14,
    "Maki Roll (1)":  6,
    "Maki Roll (2)": 12,
    "Maki Roll (3)":  8,
    "Salmon Nigiri": 10,
    "Squid Nigiri":   5,
    "Egg Nigiri":     5,
    "Pudding":       10,
    "Wasabi":         6,
    "Chopsticks":     4,
})
TOTAL_CARDS = sum(FULL_DECK.values())  # 108

# Cards dealt per player per round (standard rules)
CARDS_PER_PLAYER = {2: 10, 3: 9, 4: 8, 5: 7}

# Abbreviation → full name (handles short codes seen in server logs)
ABBREV = {
    "TMP": "Tempura",        "SSH": "Sashimi",
    "DMP": "Dumpling",       "MK1": "Maki Roll (1)",
    "MK2": "Maki Roll (2)", "MK3": "Maki Roll (3)",
    "SAL": "Salmon Nigiri", "SQD": "Squid Nigiri",
    "EGG": "Egg Nigiri",    "PUD": "Pudding",
    "WAS": "Wasabi",        "CHP": "Chopsticks",
}

def normalize(raw: str) -> str:
    s = raw.strip()
    if s in ABBREV:
        return ABBREV[s]
    for full in FULL_DECK:
        if full.lower() == s.lower():
            return full
    return s


# ── Scoring helpers ──────────────────────────────────────────────────────────

def count_maki(tableau: list) -> int:
    c = Counter(tableau)
    return c["Maki Roll (1)"] + c["Maki Roll (2)"] * 2 + c["Maki Roll (3)"] * 3


def score_maki_for_player(my_maki: int, all_maki: list, player_count: int) -> int:
    """Compute maki score for a single player given all players' maki totals."""
    if all(m == 0 for m in all_maki):
        return 0
    first_pts = 4 if player_count == 2 else 6
    second_pts = 2 if player_count == 2 else 3
    sorted_unique = sorted(set(all_maki), reverse=True)
    first_val = sorted_unique[0]
    first_tiers = [m for m in all_maki if m == first_val]
    if my_maki == first_val:
        return first_pts // len(first_tiers)
    if len(sorted_unique) > 1:
        second_val = sorted_unique[1]
        second_tiers = [m for m in all_maki if m == second_val]
        if my_maki == second_val:
            return second_pts // len(second_tiers)
    return 0


def score_nigiri_with_wasabi(tableau: list) -> int:
    """Score nigiri accounting for wasabi, highest nigiri gets wasabi."""
    c = Counter(tableau)
    wasabi_left = c["Wasabi"]
    score = 0
    for nigiri, base in [("Squid Nigiri", 3), ("Salmon Nigiri", 2), ("Egg Nigiri", 1)]:
        for _ in range(c[nigiri]):
            if wasabi_left > 0:
                score += base * 3
                wasabi_left -= 1
            else:
                score += base
    return score


def score_tableau(tableau: list) -> int:
    """Score a tableau (no maki, no pudding end-game — handled separately)."""
    c = Counter(tableau)
    score = 0
    score += (c["Tempura"] // 2) * 5
    score += (c["Sashimi"] // 3) * 10
    d = c["Dumpling"]
    DUMP = [0, 1, 3, 6, 10, 15]
    score += DUMP[min(d, 5)]
    score += score_nigiri_with_wasabi(tableau)
    return score


# ── Game state ────────────────────────────────────────────────────────────────

@dataclass
class PlayerInfo:
    name: str
    tableau: list = field(default_factory=list)
    puddings: int = 0       # accumulated across all rounds
    total_score: int = 0


@dataclass
class GameKnowledge:
    my_name: str = ""
    player_count: int = 2
    cards_per_hand: int = 10
    round_num: int = 1
    turn: int = 0           # turns elapsed this round (increments after PLAYED)

    # Ordered seat list — index 0 = first joined, clockwise order
    seats: list = field(default_factory=list)
    players: dict = field(default_factory=dict)  # name → PlayerInfo

    # My current hand
    my_hand: list = field(default_factory=list)

    # Inferred hands: after seeing our hand rotate, we know what the next
    # player holds. Maps name → list of cards (or None if unknown).
    known_hands: dict = field(default_factory=dict)

    # Previous hand (before current turn) for rotation inference
    prev_hand: list = field(default_factory=list)

    def my_info(self) -> PlayerInfo:
        return self.players.get(self.my_name, PlayerInfo(self.my_name))

    def my_tableau(self) -> list:
        return self.my_info().tableau

    def has_unused_wasabi(self, name: str) -> bool:
        t = self.players[name].tableau if name in self.players else []
        wasabi = t.count("Wasabi")
        nigiri = sum(1 for c in t if "Nigiri" in c)
        return wasabi > nigiri

    def my_maki(self) -> int:
        return count_maki(self.my_tableau())

    def all_maki(self) -> list:
        return [count_maki(p.tableau) for p in self.players.values()]

    def opponent_pudding_counts(self) -> list:
        return [p.puddings for n, p in self.players.items() if n != self.my_name]

    def next_seat(self) -> str:
        """Clockwise next player from me."""
        if not self.seats:
            return ""
        my_pos = next((i for i, n in enumerate(self.seats) if n == self.my_name), 0)
        return self.seats[(my_pos + 1) % len(self.seats)]

    def reset_for_round(self, round_num: int):
        self.round_num = round_num
        self.turn = 0
        self.prev_hand = []
        self.known_hands = {}
        for p in self.players.values():
            p.tableau = []


# ── Card value heuristic ──────────────────────────────────────────────────────

def card_value(card: str, gk: GameKnowledge, hand: list) -> float:
    """
    Estimate the marginal value of playing `card` right now.
    Returns float — higher is better.
    """
    tableau = gk.my_tableau()
    counts = Counter(tableau)
    n = gk.player_count
    hand_size = len(hand)
    # Turns remaining AFTER this play in the current round
    turns_left = gk.cards_per_hand - gk.turn - 1

    # ── Nigiri ──────────────────────────────────────────────────────────────
    if card == "Squid Nigiri":
        return 9.0 if gk.has_unused_wasabi(gk.my_name) else 3.0

    if card == "Salmon Nigiri":
        return 6.0 if gk.has_unused_wasabi(gk.my_name) else 2.0

    if card == "Egg Nigiri":
        return 3.0 if gk.has_unused_wasabi(gk.my_name) else 1.0

    # ── Wasabi ──────────────────────────────────────────────────────────────
    if card == "Wasabi":
        if gk.has_unused_wasabi(gk.my_name):
            return 0.1  # Already have unused wasabi — this one's dead weight
        nigiri_in_hand = sum(1 for c in hand if "Nigiri" in c)
        if nigiri_in_hand > 0:
            # There's a nigiri available right now — we won't need the wasabi
            # because we can just play the nigiri; but wasabi THEN nigiri next turn
            # is fine. Give medium value.
            return 3.5
        # Speculative — value decays late in round
        round_progress = gk.turn / max(1, gk.cards_per_hand)
        return max(0.3, 4.0 * (1.0 - round_progress))

    # ── Tempura ─────────────────────────────────────────────────────────────
    if card == "Tempura":
        have = counts["Tempura"]
        if have % 2 == 1:
            return 5.0  # Completes a pair — guaranteed 5 points
        # Need a pair: speculative. Value decays as hand shrinks.
        progress = gk.turn / max(1, gk.cards_per_hand)
        return max(0.3, 2.5 * (1.0 - progress))

    # ── Sashimi ─────────────────────────────────────────────────────────────
    if card == "Sashimi":
        have = counts["Sashimi"]
        mod = have % 3
        if mod == 2:
            return 10.0  # Completing a set of 3 — guaranteed 10 pts
        progress = gk.turn / max(1, gk.cards_per_hand)
        if mod == 1:
            return max(0.3, 4.0 * (1.0 - progress))
        return max(0.3, 2.0 * (1.0 - progress))

    # ── Dumpling ────────────────────────────────────────────────────────────
    if card == "Dumpling":
        have = counts["Dumpling"]
        DUMP = [0, 1, 3, 6, 10, 15]
        marginal = DUMP[min(have + 1, 5)] - DUMP[min(have, 5)]
        return float(marginal)

    # ── Maki Rolls ──────────────────────────────────────────────────────────
    if card.startswith("Maki Roll"):
        maki_pip = {"Maki Roll (1)": 1, "Maki Roll (2)": 2, "Maki Roll (3)": 3}[card]
        my_maki = gk.my_maki()
        opp_makis = [count_maki(p.tableau) for nm, p in gk.players.items()
                     if nm != gk.my_name]
        max_opp = max(opp_makis) if opp_makis else 0
        first_prize = 4 if n == 2 else 6
        second_prize = 2 if n == 2 else 3

        my_new = my_maki + maki_pip
        if my_new > max_opp:
            raw = first_prize * 0.85
        elif my_new == max_opp:
            raw = first_prize * 0.5
        else:
            second_val = sorted(set(opp_makis))[-2] if len(set(opp_makis)) > 1 else 0
            if my_new >= second_val:
                raw = second_prize * 0.6
            else:
                raw = second_prize * 0.25

        # Maki is worth more early; worth less if we already lead comfortably
        scale = max(0.3, 1.0 - (gk.turn / max(1, gk.cards_per_hand)))
        return raw * scale * (maki_pip / 2.0)  # reward higher pip cards

    # ── Pudding ─────────────────────────────────────────────────────────────
    if card == "Pudding":
        my_puds = gk.my_info().puddings
        opp_puds = gk.opponent_pudding_counts()
        max_opp = max(opp_puds) if opp_puds else 0
        min_opp = min(opp_puds) if opp_puds else 0

        # 2-player: pudding is a ±6 pt swing per pudding — very high value
        if n == 2:
            base = 5.0
        elif n <= 3:
            base = 3.5
        elif n == 4:
            base = 2.5
        else:
            base = 2.0

        # We're behind in puddings → extra urgency
        if my_puds < min_opp:
            base += 1.5
        elif my_puds <= max_opp:
            base += 0.5
        return base

    # ── Chopsticks ──────────────────────────────────────────────────────────
    if card == "Chopsticks":
        # Worthless in last 2 cards, valuable with big hand
        if hand_size <= 2:
            return 0.05
        # Value scales with hand size remaining
        return max(0.1, 2.0 * ((hand_size - 2) / gk.cards_per_hand))

    return 0.5


def should_use_chopsticks(hand: list, gk: GameKnowledge) -> Optional[tuple]:
    """
    If chopsticks are in our tableau and it's worth using them, return (i, j).
    Otherwise None.
    """
    if "Chopsticks" not in gk.my_tableau():
        return None
    if len(hand) < 2:
        return None

    values = [card_value(c, gk, hand) for c in hand]
    best_single = max(values)

    best_score = -1
    best_pair = None
    for i in range(len(hand)):
        for j in range(i + 1, len(hand)):
            combined = values[i] + values[j]
            if combined > best_score:
                best_score = combined
                best_pair = (i, j)

    # Use chopsticks only if the gain is meaningfully better than best single card
    threshold = best_single + 3.5
    if best_pair and best_score >= threshold:
        return best_pair
    return None


def pick_card(hand: list, gk: GameKnowledge) -> int:
    """Return index of the best card to play."""
    if not hand:
        return 0
    scored = [(card_value(c, gk, hand), idx) for idx, c in enumerate(hand)]
    scored.sort(reverse=True)
    return scored[0][1]


# ── Protocol helpers ──────────────────────────────────────────────────────────

def parse_hand(msg: str) -> list:
    payload = msg[len("HAND "):].strip()
    cards = []
    for m in re.finditer(r'(\d+):(.*?)(?=\s+\d+:|$)', payload):
        cards.append(normalize(m.group(2).strip()))
    return cards


def parse_played(msg: str) -> dict:
    """PLAYED p1:Card; p2:Card  →  {name: card}"""
    payload = msg[len("PLAYED "):].strip()
    result = {}
    for part in payload.split(";"):
        part = part.strip()
        if ":" not in part:
            continue
        name, _, raw = part.partition(":")
        result[name.strip()] = normalize(raw.strip())
    return result


def parse_json_scores(msg: str) -> dict:
    m = re.search(r'\{[^}]+\}', msg)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


# ── Bot ───────────────────────────────────────────────────────────────────────

class OracleGreedyBot:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = None
        self.sock_file = None
        self.gk = GameKnowledge()
        self.rejoin_token = ""

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.sock_file = self.sock.makefile("r", encoding="utf-8", errors="replace")
        print(f"Connected to {self.host}:{self.port}")

    def send(self, cmd: str):
        print(f">>> {cmd}")
        self.sock.sendall((cmd + "\n").encode("utf-8"))

    def recv(self) -> str:
        line = self.sock_file.readline()
        if line == "":
            raise ConnectionError("Server closed connection")
        msg = line.strip()
        if msg:
            print(f"<<< {msg}")
        return msg

    def recv_until(self, pred) -> str:
        while True:
            msg = self.recv()
            if msg and pred(msg):
                return msg

    def run(self, game_id: str, name: str):
        try:
            self.connect()
            self.gk.my_name = name
            self.send(f"JOIN {game_id} {name}")
            resp = self.recv_until(lambda m: m.startswith("WELCOME") or m.startswith("ERROR"))
            if not resp.startswith("WELCOME"):
                print(f"Failed to join: {resp}")
                return
            parts = resp.split()
            self.rejoin_token = parts[3]
            self.send("READY")

            # Ensure self in players/seats
            self.gk.players[name] = PlayerInfo(name)
            self.gk.seats.append(name)

            while True:
                msg = self.recv()
                if not msg:
                    continue

                if msg.startswith("JOINED"):
                    # JOINED <name> <n>/<max>
                    jname = msg.split()[1]
                    if jname not in self.gk.players:
                        self.gk.players[jname] = PlayerInfo(jname)
                        self.gk.seats.append(jname)

                elif msg.startswith("GAME_START"):
                    pc = int(msg.split()[1])
                    self.gk.player_count = pc
                    self.gk.cards_per_hand = CARDS_PER_PLAYER.get(pc, 10)

                elif msg.startswith("ROUND_START"):
                    self.gk.reset_for_round(int(msg.split()[1]))

                elif msg.startswith("HAND"):
                    hand = parse_hand(msg)
                    self.gk.my_hand = hand
                    self.gk.prev_hand = list(hand)

                    chops = should_use_chopsticks(hand, self.gk)
                    if chops:
                        i, j = chops
                        self.send(f"CHOPSTICKS {i} {j}")
                    else:
                        idx = pick_card(hand, self.gk)
                        self.send(f"PLAY {idx}")

                elif msg.startswith("PLAYED"):
                    plays = parse_played(msg)
                    # Update tableaux
                    for pname, card in plays.items():
                        if pname in self.gk.players:
                            self.gk.players[pname].tableau.append(card)
                            if card == "Pudding":
                                self.gk.players[pname].puddings += 1
                    # Infer next player's hand from our previous hand
                    next_seat = self.gk.next_seat()
                    if next_seat and self.gk.prev_hand:
                        my_play = plays.get(name, "")
                        inferred = list(self.gk.prev_hand)
                        if my_play in inferred:
                            inferred.remove(my_play)
                        self.gk.known_hands[next_seat] = inferred
                    self.gk.turn += 1

                elif msg.startswith("ROUND_END"):
                    scores = parse_json_scores(msg)
                    for nm, sc in scores.items():
                        if nm in self.gk.players:
                            self.gk.players[nm].total_score = sc

                elif msg.startswith("GAME_END"):
                    scores = parse_json_scores(msg)
                    print("\n=== GAME OVER ===")
                    for nm, sc in sorted(scores.items(), key=lambda x: -x[1]):
                        tag = " ← ME" if nm == name else ""
                        print(f"  {nm}: {sc}{tag}")
                    break

        except KeyboardInterrupt:
            print("\nDisconnecting...")
        except Exception as e:
            print(f"Error: {e}")
            import traceback; traceback.print_exc()
        finally:
            if self.sock:
                self.sock.close()


def main():
    if len(sys.argv) != 5:
        print("Usage: python oracle_greedy_bot.py <host> <port> <game_id> <player_name>")
        sys.exit(1)
    bot = OracleGreedyBot(sys.argv[1], int(sys.argv[2]))
    bot.run(sys.argv[3], sys.argv[4])


if __name__ == "__main__":
    main()
