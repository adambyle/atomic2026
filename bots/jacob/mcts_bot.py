#!/usr/bin/env python3
"""
MCTS Bot - Sushi Go AI

Strategy: Paranoid multi-player Monte Carlo Tree Search seeded with real hand
tracking. Opponents are assumed to play against us (paranoid assumption — safe
for mixed player counts). Rollouts use a greedy heuristic (not random) so each
simulation is realistic. Hard time cap of 950ms per decision.

Usage:
    python mcts_bot.py <host> <port> <game_id> <player_name>

Example:
    python mcts_bot.py localhost 7878 abc123 MCTSBot
"""

import json
import math
import random
import re
import socket
import sys
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

# ── Deck (standard Sushi Go, no extensions) ──────────────────────────────────
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

CARDS_PER_PLAYER = {2: 10, 3: 9, 4: 8, 5: 7}

ABBREV = {
    "TMP": "Tempura",        "SSH": "Sashimi",
    "DMP": "Dumpling",       "MK1": "Maki Roll (1)",
    "MK2": "Maki Roll (2)", "MK3": "Maki Roll (3)",
    "SAL": "Salmon Nigiri", "SQD": "Squid Nigiri",
    "EGG": "Egg Nigiri",    "PUD": "Pudding",
    "WAS": "Wasabi",        "CHP": "Chopsticks",
}

MAKI_PIPS = {"Maki Roll (1)": 1, "Maki Roll (2)": 2, "Maki Roll (3)": 3}

def normalize(raw: str) -> str:
    s = raw.strip()
    if s in ABBREV:
        return ABBREV[s]
    for full in FULL_DECK:
        if full.lower() == s.lower():
            return full
    return s


# ── Scoring engine ────────────────────────────────────────────────────────────

def count_maki(tableau: list) -> int:
    c = Counter(tableau)
    return c["Maki Roll (1)"] + c["Maki Roll (2)"] * 2 + c["Maki Roll (3)"] * 3


def score_maki(maki_counts: list, player_count: int) -> list:
    """Return maki score for each player slot."""
    scores = [0] * len(maki_counts)
    if all(m == 0 for m in maki_counts):
        return scores
    first_pts = 4 if player_count == 2 else 6
    second_pts = 2 if player_count == 2 else 3
    uniq = sorted(set(maki_counts), reverse=True)
    first_v = uniq[0]
    first_p = [i for i, m in enumerate(maki_counts) if m == first_v]
    for i in first_p:
        scores[i] = first_pts // len(first_p)
    if len(uniq) > 1:
        second_v = uniq[1]
        second_p = [i for i, m in enumerate(maki_counts) if m == second_v]
        for i in second_p:
            scores[i] = second_pts // len(second_p)
    return scores


def score_nigiri(tableau: list) -> int:
    c = Counter(tableau)
    wasabi = c["Wasabi"]
    score = 0
    for nigiri, base in [("Squid Nigiri", 3), ("Salmon Nigiri", 2), ("Egg Nigiri", 1)]:
        for _ in range(c[nigiri]):
            if wasabi > 0:
                score += base * 3
                wasabi -= 1
            else:
                score += base
    return score


def score_tableau_no_maki(tableau: list) -> int:
    c = Counter(tableau)
    score = (c["Tempura"] // 2) * 5
    score += (c["Sashimi"] // 3) * 10
    d = c["Dumpling"]
    DUMP = [0, 1, 3, 6, 10, 15]
    score += DUMP[min(d, 5)]
    score += score_nigiri(tableau)
    return score


def score_pudding_endgame(pudding_counts: list, player_count: int) -> list:
    """Return pudding bonus/penalty per player."""
    scores = [0] * len(pudding_counts)
    if player_count == 1:
        return scores
    max_p = max(pudding_counts)
    min_p = min(pudding_counts)
    most = [i for i, p in enumerate(pudding_counts) if p == max_p]
    least = [i for i, p in enumerate(pudding_counts) if p == min_p]
    for i in most:
        scores[i] += 6 // len(most)
    # In 2-player games, penalty still applies
    if player_count > 2 or True:
        for i in least:
            scores[i] -= 6 // len(least)
    return scores


def total_score_for_state(tableaux: list, puddings: list,
                           player_count: int, end_of_game: bool) -> list:
    """Full score for each player given their tableaux and pudding counts."""
    n = len(tableaux)
    scores = [score_tableau_no_maki(t) for t in tableaux]
    maki_c = [count_maki(t) for t in tableaux]
    maki_s = score_maki(maki_c, player_count)
    for i in range(n):
        scores[i] += maki_s[i]
    if end_of_game:
        pud_s = score_pudding_endgame(puddings, player_count)
        for i in range(n):
            scores[i] += pud_s[i]
    return scores


# ── Greedy heuristic (used for rollouts and as standalone fallback) ───────────

def has_unused_wasabi(tableau: list) -> bool:
    wasabi = tableau.count("Wasabi")
    nigiri = sum(1 for c in tableau if "Nigiri" in c)
    return wasabi > nigiri


def heuristic_card_value(card: str, tableau: list, puddings: int,
                          my_maki: int, opp_makis: list,
                          player_count: int, turn: int,
                          cards_per_hand: int, hand: list,
                          opp_puds: list) -> float:
    counts = Counter(tableau)
    hand_size = len(hand)
    turns_left = cards_per_hand - turn - 1
    progress = turn / max(1, cards_per_hand)
    n = player_count

    if card == "Squid Nigiri":
        return 9.0 if has_unused_wasabi(tableau) else 3.0
    if card == "Salmon Nigiri":
        return 6.0 if has_unused_wasabi(tableau) else 2.0
    if card == "Egg Nigiri":
        return 3.0 if has_unused_wasabi(tableau) else 1.0

    if card == "Wasabi":
        if has_unused_wasabi(tableau):
            return 0.1
        nigiri_now = sum(1 for c in hand if "Nigiri" in c)
        if nigiri_now > 0:
            return 3.5
        return max(0.3, 4.0 * (1.0 - progress))

    if card == "Tempura":
        have = counts["Tempura"]
        if have % 2 == 1:
            return 5.0
        return max(0.3, 2.5 * (1.0 - progress))

    if card == "Sashimi":
        have = counts["Sashimi"]
        mod = have % 3
        if mod == 2:
            return 10.0
        if mod == 1:
            return max(0.3, 4.0 * (1.0 - progress))
        return max(0.3, 2.0 * (1.0 - progress))

    if card == "Dumpling":
        have = counts["Dumpling"]
        DUMP = [0, 1, 3, 6, 10, 15]
        return float(DUMP[min(have + 1, 5)] - DUMP[min(have, 5)])

    if card.startswith("Maki Roll"):
        pips = MAKI_PIPS[card]
        max_opp = max(opp_makis) if opp_makis else 0
        first_pts = 4 if n == 2 else 6
        second_pts = 2 if n == 2 else 3
        my_new = my_maki + pips
        if my_new > max_opp:
            raw = first_pts * 0.85
        elif my_new == max_opp:
            raw = first_pts * 0.5
        else:
            raw = second_pts * 0.3
        scale = max(0.3, 1.0 - progress)
        return raw * scale * (pips / 2.0)

    if card == "Pudding":
        max_opp = max(opp_puds) if opp_puds else 0
        min_opp = min(opp_puds) if opp_puds else 0
        base = {2: 5.0, 3: 3.5, 4: 2.5}.get(n, 2.0)
        if puddings < min_opp:
            base += 1.5
        elif puddings <= max_opp:
            base += 0.5
        return base

    if card == "Chopsticks":
        if hand_size <= 2:
            return 0.05
        return max(0.1, 2.0 * ((hand_size - 2) / cards_per_hand))

    return 0.5


def greedy_pick(hand: list, tableau: list, puddings: int,
                my_maki: int, opp_makis: list, opp_puds: list,
                player_count: int, turn: int, cards_per_hand: int) -> int:
    """Pure greedy card picker — used in MCTS rollouts."""
    if not hand:
        return 0
    best_val = -999
    best_idx = 0
    for i, card in enumerate(hand):
        v = heuristic_card_value(card, tableau, puddings, my_maki, opp_makis,
                                 player_count, turn, cards_per_hand, hand, opp_puds)
        if v > best_val:
            best_val = v
            best_idx = i
    return best_idx


# ── Simulation state (lightweight, copyable) ─────────────────────────────────

@dataclass
class SimState:
    """
    Lightweight game state for MCTS simulation.
    player_idx=0 is always 'us' from our perspective.
    """
    player_count: int
    cards_per_hand: int
    tableaux: list          # list of lists, one per player
    puddings: list          # list of ints, one per player
    hands: list             # list of lists (cards currently held)
    turn: int               # turns elapsed this round
    round_num: int
    my_idx: int = 0

    def is_round_over(self) -> bool:
        return all(len(h) == 0 for h in self.hands)

    def current_hand_size(self) -> int:
        return max(len(h) for h in self.hands) if self.hands else 0

    def rotate_hands(self):
        """Clockwise: hand[i] goes to player[i+1]."""
        n = self.player_count
        self.hands = [self.hands[(i - 1) % n] for i in range(n)]

    def apply_plays(self, plays: list):
        """
        plays: list of card indices, one per player (by seat).
        Update tableaux, puddings, then rotate hands.
        """
        cards_played = []
        for i, idx in enumerate(plays):
            card = self.hands[i][idx]
            cards_played.append(card)
            self.tableaux[i].append(card)
            if card == "Pudding":
                self.puddings[i] += 1

        # Remove played cards from hands
        for i, idx in enumerate(plays):
            self.hands[i] = [c for j, c in enumerate(self.hands[i]) if j != idx]

        self.turn += 1
        self.rotate_hands()

    def score_current(self, end_of_game: bool) -> list:
        return total_score_for_state(self.tableaux, self.puddings,
                                     self.player_count, end_of_game)

    def clone(self):
        return SimState(
            player_count=self.player_count,
            cards_per_hand=self.cards_per_hand,
            tableaux=[list(t) for t in self.tableaux],
            puddings=list(self.puddings),
            hands=[list(h) for h in self.hands],
            turn=self.turn,
            round_num=self.round_num,
            my_idx=self.my_idx,
        )


# ── MCTS ─────────────────────────────────────────────────────────────────────

class MCTSNode:
    __slots__ = ["move", "parent", "children", "visits", "value", "untried"]

    def __init__(self, move=None, parent=None, untried_moves=None):
        self.move = move          # card index we played to reach this node
        self.parent = parent
        self.children: list = []
        self.visits: int = 0
        self.value: float = 0.0
        self.untried: list = list(untried_moves) if untried_moves else []

    def ucb1(self, c: float = 1.4) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.value / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self, c: float = 1.4) -> "MCTSNode":
        return max(self.children, key=lambda n: n.ucb1(c))

    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0

    def is_terminal(self) -> bool:
        return len(self.children) == 0 and self.is_fully_expanded()


def rollout(state: SimState, end_of_game: bool) -> float:
    """
    Simulate to round end using greedy play for all players.
    Returns our score minus average opponent score (paranoid normalization).
    """
    s = state.clone()
    while not s.is_round_over():
        plays = []
        for i in range(s.player_count):
            hand = s.hands[i]
            if not hand:
                plays.append(0)
                continue
            tableau = s.tableaux[i]
            pud = s.puddings[i]
            my_maki = count_maki(tableau)
            opp_makis = [count_maki(s.tableaux[j]) for j in range(s.player_count) if j != i]
            opp_puds = [s.puddings[j] for j in range(s.player_count) if j != i]
            idx = greedy_pick(hand, tableau, pud, my_maki, opp_makis, opp_puds,
                              s.player_count, s.turn, s.cards_per_hand)
            plays.append(idx)
        s.apply_plays(plays)

    scores = s.score_current(end_of_game)
    my_score = scores[s.my_idx]
    opp_scores = [scores[i] for i in range(s.player_count) if i != s.my_idx]
    avg_opp = sum(opp_scores) / len(opp_scores) if opp_scores else 0
    return float(my_score - avg_opp)


def mcts_search(root_state: SimState, time_budget_s: float = 0.95,
                end_of_game: bool = False) -> int:
    """
    Run MCTS from root_state.
    Returns the best move (card index) for the current player (my_idx=0).
    """
    my_hand = root_state.hands[root_state.my_idx]
    if len(my_hand) == 1:
        return 0  # No choice

    root = MCTSNode(untried_moves=list(range(len(my_hand))))
    deadline = time.monotonic() + time_budget_s
    iterations = 0

    while time.monotonic() < deadline:
        # ── Selection ──────────────────────────────────────────────────────
        node = root
        state = root_state.clone()

        # If we can expand root, do so
        path_moves: list = []

        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            path_moves.append(node.move)

        # ── Expansion ──────────────────────────────────────────────────────
        if node.untried:
            my_move = random.choice(node.untried)
            node.untried.remove(my_move)

            # Simulate one full turn: our move + greedy opponent moves
            plays = _build_plays(state, my_move)
            state.apply_plays(plays)

            child = MCTSNode(move=my_move, parent=node,
                             untried_moves=list(range(len(state.hands[state.my_idx]))))
            node.children.append(child)
            node = child

        elif not node.children:
            # Terminal node — just rollout from here
            pass

        # ── Rollout ────────────────────────────────────────────────────────
        result = rollout(state, end_of_game)

        # ── Backpropagation ────────────────────────────────────────────────
        while node is not None:
            node.visits += 1
            node.value += result
            node = node.parent

        iterations += 1

    # Pick most visited child of root
    if not root.children:
        return 0
    best = max(root.children, key=lambda n: n.visits)
    return best.move


def _build_plays(state: SimState, my_move: int) -> list:
    """
    Build a list of moves for all players this turn.
    Our move is fixed; opponents use greedy (paranoid = they try to minimize us,
    implemented here as: they play their own optimal greedy card).
    """
    plays = []
    for i in range(state.player_count):
        if i == state.my_idx:
            plays.append(my_move)
        else:
            hand = state.hands[i]
            if not hand:
                plays.append(0)
                continue
            tableau = state.tableaux[i]
            pud = state.puddings[i]
            my_maki = count_maki(tableau)
            opp_makis = [count_maki(state.tableaux[j])
                         for j in range(state.player_count) if j != i]
            opp_puds = [state.puddings[j] for j in range(state.player_count) if j != i]
            idx = greedy_pick(hand, tableau, pud, my_maki, opp_makis, opp_puds,
                              state.player_count, state.turn, state.cards_per_hand)
            plays.append(idx)
    return plays


# ── Real game knowledge tracker ───────────────────────────────────────────────

@dataclass
class PlayerInfo:
    name: str
    tableau: list = field(default_factory=list)
    puddings: int = 0
    total_score: int = 0


class GameKnowledge:
    def __init__(self):
        self.my_name = ""
        self.player_count = 2
        self.cards_per_hand = 10
        self.round_num = 1
        self.turn = 0
        self.seats: list = []           # ordered seat list (clockwise)
        self.players: dict = {}         # name → PlayerInfo
        self.my_hand: list = []
        self.known_hands: dict = {}     # name → list or None
        self.prev_hand: list = []

    def my_info(self):
        return self.players.get(self.my_name, PlayerInfo(self.my_name))

    def my_tableau(self):
        return self.my_info().tableau

    def my_seat_idx(self):
        return next((i for i, n in enumerate(self.seats) if n == self.my_name), 0)

    def next_seat(self):
        if not self.seats:
            return ""
        idx = self.my_seat_idx()
        return self.seats[(idx + 1) % len(self.seats)]

    def reset_for_round(self, round_num: int):
        self.round_num = round_num
        self.turn = 0
        self.prev_hand = []
        self.known_hands = {}
        for p in self.players.values():
            p.tableau = []

    def build_sim_state(self) -> SimState:
        """
        Construct a SimState from current knowledge.
        Our hand is exact. Opponent hands: use known if available, else sample
        from the remaining deck.
        """
        n = self.player_count
        my_idx = self.my_seat_idx()

        # Compute 'used' cards (in all tableaux + our hand + all known hands)
        used = Counter()
        for p in self.players.values():
            used.update(p.tableau)
        used.update(self.my_hand)
        for nm, hand in self.known_hands.items():
            if hand is not None:
                used.update(hand)

        # Pool of unknown cards
        pool = Counter(FULL_DECK)
        # Remove all three rounds' worth of puddings that don't appear —
        # actually just remove what's been placed in this round's distribution
        pool.subtract(used)
        # Clamp negatives to 0 (rounding errors)
        pool = Counter({k: max(0, v) for k, v in pool.items()})
        pool_list = [c for c, cnt in pool.items() for _ in range(cnt)]
        random.shuffle(pool_list)

        # Build seat-ordered hands
        hands = []
        tableaux = []
        puddings = []
        for i, seat_name in enumerate(self.seats):
            tableaux.append(list(self.players[seat_name].tableau))
            puddings.append(self.players[seat_name].puddings)
            if seat_name == self.my_name:
                hands.append(list(self.my_hand))
            elif seat_name in self.known_hands and self.known_hands[seat_name] is not None:
                hands.append(list(self.known_hands[seat_name]))
            else:
                # Sample cards from pool for this opponent's hand
                opp_hand_size = len(self.my_hand)  # same hand size (same turn)
                sampled = pool_list[:opp_hand_size]
                pool_list = pool_list[opp_hand_size:]
                hands.append(sampled)

        return SimState(
            player_count=n,
            cards_per_hand=self.cards_per_hand,
            tableaux=tableaux,
            puddings=puddings,
            hands=hands,
            turn=self.turn,
            round_num=self.round_num,
            my_idx=my_idx,
        )


# ── Protocol ──────────────────────────────────────────────────────────────────

def parse_hand(msg: str) -> list:
    payload = msg[len("HAND "):].strip()
    cards = []
    for m in re.finditer(r'(\d+):(.*?)(?=\s+\d+:|$)', payload):
        cards.append(normalize(m.group(2).strip()))
    return cards


def parse_played(msg: str) -> dict:
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

class MCTSBot:
    def __init__(self, host: str, port: int, time_budget: float = 0.95):
        self.host = host
        self.port = port
        self.time_budget = time_budget
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

    def decide(self, hand: list, end_of_game: bool = False) -> tuple:
        """
        Run MCTS and return (action_str, card_played_str).
        action_str is the full command to send (PLAY n or CHOPSTICKS i j).
        """
        if len(hand) == 1:
            card = hand[0]
            self.gk.players[self.gk.my_name].tableau.append(card)
            return f"PLAY 0", card

        t0 = time.monotonic()
        state = self.gk.build_sim_state()
        best_idx = mcts_search(state, self.time_budget, end_of_game)
        elapsed = time.monotonic() - t0
        print(f"[MCTS] {elapsed:.3f}s — chose index {best_idx} ({hand[best_idx]})")

        # Check if we should use chopsticks instead
        # (chopsticks logic is a quick greedy check on top of MCTS card choice)
        has_chops = "Chopsticks" in self.gk.my_tableau()
        if has_chops and len(hand) >= 2:
            # Score all pairs quickly
            pair = self._best_chopsticks_pair(hand, best_idx)
            if pair:
                i, j = pair
                return f"CHOPSTICKS {i} {j}", f"{hand[i]}+{hand[j]}"

        return f"PLAY {best_idx}", hand[best_idx]

    def _best_chopsticks_pair(self, hand: list, mcts_best: int) -> Optional[tuple]:
        """
        Use chopsticks if pairing the MCTS best card with another high-value
        card gives significantly more value than playing a single card.
        """
        tableau = self.gk.my_tableau()
        puddings = self.gk.my_info().puddings
        my_maki = count_maki(tableau)
        opp_makis = [count_maki(p.tableau) for n, p in self.gk.players.items()
                     if n != self.gk.my_name]
        opp_puds = [p.puddings for n, p in self.gk.players.items()
                    if n != self.gk.my_name]

        vals = [heuristic_card_value(c, tableau, puddings, my_maki, opp_makis,
                                     self.gk.player_count, self.gk.turn,
                                     self.gk.cards_per_hand, hand, opp_puds)
                for c in hand]

        best_single = max(vals)
        best_pair_val = -999
        best_pair = None
        for i in range(len(hand)):
            for j in range(i + 1, len(hand)):
                combined = vals[i] + vals[j]
                if combined > best_pair_val:
                    best_pair_val = combined
                    best_pair = (i, j)

        if best_pair and best_pair_val >= best_single + 3.5:
            return best_pair
        return None

    def run(self, game_id: str, name: str):
        try:
            self.connect()
            self.gk.my_name = name
            self.send(f"JOIN {game_id} {name}")
            resp = self.recv_until(lambda m: m.startswith("WELCOME") or m.startswith("ERROR"))
            if not resp.startswith("WELCOME"):
                print(f"Failed to join: {resp}")
                return
            self.rejoin_token = resp.split()[3]
            self.send("READY")

            self.gk.players[name] = PlayerInfo(name)
            self.gk.seats.append(name)

            is_last_round = False

            while True:
                msg = self.recv()
                if not msg:
                    continue

                if msg.startswith("JOINED"):
                    jname = msg.split()[1]
                    if jname not in self.gk.players:
                        self.gk.players[jname] = PlayerInfo(jname)
                        self.gk.seats.append(jname)

                elif msg.startswith("GAME_START"):
                    pc = int(msg.split()[1])
                    self.gk.player_count = pc
                    self.gk.cards_per_hand = CARDS_PER_PLAYER.get(pc, 10)

                elif msg.startswith("ROUND_START"):
                    rn = int(msg.split()[1])
                    self.gk.reset_for_round(rn)
                    is_last_round = (rn == 3)

                elif msg.startswith("HAND"):
                    hand = parse_hand(msg)
                    self.gk.my_hand = hand
                    self.gk.prev_hand = list(hand)

                    # For last card of last round, flag end of game
                    end_of_game = is_last_round and len(hand) == 1
                    action, played = self.decide(hand, end_of_game)
                    self.send(action)

                    # Track our play in tableau
                    if "CHOPSTICKS" in action:
                        parts = action.split()
                        i, j = int(parts[1]), int(parts[2])
                        for card in [hand[i], hand[j]]:
                            self.gk.players[name].tableau.append(card)
                            if card == "Pudding":
                                self.gk.players[name].puddings += 1
                    else:
                        idx = int(action.split()[1])
                        card = hand[idx]
                        self.gk.players[name].tableau.append(card)
                        if card == "Pudding":
                            self.gk.players[name].puddings += 1

                elif msg.startswith("PLAYED"):
                    plays = parse_played(msg)
                    # Update opponent tableaux
                    for pname, card in plays.items():
                        if pname == name:
                            continue  # already tracked above
                        if pname in self.gk.players:
                            self.gk.players[pname].tableau.append(card)
                            if card == "Pudding":
                                self.gk.players[pname].puddings += 1
                    # Infer next player's hand
                    next_seat = self.gk.next_seat()
                    if next_seat and self.gk.prev_hand:
                        my_play = plays.get(name, "")
                        inferred = list(self.gk.prev_hand)
                        if my_play in inferred:
                            inferred.remove(my_play)
                        self.gk.known_hands[next_seat] = inferred
                    self.gk.turn += 1
                    self.gk.my_hand = []  # will be refreshed on next HAND

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
        print("Usage: python mcts_bot.py <host> <port> <game_id> <player_name>")
        sys.exit(1)
    bot = MCTSBot(sys.argv[1], int(sys.argv[2]))
    bot.run(sys.argv[3], sys.argv[4])


if __name__ == "__main__":
    main()
