#!/usr/bin/env python3
"""
Smart Sushi Go Bot — card counting + opponent modeling + denial strategy.

Extends SushiGoClient by overriding handle_message to track:
  - Every card ever seen in our hand this round (card counting)
  - Every card opponents have played (from PLAYED messages)
  - Opponent progress toward sets (sashimi, tempura, maki)

Then choose_card scores each option with all of that context.

Usage (same args as sushi_go_client.py):
    python my_bot.py <host> <port> <game_id> <player_name>
    python my_bot.py <game_id> <player_name>
"""

import re
import sys
from collections import Counter, defaultdict

from ..sushi_go_client import SushiGoClient

HOST = "localhost"
PORT = 7878

# Full deck composition (fixed by the official rules)
FULL_DECK: dict[str, int] = {
    "Tempura": 14,
    "Sashimi": 14,
    "Dumpling": 14,
    "Maki Roll (1)": 6,
    "Maki Roll (2)": 12,
    "Maki Roll (3)": 8,
    "Egg Nigiri": 5,
    "Salmon Nigiri": 10,
    "Squid Nigiri": 5,
    "Pudding": 10,
    "Wasabi": 6,
    "Chopsticks": 4,
}


def maki_icons(card: str) -> int:
    return {"Maki Roll (1)": 1, "Maki Roll (2)": 2, "Maki Roll (3)": 3}.get(card, 0)


class ClaudeTwo(SushiGoClient):
    def __init__(self, host: str = HOST, port: int = PORT):
        super().__init__(host, port)
        self._player_count: int = 2

        # Per-round state (reset on ROUND_START)
        # Max count of each card we've ever seen simultaneously in our hand
        self._seen_in_hand: Counter = Counter()
        # Cards confirmed played to any table (ours + opponents), from PLAYED msgs
        self._confirmed_played: Counter = Counter()
        # Opponent name -> list of cards on their table this round
        self._opponent_table: dict[str, list[str]] = {}

        # Cross-round pudding tracking for opponents
        self._opponent_puddings: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    #  Message handling overrides                                          #
    # ------------------------------------------------------------------ #

    def handle_message(self, message: str) -> bool:
        result = super().handle_message(message)

        if message.startswith("GAME_START"):
            parts = message.split()
            if len(parts) > 1:
                try:
                    self._player_count = int(parts[1])
                except ValueError:
                    pass

        elif message.startswith("ROUND_START"):
            self._reset_round_tracking()

        elif message.startswith("HAND"):
            self._record_seen_hand(message)

        elif message.startswith("PLAYED"):
            self._parse_played(message)

        elif message.startswith("ROUND_END"):
            self._accumulate_opponent_puddings()

        return result

    def _reset_round_tracking(self):
        self._seen_in_hand = Counter()
        self._confirmed_played = Counter()
        self._opponent_table = {}

    def _record_seen_hand(self, message: str):
        """
        Card counting: track the maximum number of each card type we've
        ever held simultaneously. This gives us a lower bound on how many
        of each card exist in the circulating pool this round.
        """
        payload = message[len("HAND ") :]
        cards = []
        for match in re.finditer(r"\d+:(.*?)(?=\s\d+:|$)", payload):
            cards.append(match.group(1).strip())

        current_counts = Counter(cards)
        for card, count in current_counts.items():
            if count > self._seen_in_hand[card]:
                self._seen_in_hand[card] = count

    def _parse_played(self, message: str):
        """
        Parse PLAYED <player>:<card>[,<card>]; <player>:<card>; ...
        Update opponent tables and confirmed-played counts.
        """
        payload = message[len("PLAYED") :].strip()
        for segment in payload.split(";"):
            segment = segment.strip()
            if ":" not in segment:
                continue
            name, cards_str = segment.split(":", 1)
            name = name.strip()
            cards = [c.strip() for c in cards_str.split(",") if c.strip()]

            if name not in self._opponent_table:
                self._opponent_table[name] = []
            self._opponent_table[name].extend(cards)

            for card in cards:
                self._confirmed_played[card] += 1

    def _accumulate_opponent_puddings(self):
        """At round end, add puddings from each opponent's table to their running total."""
        for name, table in self._opponent_table.items():
            self._opponent_puddings[name] = self._opponent_puddings.get(
                name, 0
            ) + table.count("Pudding")

    # ------------------------------------------------------------------ #
    #  Card-counting helpers                                               #
    # ------------------------------------------------------------------ #

    def _cards_still_live(self) -> Counter:
        """
        Estimate cards still in circulation (in someone's hand, not yet played).
        Lower bound: deck total minus all confirmed plays.
        We DON'T subtract our current hand because they're still live for us.
        """
        live: Counter = Counter()
        for card, total in FULL_DECK.items():
            remaining = total - self._confirmed_played[card]
            live[card] = max(0, remaining)
        return live

    # ------------------------------------------------------------------ #
    #  Opponent analysis                                                   #
    # ------------------------------------------------------------------ #

    def _opponent_needs_sashimi(self) -> bool:
        """Any opponent is 1 sashimi away from completing a set of 3."""
        for table in self._opponent_table.values():
            if table.count("Sashimi") % 3 == 2:
                return True
        return False

    def _opponent_needs_tempura(self) -> bool:
        """Any opponent is 1 tempura away from completing a pair."""
        for table in self._opponent_table.values():
            if table.count("Tempura") % 2 == 1:
                return True
        return False

    def _max_opponent_maki(self) -> int:
        if not self._opponent_table:
            return 0
        return max(
            sum(maki_icons(c) for c in table) for table in self._opponent_table.values()
        )

    def _our_maki_icons(self) -> int:
        if not self.state:
            return 0
        return sum(maki_icons(c) for c in self.state.played_cards)

    def _opponent_pudding_counts(self) -> list[int]:
        return list(self._opponent_puddings.values())

    # ------------------------------------------------------------------ #
    #  Main decision                                                       #
    # ------------------------------------------------------------------ #

    def choose_card(self, hand: list[str]) -> int:
        if not hand:
            return 0

        state = self.state
        played = state.played_cards if state else []
        round_num = state.round if state else 1
        turns_left = len(hand) - 1  # turns remaining after this pick

        tempura_count = played.count("Tempura")
        sashimi_count = played.count("Sashimi")
        dumpling_count = played.count("Dumpling")
        our_puddings = played.count("Pudding") + (state.puddings if state else 0)

        # Count unused wasabis on our table (each needs its own nigiri)
        wasabi_played = sum(1 for c in played if c == "Wasabi")
        nigiri_played = sum(1 for c in played if "Nigiri" in c)
        unused_wasabis = max(0, wasabi_played - nigiri_played)

        live = self._cards_still_live()

        # Score every card in hand
        scores = {
            i: self._score(
                card,
                hand,
                turns_left,
                round_num,
                tempura_count,
                sashimi_count,
                dumpling_count,
                our_puddings,
                unused_wasabis,
                live,
            )
            for i, card in enumerate(hand)
        }

        best_idx = max(scores, key=lambda i: scores[i])
        best_score = scores[best_idx]

        # Check if denying an opponent is better than our best pick
        denial = self._denial_pick(hand, best_score)
        if denial is not None:
            return denial

        return best_idx

    def _score(
        self,
        card,
        hand,
        turns_left,
        round_num,
        tempura_count,
        sashimi_count,
        dumpling_count,
        our_puddings,
        unused_wasabis,
        live,
    ) -> float:

        # ---- Nigiri ----
        if "Nigiri" in card:
            base = {"Squid Nigiri": 3.0, "Salmon Nigiri": 2.0, "Egg Nigiri": 1.0}[card]
            return base * 3 if unused_wasabis > 0 else base

        # ---- Wasabi ----
        if card == "Wasabi":
            if unused_wasabis > 0:
                return 0.3  # second unused wasabi is nearly worthless

            nigiri_in_hand = sum(1 for c in hand if "Nigiri" in c)
            squid_live = live["Squid Nigiri"]
            salmon_live = live["Salmon Nigiri"]
            total_nigiri_live = squid_live + salmon_live + live["Egg Nigiri"]

            # Expected gain from tripling the best nigiri we'll likely see
            if squid_live + salmon_live > 0:
                expected_bonus = (squid_live * 6 + salmon_live * 4) / (
                    squid_live + salmon_live
                )
            else:
                expected_bonus = 2.0

            if nigiri_in_hand > 0:
                return expected_bonus * 2.0  # Can combo right now
            if turns_left >= 2 and total_nigiri_live > 0:
                return expected_bonus * 1.4
            if turns_left >= 1 and total_nigiri_live > 0:
                return expected_bonus * 0.7
            return 0.2

        # ---- Tempura ----
        if card == "Tempura":
            needed = 2 - (tempura_count % 2)
            if needed == 1:
                return 5.0  # Completes the pair
            # Need one more — how likely is it to come back?
            prob = min(1.0, live["Tempura"] / max(1, self._player_count * 2))
            if turns_left >= 3:
                return 5.0 * prob * 0.85
            if turns_left == 2:
                return 5.0 * prob * 0.55
            if turns_left == 1:
                return 5.0 * prob * 0.25
            return 0.0

        # ---- Sashimi ----
        if card == "Sashimi":
            progress = sashimi_count % 3
            needed = 3 - progress
            if needed == 1:
                return 10.0  # Completes the set
            sashimi_avail = live["Sashimi"]
            prob = min(1.0, sashimi_avail / max(1, self._player_count * needed))
            if needed == 2:
                if turns_left >= 3:
                    return 10.0 * prob * 0.65
                if turns_left >= 2:
                    return 10.0 * prob * 0.35
                return 0.0
            else:  # need all 3
                if turns_left >= 5:
                    return 10.0 * prob * 0.45
                if turns_left >= 4:
                    return 10.0 * prob * 0.25
                return 0.0

        # ---- Dumplings ----
        if card == "Dumpling":
            marginals = [1, 2, 3, 4, 5]
            return float(marginals[min(dumpling_count, 4)])

        # ---- Maki Rolls ----
        if "Maki Roll" in card:
            icons = maki_icons(card)
            our_total = self._our_maki_icons() + icons
            max_opp = self._max_opponent_maki()

            if our_total > max_opp:
                competitive_val = 6.0
            elif our_total == max_opp:
                competitive_val = 4.0  # Likely split (3 pts each)
            else:
                competitive_val = max(1.0, icons * 1.5)

            # Scale by how many icons this card brings
            return competitive_val * (icons / 3.0)

        # ---- Pudding ----
        if card == "Pudding":
            opp_counts = self._opponent_pudding_counts()
            max_opp_p = max(opp_counts) if opp_counts else 0
            min_opp_p = min(opp_counts) if opp_counts else 0
            after_pick = our_puddings + 1

            if round_num == 3:
                if after_pick > max_opp_p:
                    return 5.0  # Secure the +6
                if our_puddings <= min_opp_p:
                    return 4.5  # Avoid the -6
                return 2.5
            if round_num == 2:
                return 2.8 if our_puddings <= min_opp_p else 1.8
            return 2.2  # Round 1

        # ---- Chopsticks ----
        if card == "Chopsticks":
            if turns_left >= 6:
                return 3.0
            if turns_left >= 4:
                return 2.0
            if turns_left >= 2:
                return 0.8
            return 0.0

        return 0.0

    # ------------------------------------------------------------------ #
    #  Denial logic                                                        #
    # ------------------------------------------------------------------ #

    def _denial_pick(self, hand: list[str], best_score: float) -> int | None:
        """
        If we can deny an opponent a card that completes a big score,
        and the denial value exceeds what we sacrifice, take that card instead.

        Denial values:
          Sashimi (deny 10pt completion)  → denial worth ~8 pts net swing
          Tempura (deny 5pt completion)   → denial worth ~4 pts net swing
        """
        candidates: list[tuple[float, int]] = []

        for i, card in enumerate(hand):
            denial_value = 0.0

            if card == "Sashimi" and self._opponent_needs_sashimi():
                # Net swing = 10 pts we prevent them from scoring
                denial_value = 10.0

            elif card == "Tempura" and self._opponent_needs_tempura():
                denial_value = 5.0

            if denial_value > 0:
                net_gain = denial_value - best_score
                if net_gain > 1.5:  # Only deny if it's clearly worth it
                    candidates.append((net_gain, i))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        return candidates[0][1]
