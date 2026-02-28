from time import sleep

from ..sushi_go_client import SushiGoClient

HOST = "localhost"
PORT = 7878


CARD_CODES = {
    "TMP": "Tempura",
    "SSH": "Sashimi",
    "DMP": "Dumpling",
    "MK1": "Maki Roll (1)",
    "MK2": "Maki Roll (2)",
    "MK3": "Maki Roll (3)",
    "EGG": "Egg Nigiri",
    "SAL": "Salmon Nigiri",
    "SQD": "Squid Nigiri",
    "PUD": "Pudding",
    "WAS": "Wasabi",
    "CHP": "Chopsticks",
}


class OpponentState:
    hand_known: bool
    hand: list[str]
    played: list[str]

    def __init__(self):
        self.hand_known = False
        self.hand = []
        self.played = []

    def __repr__(self):
        return str(self.hand) + str(self.played)


class IterOne(SushiGoClient):
    name = "RANSACK"
    opps: dict[str, OpponentState] | None

    def __init__(self):
        self.opps = None
        super().__init__(HOST, PORT)

    def opp(self, name: str) -> OpponentState:
        if not self.opps:
            self.opps = {}
        if name not in self.opps:
            self.opps[name] = OpponentState()
        return self.opps[name]

    def all_opponent_hands(self) -> list[str]:
        cards = []
        if self.opps:
            for opp in self.opps.values():
                if opp.hand_known:
                    cards.extend(opp.hand)
        return cards

    def all_opponent_played(self) -> list[str]:
        cards = []
        if self.opps:
            for opp in self.opps.values():
                cards.extend(opp.played)
        return cards

    def choose_card(self, hand: list[str]) -> int:
        if not self.state:
            return 0

        played = self.state.played_cards
        turns_left = len(hand) - 1
        round_num = self.state.round
        opp_played = self.all_opponent_played()
        opp_hands = self.all_opponent_hands()

        # Our table counts
        our_tempura = played.count("Tempura")
        our_sashimi = played.count("Sashimi")
        our_dumplings = played.count("Dumpling")
        our_maki = sum(
            {"Maki Roll (1)": 1, "Maki Roll (2)": 2, "Maki Roll (3)": 3}.get(c, 0)
            for c in played
        )

        # Correct wasabi tracking (has_unused_wasabi in base class is buggy)
        wasabi_played = played.count("Wasabi")
        nigiri_played = sum(1 for c in played if "Nigiri" in c)
        unused_wasabis = max(0, wasabi_played - nigiri_played)

        # Opponent table totals (all opponents pooled)
        opp_tempura = opp_played.count("Tempura")
        opp_sashimi = opp_played.count("Sashimi")
        opp_maki = sum(
            {"Maki Roll (1)": 1, "Maki Roll (2)": 2, "Maki Roll (3)": 3}.get(c, 0)
            for c in opp_played
        )

        # Cards rotating toward us next turn
        incoming = opp_hands

        scores = {
            i: self._score(
                card,
                hand,
                turns_left,
                round_num,
                our_tempura,
                our_sashimi,
                our_dumplings,
                our_maki,
                unused_wasabis,
                opp_tempura,
                opp_sashimi,
                opp_maki,
                opp_played,
                incoming,
            )
            for i, card in enumerate(hand)
        }

        return max(scores, key=lambda i: scores[i])

    def _score(
        self,
        card,
        hand,
        turns_left,
        round_num,
        our_tempura,
        our_sashimi,
        our_dumplings,
        our_maki,
        unused_wasabis,
        opp_tempura,
        opp_sashimi,
        opp_maki,
        opp_played,
        incoming,
    ) -> float:
        # Chopsticks sucks!
        if card == "Chopsticks":
            return -100.0

        # ── Nigiri ──────────────────────────────────────────────────────
        if "Nigiri" in card:
            base = {"Squid Nigiri": 3.0, "Salmon Nigiri": 2.0, "Egg Nigiri": 1.0}[card]
            return base * 3 if unused_wasabis > 0 else base

        # ── Wasabi ──────────────────────────────────────────────────────
        if card == "Wasabi":
            if unused_wasabis > 0:
                return 0.3  # second unused wasabi is nearly worthless
            nigiri_in_hand = sum(1 for c in hand if "Nigiri" in c)
            nigiri_incoming = sum(1 for c in incoming if "Nigiri" in c)
            if nigiri_in_hand > 0:
                return 5.0  # can combo right now
            if turns_left >= 2 and nigiri_incoming > 0:
                return 3.5  # a nigiri is literally on its way to us
            if turns_left >= 1:
                return 1.5
            return 0.2

        # ── Tempura ─────────────────────────────────────────────────────
        if card == "Tempura":
            needed = 2 - (our_tempura % 2)
            if needed == 1:
                return 5.0  # completes the pair
            tempura_incoming = incoming.count("Tempura")
            if turns_left >= 2 and tempura_incoming > 0:
                return 4.0  # a second tempura is on its way
            if turns_left >= 2:
                return 2.5  # possible but uncertain
            if turns_left == 1:
                return 1.0
            return 0.0

        # ── Sashimi ─────────────────────────────────────────────────────
        if card == "Sashimi":
            progress = our_sashimi % 3
            needed = 3 - progress

            # Denial: opponents are one sashimi away from 10 pts — take it
            if opp_sashimi % 3 == 2:
                return 8.0

            if needed == 1:
                return 10.0  # completes our set

            sashimi_incoming = incoming.count("Sashimi")
            if needed == 2:
                if turns_left >= 3 and sashimi_incoming > 0:
                    return 4.5
                if turns_left >= 3:
                    return 2.5
                return 0.5
            else:  # need all 3
                if turns_left >= 5 and sashimi_incoming >= 2:
                    return 4.0
                if turns_left >= 5:
                    return 2.0
                return 0.0

        # ── Dumplings ───────────────────────────────────────────────────
        if card == "Dumpling":
            marginals = [1, 2, 3, 4, 5]
            return float(marginals[min(our_dumplings, 4)])

        # ── Maki Rolls ──────────────────────────────────────────────────
        if "Maki Roll" in card:
            icons = {"Maki Roll (1)": 1, "Maki Roll (2)": 2, "Maki Roll (3)": 3}[card]
            new_total = our_maki + icons
            if new_total > opp_maki:
                val = 6.0
            elif new_total == opp_maki:
                val = 4.0
            else:
                val = icons * 1.2
            return val * (icons / 3.0)

        # ── Pudding ─────────────────────────────────────────────────────
        if card == "Pudding":
            our_puddings = self.state.puddings if self.state else 0
            opp_pudding_counts = [
                opp.played.count("Pudding") for opp in (self.opps or {}).values()
            ]
            max_opp = max(opp_pudding_counts) if opp_pudding_counts else 0
            min_opp = min(opp_pudding_counts) if opp_pudding_counts else 0
            if round_num == 3:
                if our_puddings + 1 > max_opp:
                    return 5.0
                if our_puddings <= min_opp:
                    return 4.5
                return 2.5
            if round_num == 2:
                return 2.8 if our_puddings <= min_opp else 1.8
            return 2.0

        # ── Chopsticks ──────────────────────────────────────────────────
        if card == "Chopsticks":
            if turns_left >= 5:
                return 3.0
            if turns_left >= 3:
                return 1.5
            return 0.0

        return 0.0

    def post_play(self):
        # Rotate hands.
        if not self.state:
            return
        if not self.opps:
            return
        last_hand = []
        for i, opp in enumerate(self.opps.values()):
            current_hand = opp.hand
            if i == 0:
                opp.hand_known = True
                opp.hand = self.hand_after
            else:
                opp.hand = last_hand.copy()
            last_hand = current_hand
        print("\n\n")
        print(self.opps)

    def handle_message(self, message: str):
        """Handle a message from the server."""
        if message.startswith("HAND"):
            self.post_play()
            self.parse_hand(message)
        elif message.startswith("ROUND_START"):
            parts = message.split()
            if self.state:
                self.state.round = int(parts[1])
                self.state.turn = 1
                self.state.has_chopsticks = False
                if not self.state.played_cards:
                    self.state.played_cards = []
        elif message.startswith("PLAYED"):
            # Cards were revealed, next turn
            if self.state:
                self.state.turn += 1
            cards = message.split(maxsplit=1)[1]
            cards = cards.split("; ")
            for card in cards:
                card = card.split(":")
                player_name = card[0]
                card_name = card[1]
                if player_name == self.name:
                    continue
                opp = self.opp(player_name)
                for card in card_name.split(","):
                    if card == "CHP":
                        continue
                    card = CARD_CODES[card]
                    if card in opp.hand:
                        opp.hand.remove(card)
                    opp.played.append(card)
        elif message.startswith("ROUND_END"):
            # Round ended
            if self.state:
                self.state.played_cards = [
                    card for card in self.state.played_cards if card == "Pudding"
                ]
            if self.opps:
                for opp in self.opps.values():
                    opp.played = [card for card in opp.played if card == "Pudding"]

        elif message.startswith("GAME_END"):
            print("Game over!")
            return False
        elif message.startswith("WAITING"):
            # Our move was accepted, waiting for others
            pass
        return True
