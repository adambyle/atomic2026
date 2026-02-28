from bots.adam.iterone import IterOne, OpponentState

HOST = "10.8.1.191"
PORT = 7878


class Iter4(IterOne):
    name = "ChopstickHater"

    def choose_card(self, hand: list[str]) -> int:
        if not self.state:
            return 0

        played = self.state.played_cards
        turns_left = len(hand) - 1
        round_num = self.state.round
        opp_hands = self.all_opponent_hands()

        our_tempura = played.count("Tempura")
        our_sashimi = played.count("Sashimi")
        our_dumplings = played.count("Dumpling")
        our_maki = sum(
            {"Maki Roll (1)": 1, "Maki Roll (2)": 2, "Maki Roll (3)": 3}.get(c, 0)
            for c in played
        )

        wasabi_played = played.count("Wasabi")
        nigiri_played = sum(1 for c in played if "Nigiri" in c)
        unused_wasabis = max(0, wasabi_played - nigiri_played)

        # Per-opponent stats (the key addition over IterOne)
        opp_states = list((self.opps or {}).values())

        # Worst-case individual opponent counts (not pooled)
        max_opp_maki = max(
            (
                sum(
                    {"Maki Roll (1)": 1, "Maki Roll (2)": 2, "Maki Roll (3)": 3}.get(
                        c, 0
                    )
                    for c in o.played
                )
                for o in opp_states
            ),
            default=0,
        )
        max_opp_sashimi_progress = max(
            (o.played.count("Sashimi") % 3 for o in opp_states), default=0
        )
        max_opp_tempura_progress = max(
            (o.played.count("Tempura") % 2 for o in opp_states), default=0
        )
        max_opp_pudding = max(
            (o.played.count("Pudding") for o in opp_states), default=0
        )
        min_opp_pudding = min(
            (o.played.count("Pudding") for o in opp_states), default=0
        )

        incoming = opp_hands

        scores = {
            i: self._score4(
                card,
                hand,
                turns_left,
                round_num,
                our_tempura,
                our_sashimi,
                our_dumplings,
                our_maki,
                unused_wasabis,
                incoming,
                max_opp_maki,
                max_opp_sashimi_progress,
                max_opp_tempura_progress,
                max_opp_pudding,
                min_opp_pudding,
            )
            for i, card in enumerate(hand)
        }

        return max(scores, key=lambda i: scores[i])

    def _score4(
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
        incoming,
        max_opp_maki,
        max_opp_sashimi_progress,
        max_opp_tempura_progress,
        max_opp_pudding,
        min_opp_pudding,
    ) -> float:

        if card == "Chopsticks":
            return -100.0

        # ── Nigiri ──────────────────────────────────────────────────────
        if "Nigiri" in card:
            base = {"Squid Nigiri": 3.0, "Salmon Nigiri": 2.0, "Egg Nigiri": 1.0}[card]
            return base * 3 if unused_wasabis > 0 else base

        # ── Wasabi ──────────────────────────────────────────────────────
        if card == "Wasabi":
            if unused_wasabis > 0:
                return 0.3
            nigiri_in_hand = sum(1 for c in hand if "Nigiri" in c)
            nigiri_incoming = sum(1 for c in incoming if "Nigiri" in c)
            if nigiri_in_hand > 0:
                return 5.0
            if turns_left >= 2 and nigiri_incoming > 0:
                return 3.5
            if turns_left >= 1:
                return 1.5
            return 0.2

        # ── Tempura ─────────────────────────────────────────────────────
        if card == "Tempura":
            needed = 2 - (our_tempura % 2)
            if needed == 1:
                return 5.0
            # Deny if any single opponent is one away from completing a pair
            if max_opp_tempura_progress == 1:
                return 4.0
            tempura_incoming = incoming.count("Tempura")
            if turns_left >= 2 and tempura_incoming > 0:
                return 4.0
            if turns_left >= 2:
                return 2.5
            if turns_left == 1:
                return 1.0
            return 0.0

        # ── Sashimi ─────────────────────────────────────────────────────
        if card == "Sashimi":
            progress = our_sashimi % 3
            needed = 3 - progress

            # Deny if any single opponent is one away from completing a set
            if max_opp_sashimi_progress == 2:
                return 8.0

            if needed == 1:
                return 10.0

            sashimi_incoming = incoming.count("Sashimi")
            if needed == 2:
                if turns_left >= 3 and sashimi_incoming > 0:
                    return 4.5
                if turns_left >= 3:
                    return 2.5
                return 0.5
            else:
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
            # Compare against the single best opponent, not the pool
            if new_total > max_opp_maki:
                val = 6.0
            elif new_total == max_opp_maki:
                val = 4.0
            else:
                val = icons * 1.2
            return val * (icons / 3.0)

        # ── Pudding ─────────────────────────────────────────────────────
        if card == "Pudding":
            our_puddings = self.state.puddings if self.state else 0
            if round_num == 3:
                if our_puddings + 1 > max_opp_pudding:
                    return 5.0
                if our_puddings <= min_opp_pudding:
                    return 4.5
                return 2.5
            if round_num == 2:
                return 2.8 if our_puddings <= min_opp_pudding else 1.8
            return 2.0

        return 0.0
