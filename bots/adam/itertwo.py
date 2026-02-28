from bots.adam.iterone import IterOne


class IterTwo(IterOne):
    name = "RANSACK_V2"

    # ─────────────────────────────────────────────────────────────
    # Public entry
    # ─────────────────────────────────────────────────────────────

    def choose_card(self, hand: list[str]) -> int:
        if not self.state:
            return 0

        context = self._build_context(hand)

        scores = {i: self._score_card(card, context) for i, card in enumerate(hand)}

        return max(scores, key=lambda i: scores[i])

    # ─────────────────────────────────────────────────────────────
    # Context builder (cleaner than long param lists)
    # ─────────────────────────────────────────────────────────────

    def _build_context(self, hand: list[str]) -> dict:
        played = self.state.played_cards
        turns_left = len(hand) - 1

        wasabi_played = played.count("Wasabi")
        nigiri_played = sum(1 for c in played if "Nigiri" in c)
        unused_wasabis = max(0, wasabi_played - nigiri_played)

        return {
            "hand": hand,
            "turns_left": turns_left,
            "round": self.state.round,
            "our_played": played,
            "unused_wasabis": unused_wasabis,
        }

    # ─────────────────────────────────────────────────────────────
    # Core scoring
    # ─────────────────────────────────────────────────────────────

    def _score_card(self, card: str, ctx: dict) -> float:
        score = 0.0

        score += self._self_value(card, ctx)
        score += self._denial_value(card, ctx)

        return score

    # ─────────────────────────────────────────────────────────────
    # Selfish scoring (mostly IterOne logic, simplified)
    # ─────────────────────────────────────────────────────────────

    def _self_value(self, card: str, ctx: dict) -> float:
        played = ctx["our_played"]
        turns_left = ctx["turns_left"]
        unused_wasabis = ctx["unused_wasabis"]

        our_sashimi = played.count("Sashimi")
        our_tempura = played.count("Tempura")
        our_dumplings = played.count("Dumpling")

        our_maki = sum(
            {"Maki Roll (1)": 1, "Maki Roll (2)": 2, "Maki Roll (3)": 3}.get(c, 0)
            for c in played
        )

        # ── Nigiri ──
        if "Nigiri" in card:
            base = {"Squid Nigiri": 3.0, "Salmon Nigiri": 2.0, "Egg Nigiri": 1.0}[card]
            return base * 3 if unused_wasabis > 0 else base

        # ── Sashimi ──
        if card == "Sashimi":
            progress = our_sashimi % 3
            if progress == 2:
                return 10.0
            if turns_left >= 4:
                return 3.0
            return 0.5

        # ── Tempura ──
        if card == "Tempura":
            if our_tempura % 2 == 1:
                return 5.0
            return 2.0 if turns_left >= 2 else 0.5

        # ── Dumpling ──
        if card == "Dumpling":
            marginals = [1, 2, 3, 4, 5]
            return float(marginals[min(our_dumplings, 4)])

        # ── Maki ──
        if "Maki Roll" in card:
            icons = {"Maki Roll (1)": 1, "Maki Roll (2)": 2, "Maki Roll (3)": 3}[card]
            return icons * 1.5

        # ── Pudding ──
        if card == "Pudding":
            return 2.0

        # ── Wasabi ──
        if card == "Wasabi":
            if unused_wasabis > 0:
                return 0.2
            return 3.0

        return 0.0

    # ─────────────────────────────────────────────────────────────
    # Individual opponent denial scoring
    # ─────────────────────────────────────────────────────────────

    def _denial_value(self, card: str, ctx: dict) -> float:
        if not self.opps:
            return 0.0

        denial = 0.0

        for opp in self.opps.values():
            denial += self._denial_against_opponent(card, opp, ctx)

        return denial

    def _denial_against_opponent(
        self, card: str, opp: OpponentState, ctx: dict
    ) -> float:
        played = opp.played

        # ── Sashimi denial ──
        if card == "Sashimi":
            if played.count("Sashimi") % 3 == 2:
                return 8.0  # high priority block
            if played.count("Sashimi") % 3 == 1:
                return 3.0

        # ── Tempura denial ──
        if card == "Tempura":
            if played.count("Tempura") % 2 == 1:
                return 4.0

        # ── Maki denial ──
        if "Maki Roll" in card:
            icons = {"Maki Roll (1)": 1, "Maki Roll (2)": 2, "Maki Roll (3)": 3}[card]
            opp_maki = sum(
                {"Maki Roll (1)": 1, "Maki Roll (2)": 2, "Maki Roll (3)": 3}.get(c, 0)
                for c in played
            )
            our_maki = sum(
                {"Maki Roll (1)": 1, "Maki Roll (2)": 2, "Maki Roll (3)": 3}.get(c, 0)
                for c in ctx["our_played"]
            )

            if opp_maki > our_maki:
                return icons * 1.2

        # ── Pudding denial (round 3 critical) ──
        if card == "Pudding" and ctx["round"] == 3:
            opp_pudding = played.count("Pudding")
            our_pudding = self.state.puddings
            if opp_pudding >= our_pudding:
                return 4.0

        return 0.0
