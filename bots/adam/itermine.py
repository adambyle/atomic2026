from dataclasses import dataclass, field

from bots.adam.iterone import IterOne, OpponentState


@dataclass
class Weights:
    # ── Nigiri base values ───────────────────────────────────────────────
    squid_nigiri: float = 3.0
    salmon_nigiri: float = 2.0
    egg_nigiri: float = 1.0

    # Multiplier applied when we have an unused wasabi
    wasabi_nigiri_multiplier: float = 3.0

    # ── Wasabi ───────────────────────────────────────────────────────────
    # Base value when nigiri is already in hand (combo guaranteed this turn)
    wasabi_nigiri_in_hand: float = 5.5
    # Value when a good nigiri (squid/salmon) is visible in incoming hands
    wasabi_squid_salmon_incoming: float = 4.0
    # Value when any nigiri is incoming
    wasabi_any_nigiri_incoming: float = 2.5
    # Value with turns left but nothing visible
    wasabi_blind: float = 1.2
    # Penalty for a second unused wasabi (nearly worthless)
    wasabi_already_unused: float = 0.2

    # ── Tempura ──────────────────────────────────────────────────────────
    # Value when this card completes a pair (guaranteed 5 pts)
    tempura_completes_pair: float = 5.0
    # Value when a second tempura is visible in incoming hands
    tempura_pair_incoming: float = 3.5
    # Value with turns left but nothing visible
    tempura_speculative: float = 1.5
    # Value with only 1 turn left and no pair yet
    tempura_one_turn_left: float = 0.5
    # Value when it's the last card (no turns left)
    tempura_dead: float = 0.0

    # ── Sashimi ──────────────────────────────────────────────────────────
    # Value when this card completes a set of 3 (guaranteed 10 pts)
    sashimi_completes_set: float = 10.0
    # Value when we need 1 more and it's visible incoming
    sashimi_one_needed_incoming: float = 6.0
    # Value when we need 1 more but nothing visible
    sashimi_one_needed_blind: float = 2.5
    # Value when we need 2 more and enough are visible
    sashimi_two_needed_incoming: float = 4.0
    # Value when we need 2 more, speculative
    sashimi_two_needed_speculative: float = 1.5
    # Value when starting fresh on a new set
    sashimi_fresh_start: float = 0.8

    # ── Dumplings ────────────────────────────────────────────────────────
    # Marginal value of the Nth dumpling (index N = going from N to N+1 dumplings)
    dumpling_marginals: list = field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 5.0])

    # ── Maki Rolls ───────────────────────────────────────────────────────
    # Value-per-icon when this card puts us in the lead
    maki_lead_per_icon: float = 2.5
    # Value-per-icon when this card ties the leader
    maki_tie_per_icon: float = 1.2
    # Value-per-icon when trailing
    maki_trailing_per_icon: float = 0.8
    # Incoming-maki icon threshold to trigger the crowded penalty
    maki_crowded_threshold: float = 5.0
    # Penalty applied when incoming hands are maki-heavy
    maki_crowded_penalty: float = -1.5

    # ── Pudding ──────────────────────────────────────────────────────────
    pudding_round1: float = 1.5
    pudding_round2: float = 2.2
    pudding_r3_taking_lead: float = 5.5
    pudding_r3_avoiding_last: float = 4.5
    pudding_r3_neutral: float = 2.0
    # Bonus when opponents hold lots of pudding (race heats up)
    pudding_race_bonus: float = 1.2

    # ── Chopsticks ───────────────────────────────────────────────────────
    chopsticks_value: float = -100.0

    # ── Denial ───────────────────────────────────────────────────────────
    # Global multiplier applied to all denial values before adding to total
    denial_weight: float = 0.55
    denial_sashimi_complete: float = 9.0
    denial_sashimi_partial: float = 2.0  # opponent has 1/3 sashimi
    denial_tempura_complete: float = 4.5
    denial_wasabi_nigiri: float = 3.0
    denial_maki_per_icon: float = 1.2
    denial_pudding_r3: float = 3.5


class IterMine(IterOne):
    name = "MYRANSACK"

    def __init__(self, weights=None):
        self.weights = weights or Weights()
        super().__init__()

    # ── Utilities ────────────────────────────────────────────────────────

    def all_opponent_hands(self):
        cards = []
        if self.opps:
            for opp in self.opps.values():
                if opp.hand_known:
                    cards.extend(opp.hand)
        return cards

    def _maki_icons(self, card):
        return {"Maki Roll (1)": 1, "Maki Roll (2)": 2, "Maki Roll (3)": 3}.get(card, 0)

    def _count_maki(self, cards):
        return sum(self._maki_icons(c) for c in cards)

    # ── Main chooser ──────────────────────────────────────────────────────

    def choose_card(self, hand):
        incoming = self.all_opponent_hands()
        scores = {
            i: self._opportunity_score(card, hand, incoming)
            + self.weights.denial_weight * self._denial_score(card)
            for i, card in enumerate(hand)
        }
        print(scores)
        return max(scores, key=lambda i: scores[i])

    # ── Opportunity score ─────────────────────────────────────────────────

    def _opportunity_score(self, card, hand, incoming):
        if not self.state:
            return 0.0

        w = self.weights
        played = self.state.played_cards or []
        turns_left = len(hand) - 1
        round_num = self.state.round

        wasabi_played = played.count("Wasabi")
        nigiri_played = sum(1 for c in played if "Nigiri" in c)
        unused_wasabis = max(0, wasabi_played - nigiri_played)

        # ── Nigiri ──────────────────────────────────────────────────────
        if "Nigiri" in card:
            base = {
                "Squid Nigiri": w.squid_nigiri,
                "Salmon Nigiri": w.salmon_nigiri,
                "Egg Nigiri": w.egg_nigiri,
            }[card]
            return base * w.wasabi_nigiri_multiplier if unused_wasabis > 0 else base

        # ── Wasabi ──────────────────────────────────────────────────────
        if card == "Wasabi":
            if unused_wasabis > 0:
                return w.wasabi_already_unused
            nigiri_in_hand = sum(1 for c in hand if "Nigiri" in c)
            if nigiri_in_hand > 0:
                # Wasabi now, nigiri next turn from our current hand
                return w.wasabi_nigiri_in_hand
            squid_salmon_incoming = incoming.count("Squid Nigiri") + incoming.count(
                "Salmon Nigiri"
            )
            if squid_salmon_incoming > 0 and turns_left >= 1:
                return w.wasabi_squid_salmon_incoming
            if sum(1 for c in incoming if "Nigiri" in c) > 0 and turns_left >= 1:
                return w.wasabi_any_nigiri_incoming
            if turns_left >= 2:
                return w.wasabi_blind
            return w.wasabi_already_unused

        # ── Tempura ─────────────────────────────────────────────────────
        if card == "Tempura":
            if played.count("Tempura") % 2 == 1:
                return w.tempura_completes_pair
            if turns_left == 0:
                return w.tempura_dead
            if incoming.count("Tempura") >= 1 and turns_left >= 1:
                return w.tempura_pair_incoming
            if turns_left >= 2:
                return w.tempura_speculative
            if turns_left == 1:
                return w.tempura_one_turn_left
            return w.tempura_dead

        # ── Sashimi ─────────────────────────────────────────────────────
        if card == "Sashimi":
            progress = played.count("Sashimi") % 3
            needed = 3 - progress
            sashimi_incoming = incoming.count("Sashimi")

            if needed == 1:
                return w.sashimi_completes_set
            if needed == 2:
                if sashimi_incoming >= 1 and turns_left >= 2:
                    return w.sashimi_one_needed_incoming
                if turns_left >= 2:
                    return w.sashimi_one_needed_blind
                return 0.0
            # needed == 3: fresh start on a new set
            if sashimi_incoming >= 2 and turns_left >= 4:
                return w.sashimi_two_needed_incoming
            if turns_left >= 4:
                return w.sashimi_two_needed_speculative
            if turns_left >= 6:
                return w.sashimi_fresh_start
            return 0.0

        # ── Dumplings ───────────────────────────────────────────────────
        if card == "Dumpling":
            idx = min(played.count("Dumpling"), len(w.dumpling_marginals) - 1)
            return w.dumpling_marginals[idx]

        # ── Maki Rolls ──────────────────────────────────────────────────
        if "Maki Roll" in card:
            icons = self._maki_icons(card)
            our_maki = self._count_maki(played)
            all_opp_maki = [
                self._count_maki(opp.played) for opp in (self.opps or {}).values()
            ]
            max_opp_maki = max(all_opp_maki) if all_opp_maki else 0
            new_total = our_maki + icons

            crowded = (
                w.maki_crowded_penalty
                if self._count_maki(incoming) >= w.maki_crowded_threshold
                else 0.0
            )

            if new_total > max_opp_maki:
                return icons * w.maki_lead_per_icon + crowded
            if new_total == max_opp_maki:
                return icons * w.maki_tie_per_icon + crowded
            return icons * w.maki_trailing_per_icon + crowded

        # ── Pudding ─────────────────────────────────────────────────────
        if card == "Pudding":
            if round_num == 1:
                return w.pudding_round1
            if round_num == 2:
                return w.pudding_round2

            our_puddings = played.count("Pudding")
            all_opp_puddings = [
                opp.played.count("Pudding") for opp in (self.opps or {}).values()
            ]
            max_opp = max(all_opp_puddings) if all_opp_puddings else 0
            min_opp = min(all_opp_puddings) if all_opp_puddings else 0

            race_bonus = w.pudding_race_bonus if incoming.count("Pudding") >= 2 else 0.0

            if our_puddings + 1 > max_opp:
                return w.pudding_r3_taking_lead + race_bonus
            if our_puddings <= min_opp:
                return w.pudding_r3_avoiding_last + race_bonus
            return w.pudding_r3_neutral + race_bonus

        # ── Chopsticks ──────────────────────────────────────────────────
        if card == "Chopsticks":
            return w.chopsticks_value

        return 0.0

    # ── Denial score ──────────────────────────────────────────────────────

    def _denial_score(self, card):
        """How much does taking this card hurt the opponent who wants it most?"""
        if not self.opps or not self.state:
            return 0.0

        w = self.weights
        max_denial = 0.0

        for opp in self.opps.values():
            opp_played = opp.played
            val = 0.0

            if card == "Tempura":
                if opp_played.count("Tempura") % 2 == 1:
                    val = w.denial_tempura_complete

            elif card == "Sashimi":
                progress = opp_played.count("Sashimi") % 3
                if progress == 2:
                    val = w.denial_sashimi_complete
                elif progress == 1:
                    val = w.denial_sashimi_partial

            elif "Maki Roll" in card:
                icons = self._maki_icons(card)
                opp_maki = self._count_maki(opp_played)
                our_maki = self._count_maki(self.state.played_cards or [])
                if opp_maki > our_maki:
                    val = icons * w.denial_maki_per_icon

            elif "Nigiri" in card:
                opp_wasabi = opp_played.count("Wasabi")
                opp_nigiri = sum(1 for c in opp_played if "Nigiri" in c)
                if opp_wasabi > opp_nigiri:
                    base = {
                        "Squid Nigiri": w.squid_nigiri,
                        "Salmon Nigiri": w.salmon_nigiri,
                        "Egg Nigiri": w.egg_nigiri,
                    }[card]
                    # Denial value is only the *extra* swing from their wasabi
                    val = base * (w.wasabi_nigiri_multiplier - 1)

            elif card == "Wasabi":
                if sum(1 for c in opp.hand if "Nigiri" in c) > 0:
                    val = w.denial_wasabi_nigiri

            elif card == "Pudding":
                if self.state.round == 3:
                    opp_puddings = opp_played.count("Pudding")
                    our_puddings = (self.state.played_cards or []).count("Pudding")
                    if opp_puddings > our_puddings + 1:
                        val = w.denial_pudding_r3

            max_denial = max(max_denial, val)

        return max_denial
