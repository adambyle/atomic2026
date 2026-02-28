from ..sushi_go_client import SushiGoClient

HOST = "localhost"
PORT = 7878


class ClaudeBot(SushiGoClient):
    def __init__(self):
        super().__init__(HOST, PORT)

    def choose_card(self, hand: list[str]) -> int:
        """
        Smart strategy based on Sushi Go scoring rules:
        - Scores cards by expected value given current game state
        - Accounts for set completion (tempura pairs, sashimi triples)
        - Prioritizes wasabi+nigiri combos
        - Balances pudding collection across rounds
        - Uses hand size to gauge how early/late in the round we are
        """
        if not hand:
            return 0

        state = self.state
        played = state.played_cards if state else []
        round_num = state.round if state else 1
        hand_size = len(hand)

        # Count what we've already played this round
        tempura_count = played.count("Tempura")
        sashimi_count = played.count("Sashimi")
        dumpling_count = played.count("Dumpling")
        pudding_count = played.count("Pudding") + (state.puddings if state else 0)
        has_wasabi = state.has_unused_wasabi if state else False

        scores = {}
        for i, card in enumerate(hand):
            scores[i] = self._score_card(
                card,
                hand,
                hand_size,
                round_num,
                tempura_count,
                sashimi_count,
                dumpling_count,
                pudding_count,
                has_wasabi,
            )

        return max(scores, key=lambda i: scores[i])

    def _score_card(
        self,
        card,
        hand,
        hand_size,
        round_num,
        tempura_count,
        sashimi_count,
        dumpling_count,
        pudding_count,
        has_wasabi,
    ):
        """
        Score a card by its expected value in the current situation.
        Higher = better to pick.
        """

        # How many turns are left this round (rough estimate)
        turns_left = hand_size  # after picking this card, hand_size-1 remain

        # --- Nigiri ---
        if card == "Squid Nigiri":
            return 9 if has_wasabi else 3

        if card == "Salmon Nigiri":
            return 6 if has_wasabi else 2

        if card == "Egg Nigiri":
            return 3 if has_wasabi else 1

        # --- Wasabi ---
        if card == "Wasabi":
            # Only valuable if we have turns left to play a nigiri
            nigiri_in_hand = sum(1 for c in hand if "Nigiri" in c)
            if has_wasabi:
                return -1  # Already have one, second wasabi is mostly wasted
            if nigiri_in_hand > 0:
                return 5.5  # We can immediately pair it or soon will
            if turns_left >= 3:
                return 4.0  # Reasonable chance a nigiri comes around
            return 1.0

        # --- Tempura ---
        if card == "Tempura":
            needed = 2 - (tempura_count % 2)
            if needed == 1:
                return 5.0  # Completes a pair for 5 pts!
            elif turns_left >= 3:
                return 3.5  # Good chance of completing the pair
            elif turns_left == 2:
                return 2.0
            else:
                return 0.0  # Not enough turns to complete pair

        # --- Sashimi ---
        if card == "Sashimi":
            needed = 3 - (sashimi_count % 3)
            if needed == 1:
                return 10.0  # Completes a set for 10 pts!
            elif needed == 2 and turns_left >= 4:
                return 4.5
            elif needed == 2 and turns_left >= 2:
                return 2.0
            elif needed == 3 and turns_left >= 5:
                return 3.5  # Early round, can still collect 3
            else:
                return 0.0  # Too late to complete

        # --- Dumplings ---
        if card == "Dumpling":
            # Marginal value of next dumpling: 1,2,3,4,5 -> 1,3,6,10,15
            # Differences: 1st=1, 2nd=2, 3rd=3, 4th=4, 5th+=5
            marginal = min(dumpling_count + 1, 5)
            return float(marginal)

        # --- Maki Rolls ---
        # Maki is competitive — value depends on how many we have vs opponents
        # Rough heuristic: more maki icons = better, weighted by card value
        if card == "Maki Roll (3)":
            return 4.5
        if card == "Maki Roll (2)":
            return 3.0
        if card == "Maki Roll (1)":
            return 1.5

        # --- Pudding ---
        if card == "Pudding":
            # More valuable early in the game (3 rounds total)
            # Also valuable if we have very few or if round 3 (avoid penalty)
            if round_num == 3:
                return 3.5 if pudding_count == 0 else 2.5
            elif round_num == 2:
                return 2.5 if pudding_count == 0 else 1.5
            else:
                return 2.0  # Round 1

        # --- Chopsticks ---
        if card == "Chopsticks":
            # Good early in round (more turns to use), poor late
            if turns_left >= 5:
                return 2.5
            elif turns_left >= 3:
                return 1.0
            else:
                return 0.0

        return 0.0
