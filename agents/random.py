import random


class RandomAgent:
    def __init__(self, seed=None):
        self.random = random.Random(seed)

    def reseed(self, seed):
        """Reset the choice stream (eval.py calls this per game so every matchup
        faces the identical opponent randomness — common random numbers)."""
        self.random.seed(seed)

    def play(self, game):
        return self.random.choice(game.valid_moves())
