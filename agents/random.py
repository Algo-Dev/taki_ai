import random


class RandomAgent:
    def __init__(self, seed=None):
        self.random = random.Random(seed)

    def play(self, game):
        return self.random.choice(game.valid_moves())
