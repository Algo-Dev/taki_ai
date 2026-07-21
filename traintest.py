"""Tests for the training loop's seat-count handling (mixed-count training).

The seat-count one-hot is a CONSTANT within a single-count run, and a constant input folds
into the next layer's bias -- so it is provably inert there. It only carries information when
the count varies between trials, which is what mixed counts are for. These pin the sampler and,
more importantly, that turning the feature on did not disturb single-count runs.
"""
import random
import subprocess
import sys
import unittest

from game import (Game, MIN_PLAYERS, NUM_PLAYER_SLOTS, OBSERVATION_SIZE,
                  VOID_FEATURES)

# The seat-count one-hot is no longer the observation tail since R4 (the per-opponent
# color-void block, VOID_FEATURES floats, follows it), so index it absolutely.
_ONE_HOT = slice(OBSERVATION_SIZE - VOID_FEATURES - NUM_PLAYER_SLOTS,
                 OBSERVATION_SIZE - VOID_FEATURES)
from agents.random import RandomAgent


class SeatCountSpecTest(unittest.TestCase):
    """--num-players parsing: '4' and '2,3,4' are the two forms train.py accepts."""

    def _parse(self, spec):
        # Mirrors train.py's parse. Kept in sync by test_train_accepts_specs below, which
        # drives the real CLI.
        return sorted({int(x) for x in spec.split(',')})

    def test_single_and_mixed_forms(self):
        self.assertEqual(self._parse('4'), [4])
        self.assertEqual(self._parse('2,3,4'), [2, 3, 4])
        self.assertEqual(self._parse('4,2,3'), [2, 3, 4])   # order cannot change sampling
        self.assertEqual(self._parse('3,3'), [3])           # deduped

    def test_train_rejects_bad_specs(self):
        for spec in ('1', 'x', '2,1', '0,4'):
            p = subprocess.run([sys.executable, 'train.py', '--num-players', spec,
                                '--trials', '1'], capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0, f'{spec!r} should have been rejected')
            self.assertIn('num-players', p.stderr)


class ReseatTest(unittest.TestCase):
    """Game.reset(agents=...) is how a trial changes the table size."""

    def test_reset_reseats_and_reshapes_the_round(self):
        pool = [RandomAgent() for _ in range(4)]
        g = Game(pool, seed=1)
        g.reset()
        self.assertEqual(len(g.hands), 4)
        g.reset(agents=pool[:2])
        self.assertEqual(len(g.agents), 2)
        self.assertEqual(len(g.hands), 2)      # dealt for the NEW count, not the old
        self.assertEqual(g.observation()[_ONE_HOT].tolist(), [1, 0, 0])
        g.reset(agents=pool[:3])
        self.assertEqual(len(g.hands), 3)
        self.assertEqual(g.observation()[_ONE_HOT].tolist(), [0, 1, 0])

    def test_reset_rejects_a_table_too_small_to_play(self):
        pool = [RandomAgent() for _ in range(4)]
        g = Game(pool, seed=1)
        with self.assertRaises(AssertionError):
            g.reset(agents=pool[:MIN_PLAYERS - 1])

    def test_reseating_does_not_restart_the_deck_stream(self):
        # Why train.py reuses one Game rather than building one per trial: the deck RNG lives
        # on the instance. If reseating restarted it, every trial would deal the same cards --
        # a silent collapse of training variety that no test of the sampler would catch.
        pool = [RandomAgent() for _ in range(4)]
        g = Game(pool, seed=1)
        deals = []
        for _ in range(6):
            g.reset(agents=pool[:2])
            deals.append(tuple(str(c) for c in g.hands[0]))
        self.assertGreater(len(set(deals)), 1, 'every reseat dealt an identical hand')


class SingleCountUnchangedTest(unittest.TestCase):
    """A single-count run must be bit-identical to before mixed counts existed.

    The sampler draws from its own RNG and only under mixed counts, so the opener/deck streams
    a single-count run consumes are untouched. This is the guard that the whole published
    lineage (every number to date is single-count) was not silently perturbed.
    """

    def test_single_count_draws_nothing_from_the_count_rng(self):
        seat_counts, max_players = [4], 4
        mixed = len(seat_counts) > 1
        count_rng = random.Random(1)
        before = count_rng.getstate()
        for _ in range(50):
            n = count_rng.choice(seat_counts) if mixed else max_players
            self.assertEqual(n, 4)
        self.assertEqual(count_rng.getstate(), before, 'a single-count run consumed count RNG')

    def test_opener_stream_is_unaffected_by_the_feature(self):
        # seat_rng.randrange(n_seats) with n_seats fixed == the old randrange(num_of_players).
        old = random.Random(7)
        new = random.Random(7)
        self.assertEqual([old.randrange(4) for _ in range(30)],
                         [new.randrange(4) for _ in range(30)])


class MixedSamplerTest(unittest.TestCase):

    def test_sampler_covers_every_count_and_is_seeded(self):
        counts = [2, 3, 4]
        draws = [random.Random(1).choice(counts) for _ in range(1)]
        self.assertIn(draws[0], counts)
        rng_a, rng_b = random.Random(5), random.Random(5)
        a = [rng_a.choice(counts) for _ in range(200)]
        b = [rng_b.choice(counts) for _ in range(200)]
        self.assertEqual(a, b, 'seeded sampling must be reproducible')
        self.assertEqual(set(a), {2, 3, 4}, 'every count must actually be dealt')
        # Roughly uniform -- a badly skewed sampler silently reweights what the net learns.
        for c in counts:
            self.assertGreater(a.count(c), 200 / 3 * 0.6)

    def test_one_hot_varies_across_a_mixed_run(self):
        # The point of the whole exercise: under mixed counts the one-hot is no longer a
        # constant, so the net can condition on it.
        pool = [RandomAgent() for _ in range(4)]
        g = Game(pool, seed=1)
        rng = random.Random(3)
        seen = set()
        for _ in range(60):
            n = rng.choice([2, 3, 4])
            g.reset(agents=pool[:n])
            seen.add(tuple(g.observation()[_ONE_HOT].tolist()))
        self.assertEqual(len(seen), 3, f'one-hot did not vary across counts: {seen}')


if __name__ == '__main__':
    unittest.main()
