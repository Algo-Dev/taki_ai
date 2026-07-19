"""Tests for the head-to-head rotation-orbit design (eval_headtohead.py).

The orbit is what makes the promotion standard well defined at every seat count. These pin
the run set it generates, and -- most importantly -- that FOUR SEATS IS UNCHANGED, so the
published 4-seat margins still describe the same experiment.
"""
import unittest

from eval_headtohead import orbit_pairs
from game import MIN_PLAYERS


class OrbitPairsTest(unittest.TestCase):

    def test_four_seats_alternating_is_the_historical_two_run_swap(self):
        """The 4-seat standard must be bit-identical to the pre-generalization design.

        Every published 4-seat margin (incl. the M1s3 promotion, +0.0417) was measured with
        exactly these two runs. If this changes, those numbers stop describing the tree.
        """
        self.assertEqual(orbit_pairs([0, 2], 4), [([0, 2], [1, 3])])

    def test_two_seats_collapses_to_one_pair(self):
        """n=2 collapses for the same reason n=4 does: the base is alternating."""
        self.assertEqual(orbit_pairs([0], 2), [([0], [1])])

    def test_three_seats_is_the_full_orbit(self):
        """No alternating base exists at odd n, so nothing collapses: 3 pairs = 6 runs."""
        self.assertEqual(orbit_pairs([0], 3),
                         [([0], [1, 2]), ([1], [0, 2]), ([2], [0, 1])])

    def test_orbit_is_invariant_to_the_base_position(self):
        """Only the SHAPE of --team1-seats matters; rotating the base regenerates the orbit."""
        for n, base_a, base_b in [(3, [0], [2]), (4, [0, 2], [1, 3]), (5, [0], [3])]:
            with self.subTest(n=n):
                as_sets = lambda ps: {frozenset([frozenset(b), frozenset(c)]) for b, c in ps}
                self.assertEqual(as_sets(orbit_pairs(base_a, n)),
                                 as_sets(orbit_pairs(base_b, n)))

    def test_contiguous_four_seat_base_does_not_collapse(self):
        """The contiguous 2v2 is not rotationally symmetric, so its orbit is larger.

        This is why the standard names the ALTERNATING base: it is not an arbitrary pick,
        it is the one whose orbit is minimal (and friendly-fire free).
        """
        self.assertEqual(len(orbit_pairs([0, 1], 4)), 2)

    def test_every_seat_is_covered_equally_by_each_model(self):
        """Seat balance is a property of the construction, at every count."""
        for n in range(MIN_PLAYERS, 8):
            base = [0, 2] if n == 4 else [0]
            with self.subTest(n=n):
                held = [0] * n              # times model1 sits in each seat, over the orbit
                for b, c in orbit_pairs(base, n):
                    for i in b:
                        held[i] += 1        # run 1: model1 holds b
                    for i in c:
                        held[i] += 1        # run 2: model1 holds c
                self.assertEqual(len(set(held)), 1,
                                 f'n={n}: uneven seat coverage {held}')

    def test_team_sizes_are_balanced_across_each_pair(self):
        """Within a pair the two parities sum to 1, so the paired margin has parity 0.

        This is what makes the lopsided 1-vs-2 at three seats a valid paired comparison.
        """
        for n in range(MIN_PLAYERS, 8):
            base = [0, 2] if n == 4 else [0]
            with self.subTest(n=n):
                for b, c in orbit_pairs(base, n):
                    self.assertEqual(len(b) + len(c), n)

    def test_pairs_are_complementary_and_disjoint(self):
        for n in range(MIN_PLAYERS, 8):
            base = [0, 2] if n == 4 else [0]
            with self.subTest(n=n):
                for b, c in orbit_pairs(base, n):
                    self.assertEqual(set(b) & set(c), set())
                    self.assertEqual(set(b) | set(c), set(range(n)))

    def test_no_duplicate_runs(self):
        """A pair {b,c} and {c,b} name the same two runs; the orbit must dedup them."""
        for n in range(MIN_PLAYERS, 8):
            base = [0, 2] if n == 4 else [0]
            with self.subTest(n=n):
                keys = [frozenset([frozenset(b), frozenset(c)])
                        for b, c in orbit_pairs(base, n)]
                self.assertEqual(len(keys), len(set(keys)))


if __name__ == '__main__':
    unittest.main()
