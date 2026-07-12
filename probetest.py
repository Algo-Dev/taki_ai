"""Consistency tests for the behavioural-probe harness (probe.py).

A silently-inconsistent position would corrupt every Q-value and every rollout downstream while
still *looking* fine, so these run before any result is believed. Kept out of gametest.py so that
`python -m unittest gametest` stays TensorFlow-free.
"""
import collections
import itertools
import unittest

import numpy as np

from game import (Game, Card, Color, Type, State, Action, OBS_PERMS, ACT_PERMS, COLOR_PERMS,
                  action_to_scalar)
from agents.random import RandomAgent
import probe
from probe import (SCENARIOS, B1A, B1A_MIN_A, make_position, full_pool, force_line, _Dummy,
                   legal_scalars, b1c)

R, B = Color.RED, Color.BLUE
PLAY = Action.PLAY_CARD


def seats(n=4):
    return [RandomAgent(seed=i) for i in range(n)]


class MakePositionTest(unittest.TestCase):

    def test_conserves_the_120_card_multiset(self):
        for name, scen in SCENARIOS.items():
            with self.subTest(name):
                g = scen.build(seats())
                seen = collections.Counter(g.deck + g.discard + [c for h in g.hands for c in h])
                self.assertEqual(seen, collections.Counter(full_pool(0)))
                self.assertEqual(sum(seen.values()), 120)

    def test_sets_the_requested_state(self):
        g = B1A_MIN_A.build(seats())
        self.assertEqual(g.hands[0], list(B1A_MIN_A.hand))
        self.assertEqual(g.shown_card(), B1A_MIN_A.top)
        self.assertEqual(g.state, State.TAKI)
        self.assertEqual(g.taki_color, Color.RED)
        self.assertEqual(g.curr, 0)
        self.assertEqual(g.dir, 1)
        self.assertEqual([len(h) for h in g.hands[1:]], [5, 5, 5])

    def test_unreachable_position_is_rejected(self):
        # 4 cards in hand over a 1-card discard conserves the deck but no real game reaches it.
        with self.assertRaises(AssertionError):
            make_position(list(B1A.hand), B1A.top, agents=seats(), discard_size=1)

    def test_open_taki_without_a_color_is_rejected(self):
        with self.assertRaises(AssertionError):
            make_position([Card(Type.FIVE, R)], Card(Type.TAKI, R), agents=seats(),
                          state=State.TAKI, taki_color=Color.NONE)


class LegalityTest(unittest.TestCase):

    def test_every_scenario_has_its_intended_legality(self):
        for name, scen in SCENARIOS.items():
            with self.subTest(name):
                moves = scen.build(seats()).valid_moves(0)
                for m in scen.expect_legal:
                    self.assertIn(m, moves, f'{name}: {m} should be legal')
                for m in scen.expect_illegal:
                    self.assertNotIn(m, moves, f'{name}: {m} should be illegal')

    def test_b1c_legal_set_is_identical_for_every_k(self):
        # The whole sweep rests on the alternative set being held constant across k.
        sets = [set(legal_scalars(b1c(k).build(seats()), 0)) for k in range(5)]
        for k, s in enumerate(sets):
            self.assertEqual(s, sets[0], f'k={k} changed the legal set')
        self.assertEqual(len(sets[0]), 4)

    def test_b1c_deck_and_unseen_features_are_constant_across_k(self):
        obs = [b1c(k).build(seats()).observation(0) for k in range(5)]
        for k in range(1, 5):
            self.assertEqual(len(b1c(k).build(seats()).deck),
                             len(b1c(0).build(seats()).deck))
            np.testing.assert_allclose(obs[k][143:147], obs[0][143:147])   # deck + unseen counts

    def test_forced_illegal_move_raises(self):
        g = B1A_MIN_A.build(seats())
        with self.assertRaises(AssertionError):
            force_line(g, [(Action.DRAW, None)], 0)      # DRAW is illegal inside an open TAKI


class EngineFactsTest(unittest.TestCase):
    """These pin the two engine facts the report's claims rest on. If the engine ever changes,
    these fail rather than the report quietly becoming false."""

    def test_b1a_correct_line_wins_outright_from_every_determinization(self):
        for seed in range(20):
            g = B1A.build(seats(), seed=seed)
            force_line(g, list(B1A.lines['correct']), 0)
            self.assertTrue(g.done(), f'seed {seed}')
            self.assertEqual(g.hands[0], [], f'seed {seed}')

    def test_b1a_has_no_sequencing_trap_every_order_wins(self):
        # The B1 premise was that the run must END on the red 5. That was an artifact of the
        # FINISHING_TYPE_VALUES bug; the real rule finishes on anything but PLUS, so the red
        # STOP and red +2 are finishers too. Pin the corrected fact: EVERY ordering of the run
        # wins outright. If a real trap is ever wanted here, it needs a PLUS in hand.
        run = [Card(Type.FIVE, R), Card(Type.STOP, R), Card(Type.PLUSTWO, R)]
        for order in itertools.permutations(run):
            g = B1A.build(seats(), seed=0)
            force_line(g, [(PLAY, Card(Type.TAKI, R))] + [(PLAY, x) for x in order], 0)
            self.assertTrue(g.done(), f'order {order} should win')
            self.assertEqual(g.hands[0], [], f'order {order} left cards in hand')

    def test_b1a_min_both_orders_win(self):
        # Same correction at the minimal-pair scale: 5-first no longer eats a penalty, because
        # ending on the STOP is legal. The A-vs-B minimal pair therefore no longer isolates
        # anything and its Q-gap must not be interpreted.
        for order in ([Card(Type.FIVE, R), Card(Type.STOP, R)],
                      [Card(Type.STOP, R), Card(Type.FIVE, R)]):
            g = B1A_MIN_A.build(seats(), seed=0)
            force_line(g, [(PLAY, x) for x in order], 0)
            self.assertTrue(g.done(), f'order {order} should win')
            self.assertEqual(g.hands[0], [], f'order {order} left cards in hand')

    def test_cannot_finish_on_a_plus_inside_a_taki(self):
        # The one surviving finishing constraint, and the basis of any future sequencing probe:
        # emptying the hand on a PLUS costs a penalty draw and keeps the turn inside the run.
        g = make_position([Card(Type.PLUS, R)], Card(Type.TAKI, R), agents=seats(),
                          state=State.TAKI, taki_color=R, seed=0)
        force_line(g, [(PLAY, Card(Type.PLUS, R))], 0)
        self.assertFalse(g.done())
        self.assertEqual(g.curr, 0)
        self.assertEqual(len(g.hands[0]), 1)   # drew a penalty instead of winning


class MonteCarloValidityTest(unittest.TestCase):

    def test_observation_is_invariant_to_the_determinization(self):
        # The validity condition of the whole MC arm: resampling the opponents' cards and the deck
        # order must not change what the learner SEES (it sees only sizes and coarse counts). If
        # this fails, the rollouts are not sampling the learner's belief state.
        base = B1A.build(seats(), seed=0).observation(0)
        for seed in range(1, 20):
            np.testing.assert_array_equal(B1A.build(seats(), seed=seed).observation(0), base)


class ColorSymmetryTest(unittest.TestCase):

    def test_b1c_k0_is_exactly_red_green_symmetric(self):
        # At k=0 the hand holds a red TAKI and a green TAKI and nothing else of either color, so
        # swapping red<->green must leave the observation untouched while mapping the red-TAKI
        # action onto the green-TAKI one. A color-equivariant net therefore MUST give Delta(0)=0 —
        # that is the calibrated noise floor of the whole Delta(k) figure.
        idx = COLOR_PERMS.index((3, 2, 1, 4))            # RED<->GREEN, YELLOW/BLUE fixed
        obs = b1c(0).build(seats()).observation(0)
        np.testing.assert_allclose(obs[OBS_PERMS[idx]], obs, atol=0, rtol=0)
        red = action_to_scalar(PLAY, Card(Type.TAKI, R))
        green = action_to_scalar(PLAY, Card(Type.TAKI, Color.GREEN))
        self.assertEqual(int(ACT_PERMS[idx][red]), green)


class QTableTest(unittest.TestCase):

    def test_greedy_pick_matches_agent_act(self):
        agent = probe.untrained_agent(0)
        for name, scen in SCENARIOS.items():
            with self.subTest(name):
                g = scen.build([agent, RandomAgent(), RandomAgent(), RandomAgent()])
                table = probe.q_table(agent, g, 0)
                self.assertEqual(table.greedy, agent.act(g.observation(0), legal_scalars(g, 0)))


if __name__ == '__main__':
    unittest.main()
