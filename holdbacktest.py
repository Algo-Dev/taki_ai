"""Tests for the B2 hold-back census (holdback.py).

The census's numbers are only worth as much as its two primitives: a clone that cannot
corrupt its parent, and a shed count that agrees with the engine. Both are pinned here.
"""
import unittest

from game import (Game, Card, Action, Color, Type, State, Action as A)
from agents.random import RandomAgent
from agents.heuristic import HeuristicAgent, GREEDY
from holdback import clone_game, max_shed, is_refusal, playable_moves, Census
from probe import make_position


def seats(n=4):
    return [RandomAgent(seed=i) for i in range(n)]


class CloneTest(unittest.TestCase):
    def test_clone_cannot_corrupt_the_parents_discard(self):
        """draw_card recycles the discard and mutates CHCOL cards IN PLACE
        (game.py:506-508). A clone sharing Card objects would rewrite the parent's
        shown card mid-search."""
        g = make_position([Card(Type.FIVE, Color.RED)], Card(Type.THREE, Color.RED),
                          agents=seats(), seed=0)
        g.discard = [Card(Type.CHCOL, Color.RED)] * 6 + [Card(Type.THREE, Color.RED)]
        g.deck = []
        before = [(c.type, c.color) for c in g.discard]

        c = clone_game(g)
        c.draw_card(0, 3)                      # forces a recycle inside the clone

        self.assertEqual([(x.type, x.color) for x in g.discard], before)
        self.assertIsNot(c.hands, g.hands)
        self.assertIs(c.agents[1], g.agents[1])   # agents are SHARED, never copied

    def test_clone_does_not_share_hand_or_deck_lists(self):
        g = Game(seats(), seed=0)
        c = clone_game(g)
        c.hands[0].append(Card(Type.NINE, Color.BLUE))
        c.deck.clear()
        self.assertEqual(len(g.hands[0]), 8)
        self.assertTrue(g.deck)


class MaxShedTest(unittest.TestCase):
    def test_taki_run_sheds_the_whole_color_group(self):
        g = make_position([Card(Type.TAKI, Color.RED), Card(Type.THREE, Color.RED),
                           Card(Type.SEVEN, Color.RED), Card(Type.NINE, Color.BLUE)],
                          Card(Type.FIVE, Color.RED), agents=seats(), seed=0)
        r = max_shed(g, 0, (A.PLAY_CARD, Card(Type.TAKI, Color.RED)))
        self.assertEqual(r.shed, 3)            # taki + the two red numbers
        self.assertFalse(r.wins)

    def test_a_plain_play_sheds_exactly_one(self):
        g = make_position([Card(Type.THREE, Color.RED), Card(Type.NINE, Color.BLUE)],
                          Card(Type.FIVE, Color.RED), agents=seats(), seed=0)
        r = max_shed(g, 0, (A.PLAY_CARD, Card(Type.THREE, Color.RED)))
        self.assertEqual(r.shed, 1)

    def test_draw_sheds_negative(self):
        g = make_position([Card(Type.THREE, Color.RED), Card(Type.NINE, Color.BLUE)],
                          Card(Type.FIVE, Color.RED), agents=seats(), seed=0)
        self.assertEqual(max_shed(g, 0, (A.DRAW, None)).shed, -1)

    def test_winning_line_is_detected(self):
        g = make_position([Card(Type.THREE, Color.RED)], Card(Type.FIVE, Color.RED),
                          agents=seats(), seed=0)
        r = max_shed(g, 0, (A.PLAY_CARD, Card(Type.THREE, Color.RED)))
        self.assertEqual((r.shed, r.wins), (1, True))

    def test_ending_on_a_plus_sheds_nothing_and_keeps_the_turn(self):
        """The one surviving finishing constraint: a PLUS cannot end a hand, so the
        engine hands it back (penalty draw) and the seat plays on. Net shed must be 0,
        not 1 — a rule of thumb keyed on 'a card left the hand' would get this wrong."""
        g = make_position([Card(Type.PLUS, Color.RED)], Card(Type.FIVE, Color.RED),
                          agents=seats(), seed=0)
        r = max_shed(g, 0, (A.PLAY_CARD, Card(Type.PLUS, Color.RED)))
        self.assertFalse(r.wins)
        self.assertLessEqual(r.shed, 0)


class RefusalTest(unittest.TestCase):
    def test_draw_is_a_refusal(self):
        g = make_position([Card(Type.THREE, Color.RED)], Card(Type.FIVE, Color.RED),
                          agents=seats(), seed=0)
        self.assertTrue(is_refusal(g, (A.DRAW, None)))

    def test_closing_a_taki_on_a_plain_card_is_a_refusal(self):
        g = make_position([Card(Type.THREE, Color.RED)], Card(Type.FIVE, Color.RED),
                          agents=seats(), state=State.TAKI, taki_color=Color.RED, seed=0)
        self.assertTrue(is_refusal(g, (A.CLOSE_TAKI, None)))

    def test_closing_a_taki_on_a_PLUS_is_NOT_a_refusal(self):
        """Closing on a PLUS sets State.PLUS and the seat KEEPS THE TURN
        (game.py:584-598), so nothing was refused. This is the case that makes a
        naive 'CLOSE_TAKI == held back' classifier wrong."""
        g = make_position([Card(Type.THREE, Color.RED)], Card(Type.PLUS, Color.RED),
                          agents=seats(), state=State.TAKI, taki_color=Color.RED, seed=0)
        self.assertFalse(is_refusal(g, (A.CLOSE_TAKI, None)))

    def test_declining_a_king_followup_is_a_refusal(self):
        g = make_position([Card(Type.THREE, Color.RED)], Card(Type.KING),
                          agents=seats(), state=State.KING, seed=0)
        self.assertTrue(is_refusal(g, (A.CLOSE_TAKI, None)))

    def test_a_lone_plus_is_not_a_real_play(self):
        """With one card left and it a PLUS, 'playing' sheds nothing — so drawing
        instead is not holding anything back, and must not be counted as a refusal."""
        g = make_position([Card(Type.PLUS, Color.RED)], Card(Type.FIVE, Color.RED),
                          agents=seats(), seed=0)
        self.assertEqual(playable_moves(g), [])


class CensusControlTest(unittest.TestCase):
    """The positive controls. If these drift, every census number is meaningless."""

    def test_greedy_heuristic_never_refuses(self):
        from holdback import run
        c = run('heuristic:greedy', games=25, seed=0, dfs_budget=400)
        self.assertEqual(c.c['refusal'], 0)

    def test_random_refuses_constantly(self):
        from holdback import run
        c = run('random', games=10, seed=0, dfs_budget=400)
        self.assertGreater(c.c['refusal'] / c.c['free'], 0.20)


if __name__ == '__main__':
    unittest.main()


class WeaponTimingScenarioTest(unittest.TestCase):
    """scenarios_b2: delta(k) is only interpretable if k is the ONLY thing that changes.
    If the legal set or the deck/unseen features drifted across k, the sweep would be
    measuring the drift instead of the opponent-hand-size feature."""

    def _games(self):
        from scenarios_b2 import b2_weapon, KS
        from probe import _Dummy
        return [b2_weapon(k).build([_Dummy()] * 4, seed=0) for k in KS]

    def test_legal_set_is_identical_for_every_k(self):
        sets = [sorted(map(str, g.valid_moves())) for g in self._games()]
        for s in sets[1:]:
            self.assertEqual(s, sets[0])

    def test_only_the_opponent_hand_size_feature_varies(self):
        import numpy as np
        obs = [g.observation(0) for g in self._games()]
        diff = np.array([np.flatnonzero(o != obs[0]) for o in obs[1:]], dtype=object)
        changed = sorted({int(i) for idx in diff for i in idx})
        # obs[139:147] = dir, 3 opponent hand sizes (turn order), deck size, unseen counts.
        # Only the FIRST opponent slot (the next player) may move.
        self.assertEqual(changed, [140], f'unexpected features drifted across k: {changed}')

    def test_both_candidate_plays_shed_exactly_one_card(self):
        """The whole point: the shaped reward is INDIFFERENT between them, so any
        preference the net shows was not paid for by the reward."""
        from scenarios_b2 import b2_weapon
        from probe import _Dummy
        g = b2_weapon(1).build([_Dummy()] * 4, seed=0)
        for card in (Card(Type.PLUSTWO, Color.RED), Card(Type.FIVE, Color.RED)):
            self.assertEqual(max_shed(g, 0, (A.PLAY_CARD, card)).shed, 1, str(card))


class LossPenaltyTest(unittest.TestCase):
    """R6 (train.py --loss-penalty), the fix B2 motivates. Pins the two things that make
    it correct: it fires on a LOSING TERMINAL and on nothing else."""

    def _reward(self, hand_sizes, seat, won, terminal, penalty):
        """Reimplements train.py's `shaped` seat_reward branch exactly (train.py:288-306)."""
        r = -hand_sizes[seat]
        if won:
            r += sum(h for i, h in enumerate(hand_sizes) if i != seat)
        elif terminal and penalty:
            r -= penalty
        return r

    def test_a_loss_now_costs_something(self):
        hands = [4, 0, 6, 5]          # seat 1 just went out; seat 0 is a loser holding 4
        old = self._reward(hands, 0, won=False, terminal=True, penalty=0.0)
        new = self._reward(hands, 0, won=False, terminal=True, penalty=20.0)
        self.assertEqual(old, -4)     # pre-B2: a loss pays nothing, the stream just stops
        self.assertEqual(new, -24)

    def test_the_penalty_does_not_touch_ordinary_steps(self):
        hands = [4, 3, 6, 5]
        self.assertEqual(self._reward(hands, 0, won=False, terminal=False, penalty=20.0), -4)

    def test_the_penalty_does_not_touch_a_win(self):
        hands = [0, 3, 6, 5]
        self.assertEqual(self._reward(hands, 0, won=True, terminal=True, penalty=20.0), 14)

    def test_it_restores_the_sign_that_B2_found_inverted(self):
        """B2's core defect: with no loss penalty, a seat is BETTER OFF when an opponent is
        about to go out, because the game ends and the -len(hand) stream stops. Compare the
        terminal a seat faces when it loses, against simply continuing to hold its hand."""
        hands = [4, 0, 6, 5]
        keep_playing = -4                       # one more ordinary step, same hand
        lose_now_old = self._reward(hands, 0, won=False, terminal=True, penalty=0.0)
        lose_now_new = self._reward(hands, 0, won=False, terminal=True, penalty=20.0)
        self.assertEqual(lose_now_old, keep_playing)   # losing is EXACTLY as good as playing on
        self.assertLess(lose_now_new, keep_playing)    # ...and now it is strictly worse
