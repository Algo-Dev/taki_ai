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
