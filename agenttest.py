"""Tests for the public-event history log (game.py) and the R3 HeuristicAgent."""
import unittest

from game import *
from agents.random import RandomAgent
from agents.heuristic import HeuristicAgent, _OpponentModel, LACK_DECAY


def make_game(players=2, seed=0):
    return Game([RandomAgent(seed=i) for i in range(players)], seed=seed)


class HistoryLogTest(unittest.TestCase):
    def test_play_records_seat_card_and_active_color(self):
        g = make_game()
        g.curr = 0
        g.discard = [Card(Type.FIVE, Color.RED)]
        g.hands[0] = [Card(Type.THREE, Color.RED), Card(Type.ONE, Color.RED)]
        g.state = State.NORMAL
        g.history = []
        g.process_action(Action.PLAY_CARD, Card(Type.THREE, Color.RED), 0)
        self.assertEqual(g.history, [
            ('play', 0, Type.THREE.value, Color.RED.value,
             State.NORMAL.value, Color.RED.value)])

    def test_draw_records_count_state_and_active_color(self):
        g = make_game()
        g.curr = 1
        g.discard = [Card(Type.FIVE, Color.BLUE)]
        g.state = State.NORMAL
        g.history = []
        g.process_action(Action.DRAW, None, 1)
        self.assertEqual(g.history, [
            ('draw', 1, 1, State.NORMAL.value, Color.BLUE.value)])

    def test_draw_two_resolution_records_full_count(self):
        g = make_game()
        g.discard = [Card(Type.PLUSTWO, Color.GREEN)]
        g.state = State.DRAW_TWO
        g.draw_num = 2
        g.history = []
        g.process_action(Action.DRAW, None, 1)
        self.assertEqual(g.history, [
            ('draw', 1, 4, State.DRAW_TWO.value, Color.GREEN.value)])

    def test_taki_draw_uses_taki_color_not_top_card(self):
        g = make_game()
        g.discard = [Card(Type.TAKI, Color.RED)]
        g.state = State.TAKI
        g.taki_color = Color.RED
        g.history = []
        g.process_action(Action.CLOSE_TAKI, None, 0)
        self.assertEqual(g.history[0][:2], ('close', 0))

    def test_finisher_penalty_draw_is_logged(self):
        g = make_game()
        g.curr = 0
        g.discard = [Card(Type.FIVE, Color.RED)]
        # PLUS is the ONLY card a hand may not end on (game.py FINISHING_TYPE_VALUES).
        g.hands[0] = [Card(Type.PLUS, Color.RED)]
        g.state = State.NORMAL
        g.history = []
        g.agents[0] = _Scripted([(Action.PLAY_CARD, Card(Type.PLUS, Color.RED))])
        g.next_turn()
        self.assertIn(('penalty_draw', 0, 1), g.history)

    def test_history_resets_each_round(self):
        g = make_game()
        g.process_action(Action.DRAW, None, 0)
        self.assertTrue(g.history)
        g.reset()
        self.assertEqual(g.history, [])


class _Scripted:
    def __init__(self, moves):
        self.moves = list(moves)

    def play(self, game):
        return self.moves.pop(0)


class OpponentModelTest(unittest.TestCase):
    def test_informative_draw_sets_belief_with_own_decay(self):
        m = _OpponentModel(2)
        m.observe(('draw', 1, 1, State.NORMAL.value, Color.RED.value))
        # Set to 1.0, then decayed once for the drawn card itself.
        self.assertAlmostEqual(m.lacks_color(1, Color.RED), LACK_DECAY)

    def test_belief_decays_per_card_drawn_since(self):
        m = _OpponentModel(2)
        m.observe(('draw', 1, 1, State.NORMAL.value, Color.RED.value))
        m.observe(('draw', 1, 1, State.NORMAL.value, Color.NONE.value))
        m.observe(('penalty_draw', 1, 2))
        self.assertAlmostEqual(m.lacks_color(1, Color.RED), LACK_DECAY ** 4)

    def test_playing_the_color_clears_belief(self):
        m = _OpponentModel(2)
        m.observe(('draw', 1, 1, State.NORMAL.value, Color.RED.value))
        m.observe(('play', 1, Type.THREE.value, Color.RED.value,
                   State.NORMAL.value, Color.RED.value))
        self.assertEqual(m.lacks_color(1, Color.RED), 0.0)

    def test_recolored_wild_does_not_clear_belief(self):
        m = _OpponentModel(2)
        m.observe(('draw', 1, 1, State.NORMAL.value, Color.RED.value))
        m.observe(('play', 1, Type.CHCOL.value, Color.RED.value,
                   State.NORMAL.value, Color.GREEN.value))
        self.assertGreater(m.lacks_color(1, Color.RED), 0.0)

    def test_forced_plus_two_draw_is_not_color_evidence(self):
        m = _OpponentModel(2)
        m.observe(('draw', 1, 4, State.DRAW_TWO.value, Color.RED.value))
        self.assertEqual(m.lacks_color(1, Color.RED), 0.0)


class HeuristicBehaviourTest(unittest.TestCase):
    def setUp(self):
        self.agent = HeuristicAgent()

    def test_b1_stacks_plus_two(self):
        g = make_game()
        g.discard = [Card(Type.PLUSTWO, Color.GREEN)]
        g.state = State.DRAW_TWO
        g.draw_num = 1
        g.hands[0] = [Card(Type.PLUSTWO, Color.RED), Card(Type.KING)]
        action, card = self.agent.play(g)
        self.assertEqual((action, card.type), (Action.PLAY_CARD, Type.PLUSTWO))

    def test_b1_king_cancels_only_big_penalties(self):
        g = make_game()
        g.discard = [Card(Type.PLUSTWO, Color.GREEN)]
        g.state = State.DRAW_TWO
        g.hands[0] = [Card(Type.KING), Card(Type.THREE, Color.BLUE)]
        g.draw_num = 1  # 2 cards: not worth the King
        self.assertEqual(self.agent.play(g), (Action.DRAW, None))
        g.draw_num = 2  # 4 cards: cancel
        action, card = self.agent.play(g)
        self.assertEqual((action, card.type), (Action.PLAY_CARD, Type.KING))

    def test_b4_wins_on_last_finisher(self):
        g = make_game()
        g.discard = [Card(Type.FIVE, Color.RED)]
        g.state = State.NORMAL
        g.hands[0] = [Card(Type.FIVE, Color.BLUE)]
        self.assertEqual(self.agent.play(g),
                         (Action.PLAY_CARD, Card(Type.FIVE, Color.BLUE)))

    def test_b5_never_empties_hand_on_a_plus(self):
        g = make_game()
        g.discard = [Card(Type.FIVE, Color.RED)]
        g.state = State.NORMAL
        g.hands[0] = [Card(Type.PLUS, Color.RED)]
        # PLUS is the only non-finisher: playing it would trigger the penalty draw,
        # so drawing is preferred.
        self.assertEqual(self.agent.play(g), (Action.DRAW, None))

    def test_b4_wins_on_an_action_card(self):
        # The corrected finishing rule: every card EXCEPT PLUS may end the hand.
        # A last red STOP is a legal winner and must be played, not held.
        g = make_game()
        g.discard = [Card(Type.FIVE, Color.RED)]
        g.state = State.NORMAL
        g.hands[0] = [Card(Type.STOP, Color.RED)]
        self.assertEqual(self.agent.play(g),
                         (Action.PLAY_CARD, Card(Type.STOP, Color.RED)))

    def test_b7_blocks_a_near_winner_with_stop(self):
        g = make_game()
        g.discard = [Card(Type.FIVE, Color.RED)]
        g.state = State.NORMAL
        g.hands[0] = [Card(Type.STOP, Color.RED), Card(Type.THREE, Color.RED),
                      Card(Type.NINE, Color.BLUE)]
        g.hands[1] = [Card(Type.ONE, Color.GREEN)]  # about to win
        action, card = self.agent.play(g)
        self.assertEqual((action, card.type), (Action.PLAY_CARD, Type.STOP))

    def test_b8_plays_the_color_the_next_player_lacks(self):
        g = make_game()
        g.discard = [Card(Type.THREE, Color.GREEN)]
        g.state = State.NORMAL
        g.hands[0] = [Card(Type.THREE, Color.RED), Card(Type.THREE, Color.BLUE)]
        g.hands[1] = [Card(Type.ONE, Color.GREEN)] * 5
        # Seat 1 drew while red was the active color -> believed to lack red.
        g.history = [('draw', 1, 1, State.NORMAL.value, Color.RED.value)]
        self.assertEqual(self.agent.play(g),
                         (Action.PLAY_CARD, Card(Type.THREE, Color.RED)))

    def test_b3_run_dumps_action_card_first_and_wins_on_number(self):
        g = make_game()
        g.discard = [Card(Type.TAKI, Color.RED)]
        g.state = State.TAKI
        g.taki_color = Color.RED
        g.hands[0] = [Card(Type.STOP, Color.RED), Card(Type.FIVE, Color.RED)]
        action, card = self.agent.play(g)
        self.assertEqual((action, card.type), (Action.PLAY_CARD, Type.STOP))
        g.process_action(action, card, 0)
        action, card = self.agent.play(g)
        self.assertEqual((action, card.type), (Action.PLAY_CARD, Type.FIVE))

    def test_b3_closes_rather_than_end_run_on_a_plus(self):
        g = make_game()
        g.discard = [Card(Type.TAKI, Color.RED)]
        g.state = State.TAKI
        g.taki_color = Color.RED
        # A run may not END on a PLUS (the one surviving finishing constraint), so
        # the agent closes instead of spending its last card.
        g.hands[0] = [Card(Type.PLUS, Color.RED)]
        self.assertEqual(self.agent.play(g), (Action.CLOSE_TAKI, None))

    def test_b6_protects_the_hoarded_color_group(self):
        g = make_game()
        g.discard = [Card(Type.FIVE, Color.RED)]
        g.state = State.NORMAL
        g.hands[0] = [Card(Type.TAKI, Color.RED), Card(Type.THREE, Color.RED),
                      Card(Type.SEVEN, Color.RED), Card(Type.FIVE, Color.GREEN),
                      Card(Type.TWO, Color.BLUE), Card(Type.NINE, Color.BLUE)]
        g.hands[1] = [Card(Type.ONE, Color.GREEN)] * 5
        # Red TAKI + 2 reds are reserved; green 5 (type match) is the free play.
        self.assertEqual(self.agent.play(g),
                         (Action.PLAY_CARD, Card(Type.FIVE, Color.GREEN)))

    def test_b6_opens_the_hoard_when_hand_is_nearly_all_group(self):
        g = make_game()
        g.discard = [Card(Type.FIVE, Color.RED)]
        g.state = State.NORMAL
        g.hands[0] = [Card(Type.TAKI, Color.RED), Card(Type.THREE, Color.RED),
                      Card(Type.SEVEN, Color.RED), Card(Type.TWO, Color.BLUE)]
        g.hands[1] = [Card(Type.ONE, Color.GREEN)] * 5
        self.assertEqual(self.agent.play(g),
                         (Action.PLAY_CARD, Card(Type.TAKI, Color.RED)))


class SmokeEvalTest(unittest.TestCase):
    def test_beats_random_clearly(self):
        """~200 seeded games, 1 heuristic vs 3 random: must be far above 0.25."""
        heuristic = HeuristicAgent()
        wins = games = 200
        won = 0
        for g in range(games):
            heuristic.reseed()
            opponents = [RandomAgent(seed=f'{g}:{i}') for i in range(3)]
            game = Game([heuristic] + opponents, seed=g)
            turns = 0
            while not game.done() and turns < 2000:
                game.next_turn()
                turns += 1
            if game.done() and not game.hands[0]:
                won += 1
        self.assertGreater(won / games, 0.40, f'won only {won}/{games}')


if __name__ == '__main__':
    unittest.main()


class WeightsAblationTest(unittest.TestCase):
    """PLAN.md B2: the hold-back terms must be (a) inert by default and (b) really off
    when ablated. Without (a) the ablation measures a refactor; without (b) it measures
    nothing."""

    def test_default_weights_reproduce_the_r3_agent(self):
        from agents.heuristic import Weights
        w = Weights()
        self.assertEqual(
            (w.w_reserve, w.w_nofin, w.p_king, w.p_chcol, w.p_super_taki,
             w.w_save_blocker, w.score_decline_king, w.score_draw, w.hold_wilds_in_run),
            (10.0, 8.0, 6.0, 3.0, 4.0, 0.7, -2.0, -5.0, True))

    def test_default_and_explicit_default_weights_play_identically(self):
        from agents.heuristic import Weights
        a, b = HeuristicAgent(), HeuristicAgent(weights=Weights())
        for seed in range(30):
            g = Game([RandomAgent(seed=i) for i in range(4)], seed=seed)
            for _ in range(40):
                if g.done():
                    break
                self.assertEqual(a.play(g), b.play(g))
                g.next_turn()

    def test_greedy_holds_nothing_back_where_the_full_agent_does(self):
        """The three cases where SCORE_DRAW=-5 is beaten by a hold penalty."""
        from agents.heuristic import GREEDY
        full, greedy = HeuristicAgent(), HeuristicAgent(weights=GREEDY)

        # P_KING = 6.0 > 5.0: the full agent draws rather than spend a lone King.
        g = make_game()
        g.discard = [Card(Type.FIVE, Color.RED)]
        g.state = State.NORMAL
        g.hands[0] = [Card(Type.KING), Card(Type.NINE, Color.BLUE)]
        self.assertEqual(full.play(g), (Action.DRAW, None))
        self.assertEqual(greedy.play(g)[0], Action.PLAY_CARD)

        # W_RESERVE = 10.0 > 5.0: the full agent draws rather than break its hoard.
        # Two cards sit OUTSIDE the group, so the hoard-release trigger (`open_now`,
        # which needs outside <= 1) has not fired yet — the group is still being saved.
        g = make_game()
        g.discard = [Card(Type.FIVE, Color.RED)]
        g.state = State.NORMAL
        g.hands[0] = [Card(Type.TAKI, Color.RED), Card(Type.THREE, Color.RED),
                      Card(Type.SEVEN, Color.RED),
                      Card(Type.NINE, Color.BLUE), Card(Type.TWO, Color.GREEN)]
        self.assertEqual(full.play(g), (Action.DRAW, None))
        self.assertEqual(greedy.play(g)[0], Action.PLAY_CARD)

        # KING_CANCEL_MIN_PENALTY = 4: under a single +2 the full agent eats the two
        # cards rather than spend its King; the greedy one cancels.
        g = make_game()
        g.discard = [Card(Type.PLUSTWO, Color.GREEN)]
        g.state = State.DRAW_TWO
        g.draw_num = 1
        g.hands[0] = [Card(Type.KING), Card(Type.THREE, Color.BLUE)]
        self.assertEqual(full.play(g), (Action.DRAW, None))
        self.assertEqual(greedy.play(g), (Action.PLAY_CARD, Card(Type.KING)))

    def test_greedy_never_draws_when_a_legal_play_exists(self):
        """The B2 positive control: the ablated agent must have a ~0 voluntary-draw
        rate, otherwise the census's 'holds back' signal is not measuring what we think."""
        from agents.heuristic import GREEDY
        greedy = HeuristicAgent(weights=GREEDY)
        voluntary = 0
        for seed in range(20):
            g = Game([HeuristicAgent(weights=GREEDY) for _ in range(4)], seed=seed)
            for _ in range(300):
                if g.done():
                    break
                moves = g.valid_moves()
                plays = [m for m in moves if m[0] is Action.PLAY_CARD]
                action, _ = greedy.play(g)
                if action is Action.DRAW and plays:
                    # Legal only if every play would empty the hand on a PLUS.
                    if not all(len(g.hands[g.curr]) == 1
                               and c.type.value not in FINISHING_TYPE_VALUES
                               for _, c in plays):
                        voluntary += 1
                g.next_turn()
        self.assertEqual(voluntary, 0)
