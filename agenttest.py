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
        g.hands[0] = [Card(Type.STOP, Color.RED)]  # would end on an action card
        g.state = State.NORMAL
        g.history = []
        g.agents[0] = _Scripted([(Action.PLAY_CARD, Card(Type.STOP, Color.RED))])
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

    def test_b5_never_empties_hand_on_action_card(self):
        g = make_game()
        g.discard = [Card(Type.FIVE, Color.RED)]
        g.state = State.NORMAL
        g.hands[0] = [Card(Type.STOP, Color.RED)]
        # Playing would trigger the finisher penalty; drawing is preferred.
        self.assertEqual(self.agent.play(g), (Action.DRAW, None))

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

    def test_b3_closes_rather_than_end_run_on_action_card(self):
        g = make_game()
        g.discard = [Card(Type.TAKI, Color.RED)]
        g.state = State.TAKI
        g.taki_color = Color.RED
        g.hands[0] = [Card(Type.STOP, Color.RED)]
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
