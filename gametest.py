import unittest
from game import *
from agents.random import RandomAgent


class ColorTest(unittest.TestCase):
    def test_repr(self):
        redtaki = Card(Type.TAKI, Color.RED)
        self.assertEqual(str(redtaki), "red taki")
        stop = Card(Type.STOP)
        self.assertEqual(str(stop), "stop")
        supertaki = Card(Type.TAKI)
        self.assertEqual(str(supertaki), "super taki")

    def test_action_to_scalar(self):
        playredtaki = Card(Type.TAKI, Color.RED)
        action = Action.PLAY_CARD
        self.assertEqual(action_to_scalar(action, playredtaki), 0)
        yellowthree = Card(Type.THREE, Color.YELLOW)
        self.assertEqual(action_to_scalar(action, yellowthree), 18)
        chcol = Card(Type.CHCOL)
        self.assertEqual(action_to_scalar(action, chcol), 60)
        supertaki = Card(Type.TAKI)
        self.assertEqual(action_to_scalar(action, supertaki), 61)
        self.assertEqual(action_to_scalar(Action.DRAW, None), 62)
        self.assertEqual(action_to_scalar(Action.CLOSE_TAKI, None), 63)

    def test_scalar_to_action(self):
        playredtaki = Card(Type.TAKI, Color.RED)
        action = Action.PLAY_CARD
        self.assertEqual(scalar_to_action(0), (action, playredtaki))
        yellowthree = Card(Type.THREE, Color.YELLOW)
        self.assertEqual(scalar_to_action(18), (action, yellowthree))
        chcol = Card(Type.CHCOL)
        self.assertEqual((action, chcol), scalar_to_action(60))
        supertaki = Card(Type.TAKI)
        self.assertEqual((action, supertaki), scalar_to_action(61))
        self.assertEqual((Action.DRAW, None), scalar_to_action(62))
        self.assertEqual((Action.CLOSE_TAKI, None), scalar_to_action(63))

    def test_action_scalar_round_trip(self):
        # Every legal action scalar must survive a decode -> encode round trip.
        for scalar in range(ACTION_SIZE):
            act, card = scalar_to_action(scalar)
            self.assertEqual(action_to_scalar(act, card), scalar)

    def test_card_to_vector(self):
        playredtaki = Card(Type.TAKI, Color.RED)
        vec = card_to_vector(playredtaki)
        self.assertEqual(vec[0], 1)
        self.assertEqual(vec[1], 0)
        yellowthree = Card(Type.THREE, Color.YELLOW)
        vec = card_to_vector(yellowthree)
        self.assertEqual(vec[18], 1)
        self.assertEqual(vec[0], 0)
        supertaki = Card(Type.TAKI)
        vec = card_to_vector(supertaki)
        self.assertEqual(vec[61], 1)
        self.assertEqual(vec[0], 0)
        chcol = Card(Type.CHCOL)
        vec = card_to_vector(chcol)
        self.assertEqual(vec[60], 1)
        self.assertEqual(vec[0], 0)
        vec = card_to_vector(supertaki, chcol)
        self.assertEqual(vec[61], 1)
        self.assertEqual(vec[60], 1)
        self.assertEqual(vec[0], 0)


class GameFlowTest(unittest.TestCase):
    def make_game(self, players=2, seed=0):
        return Game([RandomAgent(seed=i) for i in range(players)], seed=seed)

    def test_opens_on_number_card(self):
        # The starting discard card must always be a plain number card.
        for seed in range(50):
            g = self.make_game(seed=seed)
            self.assertIn(g.shown_card().type.value, NUMBER_TYPE_VALUES)

    def test_normal_play_advances_turn(self):
        g = self.make_game()
        g.curr = 0
        g.discard = [Card(Type.FIVE, Color.RED)]
        g.hands[0] = [Card(Type.THREE, Color.RED), Card(Type.ONE, Color.RED)]
        g.state = State.NORMAL
        done, _ = self._take_turn(g, Action.PLAY_CARD, Card(Type.THREE, Color.RED))
        self.assertFalse(done)
        self.assertEqual(g.curr, 1)

    def test_stop_skips_next_player(self):
        g = self.make_game(players=3)
        g.curr = 0
        g.discard = [Card(Type.FIVE, Color.RED)]
        g.hands[0] = [Card(Type.STOP, Color.RED), Card(Type.ONE, Color.RED)]
        g.state = State.NORMAL
        done, _ = self._take_turn(g, Action.PLAY_CARD, Card(Type.STOP, Color.RED))
        self.assertFalse(done)
        self.assertEqual(g.curr, 2)  # player 1 was skipped
        self.assertEqual(g.state, State.NORMAL)

    def test_plus_two_stacks_and_draws(self):
        g = self.make_game()
        g.curr = 0
        g.discard = [Card(Type.PLUSTWO, Color.GREEN)]
        g.hands[0] = [Card(Type.PLUSTWO, Color.RED), Card(Type.ONE, Color.RED)]
        g.state = State.DRAW_TWO
        g.draw_num = 1
        # Stacking another +2 raises the pending draw to 2 and passes it on.
        self._take_turn(g, Action.PLAY_CARD, Card(Type.PLUSTWO, Color.RED))
        self.assertEqual(g.state, State.DRAW_TWO)
        self.assertEqual(g.draw_num, 2)
        self.assertEqual(g.curr, 1)
        before = len(g.hands[1])
        self._take_turn(g, Action.DRAW, None)
        self.assertEqual(len(g.hands[1]) - before, 4)  # 2 * draw_num
        self.assertEqual(g.state, State.NORMAL)

    def test_taki_chain_then_close(self):
        g = self.make_game()
        g.curr = 0
        g.discard = [Card(Type.ONE, Color.RED)]
        g.hands[0] = [Card(Type.TAKI, Color.RED), Card(Type.FIVE, Color.RED),
                      Card(Type.TWO, Color.RED), Card(Type.NINE, Color.BLUE)]
        g.state = State.NORMAL
        self._take_turn(g, Action.PLAY_CARD, Card(Type.TAKI, Color.RED))
        self.assertEqual(g.state, State.TAKI)
        self.assertEqual(g.curr, 0)  # same player keeps playing
        # Only red cards (and wilds) are playable during the open TAKI.
        plays = [c for a, c in g.valid_moves(0) if a == Action.PLAY_CARD]
        self.assertTrue(all(c.color in (Color.RED, Color.NONE) for c in plays))
        self._take_turn(g, Action.PLAY_CARD, Card(Type.FIVE, Color.RED))
        self._take_turn(g, Action.PLAY_CARD, Card(Type.TWO, Color.RED))
        self._take_turn(g, Action.CLOSE_TAKI, None)
        self.assertEqual(g.state, State.NORMAL)
        self.assertEqual(g.curr, 1)
        self.assertEqual(g.hands[0], [Card(Type.NINE, Color.BLUE)])

    def test_win_on_number_card(self):
        g = self.make_game()
        g.curr = 0
        g.discard = [Card(Type.FIVE, Color.RED)]
        g.hands[0] = [Card(Type.THREE, Color.RED)]
        g.state = State.NORMAL
        done, winner = self._take_turn(g, Action.PLAY_CARD, Card(Type.THREE, Color.RED))
        self.assertTrue(done)
        self.assertEqual(winner, 0)
        self.assertTrue(g.done())

    def test_cannot_win_on_action_card(self):
        g = self.make_game()
        g.curr = 0
        g.discard = [Card(Type.FIVE, Color.RED)]
        g.hands[0] = [Card(Type.STOP, Color.RED)]  # last card is an action card
        g.state = State.NORMAL
        done, _ = self._take_turn(g, Action.PLAY_CARD, Card(Type.STOP, Color.RED))
        self.assertFalse(done)
        self.assertFalse(g.done())
        self.assertEqual(len(g.hands[0]), 1)  # drew a penalty card instead of winning

    def _take_turn(self, game, action, card):
        """Drive one full turn (action + advancement) the way next_turn would."""
        class _Scripted:
            def play(self, _game):
                return action, card
        game.agents[game.curr] = _Scripted()
        return game.next_turn()


if __name__ == '__main__':
    unittest.main()
