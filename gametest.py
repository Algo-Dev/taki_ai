import copy
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
        king = Card(Type.KING)
        self.assertEqual(action_to_scalar(action, king), 62)
        self.assertEqual(action_to_scalar(Action.DRAW, None), 63)
        self.assertEqual(action_to_scalar(Action.CLOSE_TAKI, None), 64)

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
        king = Card(Type.KING)
        self.assertEqual((action, king), scalar_to_action(62))
        self.assertEqual((Action.DRAW, None), scalar_to_action(63))
        self.assertEqual((Action.CLOSE_TAKI, None), scalar_to_action(64))

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
        king = Card(Type.KING)
        vec = card_to_vector(king)
        self.assertEqual(vec[62], 1)
        self.assertEqual(vec[0], 0)
        vec = card_to_vector(supertaki, chcol, king)
        self.assertEqual(vec[61], 1)
        self.assertEqual(vec[60], 1)
        self.assertEqual(vec[62], 1)
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

    def test_king_cancels_pending_plus_two(self):
        g = self.make_game()
        g.curr = 0
        g.discard = [Card(Type.PLUSTWO, Color.GREEN)]
        g.hands[0] = [Card(Type.KING), Card(Type.ONE, Color.RED)]
        g.state = State.DRAW_TWO
        g.draw_num = 2
        before = len(g.hands[0])
        self._take_turn(g, Action.PLAY_CARD, Card(Type.KING))
        self.assertEqual(g.draw_num, 0)             # pending draw cancelled
        self.assertEqual(g.state, State.KING)       # player may now put one more card
        self.assertEqual(g.curr, 0)                 # same player keeps the turn
        self.assertEqual(len(g.hands[0]), before - 1)  # played King, drew nothing

    def test_king_only_legal_plus_two_response_besides_draw(self):
        g = self.make_game()
        g.curr = 0
        g.discard = [Card(Type.PLUSTWO, Color.GREEN)]
        g.hands[0] = [Card(Type.KING), Card(Type.FIVE, Color.RED)]
        g.state = State.DRAW_TWO
        g.draw_num = 1
        moves = g.valid_moves(0)
        self.assertIn((Action.PLAY_CARD, Card(Type.KING)), moves)
        self.assertIn((Action.DRAW, None), moves)
        # A non-+2, non-King card cannot be played against a pending +2.
        self.assertNotIn((Action.PLAY_CARD, Card(Type.FIVE, Color.RED)), moves)

    def test_king_grants_one_optional_follow_up(self):
        g = self.make_game()
        g.curr = 0
        g.discard = [Card(Type.FIVE, Color.RED)]
        g.hands[0] = [Card(Type.KING), Card(Type.THREE, Color.BLUE),
                      Card(Type.ONE, Color.RED)]
        g.state = State.NORMAL
        self._take_turn(g, Action.PLAY_CARD, Card(Type.KING))
        self.assertEqual(g.state, State.KING)
        self.assertEqual(g.curr, 0)                 # same player continues
        # Any card is playable after a King (even an off-color one), plus CLOSE_TAKI to
        # end the turn; a plain DRAW is not offered.
        moves = g.valid_moves(0)
        self.assertIn((Action.PLAY_CARD, Card(Type.THREE, Color.BLUE)), moves)
        self.assertIn((Action.CLOSE_TAKI, None), moves)
        self.assertNotIn((Action.DRAW, None), moves)
        # Putting the follow-up card ends the turn (state back to NORMAL, turn advances).
        self._take_turn(g, Action.PLAY_CARD, Card(Type.THREE, Color.BLUE))
        self.assertEqual(g.state, State.NORMAL)
        self.assertEqual(g.curr, 1)

    def test_king_follow_up_is_optional_via_close(self):
        g = self.make_game()
        g.curr = 0
        g.discard = [Card(Type.FIVE, Color.RED)]
        g.hands[0] = [Card(Type.KING), Card(Type.THREE, Color.BLUE)]
        g.state = State.NORMAL
        self._take_turn(g, Action.PLAY_CARD, Card(Type.KING))
        self.assertEqual(g.state, State.KING)
        self._take_turn(g, Action.CLOSE_TAKI, None)  # decline the follow-up
        self.assertEqual(g.state, State.NORMAL)
        self.assertEqual(g.curr, 1)
        self.assertEqual(g.hands[0], [Card(Type.THREE, Color.BLUE)])  # nothing else played

    def test_king_inside_taki_is_inert(self):
        # A King played mid-TAKI is playable but has no effect: it does not change
        # taki_color and does not start a King continuation; the TAKI simply goes on.
        g = self.make_game()
        g.curr = 0
        g.discard = [Card(Type.ONE, Color.RED)]
        g.hands[0] = [Card(Type.KING), Card(Type.FIVE, Color.RED)]
        g.state = State.TAKI
        g.taki_color = Color.RED
        self.assertIn((Action.PLAY_CARD, Card(Type.KING)), g.valid_moves(0))  # playable
        self._take_turn(g, Action.PLAY_CARD, Card(Type.KING))
        self.assertEqual(g.state, State.TAKI)       # still an open TAKI (inert)
        self.assertEqual(g.taki_color, Color.RED)   # color unchanged
        self.assertEqual(g.curr, 0)                 # same player keeps playing

    def test_king_closing_a_taki_grants_optional_followup(self):
        # If the TAKI is closed on a King, the player is granted an optional follow-up turn.
        g = self.make_game()
        g.curr = 0
        g.discard = [Card(Type.ONE, Color.RED)]
        g.hands[0] = [Card(Type.TAKI, Color.RED), Card(Type.KING),
                      Card(Type.NINE, Color.BLUE)]
        g.state = State.NORMAL
        self._take_turn(g, Action.PLAY_CARD, Card(Type.TAKI, Color.RED))
        self._take_turn(g, Action.PLAY_CARD, Card(Type.KING))   # King now on top, inert
        self.assertEqual(g.shown_card(), Card(Type.KING))
        self._take_turn(g, Action.CLOSE_TAKI, None)             # close ON the King
        self.assertEqual(g.state, State.KING)                   # follow-up granted
        self.assertEqual(g.curr, 0)                             # same player
        # ... and it is optional: declining with CLOSE_TAKI just ends the turn (no re-grant).
        self._take_turn(g, Action.CLOSE_TAKI, None)
        self.assertEqual(g.state, State.NORMAL)
        self.assertEqual(g.curr, 1)

    def test_can_win_on_king(self):
        # Unlike other action cards, the King is a legal finishing card.
        g = self.make_game()
        g.curr = 0
        g.discard = [Card(Type.FIVE, Color.RED)]
        g.hands[0] = [Card(Type.KING)]  # last card is a King
        g.state = State.NORMAL
        done, winner = self._take_turn(g, Action.PLAY_CARD, Card(Type.KING))
        self.assertTrue(done)
        self.assertEqual(winner, 0)
        self.assertTrue(g.done())

    def test_observation_is_normalised(self):
        g = self.make_game(players=4)
        obs = g.observation()
        # Shape and dtype: a fixed-size float vector.
        self.assertEqual(obs.shape, (OBSERVATION_SIZE,))
        self.assertTrue(np.issubdtype(obs.dtype, np.floating))
        # Count features (hand + discard) are rescaled into [0, 1].
        counts = obs[:CARD_VECTOR_SIZE * 2]
        self.assertGreaterEqual(counts.min(), 0.0)
        self.assertLessEqual(counts.max(), 1.0)
        # One-hot blocks stay 0/1: exactly one active state, and a valid colour count.
        state_block = obs[CARD_VECTOR_SIZE * 2:CARD_VECTOR_SIZE * 2 + len(State)]
        self.assertEqual(state_block.sum(), 1.0)

    def _take_turn(self, game, action, card):
        """Drive one full turn (action + advancement) the way next_turn would."""
        class _Scripted:
            def play(self, _game):
                return action, card
        game.agents[game.curr] = _Scripted()
        return game.next_turn()


class ColorSymmetryTest(unittest.TestCase):
    """The COLOR_PERMS/OBS_PERMS/ACT_PERMS tables must exactly mirror a recoloring of
    the real game: recoloring a game and re-encoding it must equal permuting the
    original encoding. This is what makes the replay augmentation a genuine
    environment transition rather than noise."""

    def _recolored(self, g, pi):
        """A deep copy of game g with every color c relabeled to pi[c-1] (NONE fixed).
        Builds new Card objects: the deck is constructed with [Card(...)] * 2, so the
        same object can appear twice and in-place recoloring would hit it twice."""
        def pc(color):
            return color if color is Color.NONE else Color(pi[color.value - 1])
        g2 = copy.deepcopy(g)
        g2.hands = [[Card(c.type, pc(c.color)) for c in h] for h in g.hands]
        g2.discard = [Card(c.type, pc(c.color)) for c in g.discard]
        g2.deck = [Card(c.type, pc(c.color)) for c in g.deck]
        g2.taki_color = pc(g.taki_color)
        return g2

    def _assert_equivariant(self, g):
        """For all 24 perms: recolored observation == gathered observation, and
        recolored valid-move scalars == forward-mapped original scalars."""
        obs = g.observation(agent=0)
        scalars = [action_to_scalar(*m) for m in g.valid_moves(agent=g.curr)]
        for pi, gobs, gact in zip(COLOR_PERMS, OBS_PERMS, ACT_PERMS):
            g2 = self._recolored(g, pi)
            np.testing.assert_array_equal(g2.observation(agent=0), obs[gobs])
            scalars2 = [action_to_scalar(*m) for m in g2.valid_moves(agent=g.curr)]
            self.assertEqual(sorted(scalars2), sorted(int(gact[a]) for a in scalars))

    def test_tables_are_bijections_with_correct_fixed_points(self):
        for k in range(len(COLOR_PERMS)):
            self.assertEqual(sorted(OBS_PERMS[k]), list(range(OBSERVATION_SIZE)))
            self.assertEqual(sorted(ACT_PERMS[k]), list(range(ACTION_SIZE)))
            # Colorless actions (CHCOL/SuperTAKI/King/DRAW/CLOSE_TAKI) never move.
            self.assertEqual(list(ACT_PERMS[k][60:]), [60, 61, 62, 63, 64])
            # State one-hot, draw_num and the extra features never move.
            # Layout: hand(63)+discard(63) -> state one-hot+draw_num at 126..134,
            # then open-TAKI color(4) + shown card(63), then extra features at 202..204.
            fixed = list(range(126, 135)) + list(range(202, 205))
            self.assertEqual(list(OBS_PERMS[k][fixed]), fixed)

    def test_identity_perm_is_noop(self):
        # itertools.permutations is lexicographic, so index 0 is the identity.
        self.assertEqual(COLOR_PERMS[0], (1, 2, 3, 4))
        np.testing.assert_array_equal(OBS_PERMS[0], np.arange(OBSERVATION_SIZE))
        np.testing.assert_array_equal(ACT_PERMS[0], np.arange(ACTION_SIZE))

    def test_known_swap_values(self):
        # RED <-> YELLOW swap: red taki (0) <-> yellow taki (15), and the gather for
        # the augmented yellow-three slot (18) reads the original red-three slot (3).
        k = COLOR_PERMS.index((2, 1, 3, 4))
        self.assertEqual(ACT_PERMS[k][0], 15)
        self.assertEqual(ACT_PERMS[k][18], 3)
        self.assertEqual(ACT_PERMS[k][60], 60)
        self.assertEqual(OBS_PERMS[k][18], 3)

    def test_observation_and_action_equivariance(self):
        # Real games at several stages of play, driven by seeded random agents.
        for seed in range(4):
            for turns in (0, 5, 20, 60):
                g = Game([RandomAgent(seed=i) for i in range(4)], seed=seed)
                for _ in range(turns):
                    if g.done():
                        break
                    g.next_turn()
                if not g.done():
                    self._assert_equivariant(g)

    def test_equivariance_in_taki_and_chcol_states(self):
        # Hand-built state covering the tricky slots: colorless CHCOL (60) and Super
        # TAKI (61) in hand, a color-carrying played CHCOL on the discard top (slot
        # 44), an active taki_color one-hot, and CHCOL's 4-way valid_moves expansion.
        g = Game([RandomAgent(seed=i) for i in range(4)], seed=0)
        g.curr = 0
        g.hands[0] = [Card(Type.CHCOL), Card(Type.TAKI),
                      Card(Type.THREE, Color.GREEN), Card(Type.PLUSTWO, Color.BLUE)]
        g.discard = [Card(Type.FIVE, Color.RED), Card(Type.CHCOL, Color.GREEN)]
        g.state = State.TAKI
        g.taki_color = Color.GREEN
        self._assert_equivariant(g)

    def test_batched_gather_matches_rowwise(self):
        # The exact vectorised expression used by AIAgent.replay(): a per-row gather.
        rng = np.random.RandomState(0)
        m = rng.rand(8, OBSERVATION_SIZE).astype(np.float32)
        ks = rng.randint(len(OBS_PERMS), size=8)
        batched = np.take_along_axis(m, OBS_PERMS[ks], axis=1)
        for i in range(8):
            np.testing.assert_array_equal(batched[i], m[i][OBS_PERMS[ks[i]]])


if __name__ == '__main__':
    unittest.main()
