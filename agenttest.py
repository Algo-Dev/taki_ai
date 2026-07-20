"""Tests for the public-event history log (game.py) and the R3 HeuristicAgent."""
import dataclasses
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


class FrozenVersionTest(unittest.TestCase):
    """The heuristic is the project's ranking metric, so a named version must be a
    FROZEN artifact: every published "vs heuristic" number refers to one. These tests
    are what make the freeze real rather than a naming convention — they fail loudly if
    an H-series edit silently redefines the yardstick past numbers were measured against.
    """

    def test_r3_is_the_agent_the_published_numbers_were_measured_against(self):
        """R3's exact scoring economy, pinned. Includes its 17-point tuning bug
        (p_king=6.0 above |score_draw|=5.0, so it DRAWS rather than play its King).
        That bug is not to be fixed here — it is what the yardstick was."""
        from agents.heuristic import R3
        self.assertEqual(
            dataclasses.asdict(R3),
            dict(w_deny=3.0, w_rich=0.8, w_block=4.0, w_chdir_block=2.0,
                 w_plus_tempo=1.0, w_taki_dump=1.2, score_draw=-5.0,
                 score_forbidden=-100.0, w_save_blocker=0.7, w_nofin=8.0,
                 w_reserve=10.0, w_open_hoard=15.0, p_chcol=3.0, p_super_taki=4.0,
                 p_king=6.0, score_decline_king=-2.0, hold_wilds_in_run=True,
                 king_cancel_min_penalty=4, block_hand_threshold=2,
                 refusal_mode='legacy',
                 # H7 added these; R3 predates the behaviour, so it must stay OFF here.
                 # (This assertion failing on a new field is the freeze working: any field
                 # added to Weights has to be consciously defaulted for the yardstick.)
                 race=False, race_hand_threshold=2, race_hold_scale=0.0,
                 # H9 added this. R3 predates the 2-seat STOP-tempo structure, so it stays
                 # on the original 'h1' tempo structure — as do b2/h1/h1b2/h7/h8/greedy.
                 structure='h1'))

    def test_b2_retuned_is_pinned(self):
        """The single definition of "retuned" — b2_block_price.py and tune_heuristic.py
        each used to carry their own copy, and they had drifted (p_king 5.0 vs 4.9)."""
        from agents.heuristic import B2_RETUNED
        self.assertEqual(
            (B2_RETUNED.p_king, B2_RETUNED.p_chcol, B2_RETUNED.p_super_taki,
             B2_RETUNED.w_reserve, B2_RETUNED.king_cancel_min_penalty,
             B2_RETUNED.hold_wilds_in_run),
            (5.0, 4.5, 4.5, 4.0, 0, False))

    def test_b2_retuned_pulled_the_WILD_holds_below_the_draw_threshold_but_not_w_nofin(self):
        """B2's whole point: a hold ABOVE |score_draw| is a REFUSAL (-13 pts), not a
        preference. B2 pulled the wild-hold penalties down — but it never touched
        `w_nofin`, which is still 8.0, i.e. still a refusal. That is PLAN.md H5, and it
        means B2's retuned point is NOT refusal-free (see the characterization test
        below). Pinned as the published artifact, defect included.

        NB `p_king == 5.0 == |score_draw|` EXACTLY: at the threshold, not below it. The
        King is still played, but only because the tie-break prefers the lower action
        scalar (King 62 < DRAW 63) — a zero-margin accident, which is precisely the
        fragility H1 removes. Pinned as published; do not 'fix' it to 4.9 here."""
        from agents.heuristic import B2_RETUNED as w
        for name in ('p_king', 'p_chcol', 'p_super_taki', 'w_reserve'):
            self.assertLessEqual(getattr(w, name), abs(w.score_draw), f'{name} refuses')
        self.assertGreater(w.w_nofin, abs(w.score_draw))   # H5: the one B2 missed

    def test_b2_retuned_STILL_REFUSES_the_defect_h1_must_remove(self):
        """CHARACTERIZATION, not an endorsement. B2's retuned agent — the one that scored
        0.899 vs random and reached parity with the 500k champion — still voluntarily
        DRAWS while holding a legal play, because `w_nofin=8.0` outvotes `score_draw=-5.0`.

        So "retuning the weights" did NOT remove the refusal cliff; it only moved it. That
        is the argument for H1 making refusal structural rather than a tuning outcome.

        This test records today's count so H1 can be shown to drive it to ZERO. When H1's
        version lands, it gets the same test asserting 0 — and this one stays, pinning
        what the old yardstick did."""
        from agents.heuristic import B2_RETUNED
        self.assertEqual(self._voluntary_draws(B2_RETUNED, games=40), 4)

    def test_r3_refuses_even_more(self):
        """The shipped yardstick, for contrast: its p_king=6.0 refusal is on top of
        w_nofin's, which is why B2 priced the whole bug at ~17 points."""
        from agents.heuristic import R3
        self.assertGreater(self._voluntary_draws(R3, games=40), 4)

    def _voluntary_draws(self, weights, games):
        """A refusal: DRAW while a legal, non-forbidden play exists. H1's invariant is
        that this is 0 for EVERY weight assignment; today it is a tuning outcome."""
        agent = HeuristicAgent(weights=weights)
        voluntary = 0
        for seed in range(games):
            g = Game([HeuristicAgent(weights=weights) for _ in range(4)], seed=seed)
            for _ in range(120):
                if g.done():
                    break
                action, _ = agent.play(g)
                plays = [(a, c) for a, c in g.valid_moves() if a is Action.PLAY_CARD]
                if action is Action.DRAW and plays:
                    if not all(len(g.hands[g.curr]) == 1
                               and c.type.value not in FINISHING_TYPE_VALUES
                               for _, c in plays):
                        voluntary += 1
                g.next_turn()
        return voluntary

    def test_the_reference_is_h8_promoted_from_r3_by_b5(self):
        """B5 (2026-07-14): the yardstick was PROMOTED r3 -> h8, deliberately, with both
        numbers re-run and recorded (R6: 0.378 vs r3 -> 0.297 vs h8; A9: 0.348 -> 0.271).

        Changing REFERENCE is never a side effect — it silently redefines every "vs
        heuristic" number the project reports. This test exists so that a future edit that
        moves it has to say so out loud."""
        from agents.heuristic import REFERENCE, resolve_weights, H8
        self.assertEqual(REFERENCE, 'h8')
        self.assertEqual(resolve_weights(), H8)
        self.assertEqual(resolve_weights(''), H8)

    #: Move-sequence fingerprints: the sha256 of every (seat, action, card) each version
    #: chooses across 60 seeded 4-player games — thousands of decisions, hashed.
    #:
    #: WHY THIS EXISTS, when `test_r3_is_the_agent_...` already pins the weight vector:
    #: the pin catches a WEIGHT changing; this catches BEHAVIOUR changing while the
    #: weights look untouched. That is not hypothetical — it is precisely what H1 did.
    #: H1 changed no weight at all and moved the agent by +0.183, by editing the scoring
    #: STRUCTURE. Had it been done without a version flag, every pin above would have
    #: stayed green while `--opponent heuristic` silently began measuring a different
    #: opponent, and R6's published 0.378 would have quietly become incomparable to
    #: itself. Edit `_score_play`, reorder a tie-break, add a branch to the scoring loop:
    #: the weights still read 6.0, and only this test notices.
    #:
    #: A characterization test: it asserts nothing is CORRECT, only that it is UNCHANGED.
    #: That is the right instrument for an artifact whose whole value is being stable.
    #:
    #: IF THIS FAILS: do not reach for the new hash. A frozen version drifting means the
    #: numbers published against it no longer describe the agent in the tree. Either the
    #: change was unintended (fix it), or it is a real behaviour change and belongs in a
    #: NEW version (add it to VERSIONS; leave the old one alone). Updating a hash here is
    #: only correct when the fingerprint HARNESS below changed — never the agent.
    FINGERPRINTS = {
        'r3':     '0d4600063ed4c97e6a5fa086e64c3b58',
        'b2':     'ae58704e10f83caec60f3e2305cdbff2',
        'h1':     '928ecd4889b8324fefa7e7ce5fec3338',
        'h1b2':   '1e69928cb5851479969e99eda1773e14',
        'h7':     '6e8b0616229f20d10dcb7e1d8dc21f90',
        'h8':     'f0bad5eb23c388d0ee41fabe209ece7f',
        # H9 is H8 plus a rule that can only fire at two seats, so at four it MUST hash
        # identically to h8. That equality is asserted outright below — if this line ever
        # has to differ from h8's, the structure leaked past n=2 and H9 is not what it says.
        'h9':     'f0bad5eb23c388d0ee41fabe209ece7f',
        # h10 DOES differ from h8 at four seats — it changes weights, not just structure.
        # Recorded so the 2-seat yardstick cannot drift; NOT an endorsement of it at n=4,
        # where it measures -0.0015 +/- 0.0046 against h8 (i.e. nothing).
        'h10':    '93f35c7e08db4885940e353317385166',
        # h11: REFUTED (see RESEARCH_LOG 2026-07-20). Pinned, not endorsed — same status
        # as h7. Its 4-seat behaviour differs from h8 and measures -0.0026 +/- 0.0028.
        'h11':    'dbadb2ef3af70d4dcbf469663f8ac09c',
        'greedy': 'fbe35ae20ec60454be487304467eb5a1',
    }

    #: The same instrument at TWO seats. It exists because the 4-seat pins above are
    #: structurally blind to H9: its whole content is a branch that cannot fire at n=4, so
    #: every hash above stays green no matter what H9 does at two seats. A yardstick now
    #: quoted at 2, 3 and 4 seats needs its 2-seat behaviour frozen too.
    #:
    #: These hashes were written fresh, which is normally forbidden — legitimate here
    #: because the HARNESS is new, not the agent (see FINGERPRINTS above). The versions
    #: other than h9 are unchanged code; their 2-seat behaviour is simply being recorded
    #: for the first time. From here they are frozen on the same terms as the 4-seat ones.
    FINGERPRINTS_2P = {
        'r3':     'cbb5e7185635f22ef58d07e21db04bac',
        'b2':     '08baebf4ee2ff06a31034fb99e272d0c',
        'h1':     '367eb91b82210d84e8674a466dba9c18',
        'h1b2':   '730b7ffd6da4bcc6ebe88e8d62a65c00',
        'h7':     '127b1b5863aa5e038e5970fc17746279',
        'h8':     '61a458889cb917cf1bdc818a3d8c6fea',
        # The one entry that differs from h8 — that difference IS H9.
        'h9':     'b7d79754c0359c999cc6199306fb2072',
        'h10':    '2b4b03421edbffab687c70f5747f10ad',
        # Identical to h9's by construction (1/(n-1) == 1 at n=2) — asserted below too.
        'h11':    'b7d79754c0359c999cc6199306fb2072',
        'greedy': '6ac8a81247de982c82721d979f431f50',
    }

    def test_every_registered_version_is_fingerprinted(self):
        """A new version must not be able to join VERSIONS without getting pinned — that is
        how a yardstick quietly stops being frozen. Both seat counts, since a version can
        now differ from another at one count and not the other."""
        from agents.heuristic import VERSIONS
        self.assertEqual(set(self.FINGERPRINTS), set(VERSIONS))
        self.assertEqual(set(self.FINGERPRINTS_2P), set(VERSIONS))

    def _fingerprint(self, version, players=4):
        """Hash the full move sequence `version` plays across 60 seeded games."""
        import hashlib
        from agents.heuristic import make_heuristic
        h = hashlib.sha256()

        class Tap:
            """Wraps an agent, hashing every move it chooses."""
            def __init__(self, inner, tag):
                self.inner, self.tag = inner, tag

            def play(self, game):
                move = self.inner.play(game)
                h.update(f'{self.tag}:{game.curr}:{move[0]}:{move[1]}|'.encode())
                return move

        for seed in range(60):
            game = Game([Tap(make_heuristic(version), i) for i in range(players)],
                        seed=seed)
            turns = 0
            while not game.done() and turns < 600:
                game.next_turn()
                turns += 1
            winners = [i for i, hand in enumerate(game.hands) if not hand]
            h.update(f'END{seed}:{winners}:{turns}|'.encode())
        return h.hexdigest()[:32]

    def test_versions_are_behaviourally_frozen(self):
        """Every frozen version must still play move-for-move as it did when its numbers
        were published. See FINGERPRINTS above — including what to do if this fails."""
        for version, expected in self.FINGERPRINTS.items():
            self.assertEqual(
                self._fingerprint(version), expected,
                f'\n\n*** FROZEN VERSION {version!r} HAS DRIFTED ***\n'
                f'It no longer plays the game it played when its numbers were published,\n'
                f'so every result measured against it is now describing a different agent.\n'
                f'Do NOT just paste in the new hash — see FINGERPRINTS in agenttest.py.\n')

    def test_versions_are_behaviourally_frozen_at_two_seats(self):
        """The 4-seat pins cannot see a 2-seat change; this is the other half."""
        for version, expected in self.FINGERPRINTS_2P.items():
            self.assertEqual(
                self._fingerprint(version, players=2), expected,
                f'\n\n*** FROZEN VERSION {version!r} HAS DRIFTED AT TWO SEATS ***\n'
                f'Do NOT just paste in the new hash — see FINGERPRINTS in agenttest.py.\n')

    def test_h9_is_h8_at_three_and_four_seats_and_differs_at_two(self):
        """H9's central claim, as a test.

        H9 adds tempo credit for a card that returns the turn to us. STOP does that only
        when skipping the next player wraps back around to us — `(me + 2d) % n == me`,
        true iff n == 2. So H9 must be H8 move-for-move at every count above two, and must
        NOT be H8 at two. Both halves matter: without the first, H9 silently redefines the
        yardstick at the counts where h8's numbers were published; without the second, H9
        does nothing at all and the structural claim is empty."""
        for players in (3, 4):
            self.assertEqual(
                self._fingerprint('h9', players), self._fingerprint('h8', players),
                f'h9 diverges from h8 at {players} seats — the 2-seat structure leaked, '
                f'so every published {players}-seat number vs h8 is no longer comparable')
        self.assertNotEqual(
            self._fingerprint('h9', 2), self._fingerprint('h8', 2),
            'h9 plays two seats identically to h8 — the STOP-tempo rule never fires')

    def test_h11_is_h9_at_two_seats(self):
        """H11 prices a STOP at `1/(n-1)` of a full extra turn. At two seats that fraction
        is exactly 1, so H11 must BE H9 there — the generalization has to contain the
        special case, or the +0.026 H9 measured would not carry over."""
        self.assertEqual(self._fingerprint('h11', 2), self._fingerprint('h9', 2))
        for players in (3, 4):
            self.assertNotEqual(self._fingerprint('h11', players),
                                self._fingerprint('h9', players),
                                f'h11 adds nothing at {players} seats')

    def test_only_two_seats_returns_the_turn_on_a_STOP(self):
        """The arithmetic H9 rests on, isolated from the agent: a STOP skips one seat, so
        it lands back on the player who played it exactly when the table has two seats."""
        for n in range(2, 7):
            for d in (1, -1):
                returns_to_us = (0 + 2 * d) % n == 0
                self.assertEqual(returns_to_us, n == 2,
                                 f'STOP self-return at n={n}, dir={d}')

    def test_h9_prices_a_two_seat_STOP_exactly_one_tempo_above_h8(self):
        """The behavioural content, priced. Same game, same card, same everything except
        the structure flag: H9 must score a 2-seat STOP higher than H8 by exactly
        `w_plus_tempo` — the same credit PLUS already earned — and must score it
        identically at four seats."""
        from agents.heuristic import H8, H9, HeuristicAgent
        from game import Card, Color, Type

        stop = Card(Type.STOP, Color.RED)
        for players, expected_gap in ((2, H9.w_plus_tempo), (4, 0.0)):
            game = Game([HeuristicAgent(weights=H8) for _ in range(players)], seed=7)
            hand = [stop, Card(Type.FIVE, Color.RED), Card(Type.THREE, Color.BLUE)]
            scores = {}
            for name, weights in (('h8', H8), ('h9', H9)):
                agent = HeuristicAgent(weights=weights)
                agent._sync_model(game)
                scores[name] = agent._score_play(game, hand, stop, set(), None, False)
            self.assertAlmostEqual(scores['h9'] - scores['h8'], expected_gap, places=9,
                                   msg=f'STOP mispriced at {players} seats')

    def test_the_fingerprint_actually_detects_a_behaviour_change(self):
        """The guard on the guard. A fingerprint that cannot fail is decoration: a harness
        bug making `_fingerprint` constant (or no longer exercising the agent) would leave
        every assertion above passing vacuously, forever. So perturb one weight by 0.1 and
        require the hash to move.

        It must be `w_rich`, and the reason is a finding rather than a detail. The holds
        SATURATE (H8, RESEARCH_LOG 2026-07-14): `p_king` 5.0 -> 6.0 is INVISIBLE here —
        the hold is already decisive, so more of it flips no argmax and not one move
        changes. `w_rich` is the sensitive dimension (the tuner measured 0.8 -> 1.6 at
        -0.14), and it detects 0.1. A mutation test has to perturb somewhere the agent can
        actually feel."""
        import dataclasses as dc
        from agents.heuristic import VERSIONS, H8
        try:
            VERSIONS['_mutant'] = dc.replace(H8, w_rich=H8.w_rich + 0.1)
            self.assertNotEqual(self._fingerprint('_mutant'), self._fingerprint('h8'),
                                'the fingerprint cannot detect a behaviour change')
        finally:
            VERSIONS.pop('_mutant', None)

    def test_the_weight_pin_and_the_fingerprint_cover_each_others_blind_spots(self):
        """Why BOTH guards are needed — neither is sufficient alone, and each one's blind
        spot is exactly the other's job:

          * the fingerprint is blind to a behaviourally-inert weight edit (`p_king` 5.0 ->
            6.0 on a saturated hold: same hash, different data). The WEIGHT PIN catches it.
          * the weight pin is blind to a structural edit that leaves the weights alone
            (H1: same data, +0.183 of different behaviour). The FINGERPRINT catches it.

        Together they pin the version as both a value and an agent."""
        import dataclasses as dc
        from agents.heuristic import VERSIONS, H8
        try:
            # Inert to behaviour, visible in the data.
            VERSIONS['_inert'] = dc.replace(H8, p_king=6.0)
            self.assertEqual(self._fingerprint('_inert'), self._fingerprint('h8'))
            self.assertNotEqual(dc.asdict(VERSIONS['_inert']), dc.asdict(H8))
        finally:
            VERSIONS.pop('_inert', None)

    def test_r3_still_runs_so_the_old_published_numbers_stay_reproducible(self):
        """Promotion must not orphan the history. Everything published before 2026-07-14
        was measured against r3, and `heuristic:r3` must go on meaning exactly that."""
        from agents.heuristic import resolve_weights, R3, VERSIONS
        self.assertEqual(resolve_weights('r3'), R3)
        self.assertEqual(R3.refusal_mode, 'legacy')     # bug and all
        self.assertEqual(R3.p_king, 6.0)                # the 17-point tuning bug, preserved
        for name in ('r3', 'b2', 'h1', 'h1b2', 'h7', 'h8', 'greedy'):
            self.assertIn(name, VERSIONS)

    def test_spec_grammar(self):
        from agents.heuristic import resolve_weights, R3, B2_RETUNED, GREEDY
        self.assertEqual(resolve_weights('r3'), R3)
        self.assertEqual(resolve_weights('b2'), B2_RETUNED)
        self.assertEqual(resolve_weights('greedy'), GREEDY)
        # ablations and overrides apply to the reference
        self.assertEqual(resolve_weights('-hoard').w_reserve, 0.0)
        self.assertEqual(resolve_weights('p_king=1.5').p_king, 1.5)
        self.assertEqual(resolve_weights('block_hand_threshold=1').block_hand_threshold, 1)
        self.assertIs(resolve_weights('hold_wilds_in_run=false').hold_wilds_in_run, False)
        # unchanged fields keep the reference's values
        self.assertEqual(resolve_weights('p_king=1.5').w_nofin, R3.w_nofin)
        for bad in ('nonesuch', '-nonesuch', 'not_a_weight=3'):
            with self.assertRaises(ValueError):
                resolve_weights(bad)

    def test_a_spec_can_be_based_on_a_version_other_than_the_reference(self):
        """H-series ablations must be measured on top of H1's STRUCTURE. Without this, an
        override is implicitly based on REFERENCE ('r3' = legacy), so testing e.g.
        `block_hand_threshold=1` would price it on top of the refusal cliff H1 removed —
        measuring the cliff, not the blocker."""
        from agents.heuristic import resolve_weights, H1_B2
        w = resolve_weights('h1b2,block_hand_threshold=1')
        self.assertEqual(w.refusal_mode, 'structural')      # the base's structure, kept
        self.assertEqual(w.block_hand_threshold, 1)         # the override, applied
        self.assertEqual(w.p_king, H1_B2.p_king)            # everything else from the base
        self.assertEqual(resolve_weights('h1,-hoard').refusal_mode, 'structural')
        self.assertEqual(resolve_weights('h1,-hoard').w_reserve, 0.0)
        self.assertEqual(resolve_weights('h1b2,'), H1_B2)   # trailing comma is a no-op
        # An un-based override still resolves against REFERENCE — which B5 promoted from r3
        # to h8, so it now inherits the STRUCTURAL mode. (Before the promotion this same
        # spec silently carried r3's legacy structure, which is exactly why the leading-
        # version form had to exist.)
        self.assertEqual(resolve_weights('block_hand_threshold=1').refusal_mode, 'structural')
        self.assertEqual(resolve_weights('r3,block_hand_threshold=1').refusal_mode, 'legacy')

    def test_refusal_mode_is_validated(self):
        """The version anchor: H1 changes the DECISION STRUCTURE, so freezing the weight
        vector alone would not freeze behaviour. A version pins its mode; an unknown mode
        must fail loudly rather than silently fall back to the legacy scoring path."""
        from agents.heuristic import Weights
        with self.assertRaises(ValueError):
            Weights(refusal_mode='typo')


class H7RaceTest(unittest.TestCase):
    """H7: the agent knows how to BLOCK a near-winner but had no answer when it CANNOT
    block one — its scoring did not change at all in that situation. A human speeds up: a
    card kept for later has no later to be kept for."""

    def _game_with_near_winner_at(self, seat, hand):
        g = make_game(players=4)
        g.discard = [Card(Type.THREE, Color.BLUE)]
        g.state = State.NORMAL
        g.hands[0] = hand
        for s in (1, 2, 3):
            g.hands[s] = [Card(Type.FIVE, Color.RED)] * 6
        g.hands[seat] = [Card(Type.FIVE, Color.RED)]     # one card: about to win
        return g

    def test_racing_stops_charging_a_hold_penalty_for_spending_a_precious_card(self):
        """The mechanism. While racing, the hold-back terms are scaled away, so a card the
        agent was hoarding stops looking expensive.

        NB what this does NOT do: make the agent PREFER dumping the King over an equivalent
        number. Both shed exactly one card, so with the penalty gone they tie at 0.0 and the
        tie-break decides. That is correct — in Taki the only move that sheds MORE than one
        card is opening a TAKI run, which is why the hoard release (below) is the part of H7
        that can actually 'dump maximally'."""
        from agents.heuristic import H1_B2, H7
        hand = [Card(Type.KING), Card(Type.NINE, Color.BLUE)]
        g = self._game_with_near_winner_at(2, list(hand))     # NOT the next seat: unstoppable
        king = Card(Type.KING)
        plan = (frozenset(), None, False)

        held = HeuristicAgent(weights=H1_B2)._score_play(g, hand, king, *plan, racing=False)
        raced = HeuristicAgent(weights=H7)._score_play(g, hand, king, *plan, racing=True)
        self.assertEqual(held, -H1_B2.p_king)    # normally: "the King is precious"
        self.assertEqual(raced, 0.0)             # racing: "there is no later"
        self.assertGreater(raced, held)

    def test_racing_cashes_the_hoarded_taki_run_instead_of_guarding_it(self):
        """The half of H7 that actually sheds cards. A colored TAKI opens a run that dumps
        the whole color group in ONE turn; normally the agent guards that group (w_reserve)
        until the release trigger fires. With a near-winner it cannot stop, the trigger is
        now — it is the last turn that will ever come."""
        from agents.heuristic import H1_B2, H7
        taki = Card(Type.TAKI, Color.RED)
        # The red group is hoarded; the two blues are the "rest of the hand", so the normal
        # release trigger (rest nearly gone) does NOT fire and only racing can open the run.
        hand = [taki, Card(Type.FIVE, Color.RED), Card(Type.SEVEN, Color.RED),
                Card(Type.NINE, Color.RED), Card(Type.EIGHT, Color.BLUE),
                Card(Type.SIX, Color.BLUE)]

        g = self._game_with_near_winner_at(2, list(hand))
        g.discard = [Card(Type.THREE, Color.RED)]            # the TAKI is playable
        agent = HeuristicAgent(weights=H7)
        agent._sync_model(g)
        reserved, hoard_color, open_now, _legal = agent._hoard_plan(g, g.hands[0])
        self.assertEqual(hoard_color, Color.RED)             # the group IS being hoarded
        self.assertFalse(open_now)                           # and the trigger has NOT fired
        self.assertTrue(agent._racing(g, g.hands[0]))        # but we cannot stop the winner

        # Racing overrides the trigger: the run is opened NOW. Worth `w_open_hoard` (the
        # release bonus) plus `w_reserve` (the guard penalty, no longer charged).
        guarded = agent._score_play(g, hand, taki, reserved, hoard_color, False, racing=False)
        cashed = agent._score_play(g, hand, taki, reserved, hoard_color, True, racing=True)
        self.assertEqual(cashed - guarded, H7.w_open_hoard + H7.w_reserve)
        self.assertEqual(agent.play(g), (Action.PLAY_CARD, taki))

        # (h1b2 opens this particular run too — `w_taki_dump` x 3 already outweighs the
        # reserve here. H7's edge is that it does so DELIBERATELY, and in the positions
        # where the guard would otherwise win.)
        self.assertFalse(HeuristicAgent(weights=H1_B2)._racing(g, g.hands[0]))

    def test_not_racing_when_the_near_winner_is_the_next_seat_and_we_can_block(self):
        """The trigger's subtle half. STOP and +2 hit the NEIGHBOUR, so a blocker is only
        an answer when the near-winner IS the next seat. Then we are not racing — we block,
        and the hold-backs go on applying normally."""
        from agents.heuristic import H7
        agent = HeuristicAgent(weights=H7)
        hand = [Card(Type.STOP, Color.BLUE), Card(Type.KING), Card(Type.NINE, Color.BLUE)]

        g = self._game_with_near_winner_at(1, list(hand))     # next seat: blockable
        self.assertFalse(agent._racing(g, g.hands[0]))
        self.assertEqual(agent.play(g)[1], Card(Type.STOP, Color.BLUE))   # block them

        # Same blocker, but the near-winner sits elsewhere: the STOP cannot reach them, so
        # holding it is worthless and we are racing after all.
        g = self._game_with_near_winner_at(2, list(hand))
        self.assertTrue(agent._racing(g, g.hands[0]))

    def test_race_is_off_in_every_frozen_version_that_predates_it(self):
        from agents.heuristic import R3, B2_RETUNED, H1, H1_B2, GREEDY, H7
        for w in (R3, B2_RETUNED, H1, H1_B2, GREEDY):
            self.assertFalse(w.race)
        self.assertTrue(H7.race)

    def test_racing_cannot_buy_a_refusal_even_though_it_rescales_holds(self):
        """H7 rescales hold weights at runtime — exactly the unbounded move that would have
        been unsafe before H1. Under 'structural' no scale can produce a voluntary draw."""
        import dataclasses as dc
        from agents.heuristic import H7
        for scale in (0.0, 1.0, 100.0):      # 100x every hold: far past the old cliff
            w = dc.replace(H7, race_hold_scale=scale)
            agent = HeuristicAgent(weights=w)
            voluntary = 0
            for seed in range(12):
                g = Game([HeuristicAgent(weights=w) for _ in range(4)], seed=seed)
                for _ in range(120):
                    if g.done():
                        break
                    action, _ = agent.play(g)
                    plays = [(a, c) for a, c in g.valid_moves() if a is Action.PLAY_CARD]
                    if action is Action.DRAW and plays and not all(
                            len(g.hands[g.curr]) == 1
                            and c.type.value not in FINISHING_TYPE_VALUES for _, c in plays):
                        voluntary += 1
                    g.next_turn()
            self.assertEqual(voluntary, 0, f'refused at race_hold_scale={scale}')


class H1StructuralInvariantTest(unittest.TestCase):
    """H1 (PLAN.md): "never refuse to play" is a STRUCTURAL INVARIANT, not a tuning
    accident. Under 'legacy', `score_draw = -5.0` is a finite score competing in the same
    `max` as the plays, so ANY hold penalty above 5.0 silently converts "I would rather
    keep this" into "I would rather not play at all" — the 13-point cliff that R3's shipped
    `p_king = 6.0` falls straight off, and that B2's retune moved rather than removed.

    The invariant PLAN pre-registered: *no weight assignment can produce a voluntary draw.*
    That is a property of the STRUCTURE, so it is tested as one — over random weight
    vectors, including absurd ones — rather than by spot-checking the versions we ship.
    """

    def _voluntary_draws(self, weights, games=40, turn_cap=120):
        """DRAW while a legal, non-forbidden play exists (holdback.is_refusal's rule)."""
        agent = HeuristicAgent(weights=weights)
        voluntary = 0
        for seed in range(games):
            g = Game([HeuristicAgent(weights=weights) for _ in range(4)], seed=seed)
            for _ in range(turn_cap):
                if g.done():
                    break
                action, _ = agent.play(g)
                plays = [(a, c) for a, c in g.valid_moves() if a is Action.PLAY_CARD]
                if action is Action.DRAW and plays:
                    if not all(len(g.hands[g.curr]) == 1
                               and c.type.value not in FINISHING_TYPE_VALUES
                               for _, c in plays):
                        voluntary += 1
                g.next_turn()
        return voluntary

    def test_h1_drives_r3s_refusals_to_zero_without_touching_a_single_weight(self):
        """The headline. H1 carries R3's exact weight vector — p_king=6.0 and all — and
        yet cannot refuse. The cliff was never in the weights; it was in the structure."""
        from agents.heuristic import H1, R3
        self.assertEqual(dataclasses.asdict(H1) | {'refusal_mode': 'legacy'},
                         dataclasses.asdict(R3))       # same weights, different structure
        self.assertGreater(self._voluntary_draws(R3), 4)      # the yardstick refuses
        self.assertEqual(self._voluntary_draws(H1), 0)        # H1 does not

    def test_h1_also_removes_the_cliff_b2s_retune_left_behind(self):
        """B2's retuned point still drew 4 times in 40 games via `w_nofin=8.0` (H5). Under
        H1 that weight keeps its meaning as a PREFERENCE and loses its power to refuse."""
        from agents.heuristic import H1_B2, B2_RETUNED
        self.assertEqual(self._voluntary_draws(B2_RETUNED), 4)
        self.assertEqual(self._voluntary_draws(H1_B2), 0)

    def test_no_weight_assignment_can_produce_a_voluntary_draw(self):
        """The pre-registered invariant, as a property test. Hold weights become
        UNBOUNDED-SAFE: even at values far past the old cliff (a 500-point hold on the
        King!), 'structural' cannot be made to draw while a legal play exists. Under
        'legacy' these same vectors refuse constantly — asserted below, so this test
        cannot pass by accident (e.g. if the games ended before a hold ever bound)."""
        import random as _random
        from agents.heuristic import Weights
        rng = _random.Random(7)
        legacy_refusals = 0
        for _ in range(12):
            holds = dict(
                w_nofin=rng.uniform(0, 500), w_reserve=rng.uniform(0, 500),
                w_open_hoard=rng.uniform(0, 500), p_chcol=rng.uniform(0, 500),
                p_super_taki=rng.uniform(0, 500), p_king=rng.uniform(0, 500),
                w_save_blocker=rng.uniform(0, 500),
                king_cancel_min_penalty=rng.randint(0, 40),
                hold_wilds_in_run=bool(rng.getrandbits(1)),
            )
            self.assertEqual(
                self._voluntary_draws(Weights(refusal_mode='structural', **holds), games=8),
                0, f'structural refused with {holds}')
            legacy_refusals += self._voluntary_draws(
                Weights(refusal_mode='legacy', **holds), games=8)
        self.assertGreater(legacy_refusals, 0, 'legacy never refused — test is vacuous')

    def test_structural_still_draws_when_the_RULES_leave_nothing_to_play(self):
        """The invariant is "never refuse", not "never draw". With no legal play the agent
        must still draw — and it must also draw on the one rule-forced case: a lone PLUS,
        which the finishing rule forbids playing (the engine hands it straight back)."""
        from agents.heuristic import H1
        agent = HeuristicAgent(weights=H1)

        g = make_game()                       # nothing playable: no match, no wild
        g.discard = [Card(Type.THREE, Color.RED)]
        g.state = State.NORMAL
        g.hands[0] = [Card(Type.FIVE, Color.BLUE), Card(Type.SEVEN, Color.GREEN)]
        self.assertEqual(agent.play(g), (Action.DRAW, None))

        g = make_game()                       # a lone PLUS: legal to play, forbidden to end on
        g.discard = [Card(Type.PLUS, Color.RED)]
        g.state = State.NORMAL
        g.hands[0] = [Card(Type.PLUS, Color.RED)]
        self.assertIn((Action.PLAY_CARD, Card(Type.PLUS, Color.RED)), g.valid_moves())
        self.assertEqual(agent.play(g), (Action.DRAW, None))

    def test_structural_cancels_a_plus_two_with_the_King_rather_than_eat_the_pile(self):
        """The costliest refusal in the agent, and the one a `max()` never saw: under a
        pending +2, declining to cancel is a voluntary draw of `2 * draw_num` CARDS bought
        by `king_cancel_min_penalty`. There is no other legal play, so nothing exists for a
        preference to reorder — the invariant leaves only one honest answer."""
        from agents.heuristic import H1, R3
        g = make_game()
        g.discard = [Card(Type.PLUSTWO, Color.GREEN)]
        g.state = State.DRAW_TWO
        g.draw_num = 1
        g.hands[0] = [Card(Type.KING), Card(Type.THREE, Color.BLUE)]
        self.assertEqual(HeuristicAgent(weights=R3).play(g), (Action.DRAW, None))
        self.assertEqual(HeuristicAgent(weights=H1).play(g),
                         (Action.PLAY_CARD, Card(Type.KING)))

    def test_h4_structural_never_declines_a_kings_follow_up_while_a_play_exists(self):
        """H4: declining the King's free follow-up is a REFUSAL, so under H1 no weight may
        buy it. It was already dead code (0 fires in 200 games at the shipped -2.0) — but
        the measurement showed that ZEROING the weight wakes it up and costs -0.0004,
        because at 0 it outbids negative-scoring plays. So the branch is gone under
        'structural', and `score_decline_king` cannot resurrect it at any value."""
        from agents.heuristic import H1
        import dataclasses as dc
        for value in (0.0, 50.0, 500.0):     # any of these would win a legacy max()
            w = dc.replace(H1, score_decline_king=value)
            g = make_game()
            g.discard = [Card(Type.KING)]
            g.state = State.KING
            g.hands[0] = [Card(Type.NINE, Color.BLUE), Card(Type.TWO, Color.RED)]
            moves = g.valid_moves()
            self.assertIn((Action.CLOSE_TAKI, None), moves)     # declining IS on offer
            action, _ = HeuristicAgent(weights=w).play(g)
            self.assertEqual(action, Action.PLAY_CARD, f'declined at {value}')

    def test_holds_still_RANK_plays_under_h1_they_just_cannot_veto_playing(self):
        """H1 must not flatten the agent into `greedy`. The hold weights keep doing their
        job — reordering the plays — they simply cannot outvote playing at all. Here a
        precious King and a plain number are both legal: the agent plays the number."""
        from agents.heuristic import H1
        g = make_game()
        g.discard = [Card(Type.THREE, Color.BLUE)]
        g.state = State.NORMAL
        g.hands[0] = [Card(Type.KING), Card(Type.NINE, Color.BLUE),
                      Card(Type.TWO, Color.RED)]
        action, card = HeuristicAgent(weights=H1).play(g)
        self.assertEqual(action, Action.PLAY_CARD)
        self.assertEqual(card, Card(Type.NINE, Color.BLUE))   # kept the King, still played
