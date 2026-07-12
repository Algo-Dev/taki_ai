"""Hand-crafted heuristic Taki agent (PLAN.md R3).

An independent, non-lineage yardstick: rule-based play with no trained network.

INFORMATION CONTRACT — this agent deliberately restricts itself to the human
information set (the project's design constraint, see CLAUDE.md). Although the
`game` object exposes everything, the agent reads ONLY:
  - its own hand (`game.hands[game.curr]`),
  - opponents' hand SIZES (`len(game.hands[i])`), never their contents,
  - `game.shown_card()`, `game.taki_color`, `game.state`, `game.draw_num`,
    `game.dir`, `len(game.deck)`,
  - `game.valid_moves()`,
  - `game.history` — the public table-event log (who played/drew what, and the
    active color at the time), i.e. what a human at the table would remember.
It never reads opponents' hand contents, the deck's contents/order, or the
full discard list.

Behaviours (numbering matches the R3 plan):
  B1  +2 stacking / King cancel under a pending +2
  B2  King follow-up: play the scored-best card, decline only if all plays hurt
  B3  open-TAKI runs: dump the color, inert action cards first, close on a
      number if the run empties the hand, save a blocker for last vs a threat
  B4  win immediately when a legal finisher empties the hand
  B5  never empty the hand on a non-finisher; with a small hand keep a finisher
  B6  colored-TAKI hoard: reserve the TAKI + its color group for one endgame run
  B7  block a near-winner with STOP/+2 (CHDIR when the player behind is safer)
  B8  color denial: play into colors the next player is believed to lack,
      inferred from their draws (belief decays per card they draw since)
  B9  color richness: stay in colors we hold many of
  B10 hold wilds (CHCOL / Super TAKI / King) unless nothing better exists
  B11 play over draw
  B12 deterministic tie-break by action scalar (bit-reproducible, no RNG)
"""

from game import (Action, Card, Color, Type, State, FINISHING_TYPE_VALUES,
                  NUMBER_TYPE_VALUES, action_to_scalar)

# --- Opponent model (B8) ---------------------------------------------------
# P(a random unseen card is a given color) = 28/120 in this deck (14 types x 2
# copies per color, 120 cards total): each card an opponent draws keeps a
# "lacks color c" belief alive with probability ~1 - 28/120.
LACK_DECAY = 1.0 - 28.0 / 120.0

# --- Thresholds -------------------------------------------------------------
BLOCK_HAND_THRESHOLD = 2   # B7: next player with <= this many cards is a threat
SMALL_HAND = 3             # B5: soft finisher-preservation kicks in at <= this
HOARD_MIN_GROUP = 2        # B6: reserve a colored TAKI with >= this many same-color cards
KING_CANCEL_MIN_PENALTY = 4  # B1: burn a King on a pending +2 only if it saves >= this

# --- Scoring weights (PLAY_CARD base score is 0) ----------------------------
W_DENY = 3.0        # B8: next player believed to lack the resulting color
W_RICH = 0.8        # B9: per card of the resulting color remaining in hand
W_BLOCK = 4.0       # B7: STOP/+2 against a threat
W_CHDIR_BLOCK = 2.0  # B7: CHDIR when the player behind is safer than the threat
W_SAVE_BLOCKER = 0.7  # B7: mild hold-back for STOP/+2 when no one is a threat
W_PLUS_TEMPO = 1.0  # PLUS gives a free extra turn
W_NOFIN = 8.0       # B5 soft: play leaves a small hand with no finisher
W_RESERVE = 10.0    # B6: breaking the hoard group
W_OPEN_HOARD = 15.0  # B6: opening the hoarded TAKI run when the trigger is met
W_TAKI_DUMP = 1.2   # a colored TAKI is worth ~this per same-color card it can dump
P_CHCOL = 3.0       # B10 hold-penalties for wilds
P_SUPER_TAKI = 4.0
P_KING = 6.0
SCORE_DRAW = -5.0   # B11: drawing loses to almost any play
SCORE_DECLINE_KING = -2.0  # B2: declining the follow-up beats only bad plays
SCORE_FORBIDDEN = -100.0   # B5 hard: emptying the hand on a non-finisher

_INFORMATIVE_DRAW_STATES = (State.NORMAL.value, State.PLUS.value)
_BLOCKER_TYPES = (Type.STOP, Type.PLUSTWO)
_WILD_TYPES = (Type.CHCOL, Type.KING)  # colorless; Super TAKI is Card(TAKI, NONE)


class _OpponentModel:
    """Per-seat 'lacks color' beliefs, built from the public event log (B8).

    A draw while color c was active (and a play was freely refusable, i.e. not
    under a pending +2) sets lacks[seat][c] = 1, then every card that seat
    draws — including that one — decays all their beliefs by LACK_DECAY per
    card. Playing a genuinely colored card (not a recolored wild) clears the
    belief for that color. NB: DRAW is always legal in this engine even with
    playable cards, so vs a random opponent this is weak evidence — the belief
    is a scoring weight, not a certainty.
    """

    def __init__(self, num_players):
        self.lacks = [{c.value: 0.0 for c in Color if c is not Color.NONE}
                      for _ in range(num_players)]

    def observe(self, event):
        kind, seat = event[0], event[1]
        if kind == 'draw':
            _, _, n_drawn, prev_state, active_color = event
            if prev_state in _INFORMATIVE_DRAW_STATES \
                    and active_color != Color.NONE.value:
                self.lacks[seat][active_color] = 1.0
            self._decay(seat, n_drawn)
        elif kind == 'penalty_draw':
            self._decay(seat, event[2])
        elif kind == 'play':
            _, _, type_value, color_value, _, _ = event
            # A colored card proves they held that color — unless the "color"
            # was chosen (CHCOL) rather than held.
            if color_value != Color.NONE.value and type_value != Type.CHCOL.value:
                self.lacks[seat][color_value] = 0.0

    def _decay(self, seat, n_drawn):
        factor = LACK_DECAY ** n_drawn
        for c in self.lacks[seat]:
            self.lacks[seat][c] *= factor

    def lacks_color(self, seat, color):
        if color is Color.NONE:
            return 0.0
        return self.lacks[seat][color.value]


class HeuristicAgent:
    def __init__(self):
        self._model = None
        self._cursor = 0

    def reseed(self, seed=None):
        """Eval-loop hook (called once per game): forget the previous game."""
        self._model = None
        self._cursor = 0

    # --- main entry ---------------------------------------------------------

    def play(self, game):
        me = game.curr
        self._sync_model(game)
        hand = game.hands[me]
        moves = game.valid_moves()

        if game.state is State.DRAW_TWO:
            return self._play_draw_two(game, moves)
        if game.state is State.TAKI or game.state is State.SUPER_TAKI:
            return self._play_in_run(game, hand, moves)

        # B4: win now if any single play legally empties the hand.
        if len(hand) == 1:
            for action, card in moves:
                if action is Action.PLAY_CARD \
                        and card.type.value in FINISHING_TYPE_VALUES:
                    return action, card

        return self._score_and_pick(game, hand, moves)

    # --- opponent-model upkeep ----------------------------------------------

    def _sync_model(self, game):
        history = game.history
        if self._model is None or len(history) < self._cursor \
                or len(self._model.lacks) != len(game.agents):
            self._model = _OpponentModel(len(game.agents))
            self._cursor = 0
        # Observe every seat's public events (including our own — harmless, we
        # never query beliefs about ourselves). This keeps the model correct
        # when one HeuristicAgent object fills several opponent seats, as
        # eval.play_match does: the model is a neutral table observer.
        for event in history[self._cursor:]:
            self._model.observe(event)
        self._cursor = len(history)

    # --- state handlers -------------------------------------------------------

    def _play_draw_two(self, game, moves):
        """B1: stack a +2 if held; King-cancel only when the penalty is big."""
        plus_twos = [(a, c) for a, c in moves
                     if a is Action.PLAY_CARD and c.type is Type.PLUSTWO]
        if plus_twos:
            return self._best_by_color(game, plus_twos)
        if 2 * game.draw_num >= KING_CANCEL_MIN_PENALTY:
            for a, c in moves:
                if a is Action.PLAY_CARD and c.type is Type.KING:
                    return a, c
        return Action.DRAW, None

    def _play_in_run(self, game, hand, moves):
        """B3: dump the run's color; never spend wilds; order so inert action
        cards go first, a number closes out a hand-emptying run, and a blocker
        lands last (its effect applies at CLOSE_TAKI) when a threat looms."""
        colored = [(a, c) for a, c in moves
                   if a is Action.PLAY_CARD and c.color is not Color.NONE
                   and c.type is not Type.CHCOL]
        # Excluded: hand-emptying plays on a non-finisher (B5 hard rule).
        playable = [(a, c) for a, c in colored
                    if not (len(hand) == 1 and c.type.value not in FINISHING_TYPE_VALUES)]
        if not playable:
            return Action.CLOSE_TAKI, None
        threat = self._threat_seat(game) is not None
        # The run empties the hand only if every card is of the run's color
        # (colorless wilds don't count — we refuse to spend them in-run).
        will_empty_hand = all(c.color is game.taki_color for c in hand)

        def order(move):
            _, card = move
            is_number = card.type.value in NUMBER_TYPE_VALUES
            is_blocker = card.type in _BLOCKER_TYPES
            if will_empty_hand:
                last = is_number          # close the run winning on a number
            elif threat and any(c.type in _BLOCKER_TYPES for _, c in playable):
                last = is_blocker         # close on the blocker for its effect
            else:
                last = is_number          # default: keep numbers (finishers) late
            return (last, action_to_scalar(*move))

        return min(playable, key=order)

    # --- general scoring ------------------------------------------------------

    def _score_and_pick(self, game, hand, moves):
        reserved, hoard_color, open_hoard = self._hoard_plan(game, hand)
        scored = []
        for move in moves:
            action, card = move
            if action is Action.DRAW:
                score = SCORE_DRAW
            elif action is Action.CLOSE_TAKI:  # only reachable declining a King
                score = SCORE_DECLINE_KING
            else:
                score = self._score_play(game, hand, card, reserved,
                                         hoard_color, open_hoard)
            scored.append((score, move))
        best = max(scored, key=lambda s: (s[0], -action_to_scalar(*s[1])))
        return best[1]

    def _score_play(self, game, hand, card, reserved, hoard_color, open_hoard):
        # B5 hard rule: never end the hand on a non-finisher (engine penalty).
        if len(hand) == 1 and card.type.value not in FINISHING_TYPE_VALUES:
            return SCORE_FORBIDDEN

        score = 0.0
        next_seat = self._seat_after_playing(game, card)
        result_color = card.color  # what the next player must answer

        # B8 color denial + B9 richness (only when an opponent acts next).
        if next_seat != game.curr and result_color is not Color.NONE:
            score += W_DENY * self._model.lacks_color(next_seat, result_color)
            # `c is not card` (identity) excludes the played card; an expanded
            # CHCOL choice is a fresh object, so nothing is excluded — correct,
            # since playing a recolored wild spends no card of that color.
            remaining = sum(1 for c in hand
                            if c.color is result_color and c is not card)
            score += W_RICH * remaining

        # B7 blocking economy.
        threat = self._threat_seat(game)
        if card.type in _BLOCKER_TYPES:
            score += W_BLOCK if threat is not None else -W_SAVE_BLOCKER
        elif card.type is Type.CHDIR and threat is not None:
            behind = (game.curr - game.dir) % len(game.agents)
            if len(game.hands[behind]) > len(game.hands[threat]):
                score += W_CHDIR_BLOCK
        elif card.type is Type.PLUS:
            score += W_PLUS_TEMPO

        # B10 wild hold-penalties.
        if card.type is Type.CHCOL:
            score -= P_CHCOL
        elif card.type is Type.KING:
            score -= P_KING
        elif card.type is Type.TAKI and card.color is Color.NONE:
            score -= P_SUPER_TAKI
        elif card.type is Type.TAKI:
            # A colored TAKI opens a run: worth roughly the color group it dumps.
            group = sum(1 for c in hand
                        if c.color is card.color and c is not card)
            score += W_TAKI_DUMP * group
            if open_hoard and card.color is hoard_color:
                score += W_OPEN_HOARD

        # B6 hoard reserve.
        if card in reserved and not (open_hoard and card.type is Type.TAKI
                                     and card.color is hoard_color):
            score -= W_RESERVE

        # B5 soft: with a small hand, keep at least one finisher.
        if len(hand) <= SMALL_HAND:
            has_finisher_after = any(
                c.type.value in FINISHING_TYPE_VALUES for c in hand if c is not card)
            if not has_finisher_after:
                score -= W_NOFIN

        return score

    # --- B6 hoard plan --------------------------------------------------------

    def _hoard_plan(self, game, hand):
        """Pick the colored TAKI with the largest same-color group to reserve.

        Returns (reserved_cards, hoard_color, open_now). open_now is True when
        the rest of the hand is nearly gone AND the run can end legally (it
        contains a number, or the run won't empty the hand)."""
        best = (None, [])
        for card in hand:
            if card.type is Type.TAKI and card.color is not Color.NONE:
                group = [c for c in hand
                         if c.color is card.color and c is not card]
                if len(group) > len(best[1]):
                    best = (card, group)
        taki, group = best
        if taki is None or len(group) < HOARD_MIN_GROUP:
            return frozenset(), None, False
        reserved = frozenset([id(taki)] + [id(c) for c in group])
        # Wrap membership test: Card defines value equality, so use identity.
        reserved_cards = _IdentitySet(reserved)
        outside = len(hand) - 1 - len(group)
        run_can_finish = any(c.type.value in NUMBER_TYPE_VALUES for c in group)
        open_now = outside <= 1 and (run_can_finish or outside > 0)
        return reserved_cards, taki.color, open_now

    # --- helpers ----------------------------------------------------------------

    def _seat_after_playing(self, game, card):
        """Which seat faces the table after we play `card` (for denial scoring)."""
        n = len(game.agents)
        me, d = game.curr, game.dir
        if card.type is Type.STOP:
            return (me + 2 * d) % n
        if card.type is Type.CHDIR:
            return (me - d) % n
        if card.type in (Type.PLUS, Type.KING):
            return me  # we act again
        if card.type is Type.TAKI:
            return me  # run continues on our turn
        return (me + d) % n

    def _threat_seat(self, game):
        """The next opponent, if they are close to winning (B7)."""
        nxt = (game.curr + game.dir) % len(game.agents)
        if len(game.hands[nxt]) <= BLOCK_HAND_THRESHOLD:
            return nxt
        return None

    def _best_by_color(self, game, plays):
        """Among same-type plays (e.g. +2s), pick by denial+richness, then B12."""
        hand = game.hands[game.curr]

        def key(move):
            _, card = move
            s = 0.0
            nxt = self._seat_after_playing(game, card)
            if nxt != game.curr and card.color is not Color.NONE:
                s += W_DENY * self._model.lacks_color(nxt, card.color)
                s += W_RICH * sum(1 for c in hand
                                  if c.color is card.color and c is not card)
            return (-s, action_to_scalar(*move))

        return min(plays, key=key)


class _IdentitySet:
    """Membership by object identity (Card defines equality by value)."""

    def __init__(self, ids):
        self._ids = ids

    def __contains__(self, card):
        return id(card) in self._ids

    def __len__(self):
        return len(self._ids)
