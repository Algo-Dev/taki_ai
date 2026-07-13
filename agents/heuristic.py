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

import dataclasses

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


@dataclasses.dataclass(frozen=True)
class Weights:
    """The agent's scoring economy, as data (PLAN.md B2).

    Defaults reproduce the R3 agent exactly. Broken out so the HOLD-BACK terms can be
    switched off individually: B2 asks whether playing fewer cards now ever pays, and
    this agent is the only one in the repo that does it on purpose, in terms we control.

    The hold-back terms are the ones marked HOLD below. They work only because
    `score_draw` is a FINITE score competing in the same `max`: any hold penalty above
    5.0 makes the agent voluntarily DRAW rather than play a legal card.

    `score_forbidden` is NOT a hold-back term — it encodes the engine's finishing rule
    (you may not end on a PLUS) and must stay on in every ablation.
    """
    w_deny: float = W_DENY
    w_rich: float = W_RICH
    w_block: float = W_BLOCK
    w_chdir_block: float = W_CHDIR_BLOCK
    w_plus_tempo: float = W_PLUS_TEMPO
    w_taki_dump: float = W_TAKI_DUMP
    score_draw: float = SCORE_DRAW
    score_forbidden: float = SCORE_FORBIDDEN   # rules, not strategy — never ablate

    # --- HOLD-BACK terms (B2 ablates these) ---
    w_save_blocker: float = W_SAVE_BLOCKER     # HOLD: keep STOP/+2 with no threat
    w_nofin: float = W_NOFIN                   # HOLD: keep a finisher with a small hand
    w_reserve: float = W_RESERVE               # HOLD: don't break the hoarded color group
    w_open_hoard: float = W_OPEN_HOARD         # HOLD: (the hoard's release trigger)
    p_chcol: float = P_CHCOL                   # HOLD: keep Change Color
    p_super_taki: float = P_SUPER_TAKI         # HOLD: keep Super TAKI
    p_king: float = P_KING                     # HOLD: keep the King
    score_decline_king: float = SCORE_DECLINE_KING  # HOLD: decline the King's follow-up
    hold_wilds_in_run: bool = True             # HOLD: never spend a wild inside a TAKI run
    # HOLD: under a pending +2, keep the King unless it cancels at least this much.
    # A threshold rather than a weight, but the same behaviour — eat 2 cards now to keep
    # the King for later. Ablating it means: always cancel with the King if you hold one.
    king_cancel_min_penalty: int = KING_CANCEL_MIN_PENALTY
    # B7: a next player holding <= this many cards counts as a threat worth blocking.
    # NOT a hold-back term — it decides WHEN to spend a blocker, i.e. selectivity.
    block_hand_threshold: int = BLOCK_HAND_THRESHOLD

    # --- Behaviour version (NOT a weight) -------------------------------------
    # Which DECISION STRUCTURE the agent uses, i.e. how refusing to play is decided:
    #   'legacy'     — `score_draw` is a finite score competing in the same `max` as
    #                  the plays, so a hold penalty above 5.0 turns "I would rather
    #                  keep this" into "I would rather not play at all" (the 13-point
    #                  refusal cliff B2 measured). This is what R3 shipped.
    #   'structural' — H1: DRAWING is decided by the RULES alone, never by a weight. The
    #                  agent draws only when it has no play the rules allow it to make;
    #                  hold penalties rank the plays against each other and can never
    #                  outvote playing at all. Hold weights become UNBOUNDED-SAFE: you can
    #                  say "I really want to keep the King" without that ever meaning "so I
    #                  will draw instead". Covers both places the agent can literally draw
    #                  (`_score_and_pick` and `_play_draw_two`); the CLOSE_TAKI refusals
    #                  (decline-King, hold-wilds-in-run) are untouched — see H4.
    # Pinned per named version below, so a frozen yardstick stays reproducible even
    # after the default flips.
    refusal_mode: str = 'legacy'

    def __post_init__(self):
        if self.refusal_mode not in REFUSAL_MODES:
            raise ValueError(f'unknown refusal_mode {self.refusal_mode!r}; '
                             f'choose from {sorted(REFUSAL_MODES)}')


#: Decision structures the scoring code implements.
REFUSAL_MODES = ('legacy', 'structural')


#: Every hold-back term off. Same agent, no patience: it plays the highest-scoring card
#: it can and never draws when a legal play exists. This is the B2 control.
GREEDY = Weights(
    w_save_blocker=0.0,
    w_nofin=0.0,
    w_reserve=0.0,
    w_open_hoard=0.0,
    p_chcol=0.0,
    p_super_taki=0.0,
    p_king=0.0,
    score_decline_king=-100.0,   # never decline a free card
    hold_wilds_in_run=False,
    king_cancel_min_penalty=0,   # always spend the King to cancel a +2
)

#: Single-term ablations: each turns exactly ONE hold-back behaviour off, so the
#: head-to-head margin against the full agent prices that behaviour on its own.
ABLATIONS = {
    'hoard': dict(w_reserve=0.0, w_open_hoard=0.0),
    'wilds': dict(p_chcol=0.0, p_super_taki=0.0, p_king=0.0, hold_wilds_in_run=False),
    'blocker': dict(w_save_blocker=0.0),
    'finisher': dict(w_nofin=0.0),
    'king_follow': dict(score_decline_king=-100.0),
    'king_cancel': dict(king_cancel_min_penalty=0),
}


def ablated(name):
    """The full agent with exactly one hold-back behaviour removed."""
    return dataclasses.replace(Weights(), **ABLATIONS[name])


# --- Named versions: the yardstick, frozen ----------------------------------
# The heuristic is the project's ranking metric (CLAUDE.md: "rank on the heuristic,
# not on random"), so CHANGING IT SILENTLY REWRITES THE PROJECT'S HISTORY: every
# published "vs heuristic" number (R3's 0.343 for A8, A9-rules' 0.348, R6's 0.378)
# was measured against R3 as shipped. The H-series improves this agent, so each
# version it passes through is pinned here by name and never edited again.
#
# A version pins BEHAVIOUR, not just numbers: `refusal_mode` freezes the decision
# structure too, so H1's rewrite cannot retroactively change what 'r3' means.

#: R3 as shipped (2026-07-12) — the opponent every pre-H-series number was measured
#: against. It carries B2's 17-point tuning bug (p_king=6.0 > |score_draw|=5.0, so it
#: DRAWS rather than play its King). Preserved deliberately: it is the yardstick those
#: numbers refer to. Do not "fix" it — fix its successor.
R3 = Weights(refusal_mode='legacy')

#: B2's retuned point (RESEARCH_LOG 2026-07-13): the WILD hold-penalties pulled down to
#: the draw threshold. Worth 0.850 -> 0.899 vs random and parity with the DQN champion.
#: The single source of truth for "retuned" — b2_block_price.py and tune_heuristic.py
#: each had their own copy, and they had already drifted apart (p_king 5.0 vs 4.9).
#:
#: IT IS NOT REFUSAL-FREE, contrary to how B2 described it. `w_nofin` is still 8.0 —
#: above |score_draw|=5.0 — and it still makes the agent DRAW while holding a legal play
#: (4 voluntary draws in 40 games; `agenttest.FrozenVersionTest`). B2 moved the cliff
#: rather than removing it. That is PLAN.md H5, and it is the case for H1: as long as
#: refusal is a scoring outcome, retuning can always leave one behind.
B2_RETUNED = Weights(
    refusal_mode='legacy',
    p_king=5.0,
    p_chcol=4.5,
    p_super_taki=4.5,
    w_reserve=4.0,
    king_cancel_min_penalty=0,
    hold_wilds_in_run=False,
)

#: H1 (PLAN.md): R3's weights, UNCHANGED — only the decision structure differs. Refusing
#: to play is no longer something a weight can buy, so R3's `p_king = 6.0 > 5.0` stops
#: meaning "draw rather than spend the King" and goes back to meaning what it reads like:
#: "keep the King if there is anything else to play". The point of isolating it this way is
#: that H1 vs R3 measures the STRUCTURE alone, with no retuning confounded into it — where
#: B2 measured the retuning alone, and (we now know) left a cliff behind at `w_nofin`.
H1 = dataclasses.replace(R3, refusal_mode='structural')

#: H1's structure applied to B2's retuned weights: the two fixes together. B2 pulled the
#: WILD holds under the threshold but missed `w_nofin`, so this is also the version where
#: that leftover cliff finally cannot bite.
H1_B2 = dataclasses.replace(B2_RETUNED, refusal_mode='structural')

VERSIONS = {
    'r3': R3,
    'b2': B2_RETUNED,
    'h1': H1,
    'h1b2': H1_B2,
    'greedy': GREEDY,
}

#: The reference opponent: what a bare `heuristic` spec means, and therefore what
#: `eval.py --opponent heuristic` measures. STAYS 'r3' until B5 formally promotes a
#: successor — so no existing command changes meaning while the H-series is in flight.
#: Promoting = change this line, then re-run the champion against the new reference and
#: record BOTH numbers in RESEARCH_LOG.md.
REFERENCE = 'r3'


def resolve_weights(spec=''):
    """Resolve a heuristic weight spec. The one grammar, shared by every entry point.

      ''  / 'reference'   -> the current reference version (REFERENCE)
      '<version>'         -> a named version: r3, b2, greedy
      '-<name>'           -> the reference with ONE hold-back behaviour ablated
      'k=v,k=v'           -> the reference with individual weights overridden, which is
                             how a weight gets SWEPT rather than merely switched off
    """
    spec = (spec or '').strip()
    if spec in ('', 'reference'):
        return VERSIONS[REFERENCE]
    if spec in VERSIONS:
        return VERSIONS[spec]
    base = VERSIONS[REFERENCE]
    if spec.startswith('-'):
        name = spec[1:]
        if name not in ABLATIONS:
            raise ValueError(f'unknown ablation {name!r}; choose from {sorted(ABLATIONS)}')
        return dataclasses.replace(base, **ABLATIONS[name])
    fields = {f.name: f.type for f in dataclasses.fields(Weights)}
    overrides = {}
    for part in spec.split(','):
        key, sep, val = part.partition('=')
        key = key.strip()
        if not sep:
            raise ValueError(
                f'unknown heuristic spec {spec!r}; expected a version '
                f'{sorted(VERSIONS)}, an ablation -<name>, or k=v overrides')
        if key not in fields:
            raise ValueError(f'unknown weight {key!r}; choose from {sorted(fields)}')
        overrides[key] = _coerce(fields[key], val.strip())
    return dataclasses.replace(base, **overrides)


def _coerce(ftype, val):
    if ftype is bool:
        return val.lower() in ('1', 'true', 'yes')
    if ftype is int:
        return int(val)
    if ftype is str:
        return val
    return float(val)


def make_heuristic(spec=''):
    """A HeuristicAgent from a spec string (see resolve_weights)."""
    return HeuristicAgent(weights=resolve_weights(spec))


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
    def __init__(self, weights=None):
        self.w = weights if weights is not None else Weights()
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
        """B1: stack a +2 if held; King-cancel only when the penalty is big.

        H1: under 'structural', `king_cancel_min_penalty` is INERT here. Declining to
        cancel means eating the pile to keep the King — a hold weight buying a voluntary
        draw, i.e. the exact cliff, and the most expensive one in the agent (the draw is
        `2 * draw_num` cards, not one). With no other legal play there is nothing for a
        preference to reorder, so the invariant leaves only one honest answer: cancel.
        """
        plus_twos = [(a, c) for a, c in moves
                     if a is Action.PLAY_CARD and c.type is Type.PLUSTWO]
        if plus_twos:
            return self._best_by_color(game, plus_twos)
        if self.w.refusal_mode == 'structural' \
                or 2 * game.draw_num >= self.w.king_cancel_min_penalty:
            for a, c in moves:
                if a is Action.PLAY_CARD and c.type is Type.KING:
                    return a, c
        return Action.DRAW, None

    def _play_in_run(self, game, hand, moves):
        """B3: dump the run's color; never spend wilds; order so inert action
        cards go first, a number closes out a hand-emptying run, and a blocker
        lands last (its effect applies at CLOSE_TAKI) when a threat looms."""
        if self.w.hold_wilds_in_run:
            colored = [(a, c) for a, c in moves
                       if a is Action.PLAY_CARD and c.color is not Color.NONE
                       and c.type is not Type.CHCOL]
        else:
            # Ablated: spend wilds in the run like any other card.
            colored = [(a, c) for a, c in moves if a is Action.PLAY_CARD]
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
        structural = self.w.refusal_mode == 'structural'
        scored = []
        for move in moves:
            action, card = move
            if action is Action.DRAW:
                if structural:
                    # H1: drawing is not an option to be outbid — it is the fallback when
                    # the rules leave nothing to play. Keep it out of the `max` entirely.
                    continue
                score = self.w.score_draw
            elif action is Action.CLOSE_TAKI:  # only reachable declining a King
                score = self.w.score_decline_king
            else:
                score = self._score_play(game, hand, card, reserved,
                                         hoard_color, open_hoard)
                if structural and score == self.w.score_forbidden:
                    # The finishing rule forbids it (a lone PLUS). Not a preference — a
                    # rule. Dropping it here is what lets DRAW remain reachable when it is
                    # the ONLY thing the rules permit, without any weight voting for it.
                    continue
            scored.append((score, move))
        if not scored:                       # structural: every play is rule-forbidden
            return Action.DRAW, None
        best = max(scored, key=lambda s: (s[0], -action_to_scalar(*s[1])))
        return best[1]

    def _score_play(self, game, hand, card, reserved, hoard_color, open_hoard):
        w = self.w
        # B5 hard rule: never end the hand on a non-finisher (engine penalty).
        if len(hand) == 1 and card.type.value not in FINISHING_TYPE_VALUES:
            return w.score_forbidden

        score = 0.0
        next_seat = self._seat_after_playing(game, card)
        result_color = card.color  # what the next player must answer

        # B8 color denial + B9 richness (only when an opponent acts next).
        if next_seat != game.curr and result_color is not Color.NONE:
            score += w.w_deny * self._model.lacks_color(next_seat, result_color)
            # `c is not card` (identity) excludes the played card; an expanded
            # CHCOL choice is a fresh object, so nothing is excluded — correct,
            # since playing a recolored wild spends no card of that color.
            remaining = sum(1 for c in hand
                            if c.color is result_color and c is not card)
            score += w.w_rich * remaining

        # B7 blocking economy.
        threat = self._threat_seat(game)
        if card.type in _BLOCKER_TYPES:
            score += w.w_block if threat is not None else -w.w_save_blocker
        elif card.type is Type.CHDIR and threat is not None:
            behind = (game.curr - game.dir) % len(game.agents)
            if len(game.hands[behind]) > len(game.hands[threat]):
                score += w.w_chdir_block
        elif card.type is Type.PLUS:
            score += w.w_plus_tempo

        # B10 wild hold-penalties.
        if card.type is Type.CHCOL:
            score -= w.p_chcol
        elif card.type is Type.KING:
            score -= w.p_king
        elif card.type is Type.TAKI and card.color is Color.NONE:
            score -= w.p_super_taki
        elif card.type is Type.TAKI:
            # A colored TAKI opens a run: worth roughly the color group it dumps.
            group = sum(1 for c in hand
                        if c.color is card.color and c is not card)
            score += w.w_taki_dump * group
            if open_hoard and card.color is hoard_color:
                score += w.w_open_hoard

        # B6 hoard reserve.
        if card in reserved and not (open_hoard and card.type is Type.TAKI
                                     and card.color is hoard_color):
            score -= w.w_reserve

        # B5 soft: with a small hand, keep at least one finisher.
        if len(hand) <= SMALL_HAND:
            has_finisher_after = any(
                c.type.value in FINISHING_TYPE_VALUES for c in hand if c is not card)
            if not has_finisher_after:
                score -= w.w_nofin

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
        if len(game.hands[nxt]) <= self.w.block_hand_threshold:
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
                s += self.w.w_deny * self._model.lacks_color(nxt, card.color)
                s += self.w.w_rich * sum(1 for c in hand
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
