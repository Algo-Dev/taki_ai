"""Hand-crafted heuristic Taki agent (PLAN.md R3) — and the project's VERSIONED, FROZEN yardstick.

An independent, non-lineage yardstick: rule-based play with no trained network.

THIS FILE IS A REGISTRY OF FROZEN VERSIONS, NOT ONE AGENT. Because it is the ranking metric,
editing the live behaviour would silently rewrite every "vs heuristic" number the project has
published. So named versions are pinned (`VERSIONS`, see the table in CLAUDE.md) and one spec
grammar (`resolve_weights`) is shared by every harness — eval.py, eval_headtohead.py,
holdback.py, tune_heuristic.py. `REFERENCE` names the current one (h9 since 2026-07-21,
h8 before that, r3 before 2026-07-14); changing it is a deliberate act that requires re-running
the champions and recording both numbers.

A version pins BEHAVIOUR, not just numbers. Two fields carry that and neither is a weight:
  `refusal_mode`  'legacy' lets `score_draw` compete in the same max() as the plays, so a
                  hold-back weight >= 5.0 bought a REFUSAL (-13 pts) instead of a preference.
                  That was an 18-point STRUCTURAL defect, not a tuning error; 'structural'
                  makes refusing rules-only, so no weight assignment can produce a voluntary
                  draw (pinned as a property test). H1 carries R3's exact weights and beats it
                  by +0.183 on that change alone.
  `structure`     which cards are recognised as returning the turn to us (tempo). See
                  `Weights.structure`.

The freeze needs BOTH guards in `agenttest.FrozenVersionTest`, because each is blind where the
other sees: the WEIGHT PIN asserts every version's values (blind to a structural edit — H1 moved
+0.183 with zero weights touched), and the MOVE-SEQUENCE FINGERPRINT hashes every decision across
60 seeded games (blind to a saturated hold — p_king 5.0->6.0 flips no argmax). If a fingerprint
fails, do NOT paste in the new hash: a frozen version drifting means the published numbers no
longer describe the agent in the tree. Real behaviour changes get a NEW version; old ones are
never edited.

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

    # --- H7: RACE — MEASURED AND REFUTED (2026-07-14). Off, and it should stay off. ------
    # The idea: the agent knows how to BLOCK a near-winner but has no answer when it CANNOT
    # block one. A human speeds up — a card kept for later has no later to be kept for — so
    # racing scales the hold-backs by `race_hold_scale` (0.0 = ignore them) and cashes the
    # hoarded TAKI run instead of guarding it.
    #
    # It LOSES, monotonically in both knobs (vs h1b2, 20k games x 2):
    #     race_hold_scale  0.0 -> -0.035   0.25 -> -0.023   0.5 -> -0.014   0.75 -> -0.003
    #     race_hand_thresh   1 -> -0.015      2 -> -0.035     3 -> -0.054     4 -> -0.066
    # Every increment of racing costs points; the optimum is not to race at all. This is
    # H5/H6 seen from the other side: the hold-backs are worth +1.1 (w_nofin), +2.9 (hoard)
    # and +2.2 (blocker-spend), and they are worth MOST in the endgame — which is exactly
    # what racing throws away. Kept (off) because "why is racing bad?" is a live question
    # for behavioural analysis; do not turn it on expecting points.
    race: bool = False
    race_hand_threshold: int = 2      # an opponent at/below this many cards is about to win
    race_hold_scale: float = 0.0      # multiplier on the hold-back terms while racing

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

    # --- Behaviour version (NOT a weight) -------------------------------------
    # Which TEMPO structure the agent uses, i.e. which cards are recognised as giving
    # us another turn:
    #   'h1' — only PLUS earns `w_plus_tempo`. What every version up to H8 shipped.
    #   'h11' — H9 generalized: a STOP earns `w_plus_tempo / (n - 1)` at EVERY count,
    #           which is the fraction of a full extra turn it actually buys (it skips one
    #           of the `n - 1` opponents who would otherwise act before us). At n=2 that
    #           fraction is 1.0, so H11 IS H9 at two seats, move for move.
    #   'h9' — a card that returns the turn to US earns `w_plus_tempo` as well. The
    #          predicate is `_seat_after_playing(card) is our own seat`, which is derived
    #          from the RULES, not from a weight, and STOP satisfies it only at TWO seats:
    #          it skips the next player, and with n=2 the next player is the only other
    #          seat, so the turn comes straight back. Formally `(me + 2d) % n == me` iff
    #          `n | 2d`, i.e. iff n == 2. So H9 IS H8, move for move, at 3 and 4 seats
    #          (pinned: the 3- and 4-seat fingerprints are equal).
    #
    # Why this is a structure and not a weight: at two seats STOP is a free extra turn,
    # exactly like PLUS, and the heuristic priced it only as a *blocker*. With H8's
    # `w_save_blocker=0.0` a threat-free 2-seat STOP scored exactly 0.0 — ranked BELOW a
    # plain number card of a color we hold. No assignment of the existing weights can fix
    # that, because none of them can see that the turn came back. Same shape as H1's
    # refusal bug: a structure-level defect that looks like a tuning problem.
    structure: str = 'h1'

    def __post_init__(self):
        if self.refusal_mode not in REFUSAL_MODES:
            raise ValueError(f'unknown refusal_mode {self.refusal_mode!r}; '
                             f'choose from {sorted(REFUSAL_MODES)}')
        if self.structure not in STRUCTURES:
            raise ValueError(f'unknown structure {self.structure!r}; '
                             f'choose from {sorted(STRUCTURES)}')


#: Decision structures the scoring code implements.
REFUSAL_MODES = ('legacy', 'structural')

#: Tempo structures the scoring code implements. See `Weights.structure`.
STRUCTURES = ('h1', 'h9', 'h11')


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

#: H7 (PLAN.md): H1_B2 plus the RACE behaviour — when a near-winner cannot be stopped, stop
#: saving cards for a later that is not coming. Off in every earlier version, so `h7` vs
#: `h1b2` isolates it.
H7 = dataclasses.replace(H1_B2, race=True)

#: H8: the coordinate-descent point (2026-07-14), tuned against the R6 champion from H1_B2
#: over 2 passes x 44 candidates x 12,000 games. `probes/h8_tune.log`.
#:
#: THE INTERESTING PART IS WHAT IT DID NOT CHANGE. H8 waited for H1 so the search could go
#: ABOVE the old hold cap of 5.0 (below which the tuner had been confined, because above it
#: a hold bought a refusal). Given that freedom, the tuner went there and came back: not one
#: hold-back weight moved — `p_king` 5.0, `w_reserve` 4.0, `w_nofin` 8.0, `w_open_hoard`
#: 15.0 all unchanged — and every gain came from ORDINARY scoring terms. The step-4 sweeps
#: had already shown why: above their optimum the holds SATURATE (identical margins to three
#: decimals, because the argmax stops moving), so there were never any points up there.
#: The cap was not costing anything; the cliff was.
H8 = dataclasses.replace(
    H1_B2,
    w_deny=1.5,             # was 3.0 — deny the next player's color HALF as hard
    w_block=6.0,            # was 4.0 — H3 said blocking is the biggest behaviour; press it
    w_chdir_block=0.0,      # was 2.0 — dead weight, and H3's ablation independently agreed
    w_plus_tempo=2.0,       # was 1.0 — the free extra turn is worth twice what R3 thought
    w_save_blocker=0.0,     # was 0.7 — do not hold the blocker back; spend it
    p_super_taki=4.0,       # was 4.5 — noise-level
)

#: H9 (2026-07-19): H8's weights UNCHANGED — a STRUCTURE change only, in the H1 mould.
#: At two seats a STOP returns the turn to us, exactly as PLUS does, and the scoring code
#: gave it nothing: it landed in the blocker branch, and with H8's `w_save_blocker=0.0` a
#: threat-free 2-seat STOP scored exactly 0.0, ranked below a plain number card. See
#: `Weights.structure` for why no weight assignment could fix that.
#:
#: The fix is derived from the rules (`_seat_after_playing(STOP) is our seat`, true iff
#: n == 2), so **H9 is H8 move for move at three and four seats** — pinned in agenttest by
#: equality of their 3- and 4-seat fingerprints, and inequality of their 2-seat ones.
#: Motivation: measured on the rotation-orbit standard, `h8` trails the M1s3 champion by
#: 0.0765 per seat at two seats against 0.0265 at four (RESEARCH_LOG 2026-07-19 later) —
#: the signature of a yardstick tuned at one seat count and quoted at three.
H9 = dataclasses.replace(H8, structure='h9')

#: H10 (2026-07-19): coordinate descent from H9 at TWO seats against the M1s3 champion,
#: 2 passes, 96 candidates, 3000 decks x 2 blocks per candidate.
#:
#: A TWO-SEAT YARDSTICK ONLY. It must never become REFERENCE: these weights were selected
#: at n=2 and measure ~0 at four seats (-0.0015 +/- 0.0046 vs h8) — they buy nothing where
#: h8's numbers were published. Cite it explicitly as `heuristic:h10` in 2-seat tables.
#:
#: On fresh disjoint decks (the tuning used blocks 0/777777; these are 1500000/2500000):
#:     h10 - h8  @2 seats   +0.0370, +0.0400   -> +0.0385 (5.3 SE)
#:     h10 - h9  @2 seats   +0.0133, +0.0060   -> +0.0097 (1.4 SE, NOT established)
#: **So the tuning is not what did it.** H9's one structural line is worth +0.026 of the
#: +0.0385; everything 96 candidates found on top is within noise of zero. This is the H8
#: result a second time (RESEARCH_LOG 2026-07-14: "the tuner went above the old cap and
#: came back empty"), and the third time in this project that a STRUCTURAL fix paid where
#: weight search did not. Kept anyway: the point estimate is positive against three
#: independent nets, and it is the honest 2-seat yardstick even if its margin over h9 is
#: not individually significant.
#:
#: Two knobs are of behavioural interest, both 2-seat-specific and both still uncertain:
#:   w_deny 1.5 -> 0.0        denying your ONLY opponent a color makes them draw, and a
#:                            drawn card is a card you then have to out-race; at four seats
#:                            the denial lands on someone else's problem.
#:   w_save_blocker 0.0 -> 2.0  H8 says spend the blocker; at two seats, hold it. Note this
#:                            interacts with H9 — once STOP earns tempo it is attractive
#:                            enough to need a counterweight against being dumped early.
H10 = dataclasses.replace(
    H9,
    w_deny=0.0,
    w_plus_tempo=3.0,       # was 2.0 — above the old grid's ceiling, reachable only at 2p
    w_save_blocker=2.0,     # was 0.0 — the sign of H8's advice REVERSES at two seats
    w_reserve=16.0,         # saturating (H8's lesson); +0.0005 in search
    p_king=2.0,             # saturating; +0.0015 in search
    block_hand_threshold=3,
)

#: H11 (2026-07-20): H9 generalized to every seat count, weights UNCHANGED from H8.
#:
#: H9 asked "does this card hand the turn straight back to me?" — true only at two seats.
#: H11 asks the quantitative version: a STOP skips one of the `n - 1` opponents who would
#: otherwise act before our next turn, so it buys `1 / (n - 1)` of what a PLUS buys, and
#: earns that fraction of `w_plus_tempo`. No free parameter; the fraction is the rules.
#:
#:     n=2  1/1  -> H11 IS H9 at two seats, move for move (pinned)
#:     n=3  1/2
#:     n=4  1/3
#:
#: WHY THIS EXISTS, and a CORRECTION. It was planned as "at three seats STOP and CHDIR are
#: the same move, but are scored by different terms". **That premise is false**, and the
#: engine says so — traced at n=3, seat 0 acting:
#:     STOP   0, 2, 0, 1, 2, ...   we act again after ONE opponent
#:     CHDIR  0, 2, 1, 0, 2, ...   both opponents act first
#: They agree only on who faces the table NEXT (which is all `_seat_after_playing` claims,
#: and it is right about it). CHDIR never buys tempo at any count — reversing a cycle still
#: leaves `n - 1` opponents ahead of us — so the CHDIR half of the plan was dropped and the
#: STOP half generalized instead. CLAUDE.md asserted the same wrong equivalence twice,
#: sourced to RULES.md, which never says it; both are corrected.
H11 = dataclasses.replace(H8, structure='h11')

VERSIONS = {
    'r3': R3,
    'b2': B2_RETUNED,
    'h1': H1,
    'h1b2': H1_B2,
    'h7': H7,
    'h8': H8,
    'h9': H9,
    'h10': H10,
    'h11': H11,
    'greedy': GREEDY,
}

#: The reference opponent: what a bare `heuristic` spec means, and therefore what
#: `eval.py --opponent heuristic` measures.
#:
#: PROMOTED h8 -> h9 on 2026-07-21 (RESEARCH_LOG 2026-07-20), deliberately. h9 is a STRICT
#: improvement on h8: bit-identical at three and four seats (the STOP-tempo fix cannot fire
#: above n=2), and +0.026 better at two, where h8 priced a free extra turn at zero. So the
#: promotion changes ONLY the two-seat meaning of `heuristic`; every 3- and 4-seat number
#: ever measured against h8 still describes the reference verbatim. The two-seat numbers
#: were re-run and recorded (orbit per-seat rate, 2 blocks):
#:
#:                      vs h8 (old)   vs h9 (new)   parity
#:     M1s3 champion       0.577         0.567        0.500
#:     R6                  0.417         0.402        0.500
#:
#: **Every two-seat "vs heuristic" number published before 2026-07-21 refers to `h8`.** At
#: three and four seats there is nothing to restate — h9 IS h8 there. Cite the old two-seat
#: numbers as `heuristic:h8`, which still runs and still means what it always did.
#:
#: EARLIER PROMOTION, for the record: r3 -> h8 on 2026-07-14 (B5). R6 vs r3 0.378 -> vs h8
#: 0.297 (four seats); the crippled r3 yardstick had inflated the headline by ~8 points.
#:
#: h10 must NEVER be REFERENCE (2-seat-tuned, worth ~0 at four seats). h11 is refuted.
REFERENCE = 'h9'


def resolve_weights(spec=''):
    """Resolve a heuristic weight spec. The one grammar, shared by every entry point.

      ''  / 'reference'   -> the current reference version (REFERENCE, now h9)
      '<version>'         -> a named version: r3, b2, h1, h1b2, h7, h8, h9, h10, h11, greedy
      '-<name>'           -> the base with ONE hold-back behaviour ablated
      'k=v,k=v'           -> the base with individual weights overridden, which is how a
                             weight gets SWEPT rather than merely switched off
      '<version>,...'     -> any of the above, but based on THAT version instead of the
                             reference: 'h1b2,block_hand_threshold=1', 'h1,-hoard'

    The leading-version form is not sugar. Without it every override is implicitly based on
    the REFERENCE, and the reference has moved over time (r3 -> h8 -> h9). An override with
    no leading version is therefore measured against whatever the current reference is — so
    to reproduce an older sweep, name the base version it was actually run on (e.g. the
    H-series used `h1b2` explicitly, precisely so its ablations were not priced against r3's
    refusal cliff).
    """
    spec = (spec or '').strip()
    if spec in ('', 'reference'):
        return VERSIONS[REFERENCE]
    if spec in VERSIONS:
        return VERSIONS[spec]

    base = VERSIONS[REFERENCE]
    head, sep, rest = spec.partition(',')
    if head.strip() in VERSIONS:            # '<version>,<modifiers...>'
        base = VERSIONS[head.strip()]
        spec = rest.strip()
        if not spec:
            return base

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
        reserved, hoard_color, open_hoard, legal_to_open = self._hoard_plan(game, hand)
        racing = self._racing(game, hand)
        if racing and legal_to_open:
            # H7: cash the run, it is now or never — but only when the rules allow the run
            # to be ENDED (see _hoard_plan). Forcing it open regardless would have the
            # agent start runs it cannot legally finish, which is a bug, not a race.
            open_hoard = True
        structural = self.w.refusal_mode == 'structural'
        fallback = None
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
                if structural:
                    # H4: declining the King's free follow-up is a REFUSAL, so under H1 it
                    # is not something a weight may buy either. It was already dead code —
                    # 0 fires in 200 games at score_decline_king=-2.0 — but zeroing the
                    # weight was measured to WAKE IT UP and cost -0.0004, because at 0 it
                    # outbids negative-scoring plays. Hence: remove the branch, don't
                    # neutralize the weight. Kept only as the fallback when the rules leave
                    # nothing else (there is no DRAW to fall back on in State.KING).
                    fallback = move
                    continue
                score = self.w.score_decline_king
            else:
                score = self._score_play(game, hand, card, reserved,
                                         hoard_color, open_hoard, racing)
                if structural and score == self.w.score_forbidden:
                    # The finishing rule forbids it (a lone PLUS). Not a preference — a
                    # rule. Dropping it here is what lets DRAW remain reachable when it is
                    # the ONLY thing the rules permit, without any weight voting for it.
                    continue
            scored.append((score, move))
        if not scored:
            # structural: the rules left nothing playable. Decline the King's follow-up if
            # that is the only move on offer (State.KING has no DRAW), else draw.
            return fallback if fallback is not None else (Action.DRAW, None)
        best = max(scored, key=lambda s: (s[0], -action_to_scalar(*s[1])))
        return best[1]

    def _score_play(self, game, hand, card, reserved, hoard_color, open_hoard,
                    racing=False):
        w = self.w
        # B5 hard rule: never end the hand on a non-finisher (engine penalty).
        if len(hand) == 1 and card.type.value not in FINISHING_TYPE_VALUES:
            return w.score_forbidden

        # H7: every hold-back term is a bet on a future turn. While racing there may not
        # BE a future turn, so they are scaled down (0.0 = ignored). NB this is exactly the
        # kind of unbounded rescaling that would have been unsafe before H1: shrinking a
        # hold is harmless, but the same switch inverted could push one past |score_draw|
        # and buy a refusal. Under 'structural' it simply cannot.
        hold = w.race_hold_scale if racing else 1.0

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
            score += w.w_block if threat is not None else -w.w_save_blocker * hold
            # H9/H11: a STOP buys TEMPO, which this branch never priced — it only ever
            # asked whether the card blocks someone.
            #
            # How much tempo: normally `n - 1` opponents act before our next turn. A STOP
            # skips one of them, so `n - 2` do. PLUS is the same quantity taken to the
            # limit — it saves all `n - 1`. So a STOP is worth `1 / (n - 1)` of what a PLUS
            # is worth, in the same units, with no free parameter to pick.
            #
            #   n=2  1/1 = 1.0  the turn comes straight back: a WHOLE free turn (this is
            #                   exactly H9, which is why H11 IS H9 at two seats)
            #   n=3  1/2        we act again after one opponent instead of two
            #   n=4  1/3
            #
            # H9 credits only the n=2 case (via `next_seat == game.curr`, which for a
            # blocker can only be true for a STOP at two seats). H11 credits every count.
            if w.structure == 'h9' and next_seat == game.curr:
                score += w.w_plus_tempo
            elif w.structure == 'h11' and card.type is Type.STOP:
                score += w.w_plus_tempo / (len(game.agents) - 1)
        elif card.type is Type.CHDIR and threat is not None:
            behind = (game.curr - game.dir) % len(game.agents)
            if len(game.hands[behind]) > len(game.hands[threat]):
                score += w.w_chdir_block
        elif card.type is Type.PLUS:
            score += w.w_plus_tempo

        # B10 wild hold-penalties.
        if card.type is Type.CHCOL:
            score -= w.p_chcol * hold
        elif card.type is Type.KING:
            score -= w.p_king * hold
        elif card.type is Type.TAKI and card.color is Color.NONE:
            score -= w.p_super_taki * hold
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
            score -= w.w_reserve * hold

        # B5 soft: with a small hand, keep at least one finisher.
        if len(hand) <= SMALL_HAND:
            has_finisher_after = any(
                c.type.value in FINISHING_TYPE_VALUES for c in hand if c is not card)
            if not has_finisher_after:
                score -= w.w_nofin * hold

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
            return frozenset(), None, False, False
        reserved = frozenset([id(taki)] + [id(c) for c in group])
        # Wrap membership test: Card defines value equality, so use identity.
        reserved_cards = _IdentitySet(reserved)
        outside = len(hand) - 1 - len(group)
        run_can_finish = any(c.type.value in NUMBER_TYPE_VALUES for c in group)
        # The run may only be opened when it can also be ENDED: if it would empty the hand
        # (outside == 0) the group must contain a number to close on. This is a rules
        # constraint, not a preference — H7's race must respect it too, or it opens runs it
        # cannot legally finish.
        legal_to_open = run_can_finish or outside > 0
        open_now = outside <= 1 and legal_to_open
        return reserved_cards, taki.color, open_now, legal_to_open

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

    def _racing(self, game, hand):
        """H7: someone is about to win and we CANNOT stop them, so the game is ending
        whatever we do — shed cards now.

        The "cannot stop them" half is the part that is easy to get wrong: STOP and +2 hit
        the NEIGHBOUR, so holding a blocker is worthless against a near-winner sitting
        anywhere else at the table. We are only safe from racing if the near-winner is the
        next seat AND we hold something that lands on them.
        """
        if not self.w.race:
            return False
        me = game.curr
        near = [s for s in range(len(game.agents))
                if s != me and len(game.hands[s]) <= self.w.race_hand_threshold]
        if not near:
            return False
        nxt = (me + game.dir) % len(game.agents)
        have_blocker = any(c.type in _BLOCKER_TYPES for c in hand)
        can_stop_them = have_blocker and near == [nxt]
        return not can_stop_them

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
