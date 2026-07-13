"""Behavioural probes: interrogate a trained policy in hand-constructed positions.

The project's payoff is behavioural analysis, not the win-rate number (CLAUDE.md): once the
model is good enough, the point is to probe *what it learned*. This is the harness for that —
build a Taki position directly, dump the policy's Q-values over the legal moves, watch the
line it actually plays, and compare against the strategically correct line.

Two arms, deliberately kept separate:

  Q arm  (--scenario ...)  One forward pass per position; read the RANKING over legal moves.
                           Q is a shaped return (train.py pays -len(hand) per step), NOT a win
                           probability, so only the ordering is interpretable. That is enough
                           wherever the correct line is provable from the rules.
  MC arm (--mc N)          Where correctness is NOT provable (hoard vs dump), force each
                           candidate line and play the position out N times, resampling the
                           hidden cards. Measures which line actually wins instead of asserting.

Usage:
    python probe.py --model models/checkpoint_a8_snap455000 --scenario all
    python probe.py --model models/checkpoint_a8_snap455000 --scenario all --controls
    python probe.py --model models/checkpoint_a8_snap455000 --scenario b1c --mc 1000
"""
import os
# The network is tiny; TF's default thread pools oversubscribe and thrash. Must be set before
# TensorFlow is imported (via agents.dqn below). Mirrors eval.py:25-31 — keep consistent.
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import argparse
import collections
import random
import zlib
from dataclasses import dataclass, field

import numpy as np

from game import (Game, Card, Color, Type, State, Action, OBSERVATION_SIZE, ACTION_SIZE,
                  INITIAL_HAND_SIZE, action_to_scalar, scalar_to_action)
from agents.dqn import AIAgent
from agents.random import RandomAgent

# A capped game is undecided (eval.py's convention); only degenerate policies ever hit this.
TURN_CAP = 2000


class _Dummy:
    """Seat filler for positions we never step (Q arm)."""

    def play(self, game):
        raise AssertionError('_Dummy.play must never be called')


def load_probe_agent(path):
    """Load a checkpoint greedy (epsilon=0), asserting it speaks the current encoding.

    AIAgent catches an architecture mismatch and silently adopts the checkpoint's own model
    (agents/dqn.py:70-77), so a stale-contract snapshot would load without complaint and quietly
    produce meaningless Q-values. Several run dirs on disk are multi-input A11/A12 experiments.
    """
    agent = AIAgent(epsilon=0.0, epsilon_min=0.0, load_model=path)
    shape_in, shape_out = agent.model.input_shape, agent.model.output_shape
    if not isinstance(shape_in, tuple) or shape_in[-1] != OBSERVATION_SIZE \
            or shape_out[-1] != ACTION_SIZE:
        raise RuntimeError(
            f'{path}: wrong encoding contract (in={shape_in}, out={shape_out}); '
            f'expected a single {OBSERVATION_SIZE}-float input and {ACTION_SIZE} outputs')
    return agent


def untrained_agent(seed):
    """A random-init net through the same code path — the null control for the Q arm."""
    import tensorflow as tf
    tf.random.set_seed(seed)
    return AIAgent(epsilon=0.0, epsilon_min=0.0)


# --------------------------------------------------------------------------------------------
# Position construction
# --------------------------------------------------------------------------------------------

def full_pool(seed=0):
    """The canonical 120-card multiset, recovered from a fresh Game.

    Read off a real deal rather than re-implementing _setup_round, so this stays correct if the
    deck composition ever changes.
    """
    g = Game([_Dummy(), _Dummy()], seed=seed)
    return g.deck + g.discard + [c for h in g.hands for c in h]


def _take(pool, cards):
    """Multiset-remove `cards` from `pool`, raising if a card isn't available."""
    counts = collections.Counter(pool)
    for c in cards:
        if counts[c] <= 0:
            raise ValueError(f'{c} is not available in the deck (already used up)')
        counts[c] -= 1
    return list(counts.elements())


def make_position(learner_hand, top, *, agents, opp_sizes=(5, 5, 5), discard_below=None,
                  discard_size=20, state=State.NORMAL, taki_color=Color.NONE, draw_num=0,
                  direction=1, learner_seat=0, seed=0):
    """Build a fully consistent Game at an arbitrary position.

    Consistency is not cosmetic: observation() feeds on len(deck), the opponents' hand SIZES and
    the unseen +2/King/CHCOL counts (total - in_hand - in_discard), so an ad-hoc deck silently
    corrupts the network's input and every Q-value with it.

    `discard_below` is the pile under the top card. Left None, it is filled with plain NUMBER
    cards drawn from what remains — numbers only, so the unseen +2/King/CHCOL features stay at
    their full totals and are therefore constant across variants of a scenario (comparability).
    """
    learner_hand, discard_below = list(learner_hand), list(discard_below or [])
    n = len(agents)
    pool = _take(full_pool(seed), learner_hand + discard_below + [top])
    rng = random.Random(seed)
    rng.shuffle(pool)

    if discard_below:
        filler = []
    else:
        numbers = [c for c in pool if c.type.value in range(Type.ONE.value, Type.NINE.value + 1)]
        if len(numbers) < discard_size - 1:
            raise ValueError('not enough number cards left to build the discard pile')
        filler = numbers[:discard_size - 1]
        pool = _take(pool, filler)
    discard = filler + discard_below + [top]          # top card last: shown_card() is discard[-1]

    hands = [[] for _ in range(n)]
    hands[learner_seat] = learner_hand
    opp_seats = [s for s in range(n) if s != learner_seat]
    if len(opp_sizes) != len(opp_seats):
        raise ValueError('opp_sizes must give a size for every non-learner seat')
    for seat, size in zip(opp_seats, opp_sizes):
        hands[seat] = pool[:size]
        pool = pool[size:]

    game = Game(agents, seed=seed)
    game.hands = hands
    game.discard = discard
    game.deck = pool                                  # drawn from the END (game.py:510)
    game.curr = learner_seat
    game.dir = direction
    game.state = state
    game.taki_color = taki_color
    game.draw_num = draw_num

    # --- consistency asserts: a silently-broken position invalidates every number downstream ---
    seen = collections.Counter(game.deck + game.discard + [c for h in game.hands for c in h])
    if seen != collections.Counter(full_pool(seed)):
        raise AssertionError('position does not conserve the 120-card deck')
    total_in_hands = sum(len(h) for h in game.hands)
    # Reachability: from a real deal, discard = 1 + plays and hands = n*8 - plays + draws, so
    # len(discard) >= 1 + n*8 - total_in_hands. A 4-card hand over a 1-card discard is
    # numerically consistent yet unreachable — the net would be evaluated off its manifold.
    floor = 1 + n * INITIAL_HAND_SIZE - total_in_hands
    if len(game.discard) < floor:
        raise AssertionError(
            f'unreachable position: discard {len(game.discard)} < {floor} for {total_in_hands} '
            f'cards in hand (no real game reaches it)')
    if state in (State.TAKI, State.SUPER_TAKI) and taki_color is Color.NONE:
        raise AssertionError('an open TAKI needs a taki_color')
    if state is State.DRAW_TWO and (draw_num < 1 or top.type is not Type.PLUSTWO):
        raise AssertionError('DRAW_TWO needs draw_num >= 1 and a +2 showing')
    return game


# --------------------------------------------------------------------------------------------
# Reading the policy
# --------------------------------------------------------------------------------------------

def legal_scalars(game, seat=None):
    """Legal action scalars, in valid_moves() order — the order act() breaks ties in."""
    return [action_to_scalar(*m) for m in game.valid_moves(seat)]


def q_vector(agent, game, seat=None):
    """The raw, unmasked (65,) Q-vector — reproduces agents/dqn.py:159-160 exactly."""
    obs = game.observation(seat)
    return agent.model(obs[np.newaxis, :].astype(np.float32), training=False).numpy()[0]


def move_name(scalar):
    action, card = scalar_to_action(scalar)
    return f"{action}{' ' + str(card) if card is not None else ''}"


@dataclass
class QTable:
    rows: list          # (scalar, q, is_greedy), sorted by q desc
    greedy: int         # the scalar act() would pick
    hand: list
    top: Card
    state: State

    def q(self, scalar):
        for s, qv, _ in self.rows:
            if s == scalar:
                return qv
        raise KeyError(move_name(scalar))

    def __str__(self):
        head = (f"  hand {self.hand} | showing {self.top} | {self.state.name}\n"
                f"  {'move':<24}{'Q':>9}{'dQ':>9}\n  {'-' * 42}")
        best = self.rows[0][1]
        body = '\n'.join(
            f"  {move_name(s):<24}{qv:>9.2f}{qv - best:>9.2f}"
            f"{'   <- picks this' if g else ''}"
            for s, qv, g in self.rows)
        return head + '\n' + body


def q_table(agent, game, seat=None):
    seat = game.curr if seat is None else seat
    q = q_vector(agent, game, seat)
    legal = legal_scalars(game, seat)
    greedy = legal[int(np.argmax(q[legal]))]          # same gather+argmax as act()
    rows = sorted(((s, float(q[s]), s == greedy) for s in legal), key=lambda r: -r[1])
    return QTable(rows, greedy, list(game.hands[seat]), game.shown_card(), game.state)


class _Scripted:
    """Plays a fixed list of moves. Asserts legality — gametest's _take_turn does not, and
    process_action will happily execute an illegal move (e.g. a DRAW inside an open TAKI)."""

    def __init__(self, moves):
        self.moves = list(moves)

    def play(self, game):
        move = self.moves.pop(0)
        if move not in game.valid_moves():
            raise AssertionError(f'forced move {move_name(action_to_scalar(*move))} is illegal '
                                 f'(legal: {[move_name(s) for s in legal_scalars(game)]})')
        return move


def force_line(game, moves, seat):
    """Force `moves` for `seat`, one next_turn() per move, then restore the real agent."""
    real = game.agents[seat]
    game.agents[seat] = _Scripted(moves)
    try:
        for _ in moves:
            game.next_turn()
    finally:
        game.agents[seat] = real


@dataclass
class Step:
    table: QTable
    move: int                 # scalar played
    hand_before: int
    hand_after: int
    penalty_draw: bool
    won: bool


def greedy_line(game, agent, seat, max_steps=32):
    """Step the policy while the turn stays with `seat`, recording the whole line.

    A TAKI run is a SEQUENCE of decisions, not one: playing a colored TAKI keeps the turn
    (next_turn only advances on NORMAL/DRAW_TWO/STOP), so each card of the run is its own
    play() call. That is what makes the policy's ordering of the run observable — and the
    finishing rule (game.py:692-702) is re-checked after every single card.
    """
    steps = []
    while not game.done() and game.curr == seat and len(steps) < max_steps:
        table = q_table(agent, game, seat)
        before = len(game.hands[seat])
        game.next_turn()
        after = len(game.hands[seat])
        # Emptied the hand on a PLAY_CARD but the game is not over => the non-finisher penalty
        # draw fired. Note it does NOT end the turn: the player keeps playing, still in the TAKI.
        played = scalar_to_action(table.greedy)[0] is Action.PLAY_CARD
        penalty = played and before == 1 and not game.done()
        steps.append(Step(table, table.greedy, before, after, penalty, game.done()))
    return steps


def print_line(steps):
    for i, st in enumerate(steps, 1):
        note = ''
        if st.won:
            note = '   *** WINS ***'
        elif st.penalty_draw:
            note = '   <<< emptied the hand on a non-finisher: PENALTY DRAW, turn continues'
        print(f"  {i}. {move_name(st.move):<24} hand {st.hand_before} -> {st.hand_after}{note}")


# --------------------------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------------------------

R, Y, G, B = Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE


def c(t, col=Color.NONE):
    return Card(t, col)


@dataclass(frozen=True)
class Scenario:
    name: str
    desc: str
    hand: tuple
    top: Card
    state: State = State.NORMAL
    taki_color: Color = Color.NONE
    opp_sizes: tuple = (5, 5, 5)
    discard_below: tuple = ()
    discard_size: int = 20
    expect_legal: tuple = ()
    expect_illegal: tuple = ()
    lines: dict = field(default_factory=dict)   # named forced prefixes, for the MC arm
    correct: str = ''

    def build(self, agents, seed=0):
        return make_position(
            list(self.hand), self.top, agents=agents, opp_sizes=self.opp_sizes,
            discard_below=list(self.discard_below) or None, discard_size=self.discard_size,
            state=self.state, taki_color=self.taki_color, seed=seed)


PLAY = Action.PLAY_CARD

# --- B1a: the sequencing trap -- INVALIDATED, DO NOT INTERPRET ---------------------------------
# These scenarios were built against a BUG in FINISHING_TYPE_VALUES that let a hand end only on a
# number or the King. The real rule: you may finish on ANY card except PLUS. The red STOP and the
# red +2 are therefore finishers too, so there is NO sequencing constraint here and no trap —
# every ordering of the run wins. The B1 headline drawn from these ("plan a TAKI run backwards
# from its last card") is an artifact of the bug; see probes/b1_colored_taki_hoard.md.
# A real sequencing trap needs a PLUS in hand (the one card that cannot end a hand). Kept only so
# the B1 report stays reproducible; b1c (the matched-TAKI sweep) never depended on the rule and
# remains valid.
B1A = Scenario(
    name='b1a',
    desc='Sequencing trap: red TAKI + red STOP + red +2 + red 5 on a red 3.',
    hand=(c(Type.TAKI, R), c(Type.STOP, R), c(Type.PLUSTWO, R), c(Type.FIVE, R)),
    top=c(Type.THREE, R),
    discard_size=18,
    expect_legal=((PLAY, c(Type.TAKI, R)), (PLAY, c(Type.STOP, R)),
                  (PLAY, c(Type.PLUSTWO, R)), (PLAY, c(Type.FIVE, R)), (Action.DRAW, None)),
    lines={
        'correct': ((PLAY, c(Type.TAKI, R)), (PLAY, c(Type.STOP, R)),
                    (PLAY, c(Type.PLUSTWO, R)), (PLAY, c(Type.FIVE, R))),
        'trap': ((PLAY, c(Type.TAKI, R)), (PLAY, c(Type.FIVE, R))),
        'open_taki': ((PLAY, c(Type.TAKI, R)),),
        'no_taki': ((PLAY, c(Type.FIVE, R)),),
    },
    correct='DEGENERATE under the corrected rule: open the TAKI and dump all four in ANY order '
            '-> wins outright. There is no last-card constraint.',
)

# --- B1a-min: the minimal pair -- INVALIDATED with B1a (see above) -----------------------------
# The pair was built on "5-first eats a penalty", which the finishing-rule fix makes false: the
# red STOP is a legal finisher, so BOTH orders win in A. The A-vs-B contrast no longer isolates a
# win-now condition, and the Q-gap it reports means nothing. Redesign with a PLUS before reusing.
B1A_MIN_A = Scenario(
    name='b1a_min_a',
    desc='Trap LIVE: in an open red TAKI with red STOP + red 5. Win is available this turn.',
    hand=(c(Type.STOP, R), c(Type.FIVE, R)),
    top=c(Type.TAKI, R), state=State.TAKI, taki_color=R,
    expect_legal=((PLAY, c(Type.STOP, R)), (PLAY, c(Type.FIVE, R)), (Action.CLOSE_TAKI, None)),
    expect_illegal=((Action.DRAW, None),),
    lines={'correct': ((PLAY, c(Type.STOP, R)), (PLAY, c(Type.FIVE, R))),
           'trap': ((PLAY, c(Type.FIVE, R)),)},
    correct='DEGENERATE: both orders win (the red STOP is a legal finisher). No trap here.',
)
B1A_MIN_B = Scenario(
    name='b1a_min_b',
    desc='Trap DEAD (control): same, plus an unplayable blue 9. No win available this turn.',
    hand=(c(Type.STOP, R), c(Type.FIVE, R), c(Type.NINE, B)),
    top=c(Type.TAKI, R), state=State.TAKI, taki_color=R,
    expect_legal=((PLAY, c(Type.STOP, R)), (PLAY, c(Type.FIVE, R)), (Action.CLOSE_TAKI, None)),
    expect_illegal=((PLAY, c(Type.NINE, B)), (Action.DRAW, None)),
    correct='No win this turn; the STOP-vs-5 order is near-irrelevant.',
)

# --- B1c: the matched-TAKI sweep --------------------------------------------------------------
# Showing a BLUE taki, a RED taki is legal by TYPE while red numbers stay illegal outside the run.
# That holds the legal set at exactly {red TAKI, green TAKI, blue 1, DRAW} for every k, so the
# only thing that varies is how many red cards back the red TAKI.
#   Delta(k) = Q(red TAKI) - Q(green TAKI): a within-state, matched-card contrast (both cards open
#   a run; the hand holds k reds and zero greens). If the policy understands what a TAKI IS, this
#   grows with k. At k=0 the observation is EXACTLY invariant under the red<->green relabeling and
#   that relabeling maps the red-TAKI action onto the green-TAKI one, so a color-equivariant net
#   must give Delta(0)=0 -- a free, calibrated noise floor.
_B1C_REDS = [c(Type.FIVE, R), c(Type.SEVEN, R), c(Type.THREE, R), c(Type.NINE, R)]
_B1C_YELLOWS = [c(Type.TWO, Y), c(Type.FOUR, Y), c(Type.SIX, Y), c(Type.EIGHT, Y)]
# 19 fixed filler cards under the top, chosen so no +2/King/CHCOL is ever "seen" (the unseen
# features stay pinned at their totals) and so the filler is identical for every k.
_B1C_FILLER = ([Card(t, B) for t in (Type.TWO, Type.THREE, Type.FOUR, Type.FIVE, Type.SIX,
                                     Type.SEVEN, Type.EIGHT, Type.NINE)] * 2
               + [Card(Type.ONE, B)] + [Card(Type.TWO, G), Card(Type.TWO, G)])


def b1c(k):
    hand = ([c(Type.TAKI, R), c(Type.TAKI, G), c(Type.ONE, B)]
            + _B1C_REDS[:k] + _B1C_YELLOWS[:4 - k])
    return Scenario(
        name=f'b1c_k{k}',
        desc=f'Matched TAKI sweep, k={k}: a red TAKI backed by {k} red cards, a green TAKI '
             f'backed by none, on a blue TAKI.',
        hand=tuple(hand), top=c(Type.TAKI, B),
        discard_below=tuple(_B1C_FILLER),
        expect_legal=((PLAY, c(Type.TAKI, R)), (PLAY, c(Type.TAKI, G)),
                      (PLAY, c(Type.ONE, B)), (Action.DRAW, None)),
        expect_illegal=tuple((PLAY, x) for x in _B1C_REDS[:k] + _B1C_YELLOWS[:4 - k]),
        lines={'dump': ((PLAY, c(Type.TAKI, R)),),
               'shed': ((PLAY, c(Type.ONE, B)),),
               'draw': ((Action.DRAW, None),)},
        correct='Unknown a priori -- adjudicated by rollout (--mc), not asserted.',
    )


SCENARIOS = {s.name: s for s in [B1A, B1A_MIN_A, B1A_MIN_B] + [b1c(k) for k in range(5)]}


# --------------------------------------------------------------------------------------------
# MC arm: force a candidate line, then let the policy finish. Which line actually wins?
# --------------------------------------------------------------------------------------------

@dataclass
class MCResult:
    wins: np.ndarray          # per-replicate 0/1, for PAIRED comparison across lines
    won_this_turn: np.ndarray
    hand_after_turn: np.ndarray
    undecided: int

    @property
    def rate(self):
        return float(self.wins.mean())

    @property
    def se(self):
        return float(self.wins.std(ddof=1) / np.sqrt(len(self.wins)))


def det_seed(seed, g):
    """Determinization seed for replicate `g` of a run based at `seed`.

    Must be injective in (seed, g), and NOT `seed + g`: that made runs at nearby seeds overlap
    almost completely (seed 0 draws deals 0..N-1, seed 1 draws 1..N — a 99.9% shared sample at
    N=1200), so "replicated on seeds 0/1/2" was one measurement reported three times. Hashing the
    pair gives every (seed, g) its own stream, so distinct --seed values are independent blocks
    and a spread across them is a real spread.
    """
    return zlib.crc32(f'{seed}:{g}'.encode()) & 0xFFFFFFFF


def mc_line(scen, learner, line, opponents, games=1000, seed=0):
    """Force `line`, hand back to the greedy policy, play out. Repeat, resampling hidden cards.

    Determinization: the learner's hand, the discard/top, the state and the opponents' hand SIZES
    are FIXED; which cards the opponents hold and the deck order are resampled per replicate. This
    is exactly the learner's belief state — sound because observation() sees the hidden region only
    through sizes and counts, so it is bit-identical across replicates (pinned in probetest).

    Replicate g uses determinization seed `det_seed(seed, g)` for EVERY line (common random
    numbers), so lines are compared on identical deals and the difference can be tested paired.
    """
    wins, wtt, hands_after, undecided = [], [], [], 0
    for g in range(games):
        ds = det_seed(seed, g)
        for opp in opponents:
            if hasattr(opp, 'reseed'):
                opp.reseed(f'{seed}:{g}:opp')
        np.random.seed(ds)                             # epsilon-greedy opponents draw from np
        game = scen.build([learner] + list(opponents), seed=ds)
        force_line(game, list(line), 0)
        # finish the learner's turn under its own policy
        turns = 0
        while not game.done() and game.curr == 0 and turns < 64:
            game.next_turn()
            turns += 1
        wtt.append(1 if game.done() and not game.hands[0] else 0)
        hands_after.append(len(game.hands[0]))
        while not game.done() and turns < TURN_CAP:
            game.next_turn()
            turns += 1
        if game.done():
            wins.append(1 if not game.hands[0] else 0)
        else:
            undecided += 1
            wins.append(0)
    return MCResult(np.array(wins, float), np.array(wtt, float),
                    np.array(hands_after, float), undecided)


def paired_se(a, b):
    """SE of the difference on PAIRED replicates — tighter and correct under common random
    numbers; two independent SEs would overstate the uncertainty."""
    d = a - b
    return float(d.std(ddof=1) / np.sqrt(len(d)))


# --------------------------------------------------------------------------------------------

def run_q_arm(agent, label, names):
    print(f"\n{'=' * 88}\nQ-RANKING — {label}\n{'=' * 88}")
    for name in names:
        scen = SCENARIOS[name]
        print(f"\n--- {scen.name}: {scen.desc}")
        game = scen.build([agent, RandomAgent(), RandomAgent(), RandomAgent()])
        print(q_table(agent, game, 0))
        if scen.lines and 'correct' in scen.lines:
            print(f"\n  correct line: {scen.correct}")
            print('  the policy actually plays:')
            print_line(greedy_line(scen.build([agent, RandomAgent(), RandomAgent(),
                                               RandomAgent()]), agent, 0))


def diff_in_diff(agent):
    """[Q(STOP) - Q(5)]_A - [Q(STOP) - Q(5)]_B.

    ~0 means the policy's preference for the STOP is a GENERIC dislike of playing low numbers,
    not detection of the win-now condition -- and the b1a result would then mean nothing.
    """
    stop, five = action_to_scalar(PLAY, c(Type.STOP, R)), action_to_scalar(PLAY, c(Type.FIVE, R))
    out = []
    for scen in (B1A_MIN_A, B1A_MIN_B):
        t = q_table(agent, scen.build([agent, RandomAgent(), RandomAgent(), RandomAgent()]), 0)
        out.append(t.q(stop) - t.q(five))
    return out[0], out[1], out[0] - out[1]


def main():
    p = argparse.ArgumentParser(description='Behavioural probes of a trained Taki policy.')
    p.add_argument('--model', default='models/checkpoint_a8_snap455000')
    p.add_argument('--scenario', default='all')
    p.add_argument('--controls', action='store_true',
                   help='also run the random-init / other-lineage controls')
    p.add_argument('--mc', type=int, default=0, help='rollout games per line (0 = Q arm only)')
    p.add_argument('--mc-opponents', default='both', choices=['dqn', 'random', 'both'])
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    names = list(SCENARIOS) if args.scenario == 'all' else args.scenario.split(',')
    agent = load_probe_agent(args.model)
    run_q_arm(agent, args.model, names)

    a, b, dd = diff_in_diff(agent)
    print(f"\n{'=' * 88}\nB1a-min DIFFERENCE-IN-DIFFERENCES — {args.model}\n{'=' * 88}")
    print(f"  Q(STOP)-Q(5)  trap LIVE (win available) : {a:+8.2f}")
    print(f"  Q(STOP)-Q(5)  trap DEAD (control)       : {b:+8.2f}")
    print(f"  difference-in-differences               : {dd:+8.2f}   "
          f"(~0 => generic reflex, not endgame understanding)")

    # Delta(k) sweep
    rt, gt = action_to_scalar(PLAY, c(Type.TAKI, R)), action_to_scalar(PLAY, c(Type.TAKI, G))
    print(f"\n{'=' * 88}\nB1c  Delta(k) = Q(red TAKI) - Q(green TAKI)   [Delta(0) must be ~0 by "
          f"color symmetry]\n{'=' * 88}")
    for k in range(5):
        scen = SCENARIOS[f'b1c_k{k}']
        t = q_table(agent, scen.build([agent, RandomAgent(), RandomAgent(), RandomAgent()]), 0)
        print(f"  k={k}  Delta={t.q(rt) - t.q(gt):+7.2f}   "
              f"Q(redTAKI)={t.q(rt):+7.2f}  Q(blue 1)={t.q(action_to_scalar(PLAY, c(Type.ONE, B))):+7.2f}"
              f"  Q(draw)={t.q(63):+7.2f}   picks: {move_name(t.greedy)}")

    if args.controls:
        for seed in (0, 1, 2, 3, 4):
            ua = untrained_agent(seed)
            a, b, dd = diff_in_diff(ua)
            deltas = []
            for k in range(5):
                t = q_table(ua, SCENARIOS[f'b1c_k{k}'].build(
                    [ua, RandomAgent(), RandomAgent(), RandomAgent()]), 0)
                deltas.append(t.q(rt) - t.q(gt))
            print(f"  [control random-init seed {seed}]  diff-in-diff={dd:+7.2f}   "
                  f"Delta(k)={['%+.2f' % d for d in deltas]}")
        for path in ('models/checkpoint_a4a7_snap550000',
                     'models/run1783692871.878205_bestresponse/snap0000'):
            try:
                ca = load_probe_agent(path)
            except Exception as e:
                print(f"  [control {path}] SKIPPED: {e}")
                continue
            a, b, dd = diff_in_diff(ca)
            deltas = []
            for k in range(5):
                t = q_table(ca, SCENARIOS[f'b1c_k{k}'].build(
                    [ca, RandomAgent(), RandomAgent(), RandomAgent()]), 0)
                deltas.append(t.q(rt) - t.q(gt))
            print(f"  [control {path}]  diff-in-diff={dd:+7.2f}   "
                  f"Delta(k)={['%+.2f' % d for d in deltas]}")

    if args.mc:
        kinds = ['dqn', 'random'] if args.mc_opponents == 'both' else [args.mc_opponents]
        for kind in kinds:
            if kind == 'dqn':
                # epsilon 0.1: the distribution Q was actually trained under (train.py opp_epsilon)
                opp = AIAgent(epsilon=0.1, epsilon_min=0.1, load_model=args.model)
                opps = [opp, opp, opp]
            else:
                opps = [RandomAgent(), RandomAgent(), RandomAgent()]
            print(f"\n{'=' * 88}\nMC ADJUDICATION vs {kind} ({args.mc} games/line, paired)"
                  f"\n{'=' * 88}")
            for name in [n for n in names if SCENARIOS[n].lines]:
                scen = SCENARIOS[name]
                print(f"\n--- {scen.name}")
                res = {}
                for ln, moves in scen.lines.items():
                    res[ln] = mc_line(scen, agent, moves, opps, games=args.mc, seed=args.seed)
                    r = res[ln]
                    print(f"  {ln:<10} win {r.rate:.3f} +/- {r.se:.3f}   "
                          f"win-this-turn {r.won_this_turn.mean():.3f}   "
                          f"hand after turn {r.hand_after_turn.mean():.2f}"
                          f"   ({r.undecided} undecided)")
                base = 'dump' if 'dump' in res else 'correct'
                for ln in res:
                    if ln == base:
                        continue
                    d = res[base].wins - res[ln].wins
                    print(f"  paired  {base} - {ln:<10} {d.mean():+.3f} +/- "
                          f"{paired_se(res[base].wins, res[ln].wins):.3f}")


if __name__ == '__main__':
    main()
