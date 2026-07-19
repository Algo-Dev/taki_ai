#!/usr/bin/env python3
"""Quick head-to-head eval of two checkpoints."""
import sys
import argparse
import math
import time
from agents.dqn import AIAgent
from agents.random import RandomAgent
from agents.heuristic import make_heuristic
from game import Game, MIN_PLAYERS

def load_greedy_agent(checkpoint_path):
    """Load a trained DQN agent in greedy mode.

    allow_obs_truncation lets a checkpoint from an older, shorter observation contract play
    on exactly the leading floats it was trained on (the observation has only ever grown by
    appending, so the prefix is bit-identical -- see AIAgent). This is what makes a
    cross-contract head-to-head possible at all: R6 (147 floats) can share a table with a
    seat-count-one-hot net (150).
    """
    return AIAgent(epsilon=0.0, epsilon_min=0.0, load_model=checkpoint_path,
                   allow_obs_truncation=True)


def make_agent(spec):
    """Resolve an agent spec (PLAN.md B2 needs non-checkpoint opponents here).

      random                -> RandomAgent
      heuristic             -> the reference heuristic (agents.heuristic.REFERENCE)
      heuristic:<version>   -> a named, frozen version: r3, b2, greedy
      heuristic:-<name>     -> the reference with ONE hold-back behaviour off,
                               for name in ABLATIONS (hoard, wilds, blocker,
                               finisher, king_follow, king_cancel)
      heuristic:k=v,k=v     -> the reference with weights overridden (sweeps)
      <path>                -> a DQN checkpoint

    The spec grammar lives in agents.heuristic.resolve_weights — one definition,
    shared with eval.py, so a version means the same thing in every harness.
    """
    if spec == 'random':
        return RandomAgent(seed=0)
    if spec == 'heuristic':
        return make_heuristic()
    if spec.startswith('heuristic:'):
        try:
            return make_heuristic(spec.split(':', 1)[1])
        except ValueError as e:
            raise SystemExit(str(e))
    return load_greedy_agent(spec)

def play_match(seat_agents, games, seed=0, verbose=False):
    """Play with a fixed list of 4 agents (one per seat).

    Returns (seat_wins, total_decided, winners), where winners[g] is the seat that won
    game g (or None if undecided).

    The deck RNG is reseeded to `seed + g` before every game, so game g's deal is a
    function of (seed, g) alone and not of how the earlier games happened to play out.
    That makes two runs over the same (seed, games) share decks game-for-game, which is
    what lets a seat swap be compared as a paired sample.
    """
    game = Game(seat_agents, seed=seed)
    num_players = len(seat_agents)

    seat_wins = [0] * num_players
    winners = []
    total_decided = 0

    for g in range(games):
        if verbose and (g + 1) % 500 == 0:
            print(f'  game {g + 1}/{games}...')

        # Reseed per game for common random numbers (any agent exposing reseed()).
        # Dedup by identity since the same agent object may fill multiple seats.
        for a in {id(a): a for a in seat_agents}.values():
            if hasattr(a, 'reseed'):
                a.reseed(f'{seed}:{g}:{id(a)}')
        game.random.seed(seed + g)

        game.reset()
        done = False
        while not done:
            done, _ = game.next_turn()

        winner = next((i for i in range(num_players) if len(game.hands[i]) == 0), None)
        winners.append(winner)
        if winner is not None:
            seat_wins[winner] += 1
            total_decided += 1

    return seat_wins, total_decided, winners

def fmt(wins, total):
    """Format win rate with count."""
    if total == 0:
        return "N/A"
    rate = wins / total
    return f"{wins}/{total} = {rate:.3f}"

def run_config(agent1, agent2, seats1, num_players, games, seed, label):
    """Play `games` with agent1 at `seats1` and agent2 elsewhere. Returns (seat_wins, total, winners)."""
    seat_agents = [agent2] * num_players
    for i in seats1:
        seat_agents[i] = agent1
    print(f'\n--- {label} ---')
    return play_match(seat_agents, games, seed, verbose=True)


def orbit_pairs(seats1, num_players):
    """The ROTATION ORBIT of the base partition: the run set, for any seat count.

    One rule generates the design at every count. Take the base seat-set, rotate it around
    the table, and pair each rotation with its complement; dedup unordered pairs, since a
    pair {b, c} and {c, b} name the same two runs.

        n=2, base {0}:    {0}|{1}                                    -> 1 pair  (2 runs)
        n=3, base {0}:    {0}|{1,2}   {1}|{0,2}   {2}|{0,1}          -> 3 pairs (6 runs)
        n=4, base {0,2}:  {0,2}|{1,3}                                -> 1 pair  (2 runs)

    Four seats is NOT a special case: the alternating pattern is rotationally symmetric with
    period 2, so its orbit collapses from 4 to 2 and the complement family collapses into it.
    Two seats collapses the same way, for the same reason. Odd counts have no alternating
    pattern, so nothing collapses and the orbit is the full 2n runs.

    Balance is a property of the construction, not of the count: across the orbit each model
    occupies each seat equally often, and holds each team size equally often (at odd n the
    complement family is exactly what restores parity -- a model is solo in half the runs and
    in the majority in the other half).
    """
    seen, pairs = set(), []
    for r in range(num_players):
        b = frozenset((i + r) % num_players for i in seats1)
        c = frozenset(range(num_players)) - b
        key = frozenset([b, c])
        if key in seen:
            continue
        seen.add(key)
        pairs.append((sorted(b), sorted(c)))
    return pairs


def seat_swap(agent1, agent2, name1, name2, seats1, num_players, games, seed):
    """Run the rotation orbit of the partition and report the seat-balanced, paired comparison.

    Each orbit element is a complementary pair of runs over the SAME decks, so the seat
    advantage cancels by construction rather than being averaged away. At four seats this is
    the historical two-run seat swap, unchanged.

    CAVEAT AT ODD COUNTS: with n odd there is no equal partition and no alternating pattern,
    so the larger team's seats are necessarily adjacent -- friendly fire (STOP/+2/CHDIR land
    on a teammate) is structural and no seating design removes it. A 3-seat comparison is a
    real comparison, but not a clean one; say so when quoting it.
    """
    pairs = orbit_pairs(seats1, num_players)
    print(f'\n=== Rotation orbit of {sorted(seats1)} at {num_players} seats: '
          f'{len(pairs)} complementary pair(s) = {2 * len(pairs)} runs ===')
    for b, c in pairs:
        print(f'    {name1} @ {b} vs {name2} @ {c}   +   the complement')

    results = []   # (seats_b, seats_c, wins_b, total_b, winners_b, wins_c, total_c, winners_c)
    for k, (b, c) in enumerate(pairs):
        wins_b, total_b, winners_b = run_config(
            agent1, agent2, b, num_players, games, seed,
            f'Pair {k + 1} run 1: {name1} @ {b} vs {name2} @ {c}')
        wins_c, total_c, winners_c = run_config(
            agent1, agent2, c, num_players, games, seed,
            f'Pair {k + 1} run 2: {name1} @ {c} vs {name2} @ {b}')
        results.append((b, c, wins_b, total_b, winners_b, wins_c, total_c, winners_c))

    # Paired statistic. Within a complementary pair, model1 holds b in one run and c in the
    # other, so its two parities sum to 1 and (won_1 + won_2 - 1) has parity 0 at every count
    # -- including the lopsided 1-vs-2 at three seats. Average over pairs WITHIN a game first:
    # the pairs share decks, so they are correlated, and one observation per deck is what makes
    # the SE honest.
    per_game = []
    per_pair = [[] for _ in results]   # d_k(g), kept per pair for the per-pair breakdown
    for g in range(games):
        ds = []
        for b, c, _, _, win_b, _, _, win_c in results:
            w1, w2 = win_b[g], win_c[g]
            if w1 is None or w2 is None:
                ds = None
                break
            ds.append((1.0 if w1 in b else 0.0) + (1.0 if w2 in c else 0.0) - 1.0)
        if ds:
            per_game.append(sum(ds) / len(ds))
            for k, d in enumerate(ds):
                per_pair[k].append(d)

    def mean_se(xs):
        m = sum(xs) / len(xs) if xs else 0.0
        v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0
        return m, math.sqrt(v / len(xs)) if xs else 0.0

    n = len(per_game)
    mean_d, se = mean_se(per_game)

    # Seat-balanced rate: average the per-seat win rate over every (run, seat) a model sat in.
    # The orbit covers seats evenly, so this is unbiased at parity 1/num_players.
    obs1, obs2 = [], []
    for b, c, wins_b, total_b, _, wins_c, total_c, _ in results:
        if total_b:
            obs1 += [wins_b[i] / total_b for i in b]
            obs2 += [wins_b[i] / total_b for i in c]
        if total_c:
            obs1 += [wins_c[i] / total_c for i in c]
            obs2 += [wins_c[i] / total_c for i in b]
    bal1 = sum(obs1) / len(obs1) if obs1 else 0.0
    bal2 = sum(obs2) / len(obs2) if obs2 else 0.0

    print(f'\n=== Orbit summary ({num_players} seats, {len(pairs)} pair(s) x 2 runs x '
          f'{games} games, seed {seed}) ===')
    for k, (b, c, wins_b, total_b, _, wins_c, total_c, _) in enumerate(results):
        r_b = [w / total_b for w in wins_b] if total_b else [0.0] * num_players
        r_c = [w / total_c for w in wins_c] if total_c else [0.0] * num_players
        print(f'  pair {k + 1} ({b}|{c}) per-seat rates: '
              f'run1 {[f"{r:.3f}" for r in r_b]}  run2 {[f"{r:.3f}" for r in r_c]}')

    # Per-pair margins. Each is a complete, self-contained paired comparison at one position of
    # the base -- so a SPREAD across pairs means the edge is seat-dependent, which the pooled
    # margin averages away. This is not a curiosity: the old hardcoded 1v3 could only ever run
    # pair 1, and doing so overstated the champion's margin (+0.0500 vs +0.0333 rotating).
    if len(results) > 1:
        print(f'\n  Per-pair margins (each is the full paired comparison at one base position):')
        pms = []
        for k, (b, c, *_) in enumerate(results):
            m_k, se_k = mean_se(per_pair[k])
            pms.append(m_k)
            print(f'    pair {k + 1} ({b}|{c}): {m_k:+.4f} +/- {se_k:.4f}')
        print(f'    spread (max - min): {max(pms) - min(pms):+.4f}'
              f'   <- large spread => the edge depends on WHERE, not just on skill')

    print(f'\n  Seat-balanced per-seat rate (parity {1 / num_players:.3f}):')
    print(f'    {name1}: {bal1:.3f}')
    print(f'    {name2}: {bal2:.3f}')
    print(f'\n  Paired margin ({name1} minus {name2}, same seats & same decks; parity 0.0):')
    print(f'    {mean_d:+.4f} +/- {se:.4f}  (paired SE over {n} common decks)')

    if abs(mean_d) < 2 * se:
        verdict = f'Tie (margin within 2 paired SE)'
    elif mean_d > 0:
        verdict = f'{name1} stronger (margin > 2 paired SE)'
    else:
        verdict = f'{name2} stronger (margin > 2 paired SE)'
    print(f'\nVerdict: {verdict}')


def main():
    parser = argparse.ArgumentParser(description='Head-to-head eval of two checkpoints at any '
                                                  'seat count, any team assignment (e.g. 1v3, '
                                                  '2v2, 1v2, at arbitrary seat positions).')
    parser.add_argument('model1', help='team 1: a checkpoint path, "random", "heuristic", '
                                       '"heuristic:greedy", or "heuristic:-<ablation>"')
    parser.add_argument('model2', help='team 2: same forms as model1')
    parser.add_argument('--num-players', type=int, default=4,
                        help='seat count (default 4). The rotation orbit adapts: 2 runs at an '
                             'even count with an alternating base, 2n runs at an odd count.')
    parser.add_argument('--team1-seats', default='0',
                        help='comma-separated seat indices for model1; remaining seats get '
                             'model2 (default "0" = model1 solo at seat 0). Under --seat-swap '
                             'this is the BASE of a rotation orbit, so only its shape matters, '
                             'not its position.')
    parser.add_argument('--seat-swap', action='store_true',
                        help='run the ROTATION ORBIT of --team1-seats -- every rotation of the '
                             'base paired with its complement, over the same decks -- and report '
                             'the seat-balanced, paired comparison. This is the standard '
                             'promotion test. Use the alternating base at even counts '
                             '(--team1-seats 0,2 at 4 seats, 0 at 2 seats); at odd counts no '
                             'alternating base exists, so --team1-seats 0 is forced up to '
                             'rotation and friendly fire is unavoidable.')
    parser.add_argument('--games', type=int, default=3000,
                        help='games per run (default 3000)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for reproducibility')
    args = parser.parse_args()

    n = args.num_players
    if n < MIN_PLAYERS:
        parser.error(f'--num-players must be at least {MIN_PLAYERS}')
    team1_idx = sorted(int(x) for x in args.team1_seats.split(','))
    if (not team1_idx or any(i < 0 or i >= n for i in team1_idx)
            or len(set(team1_idx)) != len(team1_idx)):
        parser.error(f'--team1-seats must be distinct seat indices in [0, {n - 1}]')
    if len(team1_idx) == n:
        parser.error('--team1-seats must leave at least one seat for model2')
    team2_idx = [i for i in range(n) if i not in team1_idx]

    print(f'Loading {args.model1}...')
    agent1 = make_agent(args.model1)
    print(f'Loading {args.model2}...')
    agent2 = make_agent(args.model2)

    name1 = args.model1.rstrip('/').split('/')[-1]
    name2 = args.model2.rstrip('/').split('/')[-1]

    baseline = 1.0 / n
    print(f'\nBaseline (1/{n}) = {baseline:.3f}')

    t0 = time.time()

    if args.seat_swap:
        seat_swap(agent1, agent2, name1, name2, team1_idx, n, args.games, args.seed)
        elapsed = time.time() - t0
        print(f'(took {elapsed:.1f}s for 2 x {args.games} games)')
        return

    seat_agents = [None] * n
    for i in team1_idx:
        seat_agents[i] = agent1
    for i in team2_idx:
        seat_agents[i] = agent2

    print(f'\n=== Head-to-Head: {name1} @ seats {team1_idx} vs '
          f'{name2} @ seats {team2_idx} ({args.games} games) ===')
    print('  (single occupancy: confounded by the seat-0 first-mover edge — '
          'use --seat-swap for a promotion decision)')

    seat_wins, total, _ = play_match(seat_agents, args.games, args.seed, verbose=True)
    elapsed = time.time() - t0

    team1_wins = sum(seat_wins[i] for i in team1_idx)
    team2_wins = sum(seat_wins[i] for i in team2_idx)
    team1_per_seat = team1_wins / total / len(team1_idx) if total > 0 else 0
    team2_per_seat = team2_wins / total / len(team2_idx) if total > 0 else 0

    print(f'\nModel1 team (seats {team1_idx}) total: {fmt(team1_wins, total)}'
          f'  per-seat: {team1_per_seat:.3f}')
    print(f'Model2 team (seats {team2_idx}) total: {fmt(team2_wins, total)}'
          f'  per-seat: {team2_per_seat:.3f}')
    print(f'  (baseline per-seat = {baseline:.3f})')
    print(f'  per-seat win rates: {[f"{w/total:.3f}" for w in seat_wins]}')

    if abs(team1_per_seat - team2_per_seat) < 0.005:
        verdict = 'Tie (within noise)'
    elif team1_per_seat > team2_per_seat:
        verdict = f'{args.model1.split("/")[-1]} stronger per-seat'
    else:
        verdict = f'{args.model2.split("/")[-1]} stronger per-seat'
    print(f'\nVerdict: {verdict}')
    print(f'(took {elapsed:.1f}s, {total/elapsed:.0f} games/sec)')

if __name__ == '__main__':
    main()
