#!/usr/bin/env python3
"""Quick head-to-head eval of two checkpoints."""
import sys
import argparse
import math
import time
from agents.dqn import AIAgent
from agents.random import RandomAgent
from agents.heuristic import make_heuristic
from game import Game

def load_greedy_agent(checkpoint_path):
    """Load a trained DQN agent in greedy mode."""
    return AIAgent(epsilon=0.0, epsilon_min=0.0, load_model=checkpoint_path)


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

def run_config(agent1, agent2, seats1, games, seed, label):
    """Play `games` with agent1 at `seats1` and agent2 elsewhere. Returns (seat_wins, total, winners)."""
    seat_agents = [agent2] * 4
    for i in seats1:
        seat_agents[i] = agent1
    print(f'\n--- {label} ---')
    return play_match(seat_agents, games, seed, verbose=True)


def seat_swap(agent1, agent2, name1, name2, seats1, games, seed):
    """Run both occupancies of one partition and report the seat-balanced, paired comparison.

    Across the two runs each seat is occupied by each model exactly once, so the seat-0
    first-mover advantage cancels by construction instead of being averaged away.
    """
    seats2 = [i for i in range(4) if i not in seats1]

    wins_a, total_a, winners_a = run_config(agent1, agent2, seats1, games, seed,
                                            f'Run A: {name1} @ {seats1} vs {name2} @ {seats2}')
    wins_b, total_b, winners_b = run_config(agent2, agent1, seats1, games, seed,
                                            f'Run B: {name2} @ {seats1} vs {name1} @ {seats2}')

    # Per game, did model1 win? In run A it holds seats1; in run B it holds seats2.
    # Same decks in both runs (deck RNG is seeded per game), so this is a paired sample.
    paired = [(1.0 if w_a in seats1 else 0.0) - (1.0 if w_b in seats1 else 0.0)
              for w_a, w_b in zip(winners_a, winners_b)
              if w_a is not None and w_b is not None]
    n = len(paired)
    mean_d = sum(paired) / n if n else 0.0
    var = sum((d - mean_d) ** 2 for d in paired) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n else 0.0

    # Seat-balanced rate: each model's win rate averaged over all four seats, one
    # observation per seat (from whichever run it occupied that seat in).
    r_a = [w / total_a for w in wins_a] if total_a else [0.0] * 4
    r_b = [w / total_b for w in wins_b] if total_b else [0.0] * 4
    bal1 = (sum(r_a[i] for i in seats1) + sum(r_b[i] for i in seats2)) / 4
    bal2 = (sum(r_a[i] for i in seats2) + sum(r_b[i] for i in seats1)) / 4

    print(f'\n=== Seat-swap summary (partition {seats1} / {seats2}, '
          f'{games} games x 2 runs, seed {seed}) ===')
    print(f'  per-seat win rates, run A: {[f"{r:.3f}" for r in r_a]}')
    print(f'  per-seat win rates, run B: {[f"{r:.3f}" for r in r_b]}')
    print(f'\n  Seat-balanced per-seat rate (parity 0.250):')
    print(f'    {name1}: {bal1:.3f}')
    print(f'    {name2}: {bal2:.3f}')
    print(f'\n  Paired same-seat margin ({name1} team win rate, run A minus run B; parity 0.0):')
    print(f'    {mean_d:+.4f} +/- {se:.4f}  (paired SE over {n} common decks)')

    if abs(mean_d) < 2 * se:
        verdict = f'Tie (margin within 2 paired SE)'
    elif mean_d > 0:
        verdict = f'{name1} stronger (margin > 2 paired SE)'
    else:
        verdict = f'{name2} stronger (margin > 2 paired SE)'
    print(f'\nVerdict: {verdict}')


def main():
    parser = argparse.ArgumentParser(description='Head-to-head eval of two checkpoints, '
                                                  'any 4-seat assignment (e.g. 1v3, 2v2, 3v1, '
                                                  'at arbitrary seat positions).')
    parser.add_argument('model1', help='team 1: a checkpoint path, "random", "heuristic", '
                                       '"heuristic:greedy", or "heuristic:-<ablation>"')
    parser.add_argument('model2', help='team 2: same forms as model1')
    parser.add_argument('--team1-seats', default='0',
                        help='comma-separated seat indices (0-3) for model1; remaining seats '
                             'get model2 (default "0" = model1 solo at seat 0)')
    parser.add_argument('--seat-swap', action='store_true',
                        help='run BOTH occupancies of the partition (model1 at --team1-seats, '
                             'then model2 at --team1-seats) over the same decks, and report the '
                             'seat-balanced, paired comparison. This is the standard promotion '
                             'test; use with --team1-seats 0,2 (alternating 2v2).')
    parser.add_argument('--games', type=int, default=3000,
                        help='games per run (default 3000)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for reproducibility')
    args = parser.parse_args()

    team1_idx = sorted(int(x) for x in args.team1_seats.split(','))
    if not team1_idx or any(i < 0 or i > 3 for i in team1_idx) or len(set(team1_idx)) != len(team1_idx):
        parser.error('--team1-seats must be distinct seat indices in [0, 3]')
    if len(team1_idx) == 4:
        parser.error('--team1-seats must leave at least one seat for model2')
    team2_idx = [i for i in range(4) if i not in team1_idx]

    print(f'Loading {args.model1}...')
    agent1 = make_agent(args.model1)
    print(f'Loading {args.model2}...')
    agent2 = make_agent(args.model2)

    name1 = args.model1.rstrip('/').split('/')[-1]
    name2 = args.model2.rstrip('/').split('/')[-1]

    baseline = 0.25  # 1/4 players
    print(f'\nBaseline (1/4) = {baseline:.3f}')

    t0 = time.time()

    if args.seat_swap:
        seat_swap(agent1, agent2, name1, name2, team1_idx, args.games, args.seed)
        elapsed = time.time() - t0
        print(f'(took {elapsed:.1f}s for 2 x {args.games} games)')
        return

    seat_agents = [None] * 4
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
