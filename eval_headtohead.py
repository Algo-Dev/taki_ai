#!/usr/bin/env python3
"""Quick head-to-head eval of two checkpoints."""
import sys
import argparse
import time
from agents.dqn import AIAgent
from agents.random import RandomAgent
from game import Game

def load_greedy_agent(checkpoint_path):
    """Load a trained DQN agent in greedy mode."""
    return AIAgent(epsilon=0.0, epsilon_min=0.0, load_model=checkpoint_path)

def play_match(seat_agents, games, seed=0, verbose=False, team1_seats=None):
    """Play with a fixed list of 4 agents (one per seat). Returns per-seat win counts.

    team1_seats: optional set of seat indices belonging to "team 1" (for aggregate
    reporting); if None, every seat is reported individually.
    """
    game = Game(seat_agents, seed=seed)
    num_players = len(seat_agents)

    seat_wins = [0] * num_players
    total_decided = 0

    for g in range(games):
        if verbose and (g + 1) % 500 == 0:
            print(f'  game {g + 1}/{games}...')

        # Reseed per game for common random numbers (any agent exposing reseed()).
        # Dedup by identity since the same agent object may fill multiple seats.
        for a in {id(a): a for a in seat_agents}.values():
            if hasattr(a, 'reseed'):
                a.reseed(f'{seed}:{g}:{id(a)}')

        game.reset()
        done = False
        while not done:
            done, _ = game.next_turn()

        for i in range(num_players):
            if len(game.hands[i]) == 0:
                seat_wins[i] += 1
                break
        total_decided += 1

    return seat_wins, total_decided

def fmt(wins, total):
    """Format win rate with count."""
    if total == 0:
        return "N/A"
    rate = wins / total
    return f"{wins}/{total} = {rate:.3f}"

def main():
    parser = argparse.ArgumentParser(description='Head-to-head eval of two checkpoints, '
                                                  'any 4-seat assignment (e.g. 1v3, 2v2, 3v1, '
                                                  'at arbitrary seat positions).')
    parser.add_argument('model1', help='checkpoint for team 1')
    parser.add_argument('model2', help='checkpoint for team 2')
    parser.add_argument('--team1-seats', default='0',
                        help='comma-separated seat indices (0-3) for model1; remaining seats '
                             'get model2 (default "0" = model1 solo at seat 0)')
    parser.add_argument('--games', type=int, default=3000,
                        help='games to play (default 3000)')
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
    agent1 = load_greedy_agent(args.model1)
    print(f'Loading {args.model2}...')
    agent2 = load_greedy_agent(args.model2)

    seat_agents = [None] * 4
    for i in team1_idx:
        seat_agents[i] = agent1
    for i in team2_idx:
        seat_agents[i] = agent2

    baseline = 0.25  # 1/4 players
    print(f'\nBaseline (1/4) = {baseline:.3f}')
    print(f'\n=== Head-to-Head: {args.model1.split("/")[-1]} @ seats {team1_idx} vs '
          f'{args.model2.split("/")[-1]} @ seats {team2_idx} ({args.games} games) ===')

    t0 = time.time()
    seat_wins, total = play_match(seat_agents, args.games, args.seed, verbose=True)
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
