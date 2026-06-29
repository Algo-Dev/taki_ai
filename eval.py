"""Evaluate trained Taki DQN agents against fixed opponents.

Two questions this answers (training itself can't, because train.py syncs the
opponents to the learner -> symmetric self-play -> win rate sits near 1/num_players
regardless of skill):

  Mode A (--model):   does the trained agent beat *random* opponents, i.e. win above
                      the 1/num_players chance line?  1 trained DQN + (N-1) RandomAgent.

  Mode B (--run-dir): does the *latest* model improve over earlier ones?  For every
                      snapshot S_t saved during training, play 1x S_t + (N-1)x S_0
                      (S_0 = the untrained snap0000).  A win-rate curve rising above the
                      chance line shows genuine improvement over the earliest model.
                      The first point (S_0 vs S_0) doubles as a sanity control: equal
                      policies should land near 1/num_players.

All agents play fully greedily (epsilon=0) so the comparison reflects learned policy,
not exploration.  Games share a deterministic deck/seating sequence across matchups
(seeded) so snapshots are compared on identical games (variance reduction).
"""
import os
# The network is tiny; TF's default thread pools oversubscribe and thrash (huge system
# time / context-switch counts on this box), so single-sample predict is far faster with
# a small pool. Must be set before TensorFlow is imported (via agents.dqn below).
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import argparse
import glob
import re
import random
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe; we only save PNGs
from matplotlib import pyplot as plt

from game import Game
from agents.dqn import AIAgent
from agents.random import RandomAgent

# A game only ends when a hand empties (State.FINISHED); a deck-exhaustion stalemate
# never sets done(), so cap the turns per game and treat an over-cap game as undecided.
# Legit 4-player games finish well under this; the cap only bounds degenerate stalls.
TURN_CAP = 400
# If the first STALL_CHECK games of a matchup are ALL undecided (capped), the matchup is
# degenerate (e.g. untrained greedy nets that only ever draw) -> bail out instead of
# burning every game on 400-turn stalls.
STALL_CHECK = 8


def load_greedy_agent(path):
    """Load a snapshot as a fully-greedy (epsilon=0) AIAgent."""
    return AIAgent(epsilon=0.0, epsilon_min=0.0, load_model=path)


def play_match(test_agent, opponent_agent, num_players, games, seed=0):
    """Play `games` games of 1x test_agent + (num_players-1)x opponent_agent.

    A single opponent object fills every opponent seat (play() is stateless across
    seats: it reads the current seat's observation, and last_state/last_action are
    overwritten each call). Seating and deck are seeded so every call to play_match
    with the same seed sees identical games.

    Returns (test_wins, decided, undecided).
    """
    seat_rng = random.Random(seed)
    test_wins = decided = undecided = 0
    for g in range(games):
        agents = [test_agent] + [opponent_agent] * (num_players - 1)
        order = list(range(num_players))
        seat_rng.shuffle(order)                 # randomise seating, deterministically
        seat_agents = [agents[i] for i in order]
        test_seat = order.index(0)              # where the test agent ended up
        game = Game(seat_agents, seed=seed + g)  # identical decks across matchups
        turns = 0
        while not game.done() and turns < TURN_CAP:
            game.next_turn()
            turns += 1
        if game.done():
            decided += 1
            winner = [i for i, h in enumerate(game.hands) if not h]
            if winner and winner[0] == test_seat:
                test_wins += 1
        else:
            undecided += 1
            # Degenerate matchup (only ever draws): stop early rather than running every
            # game out to TURN_CAP.
            if undecided >= STALL_CHECK and decided == 0:
                break
    return test_wins, decided, undecided


def fmt(wins, decided, undecided, baseline):
    rate = wins / decided if decided else float('nan')
    verdict = 'ABOVE' if decided and rate > baseline else 'at/below'
    extra = f' ({undecided} undecided/capped)' if undecided else ''
    return (f'{wins}/{decided} = {rate:.3f}  vs baseline {baseline:.3f}  -> {verdict}'
            f' chance{extra}')


def discover_snapshots(run_dir):
    """Return [(trial_int, path), ...] sorted by trial for snap<NNNN> dirs in run_dir."""
    snaps = []
    for p in glob.glob(os.path.join(run_dir, 'snap*')):
        m = re.search(r'snap(\d+)', os.path.basename(p))
        if m:
            snaps.append((int(m.group(1)), p))
    return sorted(snaps)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Taki DQN agents vs fixed opponents.')
    parser.add_argument('--model', default=None,
                        help='mode A: a single checkpoint to test against random opponents')
    parser.add_argument('--run-dir', default=None,
                        help='mode B: a run directory of snap<NNNN> checkpoints to chart progression')
    parser.add_argument('--games', type=int, default=500,
                        help='games per matchup for mode A')
    parser.add_argument('--games-b', type=int, default=150,
                        help='games per snapshot for mode B (both sides run a network, so keep modest)')
    parser.add_argument('--num-players', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not args.model and not args.run_dir:
        parser.error('pass --model (mode A) and/or --run-dir (mode B)')

    baseline = 1.0 / args.num_players
    print(f'Baseline (1/num_players) = {baseline:.3f}  [{args.num_players} players]')

    # ---- Mode A: trained DQN vs random ------------------------------------------------
    if args.model:
        print(f'\n=== Mode A: {args.model} vs {args.num_players - 1} random opponents '
              f'({args.games} games) ===')
        t0 = time.time()
        dqn = load_greedy_agent(args.model)
        rnd = RandomAgent(seed=args.seed)
        wins, decided, undecided = play_match(dqn, rnd, args.num_players, args.games, args.seed)
        print('DQN vs random:', fmt(wins, decided, undecided, baseline))
        print(f'(took {time.time() - t0:.1f}s)')

    # ---- Mode B: snapshot-vs-earliest progression -------------------------------------
    if args.run_dir:
        snaps = discover_snapshots(args.run_dir)
        if not snaps:
            parser.error(f'no snap<NNNN> checkpoints found in {args.run_dir}')
        print(f'\n=== Mode B: progression in {args.run_dir} '
              f'(each snapshot vs earliest, {args.games_b} games each) ===')
        s0_trial, s0_path = snaps[0]
        s0_agent = load_greedy_agent(s0_path)
        trials, rates = [], []
        t0 = time.time()
        for trial, path in snaps:
            # snap0000 vs itself is the sanity control (~baseline); later snaps test improvement.
            agent = s0_agent if path == s0_path else load_greedy_agent(path)
            wins, decided, undecided = play_match(agent, s0_agent, args.num_players,
                                                  args.games_b, args.seed)
            rate = wins / decided if decided else float('nan')
            trials.append(trial)
            rates.append(rate)
            tag = '  <- control (equal policies)' if path == s0_path else ''
            print(f'snap{trial:04d} vs snap{s0_trial:04d}: '
                  f'{fmt(wins, decided, undecided, baseline)}{tag}')
        print(f'(took {time.time() - t0:.1f}s)')

        out = os.path.join(args.run_dir, 'progression.png')
        # Skip undecided (degenerate / capped) matchups, which come back as nan.
        pts = [(t, r) for t, r in zip(trials, rates) if not np.isnan(r)]
        plt.figure(figsize=(8, 5))
        if pts:
            plt.plot([t for t, _ in pts], [r for _, r in pts],
                     marker='o', label='win rate vs snap%04d' % s0_trial)
        plt.axhline(baseline, ls='--', color='gray', label=f'baseline {baseline:.3f}')
        plt.xlabel('training trial (snapshot)')
        plt.ylabel('win rate vs earliest snapshot')
        plt.title('Self-play improvement over training')
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out)
        print(f'Saved progression plot to {out}')


if __name__ == '__main__':
    main()
