"""Evaluate trained Taki DQN agents against fixed opponents.

Two questions this answers (training itself can't, because train.py syncs the
opponents to the learner -> symmetric self-play -> win rate sits near 1/num_players
regardless of skill):

  Mode A (--model):   does the trained agent beat *random* opponents, i.e. win above
                      the 1/num_players chance line?  1 trained DQN + (N-1) RandomAgent.

  Mode B (--run-dir): how does skill progress over training?  For a sampled subset of
                      snapshots (--snap-stride), measure each one two ways: 1x S_t vs random,
                      and (with --baseline) 1x S_t vs a frozen reference model.  vs-random
                      shows convergence speed; vs-baseline above 1/num_players means S_t has
                      genuinely surpassed that reference (e.g. our current-best model).  A
                      snapshot equal to the baseline lands at ~1/num_players (parity).

All agents play fully greedily (epsilon=0) so the comparison reflects learned policy,
not exploration.  Every game is seeded individually — deck, seating AND the random
opponents' choice stream (reseeded per game via RandomAgent.reseed) — so game g of any
matchup replays identical randomness regardless of which snapshot is being tested or how
earlier games unfolded.  Snapshots are therefore compared on identical games (common
random numbers, variance reduction), and any (model, seed, games) result is reproducible
independent of the surrounding sweep.
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
from agents.heuristic import REFERENCE
from eval_headtohead import make_agent as resolve_agent

# A game only ends when a hand empties (State.FINISHED); a deck-exhaustion stalemate
# never sets done(), so cap the turns per game and treat an over-cap game as undecided.
# Legit 4-player games finish well under this (trained-vs-random tops out ~476 turns), so
# 2000 leaves ample headroom; the cap only bounds genuinely degenerate (untrained) stalls.
TURN_CAP = 2000
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
    overwritten each call). Every game is individually seeded: the deck, the seating
    and — for opponents exposing reseed() (RandomAgent) — the opponent's choice stream,
    so game g is played on identical randomness in every matchup that shares `seed`,
    no matter how games 0..g-1 went. Greedy nets are deterministic and need no reseed.

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
        if hasattr(opponent_agent, 'reseed'):
            # Common random numbers: restart the opponent's choice stream per game
            # (string seed: distinct from the deck's integer stream seed+g below).
            opponent_agent.reseed(f'{seed}:{g}:opp')
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
    return (f'{wins}/{decided} = {rate:.3f}  vs 1/N {baseline:.3f}  -> {verdict}'
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
                        help='mode A: a single checkpoint to test against the opponents, or the '
                             'sentinel "heuristic" (optionally "heuristic:<version>") to test the '
                             'hand-crafted agent itself')
    parser.add_argument('--run-dir', default=None,
                        help='mode B: a run directory of snap<NNNN> checkpoints to chart progression')
    parser.add_argument('--baseline', default=None,
                        help='mode B: a frozen checkpoint (e.g. the current-best model) to also '
                             'measure each snapshot against; above 1/num_players means it beats it')
    parser.add_argument('--snap-stride', type=int, default=100,
                        help='mode B: only evaluate snapshots whose trial index is a multiple of '
                             'this (snapshots are saved every 25; default 100 -> every-100 subset)')
    parser.add_argument('--games', type=int, default=3000,
                        help='games per matchup for mode A (default 3000 -> SE ~= 0.008, the '
                             'precision needed to rank near-equal models; see RESEARCH_LOG.md)')
    parser.add_argument('--games-b', type=int, default=150,
                        help='games per matchup for mode B (per snapshot, per reference)')
    parser.add_argument('--opponent', default='random',
                        help='what fills the N-1 opponent seats (modes A and B\'s vs-opponent '
                             'curve). "heuristic" = the hand-crafted reference agent (currently '
                             f'{REFERENCE!r}) — a harder, non-lineage yardstick than random. '
                             '"heuristic:<version>" pins a specific frozen version (r3, b2, '
                             'greedy); every published "vs heuristic" number to date is vs r3. '
                             'Default: random.')
    parser.add_argument('--num-players', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not args.model and not args.run_dir:
        parser.error('pass --model (mode A) and/or --run-dir (mode B)')

    baseline = 1.0 / args.num_players
    print(f'Baseline (1/num_players) = {baseline:.3f}  [{args.num_players} players]')

    # Ranking near-equal snapshots needs >=3000 games (SE ~= 0.008); fewer games produced a
    # false "plateau" read historically (see RESEARCH_LOG.md). Warn rather than block.
    PRECISION_GAMES = 3000
    if args.model and args.games < PRECISION_GAMES:
        print(f'  [warning] mode A --games={args.games} < {PRECISION_GAMES}: too noisy to rank '
              f'near-equal models (SE ~= {0.5 / args.games ** 0.5:.4f}).')
    if args.run_dir and args.games_b < PRECISION_GAMES:
        print(f'  [warning] mode B --games-b={args.games_b} < {PRECISION_GAMES}: fine for the '
              f'vs-random convergence curve, too noisy for ranking near-equal snapshots.')

    # ---- Mode A: trained DQN vs random ------------------------------------------------
    def make_opponent():
        if args.opponent.startswith('heuristic'):
            return resolve_agent(args.opponent)
        return RandomAgent(seed=args.seed)

    if args.model:
        print(f'\n=== Mode A: {args.model} vs {args.num_players - 1} {args.opponent} opponents '
              f'({args.games} games) ===')
        t0 = time.time()
        test = resolve_agent(args.model) if args.model.startswith('heuristic') \
            else load_greedy_agent(args.model)
        opp = make_opponent()
        wins, decided, undecided = play_match(test, opp, args.num_players, args.games, args.seed)
        print(f'{args.model} vs {args.opponent}:', fmt(wins, decided, undecided, baseline))
        print(f'(took {time.time() - t0:.1f}s)')

    # ---- Mode B: progression vs random and vs a frozen baseline -----------------------
    if args.run_dir:
        snaps = discover_snapshots(args.run_dir)
        if not snaps:
            parser.error(f'no snap<NNNN> checkpoints found in {args.run_dir}')
        # Saved every 25, but evaluating all of them is costly; sample the every-stride subset.
        snaps = [(t, p) for t, p in snaps if t % args.snap_stride == 0]
        if not snaps:
            parser.error(f'no snapshots with trial %% {args.snap_stride} == 0 in {args.run_dir}')

        rnd = make_opponent()
        # The frozen reference (e.g. the current-best model): each snapshot above 1/num_players
        # against it has genuinely surpassed it. Constant model -> no "moving reference" artifact.
        base_agent = load_greedy_agent(args.baseline) if args.baseline else None
        refs = ['random'] + (['baseline'] if base_agent else [])
        print(f'\n=== Mode B: progression in {args.run_dir} '
              f'({len(snaps)} snapshots vs {", ".join(refs)}, {args.games_b} games each) ===')
        if args.baseline:
            print(f'    baseline = {args.baseline}')

        trials, rnd_rates, base_rates = [], [], []
        t0 = time.time()
        for trial, path in snaps:
            agent = load_greedy_agent(path)
            rw, rd, ru = play_match(agent, rnd, args.num_players, args.games_b, args.seed)
            rnd_rate = rw / rd if rd else float('nan')
            trials.append(trial)
            rnd_rates.append(rnd_rate)
            line = f'snap{trial:04d}  vs random: {fmt(rw, rd, ru, baseline)}'
            if base_agent is not None:
                bw, bd, bu = play_match(agent, base_agent, args.num_players, args.games_b, args.seed)
                base_rates.append(bw / bd if bd else float('nan'))
                line += f'\n           vs baseline: {fmt(bw, bd, bu, baseline)}'
            print(line)
        print(f'(took {time.time() - t0:.1f}s)')

        out = os.path.join(args.run_dir, 'progression.png')
        plt.figure(figsize=(8, 5))

        def _plot(rates, label):
            pts = [(t, r) for t, r in zip(trials, rates) if not np.isnan(r)]  # skip undecided
            if pts:
                plt.plot([t for t, _ in pts], [r for _, r in pts], marker='o', label=label)

        _plot(rnd_rates, 'win rate vs random')
        if base_rates:
            _plot(base_rates, 'win rate vs baseline')
        plt.axhline(baseline, ls='--', color='gray', label=f'1/num_players = {baseline:.3f}')
        plt.xlabel('training trial (snapshot)')
        plt.ylabel('win rate')
        plt.title('Skill over training (vs random; vs baseline = beats current best above the line)')
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out)
        print(f'Saved progression plot to {out}')


if __name__ == '__main__':
    main()
