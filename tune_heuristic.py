#!/usr/bin/env python3
"""Coordinate-descent tuner for the heuristic, scored directly against the DQN champion.

PLAN.md B5. B2 showed R3's heuristic ships with a ~17-point tuning bug (hold-back weights
above its own draw threshold, so it draws rather than plays). Un-breaking that reached
parity with the champion. This asks the obvious next question: can hand-written rules
actually BEAT a 500k-trial DQN?

Objective: the seat-swapped 2v2 paired margin (heuristic - DQN), averaged over two
DISJOINT deck blocks. Disjoint matters: eval_headtohead seeds deck g with `seed + g`, so
seeds must be >= `games` apart or the "replication" reuses the same decks.
"""
import argparse
import dataclasses
import itertools
import time

from agents.heuristic import Weights, HeuristicAgent, B2_RETUNED
from eval_headtohead import play_match, load_greedy_agent

SEATS1 = [0, 2]          # alternating 2v2: STOP/+2/CHDIR never hit a teammate


def margin(weights, dqn, games, seeds):
    """Mean paired margin (heuristic - DQN) over both occupancies, averaged over seeds."""
    total = 0.0
    for seed in seeds:
        h = HeuristicAgent(weights=weights)
        seats_a = [dqn] * 4
        for i in SEATS1:
            seats_a[i] = h
        _, _, win_a = play_match(seats_a, games, seed)

        seats_b = [h] * 4
        for i in SEATS1:
            seats_b[i] = dqn
        _, _, win_b = play_match(seats_b, games, seed)

        paired = [(1.0 if a in SEATS1 else 0.0) - (1.0 if b in SEATS1 else 0.0)
                  for a, b in zip(win_a, win_b) if a is not None and b is not None]
        total += sum(paired) / len(paired) if paired else 0.0
    return total / len(seeds)


GRID = {
    # hold-back terms. Keep the p_* strictly BELOW score_draw=5.0 (B2's threshold
    # result) — at or above it the agent starts refusing to play, which costs ~13 pts.
    'p_king':         [3.0, 4.0, 4.9],
    'p_chcol':        [3.0, 4.0, 4.9],
    'p_super_taki':   [3.0, 4.0, 4.9],
    'w_reserve':      [2.0, 4.0, 4.9],
    'w_open_hoard':   [5.0, 15.0, 25.0],
    'w_save_blocker': [0.0, 0.7, 2.0],
    'w_nofin':        [2.0, 4.0, 8.0],
    # ordinary scoring terms — never tuned before, all hand-set in R3.
    'w_deny':         [0.0, 1.5, 3.0, 5.0],
    'w_rich':         [0.0, 0.8, 1.6, 2.5],
    'w_block':        [2.0, 4.0, 6.0, 8.0],
    'w_chdir_block':  [0.0, 2.0, 4.0],
    'w_plus_tempo':   [0.0, 1.0, 2.0],
    'w_taki_dump':    [0.5, 1.2, 2.0, 3.0],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/checkpoint_a9rules_snap500000')
    p.add_argument('--games', type=int, default=1500)
    p.add_argument('--seeds', default='0,500000')
    p.add_argument('--passes', type=int, default=2)
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    dqn = load_greedy_agent(args.model)

    # Start from B2's retuned point: every preference hold-back kept, every refusal removed.
    # Now the named, frozen version rather than a local copy (this copy had drifted to
    # p_king=4.9 against b2_block_price.py's 5.0 — the drift the registry exists to stop).
    best = B2_RETUNED
    t0 = time.time()
    best_m = margin(best, dqn, args.games, seeds)
    print(f'start (B2 retuned): {best_m:+.4f}   [{time.time()-t0:.0f}s]', flush=True)

    for it in range(args.passes):
        print(f'\n--- pass {it + 1} ---', flush=True)
        for name, values in GRID.items():
            cur = getattr(best, name)
            for v in values:
                if v == cur:
                    continue
                cand = dataclasses.replace(best, **{name: v})
                m = margin(cand, dqn, args.games, seeds)
                flag = ''
                if m > best_m + 1e-9:
                    best, best_m, flag = cand, m, '  <- keep'
                print(f'  {name:16s} {cur:>5} -> {v:<5}  {m:+.4f}{flag}', flush=True)
                if flag:
                    cur = v
        print(f'\n  best after pass {it + 1}: {best_m:+.4f}', flush=True)
        print(f'  {ovr(best)}', flush=True)

    print(f'\n=== BEST (margin heuristic - DQN = {best_m:+.4f}) ===')
    print(f'  spec: heuristic:{ovr(best)}')
    print(f'  (took {time.time() - t0:.0f}s)')


def ovr(w):
    d = dataclasses.asdict(w)
    base = dataclasses.asdict(Weights())
    return ','.join(f'{k}={v}' for k, v in d.items() if v != base[k])


if __name__ == '__main__':
    main()
