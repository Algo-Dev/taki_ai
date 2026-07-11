"""Sharded exploitability eval for the A12 best-response probe.

Reuses eval.py's play_match / loaders (identical common-random-numbers protocol,
per-game seat shuffle so no seat-0 confound) but evaluates an EXPLICIT list of
snapshots, so several thread-capped workers can each take a disjoint subset and
run in parallel without overloading the box. Each worker appends one line per
snapshot to a shared results file; a separate pass sorts and reports.

Usage:
    python eval_shard.py --baseline <ckpt> --games <N> --out <file> <snapdir> ...
"""
import argparse
import os
import re

# eval.py sets TF thread env vars at import time; import it first so those (and our
# TAKI_* overrides from the environment) are in force before any TF op runs.
from eval import load_greedy_agent, play_match
from agents.random import RandomAgent


def trial_of(snapdir):
    m = re.search(r'snap(\d+)', os.path.basename(snapdir.rstrip('/')))
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True, help='frozen reference checkpoint (A8)')
    ap.add_argument('--games', type=int, default=3000)
    ap.add_argument('--num-players', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True, help='results file to append to')
    ap.add_argument('snapshots', nargs='+')
    args = ap.parse_args()

    base = load_greedy_agent(args.baseline)
    rnd = RandomAgent()
    for snap in args.snapshots:
        t = trial_of(snap)
        agent = load_greedy_agent(snap)
        bw, bd, bu = play_match(agent, base, args.num_players, args.games, seed=args.seed)
        rw, rd, ru = play_match(agent, rnd, args.num_players, args.games, seed=args.seed)
        brate = bw / bd if bd else float('nan')
        rrate = rw / rd if rd else float('nan')
        line = (f'{t}\tvs_a8={brate:.4f}\t({bw}/{bd}, undec {bu})\t'
                f'vs_random={rrate:.4f}\t({rw}/{rd}, undec {ru})')
        with open(args.out, 'a', buffering=1) as f:
            f.write(line + '\n')
        print(f'[pid {os.getpid()}] {line}', flush=True)


if __name__ == '__main__':
    main()
