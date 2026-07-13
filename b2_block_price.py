"""Re-price the block-vs-number decision against a COMPETENT opponent pool.

B2's pricing table (RESEARCH_LOG 2026-07-13: block is +0.050 at k=1, negative at every
other k) was rolled out against `3x heuristic` with SHIPPED weights — the same weights B2
then found to be crippled by the refusal cliff (14.5% voluntary draws). A pool that draws
when it should play ends games slowly, which is exactly the variable a blocker's value is
sensitive to. That table is the sole evidence for H2 (`block_hand_threshold` should be 1,
not 2), so it is worth knowing whether it survives a pool that actually plays.

Same scenario, same lines, same paired determinizations as B2 — only the opponents change.
"""
import argparse
import dataclasses as dc

import numpy as np

from agents.heuristic import HeuristicAgent, Weights
from probe import load_probe_agent, mc_line, paired_se
from scenarios_b2 import KS, b2_weapon

# The retuned weights from B2 Result 5: hold-back terms pulled below the draw threshold.
# 0.850 -> 0.899 vs random, and parity with the 500k champion.
RETUNED = Weights(p_king=5.0, p_chcol=4.5, p_super_taki=4.5, w_reserve=4.0,
                  king_cancel_min_penalty=0, hold_wilds_in_run=False)

POOLS = {'shipped': Weights(), 'retuned': RETUNED}


def price(learner, weights, games, seed):
    """block - number, per k, on paired deals. Positive => the +2 is worth spending."""
    rows = []
    for k in KS:
        scen = b2_weapon(k)
        opps = [HeuristicAgent(weights=dc.replace(weights)) for _ in range(3)]
        res = {ln: mc_line(scen, learner, moves, opps, games=games, seed=seed)
               for ln, moves in scen.lines.items()}
        b, n = res['block'].wins, res['number'].wins
        rows.append((k, b.mean(), n.mean(), b.mean() - n.mean(), paired_se(b, n)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/checkpoint_a9rules_snap500000')
    p.add_argument('--games', type=int, default=1200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pools', default='shipped,retuned')
    args = p.parse_args()

    learner = load_probe_agent(args.model)
    for name in args.pools.split(','):
        rows = price(learner, POOLS[name], args.games, args.seed)
        print(f"\n=== block - number vs 3x heuristic ({name} weights), "
              f"{args.games} paired determinizations, seed {args.seed} ===")
        print(f"  {'k':>3}{'block':>9}{'number':>9}{'delta':>10}{'+/- SE':>9}   verdict")
        for k, bm, nm, d, se in rows:
            v = 'BLOCK' if d > 2 * se else ('do not block' if d < -2 * se else 'wash')
            print(f"  {k:>3}{bm:>9.3f}{nm:>9.3f}{d:>+10.3f}{se:>9.3f}   {v}")


if __name__ == '__main__':
    main()
