#!/usr/bin/env python3
"""R6 acceptance test — the prediction B2 registered BEFORE the retrain.

B2 found the champion's defect and its mechanism, then predicted exactly what fixing the
mechanism must do. This script checks that prediction. It is written to be able to FAIL:

  1. delta(k=1) = Q(red +2) - Q(red 5) must FLIP SIGN (become positive, i.e. the agent
     wants to block a one-card opponent). This is the defect R6 targets. Rollout says
     blocking at k=1 is worth +0.050 +/- 0.017, and at k>=2 it is worth -0.04 to -0.05,
     so a correct policy's delta(k) crosses zero between k=1 and k=2.

  2. The REFUSAL rate must NOT rise. The -len(hand) term is what teaches "never refuse to
     play" (worth ~13 pts, B2 Arm 1). If the loss penalty broke that half of the reward,
     R6 is a regression no matter what it did to defence. This is the guard-rail.

  3. Strength must not collapse: vs the heuristic yardstick, not vs random (R3/B2).
"""
import argparse
import subprocess
import sys


def q_sweep(model):
    from probe import load_probe_agent
    from scenarios_b2 import sweep
    return sweep(load_probe_agent(model), model.rstrip('/').split('/')[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('--baseline', default='models/checkpoint_a9rules_snap500000')
    p.add_argument('--games', type=int, default=200)
    args = p.parse_args()

    print('=' * 78)
    print('R6 ACCEPTANCE TEST — checking B2\'s pre-registered prediction')
    print('=' * 78)

    print('\n### 1. Weapon timing: delta(k=1) must FLIP SIGN (the defect R6 targets)\n')
    base = q_sweep(args.baseline)
    new = q_sweep(args.model)

    print('\n--- verdict on prediction 1 ---')
    print(f'  baseline delta(k=1) = {base[0]:+.2f}   (champion: wrong sign — will not block)')
    print(f'  R6 model delta(k=1) = {new[0]:+.2f}')
    d_ok = new[0] > 0
    print(f'  monotone in k? baseline {"falls" if base[0] < base[-1] else "rises"} as k falls; '
          f'R6 {"falls" if new[0] < new[-1] else "rises"} as k falls (should RISE)')
    print(f'  => {"PASS: R6 blocks the near-winner" if d_ok else "FAIL: still will not block"}')

    print('\n### 2. Guard-rail: the REFUSAL rate must NOT rise\n')
    from holdback import run
    c_base = run(args.baseline, args.games, 0, 400)
    c_new = run(args.model, args.games, 0, 400)
    r_base = c_base.c['refusal'] / max(c_base.c['free'], 1)
    r_new = c_new.c['refusal'] / max(c_new.c['free'], 1)
    print(f'  baseline refusal {r_base:.2%}   R6 refusal {r_new:.2%}')
    r_ok = r_new <= r_base + 0.03
    print(f'  => {"PASS: the good half of the reward survived" if r_ok else "FAIL: it now refuses to play — regression"}')

    print('\n### 3. Strength vs the heuristic yardstick (NOT vs random — see R3/B2)\n')
    for m in (args.baseline, args.model):
        out = subprocess.run([sys.executable, 'eval.py', '--model', m, '--opponent', 'heuristic',
                              '--games', '3000', '--seed', '0'],
                             capture_output=True, text=True).stdout
        # Match eval.py's fmt() signature ('... = rate  vs 1/N ...'), not the label before it:
        # this grepped for 'DQN vs' until 2026-10-08, but 440bf60 renamed that print to
        # '{model} vs {opponent}:', so section 3 silently printed '?' for months. Only the Mode A
        # result line carries 'vs 1/N' (the '=== Mode A ===' header does not), and --run-dir is
        # never passed here, so there is exactly one match.
        line = [l for l in out.splitlines() if 'vs 1/N' in l]
        result = line[0].partition(':')[2].strip() if line else 'NO RESULT LINE PARSED'
        print(f'  {m.rstrip("/").split("/")[-1]:<34} {result}')

    print('\n' + '=' * 78)
    print(f'PREDICTION 1 (blocks a near-winner): {"PASS" if d_ok else "FAIL"}')
    print(f'PREDICTION 2 (did not break shedding): {"PASS" if r_ok else "FAIL"}')
    print('=' * 78)


if __name__ == '__main__':
    main()
