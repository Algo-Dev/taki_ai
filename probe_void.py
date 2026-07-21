"""R4 behavioural probe: does the policy USE the per-opponent color-void feature?

One fixed position (see probe.SCENARIOS['void_*']); three history variants differing ONLY in the
color the next player (seat 1) has revealed a void in. The learner chooses to leave seat 1 BLUE or
GREEN by playing blue-5 or green-5 on a red 5. Correct B8 denial: hand them a color they LACK.

  D(variant) = Q(play blue-5) - Q(play green-5)

A void-USING net raises D when seat 1 lacks blue and lowers it when seat 1 lacks green, so the
signal is  S = D(void_blue) - D(void_green) > 0.  Because only `history` varies, the deck / discard
/ hands are byte-identical across variants, so the shift isolates the feature. A pre-R4 net (150
floats) cannot see the void block at all -> S == 0 by construction (the built-in negative control).

Usage:
    python probe_void.py --model <r4_ckpt> [--control <pre_r4_ckpt>] [--seeds 0,1,2,3]
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import argparse
import numpy as np

from game import Card, Color, Type, Action, action_to_scalar
from agents.dqn import AIAgent
from probe import SCENARIOS, legal_scalars, _Dummy

BLUE5 = action_to_scalar(Action.PLAY_CARD, Card(Type.FIVE, Color.BLUE))
GREEN5 = action_to_scalar(Action.PLAY_CARD, Card(Type.FIVE, Color.GREEN))


def load(path):
    return AIAgent(epsilon=0.0, epsilon_min=0.0, load_model=path, allow_obs_truncation=True)


def qv(agent, game, seat=0):
    """Raw Q-vector, fed on exactly the leading floats this net was trained on (so a pre-R4
    control genuinely cannot see the appended void block)."""
    obs = game.observation(seat)
    x = obs[np.newaxis, :agent.obs_size].astype(np.float32)
    return agent.model(x, training=False).numpy()[0]


def D(agent, variant, seed):
    scen = SCENARIOS[f'void_{variant}']
    game = scen.build([_Dummy() for _ in range(4)], seed=seed)
    legal = legal_scalars(game, 0)
    assert BLUE5 in legal and GREEN5 in legal, [l for l in legal]
    q = qv(agent, game, 0)
    return float(q[BLUE5] - q[GREEN5])


def report(name, agent, seeds):
    print(f'\n=== {name}  (obs_size={agent.obs_size}) ===')
    print(f'  {"variant":<12}{"D=Q(blue5)-Q(green5)":>24}')
    means = {}
    for v in ('blue', 'none', 'green'):
        ds = [D(agent, v, s) for s in seeds]
        means[v] = np.mean(ds)
        print(f'  seat1 lacks {v:<7}{means[v]:>18.4f}   (per-seed sd {np.std(ds):.4f})')
    signal = means['blue'] - means['green']
    print(f'  --> void-usage signal S = D(blue) - D(green) = {signal:+.4f}')
    if agent.obs_size < 162:
        print('      [pre-R4 net: cannot see the void block; S must be ~0 by construction]')
    else:
        verdict = ('USES the void feature (leaves the lacked color)' if signal > 0.05
                   else 'does NOT meaningfully use it' if abs(signal) <= 0.05
                   else 'ANTI-uses it (wrong sign)')
        print(f'      verdict: policy {verdict}')
    return signal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='the R4 (162-float) candidate')
    ap.add_argument('--control', default='', help='a pre-R4 net (e.g. M1s3) as negative control')
    ap.add_argument('--seeds', default='0,1,2,3,4,5,6,7',
                    help='deck seeds to average D over (the fixed hand is constant; the deck/'
                         'discard filler varies with seed)')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    report('R4 candidate', load(args.model), seeds)
    if args.control:
        report('CONTROL (pre-R4)', load(args.control), seeds)


if __name__ == '__main__':
    main()
