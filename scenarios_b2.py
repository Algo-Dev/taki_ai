#!/usr/bin/env python3
"""B2, scenario arm — weapon timing. Does the champion save a +2 for a near-winner?

The census (holdback.py) cannot answer this, and neither could B1. Both a +2 and a plain
number shed EXACTLY ONE card, so the choice between them is invisible to a shed count —
and, more importantly, invisible to the shaped reward, which pays `-len(hand)` and is
therefore *exactly indifferent* between them.

That is what makes this the sharpest probe in the B-series: it isolates the VALUE FUNCTION
from the reward. Whatever preference shows up here was not paid for. And it is the one place
R6 could still be motivated, since PLAN.md's R6 argument is precisely that defensive play
(holding a blocker for the player about to go out) is under-incentivised.

Design (mirrors b1c's matched-contrast): only ONE thing varies across the sweep — k, the
number of cards in the NEXT player's hand. The legal move set, the deck size and the unseen
counts are identical for every k (pinned in scenariotest_b2), so

    delta(k) = Q(red +2) - Q(red 5)

is a within-position contrast whose only driver is the opponent-hand-size feature (present
in the observation since A7, and rewarded by nothing). Strategy says delta should RISE as k
falls: fire the +2 exactly when the next player is about to win. Flat in k => no weapon
timing at all.
"""
import argparse

from game import Action, Card, Color, Type
from probe import (Scenario, c, load_probe_agent, untrained_agent, q_table, make_position)

PLAY = Action.PLAY_CARD
R, B, G = Color.RED, Color.BLUE, Color.GREEN

# The hand is fixed for every k. A red +2 and a red 5 are both playable on a red 7 and both
# shed one card; a blue 9 is the off-colour filler that keeps the hand size constant.
_HAND = (c(Type.PLUSTWO, R), c(Type.FIVE, R), c(Type.NINE, B))


def b2_weapon(k):
    """The next player holds k cards; the two other opponents hold 5 each.

    `discard_size = 27 - k` is NOT cosmetic. The deck is whatever the pool has left, so
    shrinking the next player's hand would otherwise GROW the deck — and `len(deck)/120`
    is an observation feature (obs[143]). The sweep would then vary two things at once and
    delta(k) could be a deck-size artefact rather than a threat response. Absorbing the
    difference into the discard holds obs[143] fixed; the discard is filled with NUMBER
    cards only (make_position's default), so the unseen +2/King/CHCOL counts stay at their
    full totals and do not move either. Pinned by holdbacktest.WeaponTimingScenarioTest:
    obs[140] (the next player's hand size) must be the ONLY feature that changes.
    """
    return Scenario(
        name=f'b2_weapon_k{k}',
        desc=f'Weapon timing, k={k}: the NEXT player holds {k} cards. A red +2 and a red 5 '
             f'are both legal and both shed exactly one card.',
        hand=_HAND, top=c(Type.SEVEN, R),
        opp_sizes=(k, 5, 5), discard_size=27 - k,
        expect_legal=((PLAY, c(Type.PLUSTWO, R)), (PLAY, c(Type.FIVE, R)),
                      (Action.DRAW, None)),
        expect_illegal=((PLAY, c(Type.NINE, B)),),
        lines={'block': ((PLAY, c(Type.PLUSTWO, R)),),
               'number': ((PLAY, c(Type.FIVE, R)),)},
        correct='Fire the +2 when k is small; keep it when k is large.',
    )


KS = (1, 2, 3, 4, 5, 6, 7)
SCENARIOS_B2 = {s.name: s for s in [b2_weapon(k) for k in KS]}


def sweep(agent, label, seed=0):
    from probe import _Dummy
    print(f'\n=== weapon timing: {label} ===')
    print('  delta(k) = Q(red +2) - Q(red 5).  Strategy: delta should RISE as k falls.')
    print(f'  {"k (next player cards)":<24}{"Q(+2)":>9}{"Q(red 5)":>10}{"delta":>9}   picks')
    print('  ' + '-' * 62)
    deltas = []
    for k in KS:
        scen = b2_weapon(k)
        game = scen.build([agent] + [_Dummy()] * 3, seed=seed)
        t = q_table(agent, game)
        q_p2 = t.q(_scalar(c(Type.PLUSTWO, R)))
        q_n5 = t.q(_scalar(c(Type.FIVE, R)))
        d = q_p2 - q_n5
        deltas.append(d)
        from probe import move_name
        print(f'  {k:<24}{q_p2:>9.2f}{q_n5:>10.2f}{d:>+9.2f}   {move_name(t.greedy)}')
    span = max(deltas) - min(deltas)
    print(f'\n  spread over k: {span:.2f}   '
          f'(a flat delta means the opponent-hand-size feature is not used here)')
    return deltas


def _scalar(card):
    from game import action_to_scalar
    return action_to_scalar(PLAY, card)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/checkpoint_a9rules_snap500000')
    p.add_argument('--controls', action='store_true',
                   help='also sweep untrained nets — the null: a random-init net has no '
                        'weapon timing, so its spread is the noise floor.')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    sweep(load_probe_agent(args.model), args.model.rstrip('/').split('/')[-1], args.seed)
    if args.controls:
        for s in range(3):
            sweep(untrained_agent(s), f'untrained net (seed {s}) — NULL', args.seed)


if __name__ == '__main__':
    main()
