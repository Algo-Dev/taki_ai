#!/usr/bin/env python3
"""Coordinate-descent tuner for the heuristic, scored directly against the DQN champion.

PLAN.md H8. The heuristic's 18-point defect turned out to be STRUCTURAL, not a tuning
error (H1): while `score_draw` competed in the same max() as the plays, any hold above 5.0
bought a refusal. That is why this tuner waited for H1 -- searching a space where a
candidate can fall off a 13-point cliff makes a bad IDEA indistinguishable from a good idea
that tripped it. Under 'structural' the holds are unbounded-safe, so the grid below can go
where the old one was forbidden to look. This asks the obvious next question: can
hand-written rules actually BEAT the DQN champion?

CAVEAT (winner's curse): coordinate descent takes a max over many noisy estimates, so the
final in-search margin is BIASED UPWARD. Always re-measure the winner on fresh, disjoint
decks before believing it -- and against a heuristic opponent too, since tuning against one
DQN can simply overfit to that DQN.

Objective: the seat-swapped 2v2 paired margin (heuristic - DQN), averaged over two
DISJOINT deck blocks. Disjoint matters: eval_headtohead seeds deck g with `seed + g`, so
seeds must be >= `games` apart or the "replication" reuses the same decks.
"""
import argparse
import dataclasses
import itertools
import time

from agents.heuristic import Weights, HeuristicAgent, resolve_weights
from eval_headtohead import play_match, load_greedy_agent, orbit_pairs


def base_seats(num_players):
    """The seat-set to rotate. Alternating at even counts so STOP/+2/CHDIR never hit a
    teammate; at odd counts no alternating partition exists, so `[0]` is forced (and the
    orbit's complement family is what restores parity). Matches eval_headtohead's own
    convention -- the tuner must optimise the SAME quantity the promotion standard reports."""
    return [0, 2] if num_players == 4 else [0]


def margin(weights, dqn, games, seeds, num_players=4):
    """Mean paired margin (heuristic - DQN) over the rotation orbit, averaged over seeds.

    Built on `orbit_pairs` rather than a hardcoded 2v2 so one rule covers every seat count
    -- the same generalization eval_headtohead got on 2026-07-19. At 4 seats this is
    bit-identical to the old alternating swap (the orbit collapses to that one pair).

    Pairs SHARE decks, so the per-pair margins are averaged WITHIN a deck before being
    averaged across decks, never pooled as one flat list: pooling assumes an independence
    the harness does not control (measured rho = +0.097 at 3 seats)."""
    n = num_players
    pairs = orbit_pairs(base_seats(n), n)
    total = 0.0
    for seed in seeds:
        per_game = None
        for seats1, seats2 in pairs:
            h = HeuristicAgent(weights=weights)
            seats_a = [dqn] * n
            for i in seats1:
                seats_a[i] = h
            _, _, win_a = play_match(seats_a, games, seed)

            seats_b = [h] * n
            for i in seats1:
                seats_b[i] = dqn
            _, _, win_b = play_match(seats_b, games, seed)

            paired = [((1.0 if a in seats1 else 0.0) - (1.0 if b in seats1 else 0.0))
                      if a is not None and b is not None else None
                      for a, b in zip(win_a, win_b)]
            if per_game is None:
                per_game = [[] for _ in paired]
            for slot, p in zip(per_game, paired):
                if p is not None:
                    slot.append(p)
        decided = [sum(s) / len(s) for s in per_game if s]
        total += sum(decided) / len(decided) if decided else 0.0
    return total / len(seeds)


GRID = {
    # HOLD-BACK terms. The old grid capped every one of these strictly BELOW
    # score_draw=5.0, because above it the agent started refusing to play (~13 pts). That
    # cap is GONE: under H1's 'structural' mode no weight assignment can buy a refusal
    # (`agenttest.H1StructuralInvariantTest`), so the holds are unbounded-safe and the
    # search can finally go where the cap forbade. This is the whole reason H8 waited for
    # H1. The step-4 sweeps say it matters: `w_reserve` was measured FLAT from 6 to 20
    # under H1 and catastrophic there under legacy — the "catastrophe" was the cliff.
    'p_king':         [2.0, 4.0, 6.0, 9.0, 14.0],
    'p_chcol':        [1.5, 3.0, 4.5, 7.0, 11.0],
    'p_super_taki':   [2.0, 4.0, 4.5, 7.0, 11.0],
    'w_reserve':      [2.0, 4.0, 6.0, 10.0, 16.0],
    'w_open_hoard':   [5.0, 15.0, 25.0, 40.0],
    'w_save_blocker': [0.0, 0.7, 2.0, 5.0],
    'w_nofin':        [2.0, 4.0, 8.0, 14.0, 20.0],
    # ordinary scoring terms — never tuned before, all hand-set in R3.
    'w_deny':         [0.0, 1.5, 3.0, 5.0],
    'w_rich':         [0.0, 0.8, 1.6, 2.5],
    'w_block':        [2.0, 4.0, 6.0, 8.0],
    'w_chdir_block':  [0.0, 2.0, 4.0],
    'w_plus_tempo':   [0.0, 1.0, 2.0],
    'w_taki_dump':    [0.5, 1.2, 2.0, 3.0],
    # selectivity, not a hold. H2 predicted 1 and measured WORSE (-0.011); 3 was free.
    'block_hand_threshold': [1, 2, 3],
}

#: Grid adjustments that are specific to TWO seats, because the game is different there.
#: Each is a claim about the rules, not a hope:
#:   w_chdir_block  DROPPED. At n=2 the `behind` seat IS the threat seat, so the guard
#:                  `len(hands[behind]) > len(hands[threat])` compares a value to itself
#:                  and the branch can never fire. Searching it would burn candidates
#:                  measuring pure noise.
#:   w_block        WIDENED UP. H8 priced blocking where it delays 1 of 3 threats; at two
#:                  seats it delays THE threat. Searched down as well as up, because H9
#:                  now also pays STOP a tempo credit and the two could double-count.
#:   w_plus_tempo   WIDENED UP. Under H9 this knob prices two behaviours (PLUS, and the
#:                  2-seat STOP), and a free turn is worth more in a duel.
#:   block_hand_threshold  EXTENDED. `_threat_seat` only ever looks at the next seat; at
#:                  n=2 that is the only opponent, so a looser threshold cannot waste a
#:                  blocker on the wrong player the way it can at four.
GRID_2P_OVERRIDES = {
    'w_chdir_block':        None,          # None = drop the knob entirely
    'w_block':              [2.0, 4.0, 6.0, 8.0, 10.0, 13.0],
    'w_plus_tempo':         [0.0, 1.0, 2.0, 3.0, 5.0],
    'block_hand_threshold': [1, 2, 3, 4],
}


def grid_for(num_players):
    """The search grid at `num_players` seats."""
    if num_players != 2:
        return dict(GRID)
    out = {}
    for name, values in GRID.items():
        if name in GRID_2P_OVERRIDES:
            values = GRID_2P_OVERRIDES[name]
            if values is None:
                continue
        out[name] = values
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/home/orih/taki-ai/models/checkpoint_r6L60_snap100000',
                   help='the DQN to beat. Default: the R6 champion (NOT A9-rules — B2 tuned '
                        'against A9-rules, and R6 beats it by +8.5, so that target is stale).')
    p.add_argument('--games', type=int, default=3000)
    p.add_argument('--seeds', default='0,777777')
    p.add_argument('--passes', type=int, default=2)
    p.add_argument('--base', default='h1b2',
                   help='the frozen version to start from (agents.heuristic.VERSIONS)')
    p.add_argument('--num-players', type=int, default=4,
                   help='seat count to tune AT. The counts are different games (STOP is a '
                        'free extra turn at 2, CHDIR a no-op), so a point tuned at one is '
                        'not a point tuned at another. Note the 3-seat orbit is 3 pairs, '
                        'i.e. ~3x the cost per candidate.')
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    n = args.num_players

    dqn = load_greedy_agent(args.model)

    # Start from H1_B2: B2's retuned weights ON H1's structure. Starting from B2_RETUNED
    # (legacy) would search a space where a candidate can still fall off the refusal cliff,
    # and a bad IDEA would be indistinguishable from a good one that tripped it — which is
    # exactly what makes automated tuning untrustworthy without H1.
    best = resolve_weights(args.base)
    if best.refusal_mode != 'structural':
        raise SystemExit(f'--base {args.base!r} is not structural; the grid below goes far '
                         f'above |score_draw| and would buy refusals. Use h1b2 (or h1).')
    grid = grid_for(n)
    t0 = time.time()
    best_m = margin(best, dqn, args.games, seeds, n)
    print(f'start ({args.base} @ {n} seats): {best_m:+.4f}   [{time.time()-t0:.0f}s]', flush=True)

    for it in range(args.passes):
        print(f'\n--- pass {it + 1} ---', flush=True)
        for name, values in grid.items():
            cur = getattr(best, name)
            for v in values:
                if v == cur:
                    continue
                cand = dataclasses.replace(best, **{name: v})
                m = margin(cand, dqn, args.games, seeds, n)
                flag = ''
                if m > best_m + 1e-9:
                    best, best_m, flag = cand, m, '  <- keep'
                print(f'  {name:16s} {cur:>5} -> {v:<5}  {m:+.4f}{flag}', flush=True)
                if flag:
                    cur = v
        print(f'\n  best after pass {it + 1}: {best_m:+.4f}', flush=True)
        print(f'  {ovr(best, args.base)}', flush=True)

    print(f'\n=== BEST (margin heuristic - DQN = {best_m:+.4f}) ===')
    print(f'  spec: heuristic:{ovr(best, args.base)}')
    print(f'  (took {time.time() - t0:.0f}s)')


def ovr(w, base_name='h1b2'):
    """The spec string that reproduces `w`, as a delta from its BASE version — so it can be
    pasted straight into `--opponent heuristic:<spec>`."""
    d = dataclasses.asdict(w)
    base = dataclasses.asdict(resolve_weights(base_name))
    delta = ','.join(f'{k}={v}' for k, v in d.items() if v != base[k])
    return f'{base_name},{delta}' if delta else base_name


if __name__ == '__main__':
    main()
