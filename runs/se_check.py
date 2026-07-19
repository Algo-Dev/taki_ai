"""Measure the actual correlation between orbit pairs sharing a deck, and the two SE estimators.

Claim under test: pooling the 3 pairs' per-deck differences as one flat sample understates the
SE by sqrt(1 + 2*rho). I asserted ~sqrt(3), which is the rho=1 worst case. Measure rho.
"""
import math
import sys
sys.path.insert(0, '/home/orih/taki-ai')

from eval_headtohead import make_agent, orbit_pairs, run_config

CHAMP = '/home/orih/taki-ai/models/checkpoint_M1s3_mixed_snap300000'
OPP = 'heuristic:h8'
N, GAMES, SEED = 3, 3000, 0

a1 = make_agent(CHAMP)
a2 = make_agent(OPP)

results = []
for b, c in orbit_pairs([0], N):
    _, _, wb = run_config(a1, a2, b, N, GAMES, SEED, f'{b} vs {c}')
    _, _, wc = run_config(a1, a2, c, N, GAMES, SEED, f'{c} vs {b}')
    results.append((b, c, wb, wc))

cols = [[] for _ in results]          # d_k(g), aligned across pairs
per_deck = []
for g in range(GAMES):
    ds = []
    for b, c, wb, wc in results:
        if wb[g] is None or wc[g] is None:
            ds = None
            break
        ds.append((1.0 if wb[g] in b else 0.0) + (1.0 if wc[g] in c else 0.0) - 1.0)
    if ds:
        per_deck.append(sum(ds) / len(ds))
        for k, d in enumerate(ds):
            cols[k].append(d)

def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(v / n), math.sqrt(v)

def corr(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (dx * dy) if dx and dy else 0.0

m_avg, se_avg, sd_avg = mean_se(per_deck)
pooled = [d for col in cols for d in col]
m_pool, se_pool, sd_pool = mean_se(pooled)

rhos = [corr(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]
rho = sum(rhos) / len(rhos)

print('\n================ SE ESTIMATOR COMPARISON (3 seats, vs h8, block 0) ================')
print(f'  decks used: {len(per_deck)}   pairs: {len(cols)}')
print(f'  per-deck averaged : mean {m_avg:+.4f}  SE {se_avg:.4f}   (n={len(per_deck)})')
print(f'  pooled flat       : mean {m_pool:+.4f}  SE {se_pool:.4f}   (n={len(pooled)})')
print(f'  ratio SE_avg / SE_pooled = {se_avg / se_pool:.3f}')
print(f'\n  pairwise correlations between pairs on a shared deck: '
      f'{[f"{r:+.3f}" for r in rhos]}')
print(f'  mean rho = {rho:+.4f}')
print(f'  predicted understatement sqrt(1 + 2*rho) = {math.sqrt(max(0.0, 1 + 2 * rho)):.3f}')
print(f'  worst case (rho=1) would be sqrt(3) = {math.sqrt(3):.3f}')
