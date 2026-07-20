# Taki AI — Research Log

Running log of training/eval experiments and their results. Newest entries on top.
Each entry records the setup, the headline metrics, and caveats so runs stay comparable.

Definitions used throughout:
- **Win rate (vs random):** fraction of *decided* games a greedy trained DQN wins as 1 of
  N seats, the rest `RandomAgent`. Chance baseline is `1/N`.
- **Undecided rate:** Taki has no real draw; a game only ends when a hand empties. `eval.py`
  cuts a game off at `TURN_CAP` turns and counts it **undecided**. For *trained* agents this
  is almost entirely a too-low cap on naturally long games (they finish given more turns);
  only *untrained* (random-weight) nets genuinely never terminate. See the draw-stall
  investigation below.

---

## 2026-07-21 — PROMOTION: REFERENCE h8 -> h9. A strict improvement, so only the two-seat column moves

At the user's direction, the reference yardstick (`REFERENCE` in `agents/heuristic.py`, what a bare
`heuristic` spec resolves to) is promoted from `h8` to `h9`. Unlike the r3->h8 promotion (B5), this
one is a **strict improvement with a bounded blast radius**: h9 is bit-identical to h8 at three and
four seats (the STOP-tempo fix is gated to n=2 by the rules, pinned by fingerprint equality) and
+0.026 better at two, where h8 priced a free extra turn at zero. So the promotion redefines the
meaning of `heuristic` **only at two seats**; every 3- and 4-seat number ever measured against h8
still describes the reference verbatim.

**Champions re-run against the new reference (the discipline for a REFERENCE change), orbit per-seat
rate at two seats, two deck blocks:**

| @2 seats | vs h8 (old ref) | vs h9 (new ref) |
|---|---|---|
| M1s3 champion | 0.577 | **0.567** (0.562 / 0.571) |
| R6 | 0.417 | **0.402** (0.401 / 0.402) |

The champion drops ~1 point against the sharper yardstick, as +0.026/2 per seat predicts. **No
ranking changes** — the champion still beats the reference comfortably at two seats, R6 still loses.

**What this does and does not touch.** `heuristic:h8` still runs and still means exactly what it did;
pre-2026-07-21 two-seat "vs heuristic" numbers refer to it and should be cited as `heuristic:h8`. At
3 and 4 seats there is nothing to restate. `h10` was **not** promoted (2-seat-tuned, worth ~0 at four
seats, and its edge over h9 is 1.4 SE); it remains an explicitly-cited 2-seat opponent. The freeze
guard `agenttest.FrozenVersionTest.test_the_reference_is_h9_...` now pins `REFERENCE == 'h9'`.

---

## 2026-07-20 — H9/H10: the yardstick was four-seat-tuned, and at two seats it priced a free extra turn at ZERO. The structural line is worth +0.026; 96 tuned candidates on top add +0.010 at 1.4 SE

**The premise, which only became measurable last night.** Every frozen heuristic version was
tuned or hand-set at four seats, and `tune_heuristic.py` was hardwired there. With the rotation
orbit (entry below) the cost is finally quantifiable. Using that entry's identity — paired margin
`= n x (seat-balanced rate) - 1`, so `margin / n` is the per-seat edge over parity:

| M1s3 champion vs `h8` | paired margin | `margin / n` |
|---|---|---|
| 2 seats | +0.153 | **+0.0765** |
| 3 seats | +0.104 | +0.0347 |
| 4 seats | +0.106 | +0.0265 |

`h8` is ~3x further behind the champion at two seats than at four, in per-seat terms. That is the
signature of a yardstick tuned at one count and quoted at three.

### The defect: a two-seat STOP is a free extra turn, scored as zero

`_seat_after_playing` returns `(me + 2d) % n` for STOP. That equals `me` iff `n | 2d`, i.e. **iff
n == 2**: STOP skips the next player, and with two seats the next player is the only other seat, so
the turn comes straight back. At two seats a STOP is a free extra turn — exactly what PLUS is.

The scoring code never saw it. STOP landed in the `_BLOCKER_TYPES` branch, which pays `w_block`
only against a threat, and with H8's `w_save_blocker = 0.0` a threat-free 2-seat STOP scored
**exactly 0.0** — ranked *below a plain number card* of a color we hold (`w_rich` pays for those).
Meanwhile the deny/richness guard `next_seat != game.curr` correctly suppressed those terms for a
self-returning card, so the zero was not an accident of one term; it was the whole score.

**No weight assignment can fix this**, because no weight can see that the turn came back. Same
shape as H1's refusal bug: a structure-level defect that looks like a tuning problem.

### H9: one line, derived from the rules, gated behind a version

Inside the existing blocker branch: if `_seat_after_playing(card)` is our own seat, add
`w_plus_tempo`. Rules-derived, so it fires only where it is true. Gated on a new `structure` field
(`'h1'` = every prior version, `'h9'`) so `h8` itself is untouched and its published numbers keep
describing the agent in the tree.

**Measured, heuristic-vs-heuristic, rotation orbit, 3000 decks/block:**

| h9 - h8 | blk 0 | blk 777777 | mean |
|---|---|---|---|
| 2 seats | +0.0317 +/- 0.0079 | +0.0207 +/- 0.0080 | **+0.026** |
| 3 seats | +0.0000 +/- 0.0000 | — | **exactly 0** |
| 4 seats | +0.0000 +/- 0.0000 | — | **exactly 0** |

The zeros are not ties within noise — there is no noise. Every paired difference was exactly 0, on
every deck: h9 and h8 are the same agent above two seats, as the construction requires. (This
exposed a cosmetic harness bug: `abs(0) < 2*0` is false, so the verdict fell through and reported
model2 as stronger. Now reported as IDENTICAL.)

Controls: vs `greedy` h9 is +0.147 against h8's +0.125 (the hold-backs are intact, not eaten); vs
random 0.966 against 0.963.

> **It missed its own pre-registered bar, and the bar is not being moved.** +0.04 pooled was
> registered before the run; the result is +0.026. Real, replicated, ~4.5 SE pooled — and below
> the bar. It recovers ~17% of h8's 2-seat deficit from one line.

### H10: the tuner ran, and the tuner is not what did it

Coordinate descent from H9 at two seats vs the M1s3 champion, 2 passes, **96 candidates**, 6868s.
Grid adapted at n=2: `w_chdir_block` dropped (at two seats `behind` IS `threat`, so its guard
compares a value to itself and can never fire), `w_block` / `w_plus_tempo` /
`block_hand_threshold` widened.

    heuristic:h9,w_deny=0.0,w_plus_tempo=3.0,w_save_blocker=2.0,w_reserve=16.0,p_king=2.0,block_hand_threshold=3

In-search: -0.1332 -> -0.1128. On **fresh disjoint decks** (tuning used 0/777777; these are
1500000/2500000):

| @2 seats | blk 1500000 | blk 2500000 | mean | SE |
|---|---|---|---|---|
| h10 - h8 | +0.0370 | +0.0400 | **+0.0385** | 5.3 SE |
| **h10 - h9** | +0.0133 | +0.0060 | **+0.0097** | **1.4 SE — not established** |

**So H9's one structural line is worth +0.026 of the +0.0385, and everything 96 tuned candidates
found on top is within noise of zero.** That is the H8 result a second time ("the tuner went above
the old cap and came back empty", 2026-07-14) and the third time in this project that a structural
fix paid where weight search did not. The winner's curse was mild (in-search implied ~+0.045
against a measured +0.0385), so this is not a case of the search being fooled — it simply found
little to find.

### The gain is not an artifact of tuning against one net

Paired margin (heuristic - DQN) at two seats, fresh blocks, mean of two:

| vs | `h8` | `h10` | gain |
|---|---|---|---|
| M1s3 (the tuning target) | -0.1588 | -0.1218 | +0.0370 |
| M1s2 (promotion runner-up — *seed* overfit test) | -0.1577 | -0.1277 | +0.0300 |
| P1 (2p-curriculum net — *recipe* overfit test) | -0.1381 | -0.0955 | +0.0426 |

The gain appears against a different seed and a differently-trained net, including the largest
gain against the one net that was never in the loop. Not a specialist. The primary metric
(h10 - h8) is DQN-free anyway, and the tuner never optimised it.

Controls: vs `greedy` +0.153 (h8 +0.125, h9 +0.147); vs random 0.964; refusal census 1, against 2
for both h8 and h9 — pre-existing CLOSE_TAKI declines (H4), not voluntary draws.

### Two behavioural findings, both 2-seat-specific, both worth a probe

1. **`w_deny` 1.5 -> 0.0 — colour denial is worth nothing at two seats**, the single largest step
   in the search (+0.009). Candidate mechanism: denying your *only* opponent a colour makes them
   draw, and a drawn card is a card you then have to out-race; at four seats the same denial lands
   on someone else's problem. Untested.
2. **`w_save_blocker` 0.0 -> 2.0 — H8's advice REVERSES.** H8 says spend the blocker; at two seats,
   hold it. Note the interaction with H9: once STOP earns tempo it becomes attractive enough to
   need a counterweight against being dumped early. The two changes are not independent.

Both are single coordinate-descent steps of ~1-1.5 SE. Do not cite them as established.

### H11: the same idea at three and four seats — REFUTED, and its stated premise was false

Planned as "at three seats STOP and CHDIR are the same move, yet are scored by different
terms". **That premise is wrong**, and checking it against the engine before building on it is
the only reason this entry is not also wrong. Traced at n=3, seat 0 acting:

    STOP    0, 2, 0, 1, 2, ...    we act again after ONE opponent
    CHDIR   0, 2, 1, 0, 2, ...    both opponents act before us

They agree on who faces the table **next** and on nothing else. That is all
`_seat_after_playing` ever claimed, and it is right about it. CHDIR buys no tempo at any
count — reversing a cycle still leaves `n - 1` opponents ahead of us — so there is no CHDIR
defect to fix.

> **CLAUDE.md asserted the false equivalence twice ("CHDIR is a no-op at two and *identical to
> STOP* at three"), both times sourced to RULES.md — which never says it.** RULES.md documents
> only the 2-player no-op. Both CLAUDE.md sites are corrected. The claim had been sitting in the
> file since the seat-count one-hot work and may have informed how that feature was reasoned
> about.

What *is* true is the STOP half, quantitatively: a STOP skips one of the `n - 1` opponents who
would otherwise act before our next turn, so it buys `1 / (n - 1)` of what a PLUS buys. No free
parameter — the fraction is the rules. At n=2 the fraction is 1.0, so **H11 is H9 at two seats,
move for move** (pinned by fingerprint equality). H11 is H8's weights plus that rule at every
count.

**Pre-registered before measuring:** win at >= +0.010 pooled at three seats, same sign and >= 2 SE
in each block; no regression past -2 SE at four.

| h11 - h8 | blk 0 | blk 777777 | mean | |
|---|---|---|---|---|
| 3 seats | +0.0026 +/- 0.0054 | +0.0069 +/- 0.0055 | **+0.0048** | 1.2 SE — **fails the bar** |
| 4 seats | -0.0036 +/- 0.0038 | -0.0016 +/- 0.0040 | **-0.0026** | 0.9 SE — no regression, but negative |

**Refuted.** Kept registered and pinned but not endorsed, the same status as `h7`.

**Why the linear model was wrong, most likely.** The n=2 case is not "the largest value of a
smooth quantity" — it is *categorical*. At two seats a STOP gives a complete extra turn: we act
twice in a row, and the opponent's hand does not shrink in between. At n>=3 it only moves us one
place up the queue, and the skipped opponent still holds every card they held. Whatever that is
worth is plausibly already priced by `w_block`, which pays for exactly the same skip — so the
faintly negative 4-seat point estimate may be double-counting rather than noise. *Statement about
Taki: STOP is not "a bit of a free turn" that gets better as the table shrinks; at two seats it is
a different card.*

This is the fifth H-series prediction to be refuted (H2, H5, H6, H7, H11) against two that paid
(H1, H9) — and both that paid were structural fixes to something priced at zero, not extensions
of a working term.

### Caveats

- **`h10` must never become `REFERENCE`.** Its weights were selected at n=2 and measure
  **-0.0015 +/- 0.0046 vs h8 at four seats** — nothing, where h8's numbers were published. (At
  three: +0.0120 +/- 0.0063, mildly positive, unexplained and not chased.) `REFERENCE` stays `h8`.
- **`h9` is a candidate to replace `h8` as REFERENCE and was NOT promoted here.** It is unusually
  cheap to promote — identical to h8 at 3 and 4 seats, so every published 3- and 4-seat number
  would stay valid verbatim, and only the 2-seat column would need re-running. Deliberately left
  as a separate decision, per the rule that changing REFERENCE is never a side effect.
- The 2-seat orbit is a 1v1, so "friendly fire" and composition effects do not arise; but it is
  also a single pair, so there is no cross-pair variance check the way there is at three seats.
- `h9`'s STOP now earns `w_block` *and* `w_plus_tempo` when a threat is present, for what is
  physically one effect (the opponent losing a turn IS us gaining one). Deliberate at the H9 stage
  (structure only, no weights). H10's `w_save_blocker = 2.0` is plausibly the tuner paying for it.

---

## 2026-07-19 (later) — The head-to-head standard generalizes to every seat count as ONE rule (a rotation orbit), and the 1v3 arm joins the promotion standard. The promotion survives both; the 3-seat number was ~1 pt optimistic and a seat-dependence reading of the old 1v3 was refuted on replication

**Setup.** `eval_headtohead.py` was hardcoded to four seats, so every 2- and 3-seat claim this
project has ever made rests on `eval.py` Mode A: 1-vs-N, seeded seat *shuffle*, and the mirrored
field composition (`ref, new, new`) **never run**. That is a weaker grade of evidence than the
4-seat paired swap, and the promotion quoted both side by side. Generalized the harness and
re-measured the champion at all three counts, two deck blocks (seeds 0 / 777777), 3000 games/run.

### The design: four seats was never a special case

The rule is a **rotation orbit** (`orbit_pairs()`): rotate the base seat-set around the table, pair
each rotation with its complement, dedup unordered pairs.

| count | base | orbit | runs |
|---|---|---|---|
| 2 | `0` | `{0}\|{1}` | 2 |
| 3 | `0` | `{0}\|{1,2}`, `{1}\|{0,2}`, `{2}\|{0,1}` | 6 |
| 4 | `0,2` | `{0,2}\|{1,3}` | 2 |

Two and four seats collapse to one pair *for the same reason* — an alternating base is rotationally
symmetric with period 2, so the orbit shrinks from *n* to 2 and the complement family folds into it.
Odd counts admit no alternating base, so nothing collapses: the full 2*n* runs. Seat balance and
team-size balance are properties of the construction at every count; at odd *n* the complement
family is what restores parity, which is what makes a lopsided 1-vs-2 a valid *paired* comparison
(the two parities sum to 1, so `won_1 + won_2 - 1` has parity 0). Pinned in `h2htest.py` (9 tests),
**including that the 4-seat orbit is bit-identical to the old two-run swap.**

Two statistical details that matter at odd counts: the 3 pairs share decks, so the margin is
averaged **within a game** (one observation per deck) rather than pooled as 9000 flat values. And
the variance reduction is real: the 3-seat cells come in at SE 0.0066-0.0070 against 0.011-0.012
for the 2- and 4-seat cells on the same 3000 decks.

> **Measured, because I first asserted this wrongly.** Pooling understates the SE by
> `sqrt(1 + 2*rho)` where rho is the mean correlation between pairs sharing a deck. I claimed
> "~sqrt(3)" — that is only the rho=1 worst case. Measured on the 3-seat vs-`h8` block-0 cell:
> **rho = +0.097**, so the understatement is **1.093, i.e. 9%, not 73%** (SE 0.0070 averaged vs
> 0.0064 pooled; identical means, and the observed ratio matches `sqrt(1+2*rho)` to three
> decimals). That changes no verdict here (12.9 SE becomes 14.1 SE). **Per-deck averaging is still
> the right default — not because the correction is large, but because it is correct for any rho,
> whereas pooling assumes independence the harness does not control.** Second wrong guess worth
> recording: I predicted rho would be *negative*, reasoning that a deck favouring seat 0 helps the
> champion in pair 1 but helps the opponent in pairs 2-3. It is mildly positive, so whole-deck
> properties (how many +2s circulate, how fast the deal resolves) move all pairs together and
> outweigh that seat-specific effect.

### Results (paired orbit margin, parity 0.0; seat-balanced per-seat rates)

| vs **R6** | blk 0 | blk 777777 | mean | M1s3 rate | R6 rate |
|---|---|---|---|---|---|
| 2 seats | +0.3173 ± 0.0111 | +0.2960 ± 0.0118 | **+0.307** | 0.659 / 0.648 | 0.341 / 0.352 |
| 3 seats | +0.0852 ± 0.0066 | +0.0742 ± 0.0068 | **+0.080** | 0.362 / 0.358 | 0.305 / 0.309 |
| 4 seats | +0.0417 ± 0.0112 | +0.0533 ± 0.0113 | **+0.048** | 0.260 / 0.263 | 0.240 / 0.237 |

| vs **`h8`** | blk 0 | blk 777777 | mean | M1s3 rate | `h8` rate |
|---|---|---|---|---|---|
| 2 seats | +0.1500 ± 0.0118 | +0.1560 ± 0.0119 | **+0.153** | 0.575 / 0.578 | 0.425 / 0.422 |
| 3 seats | +0.1034 ± 0.0070 | +0.1048 ± 0.0070 | **+0.104** | 0.368 / 0.368 | 0.299 / 0.298 |
| 4 seats | +0.1023 ± 0.0115 | +0.1103 ± 0.0117 | **+0.106** | 0.276 / 0.278 | 0.224 / 0.222 |

**The promotion holds.** Every cell is positive by many SE on both blocks, so *"M1s3 beats R6 at
every seat count"* now stands on the rigorous paired standard at **all three** counts, not just at
four. The 2-seat margin (+0.307) is enormous, as expected from R6 losing to a hand-written heuristic
there.

### Two corrections to the promotion entry above

1. **The published `+0.0417 ± 0.0112` was deck block 0 ALONE, not "averaged over both blocks."**
   The orbit reproduced it to the digit — which is the regression gate passing, and also proves its
   provenance, since ±0.0112 is a single-3000-deck SE. The genuine two-block mean is **+0.048**. The
   two-block averaging in that entry applied to the vs-`h8` table, not to the promotion standard.
2. **The 3-seat vs-`h8` figure of 0.376 was ~1 point optimistic: the composition-balanced value is
   0.368.** Mode A only ever ran `new, ref, ref` — M1s3 solo against two heuristics. Running the
   mirror too pulls it down, and in the direction you'd predict: being the lone strong player among
   weak ones is easier per-seat than being one of two strong players splitting the wins. Both blocks
   give 0.368 exactly, so this is a real bias in the old measurement, not noise.

At two seats the two harnesses agree essentially exactly (orbit margin +0.153 → rate 0.577 vs Mode
A's 0.579), which is the cross-check that the older 2-seat numbers were sound — there the orbit
margin reduces to `2 x rate - 1`, so the quantities are identical. **Do not attempt that mapping at
four seats:** the orbit is a 2v2 (0.276) while the table's 0.303 is 1-vs-3, and M1s3's two seats
split wins with each other. Different quantities, as the eval-discipline section already warns.

### The solo orbit joins the promotion standard — and finds the old 1v3 was measured at the champion's best seat

At the user's direction the **1v3 orbit is now part of the promotion standard, not a secondary
check** (`--team1-seats 0 --seat-swap`, which at four seats is 4 pairs = 8 runs: `A,B,B,B` and
`B,A,A,A` with the solo seat rotating through all four positions). It earned that on evidence: it
*moves the number*.

| 4 seats, two-block mean | balanced 2v2 (`0,2`) | solo 1v3 rotating (`0`) |
|---|---|---|
| vs R6 | +0.048 (per-seat 0.2615) | **+0.034** (0.2585) |
| vs `h8` | +0.106 (0.277) | **+0.087** (0.272) |

Both arms pass against both opponents on both blocks, so **the promotion holds under the stricter
standard**. Useful identity when comparing arms: the paired margin is exactly `n * (seat-balanced
rate) - 1`, so a margin gap must be divided by *n* to read as skill. The composition effect is
**0.3-0.5 points of per-seat rate**, not the 1.4-1.9 points the raw margins suggest.

**The published 1v3 secondary `+0.0500 +/- 0.0098` reproduces EXACTLY as pair 1** — a second
regression gate, confirming the old measurement is a strict *subset* of the new one rather than
irreproducible. The subset is misleading simply because it is **one position out of four, on a
quarter of the data, and it fluctuated high**.

> **A "seat-dependence" reading of this was proposed and then REFUTED — recorded because the
> refutation is the useful part.** On block 0 the per-pair margins are seat0 **+0.0500** vs
> +0.0247 / +0.0297 / +0.0290 — seat 0 nearly double, ~2 SE above the rest — which looked like
> M1s3 converting the opening seat about twice as effectively as R6. Replicated on block 777777 it
> **inverts**: seat0 **+0.0267** is now the *lowest*, against +0.0360 / +0.0383 / +0.0360. Seat 0
> minus the mean of the others is +0.0222 on one block and -0.0101 on the other, and the spread
> collapses from 0.0253 to 0.0117. **There is no seat effect; the old +0.0500 was noise in a
> quarter-sized sample.** This is the `delta(k)` lesson again: *never call a per-pair pattern from
> one deck block — the pooled margin is the stable quantity* (+0.0333 / +0.0343 across the two
> blocks, while its components swing by 2 SE).

`--seat-swap` prints per-pair margins and their spread so this is inspectable — but read the spread
as a **noise diagnostic**, not as evidence of positional skill, unless it replicates across blocks.

### The caveat that no design removes

At odd counts there is no equal partition and no alternating pattern, so the larger team's seats are
necessarily **adjacent** — STOP/+2/CHDIR land on a teammate. Friendly fire is structural at three
seats. The 3-seat numbers above are a real comparison but not a clean one, and should be quoted that
way.

---

## 2026-07-19 — PROMOTION: an all-seat mixed champion. It beats R6 at EVERY seat count, four seats included. And M1 — the net we were about to promote — was the worst of three seeds

Narrow mixed seeds 2 and 3 (`--num-players 2,3,4 --trials 300000 --reward shaped --loss-penalty 60`,
no `--wide`), run for two reasons: the capacity comparison had used 3 wide seeds against 1 narrow,
and — the real motive — we intended to promote a single all-seat model and M1 was **n=1**, with only
its `snap300000` surviving a worktree cleanup. Snapshot grid densified to every 50k, because M1's
best cells had been at 100k and assuming the last snapshot is the best would leave points on the
table.

### The finding that justified the run: M1 was the WORST of the three seeds

Final snapshots only (no selection — final-to-final), vs `h8`:

| narrow seed | 2 seats | 3 seats | 4 seats |
|---|---|---|---|
| seed 1 (M1) | 0.570 | 0.372 | 0.290 |
| seed 2 | 0.580 | **0.379** | 0.302 |
| seed 3 | **0.582** | 0.373 | **0.307** |

**Promoting M1 would have locked in the weakest of three seeds**, giving up ~1.7 points at four
seats. I had argued M1 was "almost certainly representative" by inferring narrow's seed variance
from the wide seeds' very tight 300k cluster (range 0.002). **That inference was wrong** — narrow's
spread at four seats is 0.017 (~2 SE). *Lesson: seed variance measured on one architecture does not
transfer to another; a promotion needs its own replication.*

### Deck-block heterogeneity, again

Everything above is eval seed 0. Re-measured on a fresh block (seed 777777), the ordering held but
absolute values moved by up to 1.7 points — and **R6's four-seat number fell from 0.297 to 0.268**.
All promotion numbers below are therefore averaged over both blocks (6000 games/cell).

### The promotion

**`checkpoint_M1s3_mixed_snap300000`** (narrow mixed, seed 3, 300k, from scratch):

| vs `h8` (2 blocks) | 2 seats (0.500) | 3 seats (0.333) | 4 seats (0.250) |
|---|---|---|---|
| R6 (previous champion) | 0.417 — **LOSES** | 0.340 | 0.283 |
| **M1s3** | **0.579** | **0.376** | **0.303** |

**Promotion standard (alternating 2v2, seat-swapped, paired, 4 seats): +0.0417 +/- 0.0112 per-seat
(3.7 SE)**; 1v3 secondary agrees at +0.0500 +/- 0.0098 (5.1 SE). The two field compositions
concurring is the clean case.

> **CORRECTED 2026-07-19 (later).** That +0.0417 is deck block 0 **alone** — despite the sentence
> above about averaging over both blocks, which applies only to the vs-`h8` table. The two-block
> mean is **+0.048**. Also, the 3-seat vs-`h8` value of **0.376 is ~1 point optimistic**: it is a
> Mode A number, and Mode A only ever runs the `new, ref, ref` composition. Balanced over both
> compositions it is **0.368**. Neither correction changes the promotion. See the entry above.

> **This is not a generalist trading peak strength for coverage — it is better everywhere.** It beats
> the outgoing champion on R6's own turf (four seats) *and* turns a model that **loses to a
> hand-written heuristic at two seats** into one that beats it by 8 points there.

### Choosing between seeds 2 and 3

Statistically indistinguishable, and the metrics disagreed — vs-`h8` aggregate favoured seed 2
(over-parity sum 0.183 vs 0.175, driven by three seats), while direct play favoured seed 3 (Mode A
head-to-head better at 3p and 4p; paired 2v2 swap -0.018 +/- 0.011 against seed 2). Seed 3 was taken
on the strength of the best-controlled test (the paired swap) plus its edge at the hardest count.
Runner-up preserved as `checkpoint_M1s2_mixed_snap300000`; the choice between them is a coin flip and
should not be cited as a real difference.

### Blocking, completing the record

Narrow seeds never block: `delta(k=1)` = -1.10 / -2.84 (seed 2, 200k/300k), -0.00 / -2.35 (seed 3),
-0.36 (M1). Across all eleven probed snapshots, **2 block — both wide, in different snapshots per
seed.** 2/6 wide vs 0/5 narrow is not significant, so this adds no architecture signal; it further
confirms `delta(k)` is a noisy, mostly-negative transient. The champion does **not** block.

### What did NOT replicate, and a correction to yesterday's entry

Narrow seed 2's curve rises to ~150k and then **plateaus with ~1 SE wobble** (2p 0.577-0.580 across
150k-300k), rather than peaking. The "wide peaks at 200k while narrow peaks at 100k, so width
extends the horizon" reading is therefore **withdrawn**: peak *location* varies by seed within an
architecture, so it was seed noise. Combined with the refuted blocking and the tight 300k null,
**width buys nothing at all** — the clean verdict.

---

## 2026-07-18 (later) — M2 seeds 2,3: the BLOCKING result does NOT replicate — it was a lucky snapshot. Width buys neither a robust win-rate gain nor blocking

Two more wide-mixed seeds (2, 3), else identical to M2 (`--wide --num-players 2,3,4 --trials
300000 --reward shaped --loss-penalty 60`), run to firm up M2's two n=1 findings. They resolve
both, and the headline one negatively.

### Blocking: REFUTED. The delta(k) sign is snapshot/seed noise, not an architecture property

I probed `delta(k=1)` at both the 200k and 300k snapshot of all three wide seeds (deterministic
given the net):

| net | snap200000 | snap300000 |
|---|---|---|
| M2 seed 1 | **-1.69** (no) | **+2.54** (BLOCKS) |
| seed 2 | **+1.02** (BLOCKS) | -1.64 (no) |
| seed 3 | -1.62 (no) | -2.18 (no) |

**Only 2 of 6 wide snapshots block, in a different snapshot for each seed, with the sign
swinging over a ~4-point range within a single run.** M2 seed 1 — the run the original entry
called a blocker — does **not** block at its own 200k snapshot (-1.69); it blocked only at 300k.
So "capacity buys blocking" was a lucky draw of one snapshot of one seed. **The blocking headline
of the M2 entry below is withdrawn** (that entry now carries a correction; its measurements
stand). This is exactly P2's uncalibrated-`delta(k)`-variance caveat, now quantified: `delta(k=1)`
wanders across roughly [-2.5, +2.5] over training with no stable relationship to width. The
`r6_accept` sweep reads a *transient*, not a learned skill, and a single-snapshot blocking claim
from any run should be treated as noise until replicated across seeds and snapshots.

### Win rate: a rock-stable NULL at 300k, and a small, real, sub-promotion peak at 200k

vs `h8`, 3000 games/cell:

| net | 2 seats @300k | 3 seats @300k | 4 seats @300k | 2-seat peak (@200k) |
|---|---|---|---|---|
| M1 (narrow) | 0.570 | 0.372 | 0.290 | 0.570 (@100k) |
| M2 seed 1 | 0.569 | 0.375 | 0.295 | 0.587 |
| seed 2 | 0.567 | 0.374 | 0.296 | 0.597 |
| seed 3 | 0.568 | 0.373 | 0.289 | 0.586 |

**At 300k the wide net is identical to the narrow one, to three seeds** (0.567-0.569 / 0.373-0.375
/ 0.289-0.296) — a very tight null. The one thing that *does* replicate is the **200k peak**: all
three wide seeds peak around 200k with the 300k snapshot lower, and the 2-seat peak (~0.587-0.597)
sits ~1.5 SE above narrow M1's best (0.570), at/above the P1 specialist ceiling (0.583). So width
buys a **small (~1-2 SE), real, but sub-promotion** win-rate gain that requires early stopping and
shows up only at two seats; it peaks *later* than narrow M1 (200k vs 100k), which is the "extends
the horizon" effect, on n=1 narrow so still tentative.

### Verdict on the capacity hypothesis

**Largely refuted on replication.** 2.72x width buys no robust win rate (identical to narrow at
300k across three seeds) and no blocking (noise). The one surviving effect is a delayed, slightly
higher 2-seat peak worth ~1-2 SE. This *extends* A10's four-seat single-count null (capacity inert)
into the mixed setting, rather than overturning it as the M2 entry tentatively suggested. Net: the
mixed net's ~1-SE-below-ceiling shortfall is not a capacity wall the wide net removes — consistent
with the per-count *information* ceiling. Checkpoints preserved as `checkpoint_M2seed{2,3}_widemixed_snap{200000,300000}`.

*Process caveat carried forward: n=1 narrow. To make "width vs narrow" fully symmetric one would
run narrow seeds 2,3 too — but since width's win-rate is a tight null and its blocking is noise,
the narrow side is no longer the interesting question.*

---

## 2026-07-18 — M2: 2.72x width buys ~no win rate at 300k, but it EXTENDS the training horizon and BUYS THE BLOCKING BEHAVIOUR M1 never learned. Capacity is not inert in the mixed setting the way A10 found it at four seats

> **CORRECTION (M2 seeds, 2026-07-18 later): the BLOCKING claim in this entry is WITHDRAWN — it
> did not replicate.** delta(k=1) = +2.54 was one snapshot (seed 1, 300k) of a quantity that swings
> across ~[-2.5, +2.5] over training with no stable tie to width: seed 1 does not block at its own
> 200k (-1.69), seed 2 blocks only at 200k, seed 3 never. See the entry above. The win-rate
> measurements here stand (and the 300k null replicated tightly across three seeds); the "capacity
> bought blocking" mechanism does not.

Branch `mixed-wider`. The hypothesis: M1 (narrow mixed-count net) sits ~1 SE below the
per-count best at every count — the signature of multi-task interference, a net splitting
capacity across three games. So widen and see. **M2 is identical to M1 in every way except the
network: `124->64` -> `256->128->64` (A10's exact widening, 30.5k -> 84k params, 2.72x), mixed
2/3/4, 300k, seed 1, from scratch.** Because only the hidden layers change, M1 still plays here
via the cross-architecture eval fallback, so the direct M2-vs-M1 head-to-head is available.
Pre-registered: if capacity is binding, M2 closes the gaps and reaches the per-count ceilings
together; if the mixed setting is information-bound like A10 found four-seat *single*-count to
be, M2 ties M1.

### The naive read at 300k: width bought nothing (vs `h8`, 3000 games/cell)

| net | 2 seats (0.500) | 3 seats (0.333) | 4 seats (0.250) |
|---|---|---|---|
| M1 (narrow, 300k) | 0.570 | 0.372 | 0.290 |
| **M2 (wide, 300k)** | 0.569 | 0.375 | 0.295 |
| ceilings (P1 / C / C) | 0.583 | 0.382 | 0.311 |

Cell-for-cell M2 ≈ M1 (deltas 0.001 / 0.003 / 0.005, all << SE 0.012), and both still sit ~1 SE
under the per-count ceilings. Head-to-head, M2 vs M1 1-vs-N reads 0.510 / 0.345 / 0.263 (parity
0.500 / 0.333 / 0.250) — a faint lean to M2, nothing past 2 SE. Taken alone, this is the
**information-bound** outcome, consistent with A10 (which found this same 2.72x width inert at
four seats). **But taking the 300k snapshot alone is a mistake here:**

### The progression tells a different story: width DELAYS and RAISES the peak (vs `h8`)

| snapshot | 2 seats | 3 seats | 4 seats |
|---|---|---|---|
| M1 snap100000 (its peak) | 0.570 | 0.381 | 0.306 |
| M2 snap100000 | 0.577 | 0.370 | 0.306 |
| **M2 snap200000 (its peak)** | **0.587** | **0.383** | **0.307** |
| M2 snap300000 | 0.569 | 0.375 | 0.295 |

M1 plateaus by 100k and then only drifts down. **M2 keeps improving to a peak at 200k — and at
that peak it reaches the per-count ceilings simultaneously** (0.587 / 0.383 / 0.307 vs P1 0.583,
C 0.382, C 0.311): the "one net = three specialists" outcome the capacity hypothesis predicted.
Then it degrades by 300k. So width did buy headroom — it extends the useful training horizon and
lifts the ceiling the net can touch — but the net now needs **early stopping**, and reading the
final snapshot hides the gain. *Caveat, stated up front: snap200000 as "the best" is a max over
9 cells (3 snaps x 3 counts) and so is selection-inflated; what makes it more than a lucky cell
is that the rise-to-200k-then-fall pattern is consistent across all three counts.* M2 vs the
2-seat specialist P1 head-to-head is 0.503 (dead even); vs C it is 0.314 at three seats (-2.2 SE,
C still edges it — the one gap width did not close) and 0.254 at four (tie).

### The clean result: capacity BOUGHT BLOCKING. Same recipe, only width differs

The `r6_accept` weapon-timing sweep — the project's pre-registered blocking test, deterministic
given the net:

| net | delta(k=1) | shape | verdict |
|---|---|---|---|
| R6 | -0.87 | wrong | FAIL |
| M1 (narrow mixed) | -0.36 | flat-ish | FAIL |
| C | -3.65 | wrong | FAIL |
| P1 | +2.24 | rises as k falls | PASS |
| **M2 (wide mixed)** | **+2.54** | **rises as k falls, zero-cross between k=3 and k=4** | **PASS** |

**M2 blocks a near-winner; M1 — the same training, the same seed, the same data, narrower by
2.72x — does not.** This is the defect R6 never fixed (the whole motivation for R17,
potential-based shaping), and here it fell out of **capacity alone** — no curriculum, no reward
change, no shaping. It is a clean single-lever result against M1, and it strengthens P2's point
that blocking is a *behaviour dissociated from win rate*: M2 gained the behaviour while its win
rate barely moved.

**This bears directly on R17's premise, and cuts the same way P2 did: do not spend a run on
potential-based shaping assuming it is the only route to blocking.** Two routes are now known to
produce it without shaping — two-seat training (P1) and capacity (M2).

### Caveats

- **n=1 per architecture.** M1 and M2 are single runs. The delta(k) probe's run-to-run variance
  has never been calibrated (P2's standing caveat), so "capacity *causes* blocking" is *suggested*
  by a clean single-lever contrast, not *proven* — this particular wide run may have found blocking
  where this particular narrow run did not. Multiple seeds would settle it and are the obvious
  follow-up.
- **The win-rate gain needs early stopping and is ~1 SE even at the peak.** M2 is not a promotion
  over M1 on strength; the honest strength verdict is a tie with a delayed, slightly higher peak.
- **Four seats is still the weakest count** and the one place C keeps a small edge (3-seat
  head-to-head -2.2 SE). Width did not fix it. Non-uniform sampling weighting the harder counts is
  still untested.
- **Blocking-helps-win-rate remains unproven.** M2 blocks and is a hair better at four seats
  head-to-head (0.263), but nothing clears 2 SE, so whether the blocking it learned actually pays
  is still open — exactly the question P2 left.

---

## 2026-07-17 — M1: mixed-count training matches the best sequential curriculum FROM SCRATCH, and in 100k trials not 300k. The sequential curriculum is fragile; keeping every count in the gradient is not

Branch `nplayers-onehot`. Context lives on branch `2p-selfplay` (the P1/P2 entries, not yet
merged here): training one net to play 2, 3 and 4 seats. P1/P2 established that (a) the
champion's edge is a **four-seat** phenomenon — R6 vs the frozen heuristic `h8` scores 0.413 /
0.342 / 0.297 at 2 / 3 / 4 seats (parity 0.500 / 0.333 / 0.250), i.e. it *loses* at two seats;
(b) a **sequential** curriculum `C = R6 -> 2p -> 3p` is the best all-round net on record at
**0.553 / 0.382 / 0.311**, reached in 300k trials spent 100k-per-count in sequence; and (c)
forgetting scales with count distance — Arm A (`P1 -> 4p`) destroyed the two-seat skill
(0.583 -> 0.422). P2 named two open items this entry closes: **mixed-count training** (the only
design that keeps every count in the gradient), which needed **an explicit count feature** in
the observation, and **does `C -> 4p` keep the two-seat skill?**

### Setup — the observation change the count feature required, and why single-count runs were untouched

The observation grew **147 -> 150**: a one-hot of the seat count (2 / 3 / 4-or-more, saturating)
appended at the tail. Appended, so `obs[:147]` is bit-identical to the old contract (21,107
observations checked across 2/3/4 seats), which is what lets `eval.py` play a 147-float
champion on the leading floats it was trained on — R6 vs `h8` reproduces its published **0.297**
to the digit under the 150-float build, so cross-contract comparison is exact, not approximate.

**The one-hot is INERT in a single-count run** — the input never varies, and a constant input
folds into the next layer's bias — so it earns its keep only when the count varies between
trials. `--num-players 2,3,4` samples a count uniformly per trial (one reused `Game`, reseated
via `reset(agents=...)`; a fresh Game per trial would restart the deck RNG and deal identical
cards every trial). Verified the feature did not perturb the lineage: 40 seeded trials at
`--num-players 4` give **bit-identical** weights across the commit that added mixed counts.

### M1 — 300k trials, mixed 2/3/4, FROM SCRATCH (no R6 inheritance, no curriculum)

The sampler dealt 100,249 / 99,994 / 99,757 (33.4 / 33.3 / 33.3%), so M1's budget is
double-matched to C: same 300k total, and ~100k per count — exactly C's per-count spend. The
only difference is **interleaved vs sequential**, which isolates whether the forgetting P2 saw
is a consequence of *sequencing* rather than of the counts.

| net | budget | 2 seats (0.500) | 3 seats (0.333) | 4 seats (0.250) |
|---|---|---|---|---|
| R6 (champion) | 100k @4 | 0.413 | 0.342 | 0.297 |
| C (best sequential) | 300k staged | **0.553** | **0.382** | **0.311** |
| **M1 (interleaved, scratch)** | 300k mixed | **0.570** | 0.372 | 0.290 |

**M1 ties C at every seat count** (per-cell vs-`h8`: +1.3 / -0.8 / -1.8 SE — none significant;
`eval.py`'s common random numbers make these conservative). Direct head-to-head, M1 vs C
1-vs-N over 3000 common decks:

| seats | M1 team win rate (parity 1/N) | read |
|---|---|---|
| 2 | 0.511 (0.500) | +1.2 SE — tie, M1 a hair ahead |
| 3 | 0.314 (0.333) | **-2.2 SE — C edges it** |
| 4 | 0.244 (0.250) | -0.8 SE — tie |

So the honest reading is a **tie with a faint trade**: M1 is marginally stronger at two seats,
C marginally stronger at three (the one cell that clears 2 SE) and four. Neither is a promotion
over the other. **What matters is how M1 got there:** C needed a hand-designed three-stage
curriculum that inherited R6's 100k; M1 matched it **from random init, in one run, with no
curriculum to design and no champion to start from.** And it holds **0.570 at two seats** — the
exact skill Arm A's sequential `-> 4p` stage destroyed (0.583 -> 0.422). Nothing is forgotten
when nothing leaves the gradient.

**M1 reaches full performance by 100k, not 300k** (progression vs `h8`):

| snapshot | 2 seats | 3 seats | 4 seats |
|---|---|---|---|
| 100,000 | 0.570 | 0.381 | **0.306** |
| 200,000 | 0.554 | 0.371 | 0.294 |
| 300,000 | 0.570 | 0.372 | 0.290 |

It is **flat from 100k on** — mixed training buys C's whole 300k-staged result in a third of the
trials. The one wrinkle is the four-seat column: it *peaks at 100k* (0.306) and drifts down to
0.290 by 300k. The hardest count is the one uniform sampling slightly under-serves late in
training; a non-uniform schedule weighting four seats is the obvious lever, untested.

### C -> 4p — the sequential curriculum is FRAGILE. P2's open item, answered: no

Does `2p -> 3p -> 4p` keep the two-seat skill through its final stage? **No.** `C -> 4p` (100k @4
on top of C, 400k total) vs `h8`:

| net | 2 seats | 3 seats | 4 seats |
|---|---|---|---|
| C | 0.553 | 0.382 | 0.311 |
| **C -> 4p** | **0.428** | 0.346 | 0.304 |

The final four-seat stage **destroys the two-seat skill** (0.553 -> 0.428, ~10 SE — back below
where P1 started), **erodes three seats** (0.382 -> 0.346, ~3 SE), and **buys nothing at four**
(0.311 -> 0.304, flat). This is Arm A's result again from the other end: four-seat training
specialises *away* from the smaller counts. **A sequential curriculum cannot hold all three
counts — each stage overwrites the last. Mixed training is the only design that keeps them,
because it never takes a count out of the gradient.** That is the entry's central result.

### Blocking is still dissociated from strength (delta(k) probe on M1)

M1's weapon-timing sweep: `delta(k=1) = -0.36`, **does not block**, spread 2.04. It joins R6
(-0.87, fail) and C (-3.65, fail) as strong nets that do not block, against P1 (+2.24, pass) which
does. So blocking is **not** what mixed-count training bought, exactly as P2 found: the skill is
doubly dissociated from win rate. (M1's `delta(k=1)` is the least-negative of the non-blockers,
and its spread matches R6's, but it still fails the sign test.) R17's premise remains unsettled
here — mixed training produced the strongest all-round net without blocking.

### Caveats

- **Mode A, not a paired promotion test.** `eval_headtohead.py` is four-seat-only, so the M1-vs-C
  head-to-heads above use `eval.py --opponent <ckpt>` (1-vs-N, seeded-shuffled seating), the only
  model-vs-model path at 2/3 seats. A 1v1/1v2 seat-swap path is still the missing harness piece.
- **The four-seat cell is M1's weakest**, and the one place C leans ahead (3-seat head-to-head
  -2.2 SE, 4-seat vs-`h8` -1.8 SE). Uniform sampling may under-serve the hardest count; a weighted
  schedule is untested.
- **`h8` is four-seat-tuned.** Competent at two seats (0.963 vs random, beats its own greedy
  control), so not a strawman, but a per-count-tuned yardstick would be a sterner test — and is now
  possible against these per-count champions.
- **A process note, not a result:** the eval chain was launched behind a `while tmux has-session`
  guard that watched a training session which outlived its script (a lingering pane), and wedged
  for 18h producing nothing. The nets were fine; the wait was watching a zombie. Fixed to key off
  the completion signal. Waits must watch the real work signal (a growing log / advancing counter),
  never a proxy that can outlive it.

---

## 2026-07-17 — P2: forgetting scales with COUNT DISTANCE (2->3 keeps it, 2->4 destroys it). And blocking is DOUBLY DISSOCIATED from strength — it is not what two-seat training bought

Three 100k warm-started runs, sequential, all from P1 or R6, identical hyperparameters
(`--reward shaped --loss-penalty 60 --epsilon-start 0.1 --seed 1`). **Arm A** `2p->4p` (the
curriculum), **Arm B** `R6->4p` continued (the budget control — without it Arm A's 200k total
trials confound "the curriculum did it" with "it trained twice as long"), and **C** `2p->3p`.

### The matrix — Mode A vs `h8`, 3000 games/cell, and the delta(k=1) probe

| net | total trials | 2 seats (0.500) | 3 seats (0.333) | 4 seats (0.250) | delta(k=1) | blocks? |
|---|---|---|---|---|---|---|
| R6 (champion) | 100k @4 | 0.413 | 0.342 | 0.297 | -0.87 | FAIL |
| **P1** = R6->2p | 200k | **0.583** | 0.364 | 0.284 | **+2.24** | PASS |
| **Arm A** = P1->4p | 300k | 0.422 | 0.356 | 0.308 | **+1.55** | PASS |
| **Arm B** = R6->4p | 200k | **0.365** | 0.338 | 0.302 | -3.71 | FAIL |
| **C** = P1->3p | 300k | **0.553** | **0.382** | **0.311** | -3.65 | FAIL |

### Forgetting is NOT uniform — it scales with the DISTANCE between counts

This is the headline, and it **reverses the conclusion drawn from Arm A alone**:

| starting from P1 (0.583 @ 2 seats), train 100k at... | 2-seat score after |
|---|---|
| 4 seats (Arm A) | **0.422** — 0.7 SE from R6. Gone. |
| 3 seats (C) | **0.553** — 11.0 SE above R6, only 2.3 SE below P1. **Kept.** |

C vs Arm A at 2 seats is **10.2 SE apart**. Same start, same budget, same everything except
whether the next stage had three seats or four. **Training at an ADJACENT count preserves the
skill; skipping a count destroys it.** And C did not merely retain — it *gained* 3 seats
(0.382 vs R6's 0.342, **3.2 SE**), the best 3-seat number on record, while holding 4 seats at
0.311 (the best 4-seat number on record, though only 1.2 SE over R6).

> **`C` (2p->3p) is the best all-round net the project has**: strong at 2 seats, best at 3, and
> at least as good as anything at 4. **This partially vindicates the `2p->3p->4p` proposal that
> the Arm A result appeared to kill** — the intermediate 3-seat stage is exactly the thing that
> makes stages accumulate. Whether a final `->4p` stage would then destroy the 2-seat skill
> anyway is **untested and is the obvious next run**; Arm A says 4-seat training is what does
> the damage, so expect it to.

**Four-seat training actively DEGRADES two-seat play**: Arm B (R6 + 100k more @4) drops to 0.365
from R6's 0.413, **3.8 SE**. Not drift — specialisation. It sharpens habits that are right at
four seats and wrong at two, which is the same claim P1 makes from the other direction.

### The four-seat plateau, confirmed at double the budget

Arm B is R6 with **100k more trials at four seats** and gains **+0.005 (0.4 SE)**: 0.297 ->
0.302. Doubling the budget at four seats buys nothing measurable. Converging with A10/A11/A12 —
but note P1 found **+17 points sitting at two seats** with the same net and the same reward.
**The ceiling is per-count, not global.**

### Blocking is DOUBLY DISSOCIATED from strength

Two arms, opposite dissociations, which together break the story P1 told:

- **Arm A** (2p->4p): **lost** the 2-seat strength (0.422), **kept** the blocking (+1.55).
- **C** (2p->3p): **kept** the 2-seat strength (0.553), **lost** the blocking (-3.65).

> **Therefore blocking is NOT what two-seat training bought.** C plays 2-seat Taki at 0.553 —
> +0.140 over R6, 11 SE — while failing the blocking test outright. At most ~0.030 of P1's
> +0.170 (the P1-vs-C gap, 2.3 SE) could be blocking; **the large majority of the two-seat gain
> is some other, still-unidentified two-seat competence.** The 2026-07-16 entry's framing —
> "two seats taught the missing endgame skill" — is wrong on the mechanism, though its
> *measurements* stand.

**Blocking still does not predict four-seat strength**, now from three nets: 0.308 (blocks),
0.302 (does not), 0.311 (does not). The one that blocks is in the *middle*. Whatever R17 buys by
fixing this behaviour, this is more evidence it is not four-seat win rate.

### CORRECTIONS

1. **"The 2-seat stage caused the blocking, controlled" — WITHDRAWN.** It rested on Arm A
   (blocks) vs Arm B (does not) at matched budget. But **C has the identical 2-seat stage and
   does not block**. Blocking survived 100k trials at *four* seats and died after 100k at
   *three* — which no "more seats -> less blocking" story explains. With **n=1 per arm** this
   cannot be separated from run-to-run variance in where a run lands. The honest statement is
   narrower: *P1 blocks; R6, Arm B and C do not; what makes it stick is unknown.*
2. **"2-seat training improved 3 seats (+0.022, ~2.6 SE)" — WITHDRAWN** (already corrected in
   the P1 entry). It used the single-sample SE instead of the SE of a difference: 1.8 SE, not
   significant. C's 3-seat gain (+0.040, 3.2 SE) *is* significant — but C trained at 3 seats.

### Caveats

- **n=1 per arm.** Every cell above is one run. The project's own calibration entry
  (2026-07-11) puts run-to-run sigma at ~0.5-0.7 pts on win rates; the delta(k) probe is a
  **single position** and its run-to-run variance has never been calibrated at all. The 10-SE
  win-rate contrasts are safe; **the blocking flips are not** — see correction 1.
- **Mode A, not promotion.** `eval_headtohead.py` hardcodes four seats (`[agent2] * 4`,
  `range(4)`); there is still no 1v1 or 1v2 seat-swap path. **C's 0.311 at four seats cannot be
  promoted over R6 on this evidence** (1.2 SE, and Mode A is not the promotion standard).
- **Refusal rates keep creeping**: 2.92% (R6) -> 4.39% (P1) -> 4.61% (Arm A) -> 5.21% (C). Each
  passes the guard-rail; the trend does not reverse. Worth a `holdback.py` census.
- `h8` is four-seat-tuned (competent at two — see the P1 entry — but not a two-seat optimum).
- All runs thread-uncapped and mutually consistent; see the P1 entry's caveat on `train.py`.

### Open

- **Does `C -> 4p` keep the two-seat skill?** The direct test of `2p->3p->4p`. Arm A predicts no.
- **What IS the two-seat competence**, if not blocking? P1 vs C at two seats (0.583 vs 0.553)
  is the only handle, and it is 2.3 SE.
- **Mixed-count training** remains the only design that keeps every count in the gradient. Needs
  the `Game` rebuilt per trial (constructed once, outside the loop, train.py:196-199), and
  probably an explicit count feature (147->150, invalidating every checkpoint).
- **A 1v1 / 1v2 seat-swap path in `eval_headtohead.py`** — without it nothing here is promotable.

---

## 2026-07-16 — P1: the champion's edge is a FOUR-SEAT phenomenon. At two seats R6 loses to the heuristic — and two-seat self-play fixed the blocking defect R6 could not

Branch `2p-selfplay`. The question was whether one model can play 2/3/4 players, and whether the
observation blocks it. **It does not block it — the observation was already count-agnostic and
nothing had to change.** `OPP_HAND_SLOTS` (game.py:166-169) reserves 3 opponent-hand slots and
zero-pads below 4 seats, so the **147-float contract is unchanged**: every checkpoint stays
loadable at every count, which is what made a warm-start from R6 possible. The count is encoded
only *implicitly*, by which slots are zero (a live opponent never holds 0 cards — only a winner
does, and then the game is over). `num_of_players` was a single local every downstream use
derived from, so `--num-players` is a one-line change (`c353565`).

### The survey: R6's edge exists only at four seats

Mode A, 3000 games/cell, `eval.py --num-players N`. The 4-seat column reproduces the published
numbers exactly, which is the check that the harness is faithful across counts:

| seats | R6 vs `h8` (parity) | `h8` vs `greedy` (parity) | `h8` vs random |
|---|---|---|---|
| 2 | **0.413** (0.500) — **9.7 SE BELOW parity** | 0.565 (0.500) | 0.963 |
| 3 | 0.342 (0.333) — 1.0 SE, a tie | 0.392 (0.333) | 0.925 |
| 4 | 0.297 (0.250) — the published number | 0.302 (0.250) | 0.906 |

> **The champion loses to a few hundred lines of hand-written rules at two seats, and merely
> ties them at three.** Every "R6 beats the heuristic" claim in this log is a *4-seat* claim.

**The yardstick is not flailing off its turf** — the obvious objection, and it fails. `h8` is at
its *strongest* vs random at 2 seats (0.963), and its hold-backs still beat its own `greedy`
control there (0.565 vs 0.500). R6 is losing to a competent opponent, not to a broken one. Its
structure is genuinely count-generic (every seat derives from `len(game.agents)`); only its
*tuning point* is 4-seat. Checked the paths that could break at n=2 — all correct, one by
accident: `w_chdir_block` is dead code at 2 seats (`behind` and `nxt` are the same seat, so
line 596 compares a hand to itself), which is the right behaviour since CHDIR is a no-op at
2 seats (RULES.md), but not for the intended reason.

**vs-random noticed none of this** (0.91-0.96 across every cell, every agent). The clearest
demonstration yet of why it was retired as a ranking metric — it is blind to a sign flip.

### P1: R6 -> 2 seats, 100k trials. The R6 acceptance test now PASSES

    train.py --num-players 2 --trials 100000 --seed 1 --reward shaped --loss-penalty 60 \
             --epsilon-start 0.1 --snapshot-every 2500 --model checkpoint_r6L60_snap100000

Warm-started from R6, near-greedy continue. **`snap0000` scores 0.413 — R6's cold score to
three decimals.** That is the control that makes the rest of the entry mean something: the
warm-start loaded the champion faithfully, snapshot 0 is pre-training, so the whole delta is
attributable to 2-seat *training* and not to distribution shift or a reload artifact.

| snapshot | vs `h8` @ 2 seats (parity 0.500) |
|---|---|
| 0 (= R6) | 0.413 |
| 12,500 | 0.546 |
| 25,000 | 0.557 |
| 50,000 | 0.548 |
| 75,000 | 0.563 |
| 100,000 | **0.583** (+0.170 vs R6, **13.4 SE**) |

Most of the gain lands inside the first 12,500 trials and it is **still rising at 100k**. A fast
concentrated gain is what one missing skill looks like, not a policy rebuild.

> **CORRECTION (P2, 2026-07-17): the "one missing skill" is NOT the blocking.** `C` (`2p->3p`)
> scores 0.553 at two seats — 11 SE above R6 — while *failing* the blocking test outright. The
> measurements in this entry stand; the mechanism it implies does not. At most ~0.030 of the
> +0.170 is attributable to blocking, and what the rest is remains unidentified.

**`r6_accept.py` — the test R6 itself FAILED — now passes on both pre-registered predictions:**

| k (next player's cards) | R6 delta(k) | P1 delta(k) | P1 picks |
|---|---|---|---|
| 1 | **-0.87** (wrong sign) | **+2.24** | play red +2 — **blocks** |
| 2 | -2.33 | +0.52 | play red +2 — **blocks** |
| 3 | -2.81 | -1.43 | play red 5 — sheds |
| 7 | -2.77 | -3.64 | play red 5 — sheds |
| *spread over k* | 1.95 | **5.88** | |

It blocks at k<=2 and sheds at k>=3; the zero-crossing sits between k=2 and k=3 (B2 predicted
between k=1 and k=2). The spread tripled, so it is not a shifted constant — it is *using* the
opponent-hand-size feature, not merely valuing the position differently.

> **The blocking defect is LEARNABLE inside the existing architecture, observation and reward.**
> No potential-based shaping was required. What 100k trials at four seats could not teach, two
> seats taught in 12,500. That makes it a **credit-assignment** problem — not a value-function
> problem, not an information-set limit.

### What it does NOT show (the honest half)

**Transfer is flat, not positive.** Corrected statistics — the first read of this used the
single-sample SE instead of the SE of a *difference*, which overstated the 3-seat cell:

| seats | R6 | P1 (2p-trained) | delta | SE of diff (unpaired) | verdict |
|---|---|---|---|---|---|
| 2 | 0.413 | **0.583** | +0.170 | 0.0127 | **13.4 SE — real** |
| 3 | 0.342 | 0.364 | +0.022 | 0.0123 | 1.8 SE — **NOT significant** |
| 4 | 0.297 | 0.284 | -0.013 | 0.0117 | 1.1 SE — **NOT significant** |

So: 2-seat training bought a large 2-seat gain and left 3 and 4 **statistically unchanged**.
Nothing was traded away — but nothing transferred either. (`eval.py` uses common random
numbers, so these unpaired SEs are conservative; pairing would tighten them.)

**The tempting overclaim to avoid:** the `r6_accept` probe is a *4-player* position
(`scenarios_b2`: `[agent] + [_Dummy()]*3`, `opp_sizes=(k,5,5)`). So P1 blocks correctly at a
4-seat table while its 4-seat win rate does not move — which *looks* like proof that blocking
was never the 4-seat bottleneck, and would undercut R17's premise directly. **It is not proof.**
P1 at 4 seats is itself off-distribution; it may have gained blocking while shedding 4-seat
competence, and the two could be cancelling. This run cannot separate them.

### Caveats

- **Mode A, not a promotion test.** `eval_headtohead.py` hardcodes 4 seats throughout
  (`[agent2] * 4`, `range(4)`); there is no 1v1 seat-swap path yet. 0.583 is a 1-vs-1 rate, not
  a paired promotion margin.
- **The refusal rate rose 2.92% -> 4.39%** (+50% relative). It passed the `r6_accept` guard-rail
  but is drifting the wrong way — watch it.
- **`h8` is 4-seat-tuned.** Competent at 2 seats (above), so this is not a strawman, but a
  2-seat-tuned version would be a sterner test. That was impossible before (`tune_heuristic.py`
  tunes *against the champion*, and there was no 2-seat champion); P1 removes the circularity.
- **`train.py` is thread-uncapped and inconsistent with `eval.py`.** `eval.py` sets
  `OMP_NUM_THREADS=2` before the TF import (eval.py:29-31); `train.py` sets nothing, inheriting
  only `agents/dqn.py`'s `tf.config.threading` (intra=4/inter=2). That caps TF's op pools but not
  the Eigen/OpenMP kernels: the run took **35 threads / 1112% CPU** on a network CLAUDE.md says
  "thrashes on default pools". Not fixed mid-series — every run here is uncapped, so they are at
  least consistent with each other. Fix separately; it may make runs *faster*.
- No `--num-players` sampling: this is a single-count run. Mixed-count training would need the
  `Game` rebuilt per trial (it is constructed once, outside the loop, train.py:196-199).

### In flight

Three 100k runs queued sequentially (Arm A `2p->4p`, Arm B `R6->4p` control, then `2p->3p`).
**All three landed — see the 2026-07-17 (P2) entry above, which answers the forgetting question
and CORRECTS two claims made here and in-session.**

---

## 2026-07-14 (last) — H8 + B5: the tuner went above the old cap and came back empty. And the champion's headline was inflated ~8 points by the crippled yardstick

### H8 — coordinate descent, and a clean winner's-curse lesson

2 passes x 44 candidates x 12,000 games each, from `h1b2`, against the R6 champion (`probes/
h8_tune.log`, 7,886s). **In-search: -0.0638 -> -0.0472.** Held out on decks the tuner never
saw, that gain **did not replicate**:

| | vs R6 champion |
|---|---|
| in-search (biased: a max over ~88 noisy candidates) | **-0.0472** |
| fresh deck block [1,000,000) | -0.0757 |
| fresh deck block [2,000,000) | -0.0727 |

**But the honest comparison needs the base on the SAME decks**, and that flips the reading —
`h1b2`'s champion margin swings a lot by deck block, so my first cross-block read was an
artifact:

| deck block | `h1b2` vs R6 | `h8` vs R6 |
|---|---|---|
| [1,000,000) | -0.1067 | **-0.0757** (h8 +0.031) |
| [2,000,000) | -0.0720 | -0.0727 (tie) |

**Deck-block heterogeneity is large** (`h1b2` vs R6 reads -0.065 / -0.107 / -0.072 on three
blocks, SE 0.012 each). A single-block champion comparison is weak evidence; the high-N paired
numbers are the ones to trust:

- **`h8` beats `h1b2` +0.0118 +/- 0.0035** (20,000 common decks, fresh) — real, 3.4 SE.
- **vs random: 0.906** (best of any version; `r3` 0.850, `h1b2` 0.897).
- Census: **0 refusals of any kind.** H1's invariant survives tuning, which is the point of it.

So H8 is a genuine but modest improvement — **not** the +0.017 the search advertised. *Lesson,
pre-registered and then observed: coordinate descent's own margin is biased upward; the "keep"
rule latched candidates worth +0.0001. Always re-measure on held-out decks, against the base,
on the same decks.*

### The H8 result that actually matters: the freed space was empty

H8 waited for H1 so the search could finally go **above the old hold cap of 5.0** (below which
the tuner had been confined, because above it a hold bought a refusal). Given that freedom, the
tuner went up there and **came back with nothing**:

    p_king 5.0    w_reserve 4.0    w_nofin 8.0    w_open_hoard 15.0     <- ALL UNCHANGED

**Not one hold-back weight moved.** Every gain came from ordinary scoring terms:
`w_deny 3.0->1.5`, `w_block 4.0->6.0`, `w_chdir_block 2.0->0.0`, `w_plus_tempo 1.0->2.0`,
`w_save_blocker 0.7->0.0`, `p_super_taki 4.5->4.0`.

The step-4 sweeps had already explained why, and the tuner's own log shows it directly: above
their optimum the holds **saturate** — `p_king` at 6/9/14 and `w_open_hoard` at 5/25/40 return
margins identical *to four decimal places*, because past a threshold the hold is already
decisive and more of it cannot move the `argmax`. **The cap was never costing anything. The
cliff was.** H1 was still necessary — it is what makes the search *safe* — but the points it
unlocked were not hiding above 5.0.

Two independent confirmations fell out: `w_chdir_block -> 0` (H3's ablation called it dead
weight; the tuner deleted it) and `w_save_blocker -> 0` (H3 found spending the blocker is the
agent's most valuable behaviour, so the term that *saved* it was a tax).

### B5 — PROMOTED: the yardstick is now `h8`, and the champions get smaller

`REFERENCE` moves `r3` -> `h8`. Mode A, 1-vs-3, 3000 games, parity 0.250
(`probes/b5_yardstick.log`). The r3 column reproduces the published numbers exactly, which is
the check that the harness is faithful:

| | vs `r3` (old, crippled) | vs `h1b2` | vs **`h8`** (new) |
|---|---|---|---|
| **R6 champion** | **0.378** | 0.307 | **0.297** |
| A9-rules | 0.348 | 0.263 | 0.271 |
| *parity* | 0.250 | 0.250 | 0.250 |

> **The champion's headline was inflated by ~8 points by a yardstick with a 13-point bug in it.**
> R6 beats a *competent* heuristic by **+4.7 points over chance, not +12.8**. It is still
> genuinely ahead — the DQN really does beat the best hand-written rules we have — but the
> margin is less than half what the record claimed.

**Every "vs heuristic" number published before 2026-07-14 refers to `r3`** and is not comparable
to one measured after. `heuristic:r3` still runs and still means exactly what it always meant —
that is what the freeze is for. Cite the old numbers by that name.

---

## 2026-07-14 (later still) — H7: RACING LOSES. When someone is about to win, that is when your kept cards are worth the most

H7 asked for the behaviour the agent visibly lacks: it knows how to *block* a near-winner but has no
answer when it **cannot** block one, and its scoring does not change at all in that situation. A
human speeds up. So: `race` (scale the hold-backs by `race_hold_scale`, cash the hoarded TAKI run).

**It loses, and monotonically in both knobs** (vs `h1b2`, 2v2 seat-swap, 20,000 games x 2;
`probes/h7_race.log`):

| `race_hold_scale` (how hard it drops its holds) | margin | | `race_hand_threshold` (how often it races) | margin |
|---|---|---|---|---|
| 0.75 (barely) | -0.0029 +/- 0.0015 | | 1 (only vs a 1-card winner) | -0.0148 +/- 0.0019 |
| 0.5 | -0.0144 +/- 0.0021 | | 2 | -0.0348 +/- 0.0028 |
| 0.25 | -0.0234 +/- 0.0024 | | 3 | -0.0539 +/- 0.0033 |
| **0.0** (ignore holds) | **-0.0348 +/- 0.0028** | | **4** | **-0.0664 +/- 0.0037** |

Both gradients point the same way and neither has an interior optimum: **the best amount of racing
is none.** Note `race_hold_scale = 1.0` *is* the no-race agent, so the first column is a clean dose-
response curve down from zero.

**This is H5/H6 seen from the other side, and together they make a real claim about Taki.** The
hold-backs are worth **+1.1** (`w_nofin`), **+2.9** (the hoard) and **+2.2** (spending the blocker) —
and they are worth most *in the endgame*, which is exactly the moment racing throws them away.

> **Statement about Taki: when an opponent is about to win, do not empty your hand faster. That is
> when your kept cards are worth the most** — the hoarded TAKI run, the finisher, the blocker. Tempo
> is not the scarce resource at the end; *optionality* is.

**Methodology note — a bug I nearly published.** The first cut forced the hoard open while racing
*without* checking the run could legally be **ended** (`_hoard_plan` gates that on the group holding
a number when the run would empty the hand), so a racing agent could open runs it could not finish.
Re-measuring after the fix moved the headline from **-0.0348 to -0.0349**. The result is the game,
not the bug — but the check was worth doing before publishing a negative result, and the habit is
worth keeping.

The knob ships **off**. It is kept rather than deleted because "why is racing bad?" is a live
question for the behavioural-analysis half of the project — unlike `score_decline_king` (H4), it is
explicitly off and documented as refuted, so it is not a phantom behaviour.

---

## 2026-07-14 (later) — Step 4: EVERY pre-registered H-series prediction failed, and two REVERSED. B2's hold-back ablations were measuring the refusal cliff, not the hold-back

All measured against the `h1b2` base **on H1's structure** (2v2 alternating, seat-swapped, paired,
20,000 games x 2). Log: `probes/h_ablations.log`. This is what H1 was *for*: before it, none of
these numbers meant what they appeared to mean.

| item | PLAN predicted | measured | verdict |
|---|---|---|---|
| **H2** `block_hand_threshold` 2 -> 1 | **+0.02 .. +0.045** | **-0.0112 +/- 0.0024** | **REFUTED — worse** |
| H2 control: threshold 3 (wider) | — | +0.0010 +/- 0.0027 | tie |
| **H3** `w_block=0` (the SPEND side) | untested hole in B2 | **-0.0223 +/- 0.0026** | blocking **earns +2.2** |
| H3 `w_chdir_block=0` | — | -0.0010 +/- 0.0015 | tie — **dead weight** |
| **H4** decline-King | dead code | -100: tie; **0: -0.0004** | dead as shipped; **safe to delete** |
| **H5** drop `w_nofin` (8.0) | "nearly vacuous, drop it" | **-0.0107 +/- 0.0008** | **REFUTED — earns +1.1** |
| **H6** delete the hoard (B6) | "buys nothing, delete it" | **-0.0289 +/- 0.0030** | **REFUTED — earns +2.9** |

### The finding behind the reversals

H5 and H6 did not fail by chance. **Every hold-back conclusion B2 drew was confounded by the
refusal cliff**, and H1 is what makes that visible. The same `w_reserve` sweep, the same weights,
with *only the decision structure* changed (`probes/h6_cliff_confound.log`; 10,000 games x 2; the
census column is voluntary draws per 200 games):

| `w_reserve` | LEGACY margin | draws | STRUCTURAL margin | draws |
|---|---|---|---|---|
| 0 | -0.0218 | 4 | -0.0212 | 0 |
| 2 | -0.0080 | 4 | -0.0081 | 0 |
| 4 (base) | 0 | 5 | 0 | 0 |
| 6 | -0.0033 | 4 | -0.0028 | 0 |
| **8** | **-0.0282** | **52** | **-0.0030** (tie) | **0** |
| **10** | **-0.0575** | **118** | **-0.0024** (tie) | **0** |
| **20** | **-0.0752** | **130** | **-0.0032** (tie) | **0** |

Below the cliff the two structures agree to three decimals — as they must, since nothing refuses
there. Above it, LEGACY's collapse tracks the refusal count *exactly*, while STRUCTURAL simply goes
flat: the hold saturates and costs nothing. So B2's "**catastrophic at 8/10 (where it starts
drawing)**" was never a fact about hoarding. It was the cliff, and B2 even named the mechanism in
its own parenthesis without drawing the conclusion.

**What this invalidates:** B2 concluded "the hoard buys nothing — the entire apparatus is complexity
with no measurable payoff and a large downside". Both halves are wrong. The payoff is **+2.2 to
+2.9 points** (0 vs 4, and the full `-hoard` ablation), and the "large downside" was the cliff,
which no longer exists. **B6 stays.** More generally: *any* hold-back weight B2 priced near or above
5.0 was priced against a confound, and its ablation numbers should not be trusted.

**What survives:** the hold-back *preference* story is intact and stronger — holds are worth real
points (`w_nofin` +1.1, hoard +2.9, blocker-spend +2.2). What died is the claim that big holds are
dangerous. Under H1 they are merely *saturating*.

### H2 is refuted, and that is a genuine surprise

B2's rollout priced blocking a 1-card next player at **+0.020 / +0.038 / +0.045** across three deck
blocks and a 2-card one at **-0.05**, so narrowing the threshold to 1 looked like free money. With
the **heuristic itself as the actor**, threshold 1 measures **-0.0112 +/- 0.0024** — a loss, at ~5
SE. The caveat B2 wrote down ("those numbers were measured with the DQN as the acting seat ... they
should transfer, but confirm with the heuristic as the actor") turns out to have been the whole
story: **they did not transfer.** Widening to 3 is free (+0.0010, tie), so the shipped threshold of
2 is at or near the optimum and the "fires too late AND too widely" framing is simply wrong.

### H4: dead, but do not set it to zero

`score_decline_king` fires **0 times in 200 games** in every version, and setting it to `-100`
changes nothing (+0.0001 +/- 0.0001) — so *deleting* it is free, as PLAN says. But setting it to
**0** measures **-0.0004 +/- 0.0002**, a small real loss: at 0 the option starts *winning* the
`max()` against negative-scoring plays, i.e. the phantom behaviour wakes up and it is bad. The lesson
is about how to retire dead code — remove the branch, don't neutralize the weight.

---

## 2026-07-14 — H1: the heuristic's 18-point defect was its STRUCTURE, not its weights. Fixed by construction, with zero weights changed

H-series steps 1+2 (freeze the yardstick) and H1 (structural refusal). Worktree `taki-ai-hseries`.

### The headline

**H1 carries R3's exact weight vector — `p_king = 6.0` and all — and beats it by +0.183.**

| head-to-head (2v2 alternating, seat-swapped, paired; parity 0.0) | paired margin |
|---|---|
| **`h1` vs `r3`** (structure alone; **identical weights**) | **+0.1826 +/- 0.0042** |
| `h1` vs `b2` (structure alone vs retuning alone) | -0.0066 +/- 0.0034 |
| `h1b2` vs `b2` (does structure add anything, once retuned?) | -0.0003 +/- 0.0011 (tie) |
| `h1b2` vs `h1` (does retuning add anything, once structural?) | +0.0075 +/- 0.0033 |

20,000 games x 2 runs each. Read together they say one thing: **the ~18 points were the refusal
cliff, entire.** You can remove it by retuning the weights under it (B2) or by removing the cliff
(H1) — and *they do not stack*. `b2 ~= h1b2 > h1 >> r3`.

**Prefer H1 anyway, and not for the win rate.** B2's route leaves the cliff in place and merely
parks the weights below it, which is why B2 *itself* still refused (below). H1's route means no
weight assignment can fall off it — that is what makes H8's automated tuning trustworthy, and it is
a property, not a number.

### Against the R6 champion (2v2 seat-swap, 3000 games x 2)

| heuristic version | vs R6 champion | vs 3 random |
|---|---|---|
| `r3` (the shipped yardstick) | **-0.2630 +/- 0.0117** | 0.850 |
| `h1` (structure only) | -0.0793 +/- 0.0118 | 0.896 |
| `b2` (retuned only) | -0.0647 +/- 0.0116 | 0.899 |
| **`h1b2`** (both) | **-0.0647 +/- 0.0117** | 0.897 |

The heuristic closes from **-26.3 to -6.5** points behind the champion. It does **not** reach parity
with R6 — and that is consistent, not a contradiction of B2: B2's "parity with the champion" was
measured against **A9-rules**, and R6 beats A9-rules by **+8.5**. Two independent measurements
agreeing to within a point is a genuine cross-check on both.

### The invariant, verified rather than asserted

PLAN pre-registered the test *no weight assignment can produce a voluntary draw*. As a property test
over 12 random weight vectors with holds up to **500** (far past the old cliff at 5.0): `structural`
**never** draws while a legal play exists; `legacy` on those same vectors refuses constantly (also
asserted, so the test cannot pass by being vacuous). `holdback.py` census, 200 games x 4 seats:

| | draw refusals | close-TAKI refusals | decline-King |
|---|---|---|---|
| `r3` | **1293** | 145 | 0 |
| `b2` | 15 | 0 | 0 |
| `h1` | **0** | 84 | 0 |
| `h1b2` | **0** | 0 | 0 |

### Three findings that were not the point of the exercise

1. **B2's retuned agent still refused.** `w_nofin` was never retuned — still 8.0, above the 5.0
   threshold — so the version that scored 0.899 and reached parity with A9-rules was *still* taking
   4 voluntary draws in 40 games. **Retuning moved the cliff; it did not remove it.** This is exactly
   PLAN's H5, and it is the whole argument for H1: while refusal is a scoring *outcome*, a retune can
   always leave one behind.
2. **The costliest refusal was invisible to the `max()`.** `_play_draw_two` declines to King-cancel a
   pending +2 in order to keep the King — a voluntary draw of `2 * draw_num` **cards**, bought by
   `king_cancel_min_penalty`. It lives outside `_score_and_pick`, so no amount of staring at
   `score_draw` would have found it. H1 covers it.
3. **H4 confirmed by a second instrument.** `decline-King` fires **0 times in 200 games in every
   version**. The census and the `king_follow` ablation (+0.0000 +/- 0.0000) now agree: it is dead
   code. The agent believes it has a behaviour it does not have.

**Residual, deliberately left:** `h1` still shows 84 **close-TAKI** refusals — `hold_wilds_in_run`
closing a run rather than spending a wild. That is a different cliff (it costs no card), and it is
H4/H6's business, not H1's. `h1b2` has none because B2 already turned that hold off.

### Reproduce

```bash
python eval_headtohead.py heuristic:h1 heuristic:r3 --team1-seats 0,2 --seat-swap --games 20000
python holdback.py heuristic:h1 --games 200        # draw refusals: 0
```

Full log: `probes/h1_eval.log`. **REFERENCE is still `r3`** — promoting a successor is B5, and it is
deliberately the *last* step of the series, not a side effect of this one.

---

## 2026-07-13 (later still) — R6 FINISHED (L=60, 100k): a NEW CHAMPION (+8.5 pts), but it still will not block. Strength and the behavioural fix came apart

Run `models/run1783959155.016766` (100k trials, seed 1, shaped, `--loss-penalty 60`). Promoted as
**`checkpoint_r6L60_snap100000`**. Supersedes the partial (20k) L=60 read in the entry below, which
was **misleading** — see the caveat there, which is why it was flagged.

**PROMOTED: it beats the champion decisively, on the promotion standard.**
Seat-swapped alternating 2v2 (`--team1-seats 0,2 --seat-swap`), 3000 games x 2, **three disjoint
deck blocks**: **+0.0850 / +0.0850 / +0.0907 +/- 0.011** — ~8 SE, and it replicates. Vs the heuristic
yardstick (1-vs-3, parity 0.250): **0.378** vs the champion's **0.348**. And it took **100k trials,
not 500k**.

| | champion (500k) | L=20 (100k) | **L=60 (100k)** |
|---|---|---|---|
| vs 3x heuristic | 0.348 | 0.366 | **0.378** |
| seat-swap vs champion | — | (not run) | **+0.085 +/- 0.011** |
| census REFUSAL | 1.75% | 1.85% | 2.92% (pass) |
| delta(k=1) | -4.01 | -3.23 | **-0.87** |
| delta slope | **falls** as k falls (WRONG) | falls (wrong) | **RISES as k falls (RIGHT)** |
| delta spread | 2.55 | 0.88 | 1.95 |

**But the pre-registered prediction still FAILED, and that is the interesting part.**
`delta(k=1) = -0.87`: much closer to zero, but **still negative — it does not block.** What *did*
change is the **slope**: the champion wanted the +2 *least* when the opponent was closest to winning
(backwards); L=60 wants it *most* then (correct). **R6 fixed the sign of the derivative but not the
intercept.** It learned *that a near-winner is dangerous* without learning *to spend a card on it*.

**So strength and the behavioural fix came apart.** +8.5 points did NOT come from blocking, because
it still does not block. It presumably comes from R6 removing the "an imminent loss is a relief"
pathology *everywhere* — the agent now plays to avoid losing in general, which is worth a lot even
though the one probe we built it for still fails. **A pre-registered test that fails while the model
gets much better is exactly the situation the test exists to reveal.** Had we only measured win rate
we would have declared the defect fixed.

**Why the intercept does not move, and why this is the case for R17.** Playing the +2 still
*prolongs the game*, and `-len(hand)` still charges ~3 per extra turn. A terminal penalty shifts the
whole `delta` curve but cannot pay off a *per-turn* tax on defence — the level term is an
**intercept** problem. **Out-shouting it does not work; removing it might.** That is precisely
**R17** (potential-based shaping: shape on the CHANGE in hand size, so a longer game costs nothing).
Strength is also **monotone in L** (0.348 -> 0.366 -> 0.378), so a larger L is worth one more run,
but the delta curve says the ceiling of this approach is near.

**Guard-rail passed:** refusal 2.92% vs the champion's 1.75%. Elevated but well within tolerance —
"never refuse to play" survived. (The 5.67% in the 20k partial was a training-length artefact, as
flagged.)

**Method note:** the 20k partial read of this same configuration showed `delta(k=1) = +0.26` and the
agent blocking at every k. **At 100k it settled to -0.87 and blocks at none.** Early-training reads of
a Q-landscape are not directional evidence. The caveat on that entry was correct and load-bearing.

## 2026-07-13 (later) — R6 attempt 1: a terminal loss penalty fixes the VALUATION but not the POLICY. The step term taxes defence

Branch `b2-holdback`. `train.py --loss-penalty`. Run `models/run1783898274.148395` (100k trials,
seed 1, shaped, L=20). Acceptance test `r6_accept.py`, **pre-registered in the B2 commit before this
run finished, and written to be able to fail. It did.**

**Setup.** B2 (entry above) found the champion will not block a one-card opponent even though
blocking is worth +0.050 +/- 0.017, and traced it to `seat_reward`: `-len(hand)` every step, a bonus
only on a win, **nothing on a loss** — so ending the game *stops the penalty stream* and an imminent
defeat is a relief. R6 adds a penalty to the terminal transition of a seat that did not win.

**Result: PREDICTION 1 FAILED, PREDICTION 2 PASSED.**

| | baseline champion | R6 (L=20, 100k) |
|---|---|---|
| Q(red 5) at k=1 / k=7 | -6.56 / -9.93 — a near-winning opponent is **good news** | **-20.07 / -18.84 — now correctly BAD news** |
| delta(k=1) = Q(+2)-Q(5) | -4.01 | -3.23 (moved the right way, nowhere near flipping) |
| delta spread over k | 2.55, **falling** as k falls (backwards) | 0.88, still **falling** (still backwards) |
| census REFUSAL rate | 1.93% | 1.85% (guard-rail holds) |

**The mechanism claim is CONFIRMED by intervention.** The sign of the state value flipped exactly as
B2 predicted: the identical hand is no longer worth *more* when an opponent is about to go out. B2's
causal story was right.

**But the policy did not follow, and the reason is a tension B2 missed.** Blocking with a +2 makes the
opponent draw and **prolongs the game** — and the `-len(hand)` step term charges ~3 per extra turn. So
the *same* reward term B2 praised (it teaches "never refuse to play", worth ~13 pts) **also taxes every
defensive line**. A one-off penalty of 20 does not outweigh ~15 in extra step cost times the
probability the block actually saves you. The two halves of the shaped reward are in conflict, and a
constant terminal penalty cannot resolve it.

**Guard-rail passed, which matters:** the refusal rate did not move (1.85% vs 1.93%), so the loss
penalty did not break the half of the reward that was working. Whatever the fix is, it need not
trade one against the other.

### R6 attempt 2 (L=60, partial — 20k trials): the sign FLIPS, but it over-blocks and the guard-rail cracks

| | baseline (500k) | L=20 (100k) | L=60 (20k, partial) |
|---|---|---|---|
| delta(k=1) | -4.01 | -3.23 | **+0.26 — SIGN FLIPPED** |
| delta direction | falls as k falls (wrong) | falls (wrong) | **rises (right)** |
| delta spread over k | 2.55 | 0.88 | **0.20** |
| plays at k=1 | number | number | **the +2** |
| plays at k=7 | number (correct) | number (correct) | **the +2 (WRONG — blocking costs -0.036 here)** |
| census REFUSAL | 1.93% | **1.85%** (pass) | **5.67%** (elevated) |
| vs 3x heuristic | 0.348 | **0.366** | 0.342 |

**The magnitude hypothesis is confirmed and then punished.** L=60 does flip the sign — so the "step
tax dominates" diagnosis was right — but it produces an agent that plays the +2 at **every** k,
including where blocking is measurably worse. **The k-spread is 0.20**, i.e. it is not blocking
*selectively*; it is just always blocking. A constant terminal penalty can shift **how much you fear
losing**, but it cannot teach **when the threat is real** — which is precisely the k-sensitivity the
probe asks for. L=20 under-blocks everywhere, L=60 over-blocks everywhere, and *neither has the
spread*. That is a strong argument that the problem is **form, not magnitude** -> **R17**.

**The unexpected win: L=20 is genuinely STRONGER.** 0.366 vs the champion's 0.348 against the
heuristic yardstick (SE ~0.009, so ~2 SE) — while passing the refusal guard-rail. So a terminal loss
penalty is worth keeping **even though it did not fix the behaviour it was built for**. Promotion is
NOT claimed here: it needs a seat-swapped head-to-head against the champion, which was not run.

**Caveats, load-bearing.** L=60 reached only **20k of 100k trials** before the time budget ran out, so
its 0.342 and its 5.67% refusal are **confounded by short training** (early models refuse more) and
must not be compared to the fully-trained rows as if they were peers. The sign flip and the near-zero
k-spread are the parts that are safe to read at 20k, because the L=20 run's delta at 20k already
matched its own final value. **A full L=60 run is the first thing to finish.**

**Next.** (a) Finish the L=60 run (100k) and re-run `r6_accept.py`.
(b) A larger penalty (L=60 running) directly tests the "the step tax dominates" explanation —
it must eventually flip delta if the diagnosis is right; if even a large L fails, the problem is not
magnitude but *form*. (b) The principled fix is **potential-based** (PLAN A9): shape on the *change*
in hand size rather than its level, so a longer game is not intrinsically penalised and defence stops
being taxed. That is now the leading candidate and it is what R6 should probably become.

**Process note:** two concurrent `train.py` runs (each thread-capped and niced) ran at **~2.2 trials/s
each**; killing one took the survivor to **~34 trials/s** — a **>10x** speedup, not the ~2x core
contention predicts. Concurrency is strongly negative-sum on this box: **run trainings sequentially**,
and always measure the rate before sizing a run (100k looked like 13 h concurrent, ~50 min solo).

## 2026-07-13 — B2: the shaped reward is the source of the champion's best habit AND its worst blind spot. R6 proven — on defence, not on shedding

Branch `b2-holdback`. Full write-up: `probes/b2_holding_back.md`. Harness: `holdback.py`,
`agents/heuristic.py` (`Weights`/`GREEDY`/`ABLATIONS`), `holdbacktest.py` (14 tests).

**PLAN.md B2 asked** whether the shaped reward (`-len(hand)` per step) makes the policy a greedy
hand-size minimiser that never plays fewer cards for a better endgame — which would be the argument
for **R6**. **The premise is false.**

**The key distinction, which nothing in this project had drawn:**
- **REFUSAL** hold-back — decline to put a card down at all (DRAW, or CLOSE a TAKI / King
  continuation) while a legal play exists. Keeps the card, loses the tempo. **Catastrophic.**
- **PREFERENCE** hold-back — play a *different* card instead. Keeps the card, spends the turn.
  **Valuable.**

**Method — we did NOT ask the DQN's value function** (it is the thing under suspicion, so its own
rollouts would be circular). We asked an agent whose strategy we own. R3's `HeuristicAgent` holds
cards back on purpose, and `SCORE_DRAW = -5.0` is a *finite score in the same max*, so **any hold
penalty above 5.0 flips it from preferring another play to refusing to play.** Refactored its
constants into a `Weights` dataclass (defaults byte-identical, pinned), ablated each behaviour, and
ran `eval_headtohead.py --team1-seats 0,2 --seat-swap` (3000 games x 2 occupancies). No network.

**Result 1 — every hold-back term priced** (margin = full - ablated; positive = worth having):

| behaviour | margin | kind | verdict |
|---|---|---|---|
| hoard (`w_reserve` 10->0) | **-0.0623 +/- 0.0105** | refusal (draws to protect the group) | harmful |
| king_cancel (eat a +2 to keep the King) | **-0.0353 +/- 0.0054** | refusal (draws 2) | harmful |
| wilds (`p_king`/`p_chcol`/`p_super`, in-run) | **-0.0450 +/- 0.0116** | mixed | harmful net |
| blocker (save STOP/+2, no threat) | +0.0110 +/- 0.0064 | preference | mildly good (1.7 SE) |
| finisher (keep a legal last card) | +0.0083 +/- 0.0035 | rules-driven | good (2.4 SE) |
| king_follow (decline the free card) | **+0.0000 +/- 0.0000** | — | **never fires** |

**Result 2 — a 13-point cliff exactly at the draw threshold.** Sweeping `p_king` alone:
2 -> +0.016, 4 -> +0.019, **5 -> +0.020** (holding the King is worth +2 pts), **6 -> -0.104**
(it now DRAWS rather than play it), 8 -> -0.104 (identical: the effect is the behavioural *flip*,
not the magnitude). Head-to-head `p_king=6` vs `p_king=5` = **-0.130 +/- 0.009**, replicated on
three disjoint deck blocks (-0.130 / -0.129 / -0.132). The `w_reserve` sweep has the same shape:
flat (~0) from 0 to 6, then collapses at 8 and 10 — i.e. **hoarding is free; drawing to hoard is not.**

> **Statement about Taki: never draw to protect a plan.** Keeping a card by playing something else is
> free or better; keeping it by passing costs more than the card is ever worth.

**Result 3 — this RESOLVES B1's apparent contradiction.** B1 found "dumping the TAKI never beats
keeping it"; the ablation says hoarding costs 6.2 pts. Both are right: **B1 only ever compared
play-vs-play**, which is the preference regime — and there B1 is confirmed exactly (`w_reserve` 0-6
is flat). R3's `W_RESERVE = 10.0` goes further than anything B1 tested: it *draws*. The 6.2 points
are the drawing, not the hoarding. B1's hoarding half survives, with a sharp boundary around it.

**Result 4 — the census (`holdback.py`), 200 self-play games, all 4 seats.** At every decision, does
the agent pick a move that provably sheds fewer cards this turn than a legal alternative? "Sheds
fewer" is a rules-level search (`max_shed`), not a heuristic, because action identity lies: closing a
TAKI on a PLUS or King **keeps the turn** (game.py:584-598), so CLOSE_TAKI is not always a refusal.

| agent | REFUSAL (% of free decisions) | PREFERENCE hold-back | vs 3x random |
|---|---|---|---|
| random | 35.1% | 47.1% | 0.250 |
| heuristic (R3, shipped) | 14.5% | 34.6% | 0.850 |
| **DQN champion** | **1.8%** | **20.4%** | **0.906** |
| heuristic (retuned) | 0.3% | 14.4% | 0.899 |
| heuristic:greedy (control) | **0.0%** | 13.7% | 0.879 |

**The champion holds cards back one time in five, and almost never by refusing to play.** It is not
a greedy shedder, and it is not making B2's predicted mistake. **The shaped reward is WHY:**
`-len(hand)` punishes drawing (drawing grows the hand), so the training signal directly encodes the
most valuable rule in the game. On *shedding*, it is **aligned, not biased** — R6's hold-back
justification is dead.

**Result 5 — BUT: weapon timing, where the champion fails and R6 turns out to be RIGHT after all**
(`scenarios_b2.py`). A red +2 and a red 5, both legal, **both shedding exactly one card** — so the
shaped reward is *exactly indifferent* between them, and the probe isolates the VALUE FUNCTION from
the reward. Only `k`, the NEXT player's hand size, varies (legal set / deck / unseen counts constant
by construction).

    k:              1       2       3       5       7
    Q(red +2)  -10.58  -10.72  -11.36  -11.74  -11.40
    Q(red 5)    -6.56   -7.98   -9.00   -9.66   -9.93
    delta       -4.01   -2.74   -2.36   -2.08   -1.47      <- should RISE as k falls. It FALLS.

It **does** use the feature (spread 2.55 vs an untrained noise floor of 0.01-0.06 over three seeds),
but with the **wrong sign**: it least wants to fire the +2 exactly when the next player is one card
from winning. Rollout (1200 determinizations, paired, 3x heuristic):

| k | block (+2) | number (5) | block - number |
|---|---|---|---|
| **1** | **0.310** | **0.260** | **+0.050 +/- 0.017** <- champion plays the number: WRONG |
| 2 | 0.319 | 0.369 | -0.050 +/- 0.017 (correct) |
| 3 | 0.342 | 0.378 | -0.036 +/- 0.017 (correct) |
| 5 | 0.358 | 0.394 | -0.036 +/- 0.018 (correct) |

**Right at every k except the one that matters — and most confident precisely there.**

**The mechanism, and it is not a mis-fit network.** `Q(red 5)` is **-6.56 at k=1** and **-9.93 at
k=7**: the *identical hand* is valued 3.4 HIGHER when an opponent is about to win. From
`train.py:288-290`, `r = -len(hand)` every step (always negative) `+ bonus only on a win` — **a loss
pays nothing, the penalty stream just stops.** So an imminent loss is *rewarded* (it truncates the
stream), and blocking a near-winner *prolongs* the game and therefore looks worse. The value function
is faithfully optimising an objective **indifferent to who wins**.

> **Statement about Taki: block the player about to go out.** A +2 spent on a one-card opponent is
> worth ~5 points; spent on anyone else it is worth *less* than a plain number.

**CORRECTION (2026-07-13, later) — the magnitude above is inflated ~2x; the finding survives.** The
table was rolled out against `3x heuristic` with **shipped** weights — the same weights this entry
goes on to show are crippled by the refusal cliff (14.5% voluntary draws). A pool that draws when it
should play ends games slowly, and a tempo weapon is worth more against slow opponents. Re-priced
against the **retuned** pool (`b2_block_price.py`, 1200 paired determinizations x 3 independent deck
blocks), with the shipped pool re-run alongside as a control:

| k | shipped pool (as published) | retuned pool |
|---|---|---|
| **1** | +0.050 / +0.064 / +0.081 | **+0.020 / +0.038 / +0.045** |
| 2 | -0.050 | **-0.048 / -0.049 / -0.050** |
| 3, 5, 7 | -0.036 .. -0.050 | -0.031 .. -0.077 |

Seed 0 reproduces the published row **exactly** (+0.050, -0.050, -0.036, -0.036), so the harness is
faithful and the difference is the opponent pool, not the code. **The k=1-vs-k=2 sign flip — the whole
content of the finding, and the sole evidence for H2 — is unchanged.** Only the price changes: the
statement above should read **~2-4 points**, not ~5. The champion's own win rate drops from ~0.31-0.41
to ~0.20-0.26 across the same positions, independently corroborating that the retuned heuristic is a
much stronger opponent (Result 5).

**Method bug found while doing this, and it taints every "confirmed on seeds 0/1/2" claim in B1/B2.**
`mc_line` seeded replicate `g` with `seed + g`, so a run at seed 0 drew deals 0..1199 and a run at
seed 1 drew deals 1..1200 — **a 99.9% shared sample**. Re-running at nearby seeds was not replication;
it was one measurement reported three times, and it looked *reassuringly* tight for exactly that
reason (the retuned k=1 cell read +0.020 / +0.020 / +0.019). Fixed: `probe.det_seed(seed, g)` hashes
the pair (crc32), so distinct seeds are independent blocks. The same cell now reads **+0.023 / +0.018
/ +0.038** — the honest spread, and consistent with the disjoint blocks above. Common random numbers
across *lines* (the pairing that makes the SEs tight) are unaffected and still pinned by `probetest`.
Prior logged rollout numbers will not bit-reproduce under the new seeding; that is the cost of the fix.

**So B2 both kills and proves R6, for opposite reasons.** Its *hold-back* justification ("the shaped
reward makes it greedily shed") is **dead** — that term is doing the most useful work in the model.
Its *defensive* justification ("losers get no terminal penalty") is **exactly right**, and this is the
first direct evidence: reward code -> Q-values -> behaviour -> a measured 5.2-point cost.
Pre-registered prediction for R6: add a terminal loss penalty and `delta(k)` flips sign at k=1.

**Result 6 (unplanned, and the most actionable) — R3's heuristic is badly tuned.** Keep every
preference hold-back, delete every refusal (`p_king=5, p_chcol=4.5, p_super_taki=4.5, w_reserve=4,
king_cancel_min_penalty=0, hold_wilds_in_run=false`):

| | vs 3x random | vs DQN champion (seat-swap 2v2, 3 disjoint seeds) |
|---|---|---|
| heuristic (R3, shipped) | 0.850 | -0.208 +/- 0.012 |
| heuristic (fully greedy) | 0.879 | — |
| **heuristic (retuned)** | **0.899** | **+0.018 / -0.018 / +0.005 -> PARITY** |
| DQN champion | 0.906 | — |

**A few hundred lines of rules, retuned on nothing but hold-back discipline, match the 500k-trial
champion.** R3's headline ("A8 genuinely better, 0.343 vs 3 heuristics") was measured against a
crippled opponent — the yardstick had a 17-point tuning bug. Anything ranked against the R3 heuristic
should be re-run, R3's own conclusions included.

**Two process findings, both bit us:**
- **`agenttest.py` was RED on master and nobody noticed.** R3 (agenttest) and the finishing-rule fix
  were developed in parallel; three tests still asserted the old rule (they used a red STOP as "an
  action card you cannot finish on" — it is now a legal finisher). The 2026-07-12 entry's "56 green"
  counted `gametest` + `probetest` only. Fixed here (they now use PLUS, the only real non-finisher);
  suite is 83 green. **Run the whole suite, not the subset you changed.**
- **`eval_headtohead` seeds deck `g` with `seed + g`**, so seeds 0/1/2 share 2999 of 3000 decks and
  are NOT independent replications. Seeds must be >= `games` apart. Our first "replication" produced
  three near-identical numbers, which is what exposed it.

**Caveats.** The retuned weights were swept on seed 0 and confirmed on disjoint blocks (the parity
claim rests on 3 independent seeds), but the exact values carry selection risk — they are un-broken,
not optimised. The census is one distribution (self-play). **The B4/Ĝ arm was not run:** Arm 1
answered "does holding back pay" by a stronger non-circular route, so the empirical shaped return was
not needed for the verdict. It remains the way to ask whether the champion's 20.4% preference
hold-backs are well-*calibrated*, as opposed to merely present.

## 2026-07-12 — RULES BUG: the finishing rule was wrong. Fixed, B1 half-retracted, A8 retraining

**Not an experiment — a correctness fix that invalidates a result and a champion.** Branch
`fix-finishing-rule` (worktree `/home/orih/taki-rules`).

**The bug.** `game.py` allowed a hand to END only on a number or the King:

    FINISHING_TYPE_VALUES = NUMBER_TYPE_VALUES | {Type.KING.value}      # WRONG

Every other card — TAKI, STOP, CHDIR, +2, Change Color, Super TAKI — emptied the hand but drew a
penalty and play continued. **The real rule: you may finish on any card EXCEPT PLUS.** PLUS is the
unique exception because it obliges you to put another card, which an empty hand cannot do; every
other card's effect lands on someone else and is coherent as a final play.

    FINISHING_TYPE_VALUES = frozenset(t.value for t in Type) - {Type.PLUS.value}   # correct

**This was never a house rule.** It is absent from RULES.md's house-rule section; the code comment
asserted the King exception was "standard Taki"; and `gametest.py` had a
`test_cannot_win_on_action_card` *encoding the bug*, which is why the suite stayed green for the
whole project. A test can only protect the behaviour it describes. One enforcement site
(`next_turn`), no contract change — `OBSERVATION_SIZE`/`ACTION_SIZE` untouched, so checkpoints
still load. RULES.md now has a dedicated **Finishing** section.

**Fallout 1 — B1's sequencing half is RETRACTED** (see the banner on the entry below). B1a's whole
premise was "the only finisher in hand is the red 5, so the run must end on it." Under the real
rule the red STOP and red +2 are finishers too: **every ordering wins**, and there was never a
trap. A8's deferral of the 5 was *real behaviour* — the DiD control genuinely separated it from
random-init nets — but it was skill at a game that isn't Taki. **"Statement about Taki #1" is
withdrawn.** B1's hoarding half (b1c) never touches finishing (no hand there can empty) and
survives as reasoning, though its numbers came from a policy trained on the buggy game.

**Fallout 2 — the interesting one.** With PLUS the *only* card that cannot end a hand, **there is
almost nothing to sequence in Taki**. A TAKI run needs planning only when it contains a PLUS;
otherwise dump the color group in any order and the last card wins. B1 assumed a rich sequencing
skill and found the policy had it; the corrected game says the skill barely exists. *That absence
is itself a finding*, and any redone sequencing probe must put a PLUS in hand to have a real trap.

**Fallout 3 — the champion is stale.** `checkpoint_a8_snap455000` optimised a game the engine no
longer plays, so probing it now would measure a policy on rules it never trained under. **Retrain
launched** (tmux `retrain_a9`, `nice -n 19`): `train.py --trials 500000 --seed 1
--snapshot-every 5000 --reward shaped` — A8's recipe on the corrected engine, but with the current
**rank-sym default left ON** (color-sym + rank-sym; R1 measured rank-sym at parity, so this is not
expected to move the number, and it re-tests rank-sym on the corrected game for free). ~6–7 h. On
completion: screen with `eval.py` Mode B, confirm with seat-swap-controlled `eval_headtohead.py`,
promote a new champion, then **redo B1** (hoarding first; sequencing needs a new PLUS-based
design) before B2.

**Caveat for anyone comparing across this line:** every win rate recorded *above* this entry was
measured under the buggy rule. They are not comparable to anything measured after it. The rule
change makes hands end sooner and more often on action cards, so absolute win rates vs random may
shift on their own.

Tests: 56 green (`gametest` + `probetest`). The old `test_cannot_win_on_action_card` is inverted
into per-card finishing coverage (STOP/CHDIR/+2/TAKI/Super TAKI/CHCOL/King all win) plus
`test_cannot_win_on_plus`; probetest's two B1A engine-fact tests now pin the *corrected* behaviour
and the one surviving constraint (you cannot finish on a PLUS inside a TAKI).

### RESULTS of the retrain (A9-rules): the fix costs nothing, and the run is flat after 25k trials

Run `models/run1783853033.01791`, 500k trials, seed 1, shaped, color-sym + rank-sym. **Promoted:
`checkpoint_a9rules_snap500000`** — the new champion, and the *only* usable checkpoint on disk
(see the missing-champion note below).

| yardstick | A9-rules snap500000 | A8 (old rules, for reference only) |
|---|---|---|
| vs 3 random (3000 games) | **0.906** | 0.912 |
| vs 3 **heuristic** (3000 games) | **0.348** | 0.343 |

**The rule fix is behaviourally cheap.** On the heuristic yardstick — the one R3 says to trust —
the corrected-rules agent lands at **0.348 vs A8's 0.343**, a difference of 0.005 against SE
≈0.009: *parity*. Learning the real game is neither harder nor easier than learning the buggy one.
(Both numbers are cross-rule comparisons and should be read as "same ballpark", not as a ranking.)

**The run is flat, and that is the finding.** Mode B (21 snapshots, 3000 games each, baseline =
`snap500000`): vs-random hits **0.902 at snap25000** and never improves — the whole 25k→500k tail
wanders in 0.893–0.909, i.e. inside noise. Against the internal baseline every snapshot sits in
0.237–0.269 around parity 0.250 (SE ≈0.008); the max (snap300000, 0.269) is +2.4 SE, which is
exactly what the *maximum of 21 draws* looks like under the null — winner's curse, not skill.
Confirmed head-to-head with the new standard (`--team1-seats 0,2 --seat-swap`, 3000×2):
snap300000 vs snap500000 = **+0.0073 ± 0.0109, Tie**. So no snapshot beats any other, and
`snap500000` is promoted for parsimony (final snapshot, no selection bias).
**~475k of the 500k trials bought nothing measurable.** This is the plateau of CLAUDE.md's
project-goal section showing up on the *corrected* rules too, and it sharpens the case that the
agent is at the ceiling of its information set rather than under-trained. A cheap follow-up: the
next run needs nowhere near 500k trials, which makes B-series iteration much faster.

**Two caveats, both load-bearing:**
- **vs-random is saturated and near-useless.** 0.90 by trial 25,000, then flat for 475k more. R3
  already retired it as a *ranking* metric (A8: 0.912 vs random, 0.343 vs heuristic); this run
  shows it is not even a *progress* metric past the first 5% of training. Rank on the heuristic.
- **The champion checkpoints are GONE.** `models/` held nothing when the retrain finished —
  `checkpoint_a8_snap455000` and `checkpoint_a4a7_snap550000` no longer exist anywhere on disk,
  lost in the models-symlink incident that commits `0f96882`/`62a7433` repaired. So the planned
  "new champion vs old champion" seat-swap head-to-head is **impossible** and no cross-rule
  head-to-head will ever be run. The A8 numbers above survive only as recorded history. Every
  comparison from here is against the heuristic or within-lineage.

Not yet done: **redo B1** against the new champion (hoarding half re-measures directly; the
sequencing half needs a fresh PLUS-based design), then B2.
## 2026-07-12 — R3 (half): a hand-crafted heuristic agent — the first non-lineage yardstick. A8's 0.912 vs random is worth far less than it looked.

**Setup.** New `agents/heuristic.py`: a rule-based agent with **no network**, restricted to the same
human information set as the DQN (own hand, opponents' hand *sizes*, top card, deck size, and a new
`game.history` log of **public table events** — who played/drew what, and the active color at the
time). It never reads opponents' hand contents. Behaviours: +2 stacking (King-cancel only when the
pending penalty ≥ 4), TAKI-run planning that closes the run on a legal finisher, colored-TAKI
**hoarding**, near-winner blocking with STOP/+2, wild-holding, color richness, and **color denial**:
if a player drew while color *c* was active they are believed to lack *c*, a belief that decays
×(1 − 28/120) per card they have drawn since. Deterministic (no RNG), so eval's common-random-numbers
invariant is preserved. `eval.py` gained `--opponent {random,heuristic}` and a `--model heuristic`
sentinel. 22 new tests in `agenttest.py`; `gametest` still green (the history log changes no
observation/action contract — **all checkpoints remain loadable**).

**Results** (4 players, 3000 games, seed 0, shuffled seating, parity 0.25):

| matchup | win rate |
|---|---|
| heuristic vs 3 random | **0.849** |
| A8 (`checkpoint_a8_snap455000`) vs 3 random | 0.912 (previously measured) |
| **A8 vs 3 heuristic** | **0.343** |
| random vs 3 heuristic | 0.014 |

**Reading.** The heuristic is a strong opponent — it takes a random agent from 0.25 to 0.014, i.e. it
nearly shuts random out, and it lands within 6 points of A8 on the vs-random scale that has been our
headline metric all along. A few hundred lines of rules recover most of what 500k trials of self-play
bought, which says the vs-random number has very little resolution up here: **0.849 vs 0.912 is the
gap between "sensible rules" and "our best model", so vs-random should be retired as a ranking metric**
(it is still fine as a smoke test that a run hasn't collapsed).

A8 *is* genuinely better: 0.343 vs 3 heuristics is comfortably above the 0.25 parity line
(SE ≈ 0.009, so ~10 SE). That is the first evidence of A8's skill measured against something outside
its own lineage, and it holds up. But the margin is modest, and it reframes the plateau story: the
agent is not "near the ceiling of play", it is near the ceiling of *what vs-random can see*. The
heuristic is now the recommended eval opponent for anything that needs discriminating power.

**Caveats.** (1) The heuristic's weights are hand-set, not tuned — a tuned version would likely be
stronger, which would *lower* A8's 0.343, not raise it. (2) The color-denial belief is weak evidence
against a random opponent (DRAW is always legal here even with playable cards), so it earns its keep
mainly against rational play. (3) 1-vs-3 rates are not comparable to `eval_headtohead.py`'s per-seat
rates (CLAUDE.md). (4) The oracle half of R3 (a full-information upper bound) is still open.

## 2026-07-12 — New head-to-head standard: alternating 2v2 seat swap (`--seat-swap`), + a pairing bug it exposed

Methodology change, not a training result. The head-to-head promotion test is now
**`eval_headtohead.py <cand> <champ> --team1-seats 0,2 --seat-swap --games 3000`**, which runs both
occupancies of the alternating partition over the same decks and reports the seat-balanced, paired
comparison in one command. Rationale is in CLAUDE.md; the two reasons that decided it:

- **Alternating `A,B,A,B` is the only 2v2 layout with no friendly fire.** STOP/+2/CHDIR hit your
  *neighbour*, so in a contiguous `A,A,B,B` half of each team's aggression lands on a teammate — and
  CHDIR makes that asymmetry direction-dependent. Interleaved, every attack crosses team lines.
- **Seat balance becomes exact rather than approximate.** Across the two runs each model holds each
  seat exactly once, so seat 0's first-mover edge cancels by construction; and all four seats feed the
  team indicator (parity 0.5) instead of one Bernoulli per game (parity 0.25).

**Pairing bug found and fixed while implementing it.** `play_match` built one `Game` and reused its
single RNG stream across all games. Mid-game reshuffles draw from that same stream, so game *g*'s deal
depended on how games *0..g-1* happened to play out — meaning two runs with swapped occupants silently
**diverged after game 0** and were never actually paired. Fixed by reseeding the deck RNG to `seed + g`
before every game, as `eval.py` has always done. (Also: `total_decided` used to increment on every game
including undecided ones; now only on decided ones.) Old head-to-head numbers were unpaired but not
*biased* — the seat-swap logic was still comparing like with like in expectation, just noisily.

**Validation — the new protocol reproduces the A8 promotion independently.**
`checkpoint_a8_snap455000` vs `checkpoint_a4a7_snap550000`, partition [0,2]/[1,3], 3000 games x 2 runs,
seed 0:

| | seat-balanced per-seat (parity 0.250) |
|---|---|
| `checkpoint_a8_snap455000` | **0.268** |
| `checkpoint_a4a7_snap550000` | 0.232 |

Paired same-seat margin **+0.0710 ± 0.0114** (paired SE over 3000 common decks; ~6 SE). The +3.6 pt
per-seat edge matches the **+3.7** recorded for A8's original promotion under the old ad-hoc protocol,
so the champion stands and the two protocols agree where they overlap. Runtime is cheap: 134 s for both
runs at 3000 games.

**Keep the 1v3 form (`--team1-seats 0 --seat-swap`) as a secondary check** — it is the same shape as
Mode B (one model in a homogeneous field), which is what the champion's headline 0.912 means. A
disagreement between 2v2 and 1v3 is information (the edge depends on field composition), not a bug.

---

## 2026-07-12 — B1: A8 sequences a TAKI run correctly, but dumps the TAKI too eagerly (behavioural probe)

> **⚠️ HALF RETRACTED same day — see the finishing-rule entry above.** The engine's finishing rule was
> a bug (finish only on a number/King; the real rule is *any card except PLUS*). That makes the
> **sequencing half degenerate**: in B1a's hand the STOP and +2 are finishers too, so every ordering
> wins, there was never a trap, and **"Statement about Taki #1" is withdrawn** — A8's deferral of the
> 5 is real behaviour at a game that isn't Taki. The **hoarding half (b1c) survives the rule fix** (no
> hand there can empty), but its numbers came from a policy trained on the buggy game, so it must be
> re-measured after the retrain. Details in `probes/b1_colored_taki_hoard.md`.

The first B-series probe (PLAN.md), and the first result in this log that is **not a win rate**: it is
a pair of statements about how to play Taki, each backed by the champion's own valuation and then
adjudicated by rollout. No training. Harness: `probe.py` (+ `probetest.py`, 14 tests). Full write-up:
`probes/b1_colored_taki_hoard.md`.

**Setup.** Hand-built positions against `checkpoint_a8_snap455000`, greedy. Two arms, kept separate
because they answer different questions. *(Q arm)* one forward pass; read the **ranking** over legal
moves. Q is a shaped return, **not** a win probability, so only within-position orderings are used.
*(Rollout arm)* force a candidate line, hand back to the greedy policy, play out 1500x, resampling the
hidden cards (opponent hands + deck order) while holding the learner's hand / discard / state / opponent
hand *sizes* fixed. That determinization is exactly the learner's belief state — sound because the
observation is **bit-identical** across resamples (it sees the hidden region only through sizes and
coarse counts; pinned as a test). Common random numbers across lines → all differences are **paired**.
Opponent pools: **copies of A8 at eps=0.1** (the self-play distribution Q was trained under; parity
0.25) and `RandomAgent`.

**Headline 1 — sequencing: PASS, and it is not a reflex.** A hand may only *end* on a number or the
King, so a TAKI run must close on a finisher. Given `red TAKI, red STOP, red +2, red 5` on a red 3, A8
opens the run, plays the action cards, and **saves the red 5 for last** — winning outright. It ranks
that red 5 *fourth of five* legal moves. The obvious alternative explanation is that it merely dislikes
playing low numbers, so the control kills the win (same choice, plus an unplayable blue 9):

| policy | Q(STOP)−Q(5) win live | ... win dead | **diff-in-diff** |
|---|---|---|---|
| **A8** | +18.54 | +2.91 | **+15.63** |
| `checkpoint_a4a7_snap550000` (different lineage) | | | **+7.79** |
| 5x random-init nets | | | −0.00 … +0.01 |

The reflex explanation is rejected: A8 defers the finisher **~6x more strongly when the win is actually
available**, the effect replicates in an independently trained lineage, and it is absent in every
untrained net. Rollouts: the correct line wins **1.000**; mis-sequencing costs **+0.404 ± 0.013** win
rate vs A8 copies. Forcing only the TAKI and letting A8 choose the rest also returns **1.000** — once
it opens a run it finishes it correctly every time.
*(Engine fact, pinned by test: the penalty draw does **not** end the turn — you keep playing inside the
open TAKI. Mis-sequencing forfeits a **guaranteed** win, not the win: the trap line still wins outright
35.9% of the time when the drawn card is red-and-a-number.)*

**Headline 2 — hoarding: FAIL, and the shaped reward is the culprit.** Showing a blue TAKI, a red TAKI
is legal *by type* while red numbers are not — which pins the legal set at exactly
`{red TAKI, green TAKI, blue 1, DRAW}` while `k`, the number of red backers, varies 0..4. The hand also
carries a **green TAKI backed by nothing**, so `Δ(k) = Q(red TAKI) − Q(green TAKI)` is a within-position,
matched-card contrast — and at k=0 the observation is *exactly* red↔green symmetric (pinned by test), so
a color-equivariant net must give **Δ(0)=0**: a free, calibrated noise floor.

| k | Δ(k) | A8 plays | **dump − shed** (paired) vs A8 copies | vs random |
|---|---|---|---|---|
| 0 | +0.41 *(noise floor)* | **sheds** (blue 1) | −0.019 ± 0.012 | −0.024 ± 0.013 |
| 1 | +0.65 | **sheds** | −0.007 ± 0.013 | −0.023 ± 0.012 |
| 2 | +1.57 | **dumps** | −0.005 ± 0.014 | −0.019 ± 0.013 |
| 3 | +2.05 | **dumps** | +0.019 ± 0.015 | −0.001 ± 0.012 |
| 4 | +1.86 | **dumps** | −0.017 ± 0.017 | **−0.033 ± 0.011** |

A8 prices the TAKI by its backing (Δ grows to ~5x its noise floor; random-init controls are **flat**
across k) and it does **not** dump on sight — it keeps a weakly-backed TAKI. But its crossover is in the
wrong place. **Dumping never beats keeping, at any k, against either pool.** At k=4 dumping discharges
**five cards in one turn** (hand 7→2) against shedding **one** (7→6) — and the five-card turn **wins
less often** (0.862 vs 0.895 vs random, ~3 SE). Shedding five times the cards buys nothing: the TAKI
group was never the problem, it was the guaranteed escape hatch, and cashing it early spends it while
leaving behind the cards that may be unplayable when the turn returns.

**So PLAN.md's premise — "hoarding is nearly free" — is confirmed by measurement, and the policy is what
doesn't fully believe it.** The cause is legible: `train.py`'s shaped reward pays `-len(hand)` per step,
so a five-card turn is enormously rewarded in the *training signal* while being worth nothing in *win
rate*, and A8's Q prefers the dump from k=2 up precisely where the win rate says not to.

**Caveats.** Q is a shaped return, not a win probability — cross-position Q levels are meaningless and
are never used. The rollouts assume a **uniform prior** over unseen cards (correct given the observation,
but a human would sometimes infer better from the play history). Positions are hand-built: guaranteed
*reachable* (`make_position` asserts deck conservation and a discard large enough for a real game to have
reached the position) but still hand-picked, not sampled from play. Effect sizes in the hoard table are
small (1–3 pts); k=4-vs-random is the only individually strong one, though **every** cell points the same
way, which is the substance of the claim. **This probe cannot say whether the network mis-estimated its
own objective or estimated it correctly and the objective is wrong** — separating those needs the
empirical shaped return measured beside the win rate, now filed as **PLAN.md B4**.

---

## 2026-07-11 — CALIBRATION: run-to-run sigma is ~0.5-0.7 pts — the promotion history holds up

The project had never measured **run-to-run variance**, yet every promotion was adjudicated
against a ±2.0-point bar on a single seed. Without sigma, no bar means anything, and A8's +3.7
could in principle have been noise we narrated as signal. This run settles it.

**Setup.** Two fresh 500k-trial runs at the current default recipe (color+rank sym), seeds 2 and
3 (`models/run1783774662.463957`, `models/run1783774663.490516`), joining the R1 run (seed 1). All
three are the *same* recipe, so their true edge vs A8 should be ~0; the **spread** of their edges
is sigma. **Pre-committed selection rule, fixed in writing before any result was seen:** evaluate
`snap500000`, the **final** snapshot, of each seed — no screening, no argmax. Seat-swap-controlled
head-to-head vs `checkpoint_a8_snap455000`, 3000 games x 2 seat assignments.

| Seed | cand avg per-seat | A8 avg per-seat | Edge |
|---|---|---|---|
| 1 (R1) | 0.260 | 0.258 | +0.15 |
| 2 | 0.264 | 0.256 | +0.85 |
| 3 | 0.264 | 0.253 | +1.15 |

**sigma = 0.51 pts** (sample SD, n=3). Pure *eval* noise alone predicts an SE of **0.65 pts** per
edge (3000 games; seat-0 rate + 3-seat average). The observed seed spread is **below** the
measurement floor — seed-to-seed variance is not even detectable above eval noise. Take
**sigma_total ~ 0.5-0.7 pts**.

**Consequences — every past verdict survives:**

| Result | In sigma |
|---|---|
| A8's promotion (+3.7) | **~5.7 sigma** — real |
| The +2.0 promotion bar | ~3.1 sigma — a sound bar |
| R1 best-of-5 (+1.6) | ~2.5 sigma, but a *max of 5* — correctly not promoted |
| R1 mean (-0.14) | ~-0.2 sigma — parity, as reported |

**The sharper lesson: the hazard is snapshot choice, not seed choice.** Spread across the 5
screened snapshots *within* the R1 run was **1.25 pts SD** — 2.5x the seed-to-seed spread of the
final snapshot (0.51). Argmax-over-snapshots is where the winner's curse actually enters, and it is
exactly the leak that would have sold `snap225000` (+1.60) as a win. **Standing rule going forward:
pre-commit the snapshot (the final one, or an average of post-floor snapshots) before the confirming
head-to-head, and never report the max of a selection set as the edge.**

**Footnote, honestly reported:** the 3 seeds' mean edge is **+0.72 ± 0.38** (~1.9 sigma) — a weak
hint that the current default at snap500000 sits a touch above A8. It is below the +2.0 bar, it is
confounded (A8 is snap455000 of a color-only run), and R1's own wider 5-snapshot sample said -0.14.
**Read as parity. Nothing promoted; `checkpoint_a8_snap455000` remains the champion.**

## 2026-07-11 — NEGATIVE: rank-symmetry augmentation is inert on top of color-sym (R1)

Branch `r1-rank-sym`. The nine number cards ONE..NINE are *exactly* interchangeable in TAKI
(deck holds 2 copies of every rank in every color; matching is "same color or same type"; the
only two rules that mention ranks — must OPEN on a number, may END on a number or the King —
are set-membership tests, so no rule branches on a specific rank). That makes any global
relabeling of the nine ranks an exact symmetry of the dynamics: a group of 9! = 362,880 perms
that commutes with the 24 color perms, for ~8.7M composed relabelings. R1 was the direct
sequel to color-sym — the single largest gain the project ever produced — and carried the
highest prior on the board ("data-side symmetries pay").

**It bought nothing.**

**Setup.** `train.py --trials 500000 --seed 1 --snapshot-every 5000 --rank-sym`
(`models/run1783752040.164939_ranksym`) — A8's exact config, with rank-sym as the only
difference, so the promoted `checkpoint_a8_snap455000` (color-sym only, same seed) is the
paired control. Rank-sym is *additive*: color-sym stays on, and each replayed transition gets
a single relabeling composed from both groups. Cost ~100us per replay batch — invisible next
to the TF step.

**Screening** (Mode B, stride 25000, 500 games, vs `checkpoint_a8_snap455000`): vs-baseline
ranged 0.246-0.294 across all 21 snapshots with **no trend and no standout** — the whole
spread is inside noise at 500 games (SE ~= 0.022). vs-random saturates at 0.88-0.93, the same
plateau as A8's 0.912.

**Seat-swap-controlled head-to-head** (the decisive step; top 5 screened snapshots vs A8,
3000 games x 2 seat assignments each, per CLAUDE.md's evaluation discipline):

| Snapshot | rank-sym avg per-seat | A8 avg per-seat | Edge |
|---|---|---|---|
| snap225000 | 0.264 | 0.247 | **+1.60** |
| snap500000 | 0.260 | 0.258 | +0.15 |
| snap475000 | 0.258 | 0.258 | +0.00 |
| snap75000 | 0.257 | 0.263 | -0.60 |
| snap150000 | 0.251 | 0.269 | -1.85 |

**Mean edge across the 5 candidates: -0.14 pts — i.e. exact parity.** The best (+1.60) is a
max-of-5 and so is selection-inflated; it does not clear the +2.0 bar that A10 and A11 already
failed, let alone A8's +3.7. Read as parity, not as a small win.

**Why this matters more than a normal null.** The "data-side symmetries pay" regularity was
the project's one reliable guide, and this is its cleanest possible test — an *exact* symmetry,
a 15,000x larger augmentation group than color-sym, implemented with the same machinery, and
it moved nothing. So color-sym's gain was **not** "augmentation as such": more likely it was
acting as a *stabilizer* (the `--no-color-sym` ablation diverges on long runs) and the color
axis was the one the net was actually wasting capacity on. Once training is stable, piling on
more exact relabelings is not the bottleneck. Combined with A12 (a dedicated best response
could not exploit A8) and A10/A11 (capacity levers inert), the plateau increasingly looks like
a **ceiling of this information set**, not a self-play equilibrium trap or a data-efficiency
problem. The remaining lever with real headroom is the *observation* — what the agent can see
(the discard histogram was deliberately removed in A7) — not how its data is relabeled.

**Caveats.** Single seed (seed 1). A8's checkpoint was itself the max of a 101-snapshot screen,
so the control is a selection-maximum while the treatment's candidates were screened the same
way — symmetric in protocol, but both are optimistic.

**No snapshot promoted.** `snap225000`'s +1.60 is the *max* of 5 candidates whose mean is -0.14,
i.e. ~2 SE of selection noise; promoting it would launder a null into a fake win. Precedent: A10's
`snap375000` was the top post-floor snapshot in all three seeds at screening and then failed its
gate. **`checkpoint_a8_snap455000` remains the champion.**

**Default flipped to ON anyway (2026-07-11), on principle rather than evidence.** Nothing here says
rank-sym *hurts* — it says it is inert. Since the symmetry is exact and costs ~100us/batch, leaving
it off would mean deliberately keeping a rank-labelling bias that the rules do not contain, so
`--rank-sym` became `--no-rank-sym` (default on; ablation runs tag `_noranksym`). **Read the default
as bias removal at measured parity, NOT as a win** — and note the consequence: the no-flag recipe is
no longer bit-identical to the one that produced A8, so future A/B controls drift to color+rank,
which itself has only been validated at parity on one seed.

## 2026-07-11 — RESULT: A8 is near-unexploitable in this function class (A12 exploitability probe)

Branch `a12-exploitability-probe`. **Best-response training**: a fresh learner (same
architecture as A8, cold start) trained for **200,000 trials against 3 *frozen* greedy
`checkpoint_a8_snap455000` seats** — `train.py --freeze-opponents --opponent-model
checkpoint_a8_snap455000`. No opponent sync (the frozen seats never update), and transitions
are collected from the learner seat only (A8's all-seats collection is sound only when every
seat runs the learner's own policy; here the other three run a fixed foreign policy, so their
transitions would train the learner to *imitate* A8, not exploit it). Frozen seats play greedy
(epsilon=0) — exploitability is the best response to the champion's *actual* policy, which is
also what `eval.py` measures. Reward shaped, color-sym on, seed 0. This directly asks: **is the
~0.91 plateau a self-play equilibrium trap, or the ceiling of this function class?**

Evaluated all 21 snapshots (snap0000..snap200000, stride 10k) at **3000 games** each vs A8 and
vs random, sharded across 3 thread-capped workers (`eval_shard.py`; reuses `eval.py`'s
common-random-numbers `play_match`, which shuffles seating every game so the seat-0 advantage
averages out — vs-A8 above 0.25 would be a genuine win). SE ≈ 0.008 per point.

**Headline: the best response never reaches parity against A8.** vs-A8 win rate climbs from
0.007 (untrained) and plateaus from ~trial 70k onward, **peaking at 0.236 (snap180000) — 0.014,
or 1.8 SE, *below* the 0.25 parity line** — and ending at 0.213 (snap200000). Trained
specifically to beat A8 for 200k trials, it cannot even match it, let alone clear the +0.25 bar
that would signal real headroom. The same nets sit at ~0.89 vs random (vs A8's own 0.91),
confirming they are competent policies, just not superior ones.

| trial | vs A8 | vs random |   | trial | vs A8 | vs random |
|------:|------:|----------:|---|------:|------:|----------:|
| 0     | 0.007 | 0.068     |   | 110k  | 0.228 | 0.893 |
| 10k   | 0.132 | 0.840     |   | 120k  | 0.227 | 0.902 |
| 20k   | 0.136 | 0.850     |   | 130k  | 0.229 | 0.909 |
| 40k   | 0.146 | 0.870     |   | 140k  | 0.216 | 0.893 |
| 50k   | 0.178 | 0.874     |   | 150k  | 0.229 | 0.905 |
| 60k   | 0.204 | 0.888     |   | 160k  | 0.226 | 0.899 |
| 70k   | 0.218 | 0.900     |   | 170k  | 0.223 | 0.899 |
| 90k   | 0.218 | 0.899     |   | 180k  | **0.236** | 0.897 |
| 100k  | 0.227 | 0.894     |   | 200k  | 0.213 | 0.894 |

**Interpretation.** This is evidence for the *ceiling* hypothesis over the *equilibrium-trap*
hypothesis: a dedicated best-response cannot exploit A8, so the plateau is closer to the limit
of this representation + information set than an artifact of self-play failing to find the
exploit. It aligns with the flat parity bands seen in A10 (capacity) and A11 (dueling) — all
different function forms, all landing at 0.21–0.30 vs A8. **Implication for PLAN.md:** demotes
the opponent-pool / training-scheme levers (#1) relative to *representation* (belief/memory
features — the A7 obs dropped the full discard histogram) and *search*, which change the
function class rather than search harder within it.

**Caveats.** Single seed; 200k trials, not A8's 450–500k protocol (so the best response had
less training than its target) — but vs-A8 is flat from ~70k to 200k, so more trials are
unlikely to break 0.25 by a meaningful margin. A best response could in principle overfit to
A8's specific weaknesses and still miss a *general* improvement; here it found neither. Raw
per-snapshot numbers in `models/run1783692871.878205_bestresponse/eval_exploitability_*.tsv`.

---

## 2026-07-11 — RESULT (NEGATIVE): dueling head (A11) does not beat `checkpoint_a8_snap455000`, post-floor included

Supersedes the 2026-07-10 INCONCLUSIVE entry below. That entry's sole caveat was that no A11 run
had ever trained past the epsilon floor. **This run did, and the verdict does not change: dueling
is at parity. A11 is now a genuine negative result, not an open question.**

**Setup.** Same architecture as below (`Q = V + (A - mean_legal(A))`, advantage centred over legal
actions only, heads split off the 64-wide trunk with no per-stream hidden layer, so params stay
~equal to the flat net). Only change: **`--trials 450000`** instead of 550000, which moves the
epsilon floor to **360,000** (`0.8 * 450000`) so a run *completes* with ~90k post-floor trials
inside the window this box stays up for. 3 seeds, cold start, snapshot_every 25000, reward shaped,
color-sym on. Seeds 0 and 2 reached `snap450000`; seed 1 was killed by a VM wedge at trial 442,890
(last snapshot `snap425000`) — irrelevant to the verdict, since its post-floor snapshots
(375k/400k/425k) all exist.

**Screening** (Mode B, stride 25000, games-b 500, vs `checkpoint_a8_snap455000`, 53 snapshots):
the post-floor snapshots looked genuinely promising — vs-baseline mean **0.273** (10 of 11 above
the 0.25 parity line, range 0.246-0.288) versus a pre-floor mean of 0.263 in a noisy band straddling
parity. Better still, `snap375000` was the **top post-floor snapshot in all three seeds
independently** (0.288 / 0.286 / 0.282), which looks like replication rather than a lucky draw.

**Seat-swap-controlled head-to-head** (3000 games x2 orientations, duel @ seats {0,1} then {2,3},
so every occupant covers all four seats) vs `checkpoint_a8_snap455000`:

| Candidate | Edge (pts/seat) |
|---|---|
| seed 0, snap425000 | +1.00 |
| seed 2, snap375000 | +0.95 |
| seed 0, snap375000 | +0.15 |
| seed 1, snap375000 | -0.35 |

`snap375000`, the apparent 3-seed replication, gives **+0.15 / -0.35 / +0.95 — mean +0.25
pts/seat**, straddling zero. Best candidate overall is +1.00, below the **+2.0** promotion bar and
within ~1 SE (~1 pt/seat at 3000 games) of zero. Not promoted.

**Why screening read higher than the gate — and what it is NOT.** The first version of this entry
claimed the 0.273 post-floor screening figure was a seat-0 artifact. **That was wrong and is
corrected here.** `eval.py`'s `play_match` **shuffles seating per game**
([eval.py:81](eval.py#L81), `seat_rng.shuffle(order)`), so the test agent occupies the advantaged
opening seat in only ~25% of games — its parity share. **Mode B `--baseline` numbers are fair in
expectation, not inflated.** This confirms review item **R7** in PLAN.md. (The false claim
originated in the 2026-07-06 entry below and had spread to CLAUDE.md; `git log -S` shows the shuffle
landed in `e5feaa8`, the *original* eval-harness commit, a week **before** that entry — so it was
wrong when written. **Both have since been corrected and R7 is closed.** The seat-0 *discovery* is
real, but it applies to `eval_headtohead.py`, which genuinely takes fixed seats.)

The real explanation is duller: **the two numbers measure different things.** Mode B is a
**1-vs-3** win rate (one dueling net vs three A8 copies, parity 0.25); the gate is a **2v2**
per-seat rate. A modest edge does not map between them one-for-one, so "0.273 vs +1.00 pts/seat"
was never an apples-to-apples contradiction needing a confound to explain it. Both readings agree
on the substance: **slightly above parity, comfortably below the +2.0 bar.** The screening signal
was real but small — and small is not enough to promote. See R8 (winner's curse) for why the top
of 53 screened snapshots is expected to overstate.

Current best remains `checkpoint_a8_snap455000`. Combined with A10 (capacity, negative), the
**architecture lever is now closed on both halves**: neither more width nor a dueling
parametrization beats A8 with hyperparameters held fixed. Read alongside **A12** (same day: a best
response trained against frozen A8 never reached parity, so A8 is near-unexploitable in this
function class), the two results point the same way — the ~0.91 plateau looks like the ceiling of
the current *representation*, and neither a richer head nor a better opponent distribution moves
it. The next lever is the function class itself (PLAN.md R1/R4: rank-symmetry augmentation,
belief features).

## 2026-07-10 — INCONCLUSIVE: dueling head (A11) reaches parity, does not beat `checkpoint_a8_snap455000`

Branch `a11-dueling-head`. `create_model()` split the `124->64` trunk into a scalar state-value
`V(s)` and a per-action advantage `A(s,a)`, recombined as **`Q = V + (A - mean_legal(A))`** — the
mean taken over the **legal actions only**, not all 65, since most actions are illegal in any
given state and full-vector centring would fold untrained illegal-action advantages into every
legal Q. The heads split directly off the 64-wide trunk with **no per-stream hidden layer**, so
params stay ~equal to the old flat net (+65 for the V head): this isolates the dueling
*parametrization* from width, which A10 tested separately and found negative. All other
hyperparameters held constant (lr 1e-3, batch 64, buffer 80k, reward shaped, color-sym on).

Implementation note: the net gained a **second input**, a 65-slot legality mask consumed by a new
`DuelingAggregation` layer, threaded through `act()`, `replay()` (masks for both current and next
states, remapped under color-sym) and a new `cur_valid` transition field. Architecture change ->
invalidates every prior checkpoint; trained from scratch.

**Caveat that dominates this entry: no run ever reached the epsilon floor.** Three separate
550,000-trial attempts died at trial ~375k-402k — twice because 3 concurrent CPU-bound TF
processes wedged the WSL2 VM (~5h in; CPU pegged, snapshots stop, VSCode can't attach). Epsilon
bottoms out at 440,000 (`0.8 * 550000`), so **the post-epsilon-floor phase — where A8's gains were
concentrated — was never trained.** Everything below is from pre-floor snapshots.

**Screening** (Mode B, stride 25000, games-b 500, vs random and vs `checkpoint_a8_snap455000`,
48 snapshots across 3 seeds, all ending at `snap375000`): vs-random saturates at 0.90-0.93 by
`snap25000` and stays flat. vs-baseline clustered in a flat **0.234-0.306** band around the 0.25
parity line with **no upward trend across training** — the same shape A10 produced. Top point
estimates came from *mid*-training snapshots (75k, 225k, 350k), not the latest, which is the
signature of noise (SE ~= 0.019 at 500 games; 48 snapshots screened).

**Seat-swap-controlled head-to-head** (3000 games x2 orientations, duel @ seats {0,1} then {2,3},
so each occupant covers all four seats) vs `checkpoint_a8_snap455000`:

| Candidate (seed, trial) | Edge (pts/seat) |
|---|---|
| seed 2, snap350000 | +1.40 |
| seed 1, snap225000 | +0.20 |
| seed 0, snap75000  | +0.02 |

Best candidate **+1.40 pts/seat**, below the **+2.0 pts/seat** promotion bar (the smallest margin
ever actually promoted on, `checkpoint_colorsym_snap180000`'s +2.00). Not promoted. Two reasons
the +1.40 is weaker than it looks: it was the top of 48 screened snapshots (selection bias), and
the edge's SE is ~1 pt/seat at this sample size, so it sits ~1.4 SE from zero. The seat-0 confound
was clearly present and correctly cancelled (seat 0 scored ~0.27-0.28 in *both* orientations
regardless of occupant).

An earlier interrupted attempt's `snap400000` (seed 1) scored **-0.65 pts/seat** under the same
swap-controlled protocol — consistent with parity. (Those run dirs were deleted before a relaunch,
so that checkpoint no longer exists; only the number survives.)

Current best remains `checkpoint_a8_snap455000`. **Verdict: parity, not a negative result.** Unlike
A10 (which ran to completion), A11 never trained through the phase most likely to matter. The
cheap follow-up is `--trials 450000` (moving the epsilon floor to 360k) so a run actually
*completes* with ~90k post-floor trials and lands next to A8's own `snap455000` — see PLAN.md.

**Side finding, fixed in-branch:** `act()` unconditionally passed the new two-input `[obs, mask]`
signature, which crashed with `Layer sequential expects 1 input(s), but it received 2` whenever a
dueling agent played a legacy single-input checkpoint adopted via the shape-mismatch load fallback
— i.e. *every* head-to-head against `checkpoint_a8_snap455000`. Fixed by tracking `self.dueling`
from the resolved model's input count and branching the forward call (commit `8a49d04`).

## 2026-07-08 — RESULT: bigger network (A10) does not beat `checkpoint_a8_snap455000`

Branch `a10-bigger-network`. `create_model()` widened `147->124->64->65` ->
`147->256->128->64->65` (30,577 -> 83,265 params, 2.72x), isolated from the deferred
dueling-head idea so the capacity effect isn't confounded with it (dueling remains open,
see PLAN.md). All other hyperparameters held constant (lr 1e-3, batch 64, buffer 80k, reward
shaped, color-sym on) per single-variable methodology.

3 seeds x 550,000 trials (snapshot_every 25,000), cold start (no `--model`). A mid-run power
outage killed all three processes at seed-dependent points between trial ~456k-471k; all
snapshots through `snap450000` were confirmed intact and reloadable, so each run resumed
from `snap450000` with `--epsilon-start 0.1` (epsilon had already decayed to its floor at
trial 440,000 = `0.8 * 550000`, so this continues the original schedule exactly, not an
approximation) for the remaining 100,000 trials.

**Side finding, fixed in-branch:** loading `checkpoint_a8_snap455000` (or any pre-A10
checkpoint) crashed under the new architecture — `AIAgent.__init__` always rebuilds via
today's `create_model()` and force-applies the loaded weights, so any shape mismatch threw
an unguarded `ValueError` from `set_weights()`. This blocked comparing new snapshots against
the old baseline entirely (`eval.py` and `eval_headtohead.py` alike), not just `train.py`'s
warm-start path. Fixed by falling back to the checkpoint's own saved architecture on a shape
mismatch — safe for eval (epsilon=0, no replay/optimizer use); same-shape loads (the normal
warm-start case) are unaffected.

**Screening** (Mode B, stride 25000, games-b 500, vs random and vs `checkpoint_a8_snap455000`,
66 snapshots total across 3 seeds): vs-baseline win rate clustered in a flat 0.22-0.29 band
around the 0.25 parity line for essentially every snapshot — no candidate separated from the
pack. Top 6 by point estimate (including seed 2's best, which never exceeded 0.258, to check
replication) were shortlisted for the swap-controlled gate.

**Seat-swap-controlled head-to-head** (3000 games x2 orientations) vs `checkpoint_a8_snap455000`:

| Candidate (seed, true trial) | Edge (pts/seat) |
|---|---|
| seed 0, snap100000 | -1.40 |
| seed 0, snap550000 | -0.35 |
| seed 1, snap175000 | +0.35 |
| seed 1, snap325000 | +0.05 |
| seed 1, snap550000 | +0.65 |
| seed 2, snap300000 | -1.75 |

Best candidate: +0.65 pts/seat (seed 1, snap550000) — well below the +2.0 pt/seat promotion
bar (the smallest margin ever actually promoted on, `checkpoint_colorsym_snap180000`'s +2.00).
Not promoted. Notably, the mildly promising-looking raw seat-0 numbers from the un-swapped
comparison (0.26-0.28, i.e. above the 0.25 parity line) mostly evaporated once seat-swapped —
consistent with CLAUDE.md's standing seat-0-confound finding, not a real skill edge.

Current best remains `checkpoint_a8_snap455000`. Caveat: this tests capacity alone, holding
lr/batch/buffer fixed — it's possible a bigger network needs those retuned (e.g. a lower LR,
larger buffer) to actually benefit from the extra capacity, so this doesn't rule out capacity
as a lever, only this specific unchanged-hyperparameters version of it. Dueling head (state-value
+ advantage streams) remains a separate, untested follow-up idea in PLAN.md.

## 2026-07-06 — PROMOTION: new best `checkpoint_a8_snap455000` (A8)

Promoted from the 500k-trial A8 run (`models/run1783298662.882489`, seed 1) after screening
its snapshot progression for a peak, since the *final* snapshot is not reliably the
strongest one (precedent: `checkpoint_colorsym_snap180000` beat its own run's later
snapshots too).

**Screening process:**
1. Broad scan, `eval.py` Mode B, stride 25000 (21 snapshots), 500 games each, vs random and
   vs `checkpoint_a4a7_snap550000` baseline (test agent always seat 0 vs baseline at seats
   1-3 — confounded by the seat-0 effect below, but *consistently* so across all snapshots,
   which keeps the **relative ranking** valid even though the absolute numbers are inflated).
   Top region: snap180000-225000 and snap450000-500000.
2. Fine scan, stride 5000 (101 snapshots), same setup. Top 5 by vs-baseline win rate:
   snap365000 (0.304), snap285000 (0.302), snap455000 (0.298), snap315000 (0.296),
   snap180000 (0.294) — all within noise of each other at 500 games (SE ≈ 0.022).
3. **Seat-swap-controlled head-to-head** (the decisive step — see next section) at 3000
   games for these 5 candidates vs `checkpoint_a4a7_snap550000`: each pair run twice
   (candidate at seat 0 / baseline at seats 1-3, then swapped) so each model gets equal
   exposure to the advantaged and disadvantaged seat positions; final score is the average
   of each model's seat-0 rate and its seats-1-3 rate.

| Snapshot | Candidate avg per-seat | Master avg per-seat | Edge |
|---|---|---|---|
| **snap455000** | **0.278** | 0.241 | **+3.68 pts** |
| snap285000 | 0.274 | 0.243 | +3.03 pts |
| snap315000 | 0.269 | 0.243 | +2.60 pts |
| snap365000 | 0.269 | 0.247 | +2.18 pts |
| snap180000 | 0.267 | 0.247 | +2.00 pts |

`snap455000` won by the largest and most consistent margin (strong in both the solo-seat-0
role, 0.306, and the 3-seat-minority role, 0.250 avg) and is promoted to
`models/checkpoint_a8_snap455000`. Confirming vs-random eval on the copied checkpoint:
**0.912** @ 500 games (456/500, seed 0) — bit-identical to the fine-scan reading for the same
snapshot, confirming the copy is correct.

**vs current-best comparison:**
- vs random (500 games): A8 snap455000 0.912 vs master 0.909 (near parity; not the
  discriminating readout here)
- head-to-head (seat-swap-controlled, 3000 games x2 directions): **+3.7 points** — the
  decisive signal, consistent with the color-sym precedent where head-to-head, not
  vs-random, was the readout that actually separated near-equal snapshots.

Caveat: the swap-control only cancels the seat-0 confound for a **homogeneous-opponent**
table (1 model vs 3 copies of the other, in both directions) — it does not test a mixed
4-model table, which hasn't been run.

## 2026-07-06 — DISCOVERY: seat-0 first-mover advantage confounds naive head-to-head evals

> ### ⚠ CORRECTION (2026-07-11): the `play_match` half of this entry was WRONG WHEN WRITTEN.
> This entry asserts that `eval.py`'s Mode A/B `play_match` "always seats the test agent at seat 0".
> **It does not, and never did.** `play_match` has shuffled seating per game
> (`seat_rng.shuffle(order)`, [eval.py:81](eval.py#L81)) since the *original* eval-harness commit
> `e5feaa8` (2026-06-29) — a week **before** this entry was written. The test agent occupies the
> advantaged opening seat in ~25% of games, exactly its parity share, so **Mode A/B numbers —
> including Mode-B `--baseline` head-to-heads — are fair in expectation, NOT inflated.**
> The "Not yet fixed" paragraph at the end of this entry is therefore void, and the PLAN.md
> follow-up it spawned ("fix seat-0 bias in eval.py") targeted a non-bug and has been deleted
> (see PLAN.md **R7**).
>
> **The seat-0 discovery itself is real and unaffected.** Seat 0 does win more regardless of
> occupant; it bites `eval_headtohead.py`, which genuinely takes explicit *fixed* seats. The
> seat-swap protocol remains mandatory there, and every promotion decision made with it stands.
>
> This error was not harmless: on 2026-07-11 it was used to wrongly dismiss A11's screening
> signal as a "seat-0 artifact" (see that entry's own correction). Original text kept below,
> uncorrected, as the historical record.

While running head-to-head comparisons of the 500k A8 run's final checkpoint against
`checkpoint_a4a7_snap550000`, found that `Game.reset()` ([game.py:392](game.py#L392))
defaults `start_seat=0` and **every** eval game uses that default — so the "test agent"
seat (always seat 0 in both `eval.py`'s `play_match` and the first head-to-head script) has
a real, model-independent scoring advantage. Confirmed across 14 seat configurations
(solo-seat sweep at all 4 seats, all 3 distinct 2v2 partitions in both directions): seat 0
was the single highest-scoring seat in nearly every run, **regardless of which model
occupied it** — e.g. Master-at-seat-0 outscored A8-at-seats-1-3 even in configurations where
A8 is the stronger model by every other measure.

Fix: `eval_headtohead.py` (new script) takes an explicit `--team1-seats` seat list so any
assignment can be scripted; comparing two checkpoints fairly requires running **both seat
assignments** for the pair being compared and averaging same-seat, swapped-occupant win
rates (or a full seat-permutation set). This is now documented in CLAUDE.md's "Evaluation
discipline" section as a standing eval requirement, and was the method used for the A8
snapshot-promotion screen above.

Not yet fixed: `eval.py`'s Mode A/B `play_match` still always seats the test agent at seat
0 — fine for a single model's vs-random rate (no second model to be biased against) or for
*relative* ranking within one run (bias is constant across snapshots), but any one-off
Mode-B `--baseline` head-to-head number should be read as inflated in the test agent's favor
until `eval.py` itself is updated to swap or randomize seats (not done; tracked as a PLAN.md
follow-up).

## 2026-07-05 — CHANGE: learn from all four seats (A8) + buffer 20k→80k

Branch `a8-learn-from-all-seats`. The self-play loop in `train.py` now stores **all four
seats'** transitions in the learner's replay buffer, not just seat 0's. Every seat plays the
same DQN policy (opponents synced every 5 trials) and the observation is egocentric
(`observation(agent=i)` / `valid_moves(agent=i)` are seat-relative), so a transition recorded
from any seat is valid learner training data — DQN is off-policy, so the ε mismatch (learner
anneals from high, opponents fixed 0.1) is immaterial. **~4× data per trial at ~zero extra
cost** (the extra work is a few pure-Python `observation`/`valid_moves` calls; the TF
`replay()` cadence is unchanged).

Implementation: a turn-by-turn loop keeping one *pending* transition per seat, closed when
that seat is next about to act (`new_state`/reward/`next_valid` measured at that moment) or
when the round ends (terminal close). This reproduces the old seat-0 transitions with
identical semantics — including multi-card TAKI/PLUS/KING turns, each still its own
transition — and adds seats 1–3. Only seat 0 feeds the reward/win plots.

**Replay ratio recomputed (the A8 ask):** collection rises from ~25 → ~100 transitions/trial
(steady state; early ε=1.0 runs are longer, ~55/seat observed), replay cadence held fixed, so
the ratio drops **~20 → ~5** (near Atari's ~8) — same gradient budget over 4× more diverse,
faster-refreshing data. Buffer bumped **20k → 80k** (`agents/dqn.py`) to keep the ~800-trial
history horizon the 20k buffer gave under seat-0-only collection.

Sanity-checked: `python -m unittest gametest` (27 tests, unaffected — no `game.py` change);
short seeded runs collect from all seats and run clean on the multi-card and
opponent-wins-first paths. **Not yet evaluated** — needs the standard 10k-trial screen (2–3
seeded pairs vs a current-`master` control) on the three readouts (3000-game vs-random,
head-to-head vs the control snapshot, Mode-B vs `checkpoint_shaped_snap300000`) before any
promotion.

---

## 2026-07-05 — FEATURE: King card added (encoding 62/64/201 → 63/65/205)

Added the two King cards to the deck (branch `feature/king-card`), per original Taki: a
colorless wild that (outside a TAKI) **cancels a pending +2** (`draw_num → 0`, no cards drawn)
and grants one **optional** follow-up card of any color/type (declined via CLOSE_TAKI; no DRAW
during the continuation). King = card slot 62 / action 62 / `State.KING`; the colored-block
stride is now the named constant `TYPES_PER_COLOR` (15), decoupled from `len(Type)` (=16) so
the wild-only King doesn't shift the colored slots. Rules: follow-up optional, Kings chain,
**you CAN win on a King** (`FINISHING_TYPE_VALUES` = numbers + King), and the King IS playable
inside an open TAKI but is **inert there** (no color change, no continuation) — only if the
TAKI is closed ON a King does the follow-up fire. See PLAN.md for the full interpretation.

**Encoding change → all pre-King checkpoints are unloadable** (obs 201→205, actions 64→65,
card vector 62→63, `len(State)` 7→8). Last old-contract commit that still loads the pre-King
models (incl. `checkpoint_colorsym_snap180000`) is **`344535a`** on `master`.

Verification (correctness, not a quality bar):
- `python -m unittest gametest` → **26/26 pass**, incl. 7 new King tests (cancels +2, legal
  +2 response, one optional follow-up, decline-via-close, inert-inside-TAKI, close-on-King
  grants follow-up, can-win-on-King) and the updated color-symmetry perm-table tests (tables
  now (24,205)/(24,65); King slot/action are color-invariant fixed points).
- Smoke train `--trials 1500 --reward shaped` (color-sym on): ran end-to-end, no shape errors.
  Eval of the fresh net: **0.855 vs 3 random @ 1000 games** (chance 0.25) — confirms the
  pipeline still learns with the King in the deck. A proper strong model needs a full retrain
  (deferred), since every prior checkpoint was invalidated.

## 2026-07-05 — PROMOTION: new best `checkpoint_colorsym_snap180000` (color-sym)

Promoted the peak snapshot of the 300k color-sym run (`run1783109653.234119_colorsym/snap180000`)
to `models/checkpoint_colorsym_snap180000`, replacing `checkpoint_shaped_snap300000` as the
current best. Confirming eval (3000 games, seed 0, CRN — bit-identical to the 2026-07-04 A/B):
**0.926 vs random** and **0.292 head-to-head vs the old champion** (> 0.25 parity, z ≈ 5). It is
the strongest snapshot on hand — stronger than any 100k color-sym run (0.913 / 0.270). The old
`checkpoint_shaped_snap300000` (1M-trial run, ~0.91 vs random) is retained for reference/baseline.

---

## 2026-07-05 — 3-seed 100k A/B: color-sym stable and beats champion; **vanilla control DIVERGES**

Purpose: replicate the color-sym advantage across seeds at the 100k scale, on the current
post-S1–S8 branch code (`exp-color-sym` worktree, tip after the DQN-hygiene commits, all
hygiene flags default-off). Six runs, identical except `--color-sym`:
`--trials 100000 --reward shaped --snapshot-every 10000 --epsilon-start 1.0 --seed {0,1,2}`.
Thread-capped (`OMP/OPENBLAS/MKL=1`, `TF_INTRA=2/INTER=1`) after an earlier unthrottled
6-way launch died — six uncapped `train.py` each grab ~8 cores and oversubscribe/OOM the
6-core box. All eval at seed 0, 3000 games unless noted.

Run dirs: color-sym `…357439_colorsym` (s0), `…360439_colorsym` (s1), `…358192_colorsym`
(s2); control `…369189` (s0), `…361702` (s1), `…375708` (s2) (all `models/run1783181329.*`).

**Artifacts preserved** (the `exp-color-sym` worktree these ran in was retired 2026-07-05;
its `models/` was relocated into the main worktree, gitignored): the six trained run dirs at
`models/run1783181329.*` (11 snapshots each, snap0000→snap100000); their final checkpoints,
per-run training-curve PNGs, and all A/B eval logs (`AB_final_eval.txt`, `AB_control_eval.txt`,
the `ab_s*_*.log` training logs and `launch_ab.sh`/`auto_eval*.sh` scripts) under
`models/_colorsym_ab_logs/`.

**Color-sym (snap100000) — stable and strong, all three seeds:**

| seed | vs random | vs champion `snap300000` (head-to-head) |
|---|---|---|
| 0 | 0.911 (2734/3000) | 0.279 (838/3000) |
| 1 | 0.918 (2755/3000) | 0.266 (799/3000) |
| 2 | 0.909 (2726/3000) | 0.264 (791/3000) |

vs-random ~0.913 (above the 0.907 champion reference); vs-champion **pooled 2428/9000 =
0.270, z ≈ 3.8** over 0.25 parity — color-sym at **100k** trials beats the 1M-trial champion
in every seed. Consistent with the 2026-07-04 300k result, at 1/3 the trials.

**Vanilla control (snap100000) — diverged, worse than random:**

| seed | vs random | color-sym vs control (head-to-head) |
|---|---|---|
| 0 | (not evaluated — run finished ~5 h late, see below) | — |
| 1 | 0.038 (92/2411, 589 undecided) | 0.933 (2798/3000) |
| 2 | 0.002 (6/2605, 395 undecided) | 0.9997 (2996/3000) |

**Divergence trajectory (control seed 1, 1000 games/snapshot vs random):**

| snap | 0 | 10000 | 20000–50000 | 60000 | 70000–100000 |
|---|---|---|---|---|---|
| vs random | 0.810 | **0.860** | **0.000 (all undecided)** | 0.132 | 0.00–0.05 |

The control run climbs normally to snap10000 (0.86) then **collapses between 10k and 20k**:
win rate → 0 with **every game hitting the turn cap (undecided)** — a degenerate greedy
policy stuck in non-terminating loops, the textbook signature of **Q-value divergence
(overestimation blowup)**. It never recovers. Both control seeds show it; color-sym on
identical code does not. This is exactly the instability PLAN.md **A1/A2** predicted (target
network effectively disabled + vanilla max-Q). Color-sym augmentation evidently **damps**
it (regularization / ~4× effective data).

**Caveats / open puzzle.**
- The 100k control is **not a valid skill baseline** — it's a diverged run, so the
  color-sym-vs-control head-to-head (0.93 / 0.9997) is real but **confounded**: color-sym
  isn't out-skilling a *trained* vanilla agent, it's staying stable while vanilla
  self-destructs. The vs-random and vs-champion color-sym numbers do **not** depend on the
  control arm and stand on their own.
- **Discrepancy with 2026-07-04:** that 300k control (0.892, no divergence) ran on
  *pre-S1–S8* code. This 100k control (same `color_sym=False`, all hygiene flags off) is on
  the newer branch and diverges two-for-two. Either an S1–S8 change shifted training
  dynamics (S3's `valid_moves` dedup alters the exploration + `max(next_valid)` target
  distribution — prime suspect) or divergence is partly stochastic; two seeds argue against
  pure luck. Unresolved — worth a bisect.
- Control finished ~5 h after color-sym because weaker/looping play → longer games → more
  learner steps/trial, *and* two unrelated `--color-sym --seed 2` processes (one a
  Double-DQN/Huber hygiene run) were competing for CPU during the run.

**Decision:** color-sym is verified across seeds and adopted as an integral part of training
(→ make it default, per the PLAN.md item). **Next experiment:** does the DQN-hygiene package
(Double DQN + Huber + slow/stepped target) prevent the vanilla divergence? — i.e. re-run the
control arm with `--double-dqn --loss huber --target-sync-mode steps --target-sync-every 2000`.

**Update — cross-reference to the DQN-hygiene branch (`exp-dqn-hygiene`, 2026-07-05).** That
parallel line ran the hygiene package as a 10k ablation + a 100k full-bundle A/B, always with
`--color-sym` on. Two conclusions bear on this entry: (1) **Corroboration** — its
color-sym control reproduced *these exact numbers* (vs-random .913 pooled; vs-champion .270,
+5.2 SE), independently confirming "100k color-sym beats the 1M-trial champion." (2)
**Resolution of the open question above** — the A1/A2 stability levers (Double DQN + slow
target) are **inert** (10k head-to-head 0.252, +0.4 SE; nothing at 100k); only Huber+reward÷10
helps, and only ~+1.5 pt vs-random. Since those levers do nothing *once color-sym is present*,
the evidence is that **color-sym itself is the long-run stabilizer** that A1/A2 were meant to
be — consistent with vanilla diverging here while color-sym does not. The DQN-hygiene package
is **not** the plateau-breaker; the lead levers are now structural (richer observation / bigger
network) + an opponent pool. See PLAN.md action-items #1–3 (marked done/inert).

---

## 2026-07-04 — Color-symmetry at 300k trials BEATS the 1M-trial champion; control plateaus

Follow-up to the 2026-07-03 `--color-sym` entry, at the 100k–300k scale it flagged as
"next". Two fresh runs, identical except the flag: `--trials 300000 --reward shaped
--snapshot-every 10000`. Control `run1783109650.848521`, color-sym
`run1783109653.234119_colorsym`. Both trained on identical pre-S3 code (launched before
the S1–S8 branch), so they are directly comparable to each other; **both evaluated under
the current post-S3 eval code**, so vs-random uses the 0.907 reference and the champion
`checkpoint_shaped_snap300000` is the frozen baseline. Eval seed 0. Best snapshot per run
picked from the Mode-B curve: color-sym **snap180000**, control **snap300000** (the control
policy converged — see below — so any late snapshot is equivalent).

**Headline (3000-game evals):**

| matchup | control best | color-sym best | notes |
|---|---|---|---|
| vs random | 0.892 (2675/3000) | **0.926 (2777/3000)** | color-sym +3.4 pts, z ≈ 4.6; and **above the 0.907 champion reference** (z ≈ 2.6) |
| vs champion (head-to-head, over 0.25 parity) | 0.211 (633/3000) | **0.292 (875/3000)** | control **loses** to the champion (z ≈ −5.2 *below* parity); color-sym **beats** it (z ≈ +5.0 above) |
| color-sym best vs control best (head-to-head) | — | **0.345 (1035/3000)** | +9.5 pts over 0.25 parity, **z ≈ 11** — decisive |

**The result:** a 300k-trial color-sym run surpasses `checkpoint_shaped_snap300000` — the
previous best, which took **1M** trials — both on the vs-random yardstick (0.926 > 0.907)
and in a direct 3000-game head-to-head (0.292 > 0.25, z ≈ 5). The equal-length control run
never reaches the champion (0.211 < 0.25). **~3.3× more sample-efficient** at matching, and
then exceeding, the old best.

**Control plateau (behavioral convergence).** In the 300-game Mode-B sweep every control
snapshot from snap90000 → snap300000 returned *byte-identical* rates (0.850 vs random,
0.240 vs champion). The snapshot weights **differ** (distinct md5s) — so this is not frozen
weights or an eval bug (color-sym, same eval code, keeps varying and climbing). It is a
**converged greedy policy**: the net keeps drifting but its argmax action in every state on
these common-random-number games stops changing, so play is identical. Control settles just
*below* the champion and stays there; color-sym keeps improving, peaking at snap180000.
(Note: the first-300-game window undersold control's vs-random rate — 0.850 there vs 0.892
over the full 3000 — but the ranking color-sym > champion > control holds at 3000 games.)

**Mode-B progression, vs random / vs champion (300 games/snapshot, coarse):**

| snapshot | control vs rand | color-sym vs rand | control vs champ | color-sym vs champ |
|---|---|---|---|---|
| snap90000 | 0.850 | 0.903 | 0.240 | 0.250 |
| snap120000 | 0.850 | 0.900 | 0.240 | 0.280 |
| snap180000 | 0.850 | **0.947** | 0.240 | **0.333** |
| snap300000 | 0.850 | 0.913 | 0.240 | 0.247 |

color-sym clears the champion (>0.25 vs baseline) across snap120000–snap270000; control
never does. (300-game figures are noisy — the 3000-game table above is authoritative for
ranking; e.g. snap180000-vs-champion is 0.333 at 300 games, 0.292 at 3000.)

**Caveats.** Still **n = 1 training pair**, unseeded. But the effect is now large and
multiply-confirmed: +3.4 pts vs random (z ≈ 4.6), a z ≈ 11 direct head-to-head, and — the
qualitative jump — color-sym *beats the 1M-trial champion* at <⅓ the trials while the
control plateaus short of it. The `--seed` flag (S8) now exists, so the clean next step is
2–3 **seeded** replicate pairs to pin the effect size and confirm the control-plateau /
color-sym-keeps-climbing divergence reproduces. Worth also re-timing: color-sym snapshots
kept moving out to snap180000, so a longer color-sym run may go further still.

---

## 2026-07-04 — Review fixes S1–S8 (branch `review-fixes-s1-s8`); vs-random baseline must be re-measured

Code-only follow-ups to the 2026-07-03 full review (see [PLAN.md](PLAN.md) for the S1–S8 /
A1–A9 findings). No training or eval was run here — that is a separate experiment.

**One change shifts the vs-random baseline: S3.** `valid_moves` now deduplicates exact-duplicate
`(Action, Card)` moves (`dict.fromkeys`), so a uniform chooser (`RandomAgent`, epsilon-exploration)
is uniform over *distinct* moves instead of over move *instances* — previously it over-weighted
duplicated cards in hand. **Greedy DQN play is bit-identical** (duplicate moves share a Q-value, so
`argmax` is unchanged), which means **model-vs-model head-to-head evals are unaffected**, but the
**vs-random win rate shifts** because the random opponents now play a different distribution.

**New baseline measured (post-dedup):** `checkpoint_shaped_snap300000` vs 3 random,
**0.907 (9068/10000), 0 undecided**, seed 0, SE ≈ ±0.003. This is if anything ~+0.6 pt above the
pre-dedup figures (~0.897–0.901 at 3000 games, older RandomAgent) — well within a couple SE, i.e.
the S3 dedup did **not** degrade the vs-random rate; deduping just makes the random opponents
slightly more uniform over distinct moves. Use **0.907** as the vs-random reference on this branch.
Head-to-head chains and Mode-B `--baseline` comparisons carry over unchanged (greedy play is
bit-identical).

Other fixes (no behavior change): S1 raise instead of print on an illegal play; S2 distinct deck
objects (counts unchanged); S4 comment on the unreachable colorless-CHCOL action scalar; S5 removed
dead `Card.amount()`; S6 eval mode-A `--games` default → 3000 + sub-precision warnings; S7
`--trial-len`/`--target-sync-every` flags; S8 `--seed` (verified: two `--seed 123` runs → identical
weights). Tests: `gametest` 19/19.

## 2026-07-03 — Color-symmetry replay augmentation (`--color-sym`): +3.3 pts vs random, wins head-to-head

**Idea (PLAN.md:9).** TAKI's four colors are interchangeable — only color *consistency*
matters — so every transition is equivalent under any of the 24 (4!) color relabelings.
The DQN never exploited this: each buffered transition is replayed ~16×, always with the
same color realization. New `--color-sym` flag augments each *sampled* transition in
`AIAgent.replay()` with one uniformly-random color permutation (identity included),
applied consistently to state / new_state / action / next_valid (reward and done are
color-invariant). Chosen over canonicalization (discontinuous input map) and over
expanding the buffer in `remember()` (would shrink the 20k horizon ~24×); per-draw
augmentation turns each of the ~16 replays of a transition into a different recoloring for
free.

**Implementation.** `game.py` builds `OBS_PERMS` (24×201 gather arrays) and `ACT_PERMS`
(24×64 forward maps) once at import; the observation table is the *inverse* of the action
forward map (gather vs scatter — the one real pitfall, `aug[f[i]]=orig[i]`). `replay()`
does two `np.take_along_axis` gathers on the batch + a per-row action relabel. New
`ColorSymmetryTest` (6 tests) proves equivariance directly: recoloring a real mid-game
state and re-encoding equals permuting the original encoding, for all 24 perms
(`gametest` 19/19). **Overhead: none measurable** — 200 trials 21.5 s (off) vs 20.6 s
(on); the two float gathers are noise next to the TF step.

**A/B setup.** Two fresh runs, identical except the flag: `--trials 10000 --reward shaped
--snapshot-every 500`. Control `run1783094805.690908`, treatment
`run1783094809.117728_colorsym` (each run's `config.txt` records the args). Eval seed 0.
This is the deliberately small "quick signal" scale — 10k trials, one pair.

**Headline (3000-game evals, best snapshot = snap10000 for both):**

| metric | control | color-sym | gap |
|---|---|---|---|
| vs random (Mode A) | 0.866 (2597/3000) | **0.899 (2696/3000)** | **+0.033, z ≈ 4.0** |
| head-to-head: 1 color-sym seat + 3 control seats | — | **0.323 (969/3000)** | **+0.073 over 0.25 parity, z ≈ 8.6** |

The head-to-head is the decisive one: drop the color-sym model into a table of three
control copies and it wins 32.3% of games where equal skill scores 25% — it genuinely
beats the control *policy*, not just a shared random opponent. All 3000 games decided, 0
undecided.

**Progression (Mode B, 300 games/snapshot vs random and vs the frozen current-best
`checkpoint_shaped_snap300000`):** color-sym is ahead at essentially every late snapshot.

| snapshot | control vs random | color-sym vs random | control vs best | color-sym vs best |
|---|---|---|---|---|
| snap5000 | 0.843 | 0.867 | 0.193 | 0.243 |
| snap7000 | 0.873 | 0.897 | 0.190 | 0.220 |
| snap9000 | 0.860 | 0.893 | 0.180 | 0.233 |
| snap10000 | 0.857 | 0.900 | 0.203 | 0.233 |

(Both runs stay *below* 0.25 vs the 1M-trial best — expected at 10k trials; neither has
caught the current champion, but color-sym closes the gap faster.)

**Caveats.** (1) **n = 1 training pair**, both unseeded; historical run-to-run variance is
~0.02 vs random, so the +0.033 Mode-A gap alone is ~1.6σ of *training* noise even though
it is ~4σ of *eval* noise. What lifts this above "lucky run" is the convergence of three
independent readouts: the 3000-game vs-random gap, the decisive z≈8.6 head-to-head (a
direct model-vs-model ranking, largely immune to the shared-opponent variance), and
color-sym leading at nearly every Mode-B snapshot. (2) A definitive *effect-size* estimate
still needs replicate pairs and/or a longer horizon. **Next:** run the A/B at 100k–300k
trials (and ideally 2–3 seeded pairs) to see whether the gain compounds toward / past the
current best, and whether augmentation shifts where greedy skill plateaus.

## 2026-07-03 — Eval RNG fix: per-game common random numbers; reproducibility fixed, no extra comparison precision

**Bug (found in the 2026-07-02 code review):** eval.py Mode B created ONE `RandomAgent(seed)`
and reused it across every snapshot matchup, so its choice stream carried over — snapshot k's
vs-random games depended on how snapshots 1..k−1 consumed the stream. The documented
"snapshots are compared on identical games" guarantee only covered decks/seating, not the
opponents' choices, and a result silently depended on the snapshot's *position in the sweep*
(and on `--snap-stride`).

**Fix:** reseed the opponent **per game** — `RandomAgent.reseed(f'{seed}:{g}:opp')` in
`play_match` [eval.py] — alongside the existing per-game deck seed (`seed+g`) and per-matchup
seating stream. Now game g replays identical randomness in every matchup sharing `--seed`,
regardless of the model under test or how games 0..g−1 unfolded (true common random numbers);
greedy net opponents are deterministic and need no reseed.

**Experiment A — same model in 10 Mode-B slots** (10 × `checkpoint_shaped_snap300000`,
1000 games each, seed 0). Any spread = pure opponent-stream artifact:

| | rates across the 10 duplicate slots | sd | spread |
|---|---|---|---|
| before | 0.885 … 0.922 | 0.0132 | 0.037 |
| after | **0.897 ×10 (bit-identical)** | **0.0000** | 0.000 |

Before the fix the artifact was *larger than the binomial SE at 1000 games* (±0.0094) — a
snapshot could gain/lose ~2–4 pts vs random purely from sweep position. After the fix, a
(model, seed, games) triple is one deterministic number. **Reproducibility: fixed, verified.**

**Experiment B — does CRN also tighten model *comparisons*?** best vs prior-best
(`shaped_snap300000` vs `shaped_snap10000`) in one Mode-B sweep, 300 games/matchup,
seeds 0..39; metric = sd across seeds of the paired vs-random difference. **No:**
sd(diff) 0.0252 → 0.0259 (ratio 1.03), pairing correlation ρ̂ = 0.04 before / −0.02 after
(SE ≈ 0.16), and each model's seed-to-seed sd matches pure binomial noise. Mechanism: two
policies diverge at their first differing decision and the trajectories decorrelate — common
decks/streams can't couple the outcomes. Not extended past 40 seeds: the correlation channel
(the only mechanism by which pairing could cut variance) measures empty in both arms.

**Takeaways:** (1) vs-random numbers are now exactly reproducible and independent of sweep
composition — before, up to ~±1.3 pts (sd) was position artifact. (2) CRN does **not** buy
comparison precision here; the **≥3000 games** rule for ranking near-equal snapshots stands
unchanged. (3) Calibration going forward: vs-random rates shift *within SE* under the new
stream scheme (best: 0.901 → 0.897 at these game counts); historical log numbers remain
valid/unbiased, just not bit-reproducible. Mean best-vs-prior gap was ~+2.4 pts vs random in
both arms — estimates unbiased before and after.

Same session, non-eval fixes: main.py demo previously created its DQN agents with the default
`epsilon=1.0` — i.e. **pure random play even with `--model`**; it now plays greedily when a
model is given (random otherwise, since greedy random-weight nets stall) and is fully seeded
(two runs replay bit-identical games). Verified: gametest 13/13; demo with the best model
finishes 4/4 games, play-dominated (243 plays / 140 draws).

## 2026-07-01 — Long shaped run (1M trials): real gain to ~0.90 vs random, then plateau

Continued **pure shaped** from `checkpoint_shaped_snap10000`, `--epsilon-start 0.1`,
`--snapshot-every 10000`, cap `--trials 1000000`. Ran to completion: **1M trials, 13.1 h,
~0.047 s/trial, 101 snapshots, healthy throughout** (hourly heartbeat + periodic spot-checks).
Peak-finding eval at 3000 games (SE ≈ ±0.0079), stride 100000, baseline = the start model
(`checkpoint_shaped_snap10000`):

| snap (×1000) | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vs random | .869 | .900 | .888 | .901 | .892 | .902 | .901 | .900 | **.906** | .892 | .900 |
| vs start-best | .243 | .283 | .284 | **.305** | .285 | .287 | .291 | .294 | .289 | .286 | .297 |

**Two real gains, then saturation:**
- **vs random jumps ~0.869 → ~0.90** for every snapshot from 100k on (+~3 pts absolute). This
  updates the earlier "vs-random saturates ~0.86" claim — with *far* more trials it reaches ~0.90.
  The plateau isn't where we thought; it's ~0.90.
- **vs the start-best, all long-run snapshots are +4 to +7 SE above parity**; the gain is real,
  not noise. Peak **snap300000 = 0.305** vs the prior best (also 0.901 vs random).
- But **100k→1M is flat** (vs-best 0.28–0.31, vs-random 0.888–0.906, all within ~2–3 SE): the
  improvement **saturated by ~snap100–300k**; the remaining ~700k trials (~9 h) added nothing.

**New best (promoted): `models/checkpoint_shaped_snap300000`** — beats the prior best by ~+5.5 pts
(0.305, ~7 SE) and lifts vs-random to 0.901. Use it as `--baseline` going forward.

**10,000-game recheck (vs the prior best, seed 0, SE ≈ ±0.0046)** — to be sure about the plateau:
snap0000 (control) **0.2530** ✓; snap100000 .2902, snap300000 .2998, snap500000 .2961,
snap700000 .2986, snap1000000 **.3039**. Verdict: the gain is rock-solid (+9…+12 SE over the prior
best), and 100k→1M is a **near-plateau with a faint residual creep** — .290→.304, i.e. snap1000000
is only ~2.1 SE above snap100000 (≈+1.4 pts over 900k trials). So *practically* plateaued (near-zero
returns after ~snap100k) but not dead-flat. Peak is a tie: snap1000000 (.3039) is nominally highest
but only ~0.6 SE over snap300000 (.2998) → best pick unchanged; the final model at most ties it.

**Takeaways:** (1) shaped self-play has **more headroom than the 1200-game evals implied** —
~0.90 vs random, not ~0.86 — but it *does* saturate. (2) Past the saturation point (~snap300k),
more trials are wasted; to go beyond ~0.90 the lever is **structural** (network capacity, richer
observation features, stronger/more diverse opponents), not more self-play of the same setup.

## 2026-06-30 (3000-game re-eval) — CORRECTION: shaped keeps improving; new best snap10000

Re-ran the two recent Mode-B evals at **3000 games** (SE ≈ ±0.0079 vs ±0.0125 at 1200) to resolve
borderline head-to-heads. The tighter numbers **overturn the "plateau / no gain" call below** — at
1200 games the ~2–3 pt edges were buried in noise. All figures vs the named baseline, `snap-stride
1000`, 0 undecided, vs-random ~0.85–0.88 throughout.

**snap3000 (anneal) genuinely beats the 1000-best** — not noise after all:
| anneal snap | 2000 | 3000 | 7000 | 8000 | 9000 | 10000 |
|---|---|---|---|---|---|---|
| vs 1000-best | **.285** | **.278** | .268 | **.227** | **.226** | .249 |

snap2000/3000 are ~3.5–4.4 SE above 0.25. But the **win-dominated tail (snap8000/9000) drops to
~0.226, ~3 SE *below* parity** → once the win reward dominates it *actively hurts*; the anneal net
ends ~parity (snap10000 0.249). (The earlier "snap3000 = noise" doubt came from one noisy 1200-game
reverse reading; the direct 3000-game measurement settles it.)

**Shaped continuation keeps climbing — new best:** vs snap3000, a consistent rising trend
snap7000 .259 → 8000 .257 → 9000 **.268** → **snap10000 .278 (~3.5 SE)**. So pure shaped training
past 1000 trials *does* yield real, slow gains; **`shaped-snap10000` > `snap3000` > `1000-best`**
(each ~3.5 SE, all directly measured). snap10000 also has the best vs-random (0.877).

**New best (promoted):** shaped-continuation `snap10000` → `models/checkpoint_shaped_snap10000`
(= `checkpoint1782834176`). Use it as `--baseline` going forward.

**Corrected takeaways:** (1) the model is **not** plateaued — shaped self-play still improves slowly;
the earlier "converged" entry was a 1200-game-noise artifact. (2) **Win-reward is harmful**, now
confirmed at high precision (anneal win-tail regresses below the 1000-best). (3) **Methodology:**
1200 games (SE ±0.0125) can't resolve the ~2–3 pt gaps these models differ by — use ≥3000 games
(SE ±0.0079) for ranking near-equal snapshots.

## 2026-06-30 (latest) — More shaped training: converged, no decisive gain [SUPERSEDED]

**⚠ Conclusion corrected by the 3000-game re-eval above** — the "converged / no decisive gain"
read was driven by 1200-game noise; at 3000 games snap10000 beats snap3000 by ~3.5 SE. The
healthy-run / vs-random facts below still hold; only the "no gain / plateau" verdict is wrong.

**Test:** is snap3000's edge just "extra near-shaped training" that would keep paying off?
Continue **pure shaped** training from `checkpoint1782765573` (the 1000-run best, clean lineage),
10000 trials, `--epsilon-start 0.1` (flat), `--snapshot-every 250`. **Answer: no — the model is
at a plateau; more shaped training does not produce a clearly better player.**

- **Healthy** (shaped never collapses): 594 s, ~0.07 s/trial, wins climb ~linearly to ~2580;
  reward trend essentially **flat** (~−90→−70) — i.e. converged, not climbing.
- **vs random:** flat ~0.84–0.875 across all snapshots (snap10000 0.873), 0 undecided.
- **vs the current best (snap3000), 1200 games, SE ≈ ±0.013:** every snapshot is within noise of
  0.25; the best (snap9000 0.270, snap10000 0.269) is only ~1.5 SE above — **below the ≳2–3 SE bar
  for promotion. No new best; snap3000 retained.**

| snap | 0 | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 | 9000 | 10000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vs snap3000 | .254 | .223 | .258 | .247 | .229 | .242 | .252 | .254 | .258 | .270 | .269 |

**Consistency finding (corrects the snap3000 promotion):** `snap0000` here *is*
`checkpoint1782765573`, and it scores **0.254 vs snap3000** (~parity). Last run snap3000 scored
0.290 *vs* `checkpoint1782765573`; if that edge were real the reverse matchup would be clearly
*below* 0.25, not at it. So **snap3000 ≈ `checkpoint1782765573` — the 0.290 was mostly noise**, and
the entire shaped family (1000-best, snap3000, this run) sits on one plateau at ~0.86 vs random.

**Verdict:** greedy skill saturated early (≈snap0100 in the 1000-run) and **nothing since — more
trials, annealing, or win-reward — has decisively moved it.** Breaking this plateau likely needs a
different lever (network capacity / observation features, richer opponents, or smarter exploration)
rather than more self-play trials of the same setup. Current best unchanged
(`checkpoint_anneal_snap3000`, ≈ tied with `checkpoint1782765573`).

## 2026-06-30 (later) — Curriculum reward annealing: collapse avoided, but no net gain

**Hypothesis:** the instant shaped→win switch collapsed because the value function couldn't
absorb the reward-scale shock all at once (see entry below). Anneal it slowly so the Q-values
track a moving target. **Result: the curriculum completely fixes the collapse — but the
win-only objective still doesn't beat the shaped model; it lands at parity.**

**Code:** new `--reward anneal` ([train.py](train.py)). Per trial, progress `p` ramps 0→1 over
the first `--reward-anneal-fraction` (0.8) of trials then holds; `step_coef = 1−0.99p` scales the
dense per-step penalty (1.0→0.01), `alpha = 0.99p` blends the end reward from `sum(opp)` toward
`min(opp,4)`. At `p=0` it is *exactly* the shaped reward (verified), so warm-starting has zero
initial mismatch. **Run:** warm-start `checkpoint1782765573`, 10000 trials, `--epsilon-start 0.1`,
`--snapshot-every 250`.

**Collapse fixed (the headline):**
- **687 s** total, healthy **~0.07 s/trial the whole way** (vs the collapse's 7741 s / 0.77).
  Live pace-monitoring through the near-sparse hold phase (`step_coef=0.01, alpha=0.99`, trials
  8000–10000) showed no stall.
- Accumulated training wins climb ~linearly to **~2580** (vs the collapse's flat **9**).
- Eval: **0/1200 undecided** at every snapshot; **vs random ~0.85 throughout** (snap10000 0.848).
  The policy stays strong from start to finish.

**But no improvement over the shaped best.** Win rate vs the start model (`checkpoint1782765573`,
1200 games, SE ≈ ±0.013):

| snap | 0 | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 | 9000 | 10000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vs start | .263 | .252 | **.290** | **.290** | .258 | .246 | .248 | .275 | .241 | .235 | .246 |

snap0000 (= the start model) sits at ~parity as expected. A modest bump at snap2000–3000
(0.290, ~3 SE above 0.25) occurs while the reward is still ~75% shaped (`alpha` ≈ 0.25–0.37) —
most likely just *extra near-shaped training* (the 1000-run was itself still inching up), not a
win-reward effect, since it **washes out as the reward becomes win-dominated**: snap8000–10000
fall back to/just below parity (snap10000 **0.246 ≈ start model**).

**Verdict:** annealing is the right mechanism — it turns a catastrophic collapse into a stable
run. The original motivation (drop the draw-penalty → better play) is **largely not supported**:
the fully-annealed (near win-only) model is statistically indistinguishable from the shaped best,
and the win-dominated tail erodes the small mid-run edge. If win-reward is worth another pass, the
signal says the *late* win-dominated regime is where it stops helping — try a larger per-step
floor, or the gentler "penalize only net hand growth" shaping noted in [PLAN.md](PLAN.md), rather
than driving `step_coef` to ~0.

**New best (promoted):** the highest vs-start scorer, **snap3000** (0.290 vs the prior best,
~3 SE above parity), copied to `models/checkpoint_anneal_snap3000` and adopted as the current
best. Caveat for the record: snap3000 is mid-anneal (reward still ~63% shaped, `alpha≈0.37`), so
this edge most likely reflects *extra near-shaped training* on top of the 1000-run rather than a
win-reward benefit — but it is a real, measured head-to-head improvement over `checkpoint1782765573`.
Use `--baseline ./models/checkpoint_anneal_snap3000` in future progression evals.

## 2026-06-30 — Win-only finetune: catastrophic collapse (negative result)

**Hypothesis:** the dense `-len(hand)` reward punishes drawing even when drawing is correct;
finetuning the 1000-trial best with a **win-only** reward should remove that bias. **It
destroyed the model instead.**

**Code (new, all working):** `dqn.py` now *raises* on a failed `load_model` (no silent
cold-start); `train.py` gains `--epsilon-start`, `--snapshot-every`, and `--reward {shaped,win}`.
`win` reward = 0 every step/loss, and on a win `min(fewest opponent's cards, 4)`.
**Run:** warm-start `checkpoint1782765573` (1000-trial best), 10000 trials, `--epsilon-start 0.1`
(flat = `epsilon_min`), `--snapshot-every 250`, `--reward win`.

**Result — collapse within ~50 episodes, never recovered:**
- Training plot: accumulated wins jump to 9 in the first ~50 episodes then are **dead flat for
  the remaining ~9 950**; per-episode reward is 0 after the start. The learner stopped winning
  entirely.
- Eval (`--snap-stride 1000 --games-b 500`, baseline = the start checkpoint):

  | snapshot | vs random | vs baseline (start model) |
  |---|---|---|
  | snap0000 (= warm-start) | **0.856** | 0.302 |
  | snap1000 … snap10000 | **0.000** (heavy undecided) | **0.000** |

  snap0000 confirms the warm-start loaded the good model; **every snapshot from 1000 on wins 0/500
  vs both** random and the start model, with large undecided counts (the collapsed policy stalls).
- Wall time **~7740 s (~10× a normal 10k run)**: opponents are synced to the collapsing learner,
  so all four seats stall and every trial runs to the 300-step cap.

**Diagnosis — sparse reward + warm-start scale mismatch + self-play, compounding:**
1. The loaded Q-values were fit to the *dense* reward (episode returns ~ −50…−230); the win-only
   targets live in `{0} ∪ [1,4]`. The first replay updates drag every Q toward ~0, flattening the
   action ordering → the learned policy is erased almost immediately.
2. With no shaping, the **only** learning signal is a win; once the policy degrades it stops
   winning, so the signal vanishes and there's nothing to climb back on (sparse-reward trap).
3. `OPPONENT_SYNC_EVERY=5` copies the degrading learner into the opponents, so self-play locks the
   whole table into a non-terminating stall — the win signal can't reappear even by luck.

**Takeaways / what would be needed to make win-only viable:** keep it *potential-based / shaped*
rather than fully sparse, or normalize/reset the value head before switching reward scale; use
**high** exploration when changing the objective (not `epsilon=0.1`); and during finetuning hold a
**fixed strong opponent set** instead of syncing to the learner, so a collapse can't propagate and
games still terminate. The draw-penalty concern is real, but the fix is gentler reward shaping
(e.g. only penalize *net* hand growth, or reward progress toward emptying), not removing all
per-step signal. Run kept at `models/run1782767834` (gitignored).

## 2026-06-29 (latest) — 1000-trial run + dual-yardstick progression (vs random AND vs best)

Closes the three open items below. **Code:** `eval.py` now uses `TURN_CAP = 2000` and a
reworked Mode B (`--baseline`, `--snap-stride`; each snapshot scored vs random *and* vs a
frozen reference model). **Run:** `models/run1782765573` — 1000 trials, snapshots every 25.
**Eval:** `--snap-stride 100 --games-b 1200` (11 snapshots × 1200 games × 2 references),
baseline = the prior best `checkpoint1782749825` (the 300-trial run). SE ≈ ±0.012.

### Speed
- **Training 1000 trials: 80 s** (~0.08 s/trial) — consistent with the ~28 s/300 fast path.
- **Eval: 578 s** (~9.6 min) for the full 11×1200×2 sweep. vs-random ~61 g/s, vs-baseline
  (net-vs-net) ~45 g/s.

### Cap fix validated (open item 1)
At `TURN_CAP = 2000`, **every trained snapshot has 0 undecided** vs random (incl. snap1000
**0.857 = 1028/1200, 0 undecided**). Only the untrained `snap0000` still stalls vs random
(173/1200 undecided → 0.085 over decided) — exactly the expected untrained-only behaviour.

### vs random — converges fast, then flat (open item 3, convergence)
Jumps to ~0.83 by **snap0100** and sits at **0.82–0.86 through snap1000** (no trend, no
dips). Greedy skill is essentially set within the first ~100 trials — matching the 300-run's
~snap0075. The *reward* trend in `training<ts>.png` keeps gently rising to ~1000, but that
tracks the hand-size penalty under ongoing exploration, not greedy skill.

### vs the frozen prior-best — the longer run *does* edge past it, late (open item 2)
This is the signal vs-random can't show (it's saturated). Win rate vs the 300-run best:

| snapshot | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| vs best | .245 | .245 | .245 | .229 | .242 | .233 | **.252** | **.254** | **.269** | **.279** |

(`snap0000` = 0.037, the floor.) Mid-run snapshots sit *at or just below* parity (0.25) —
the new run takes most of training just to match the old best — then the **last ~300 trials
climb above the line**: snap1000 **0.279** is ~2.3 SE above parity (snap0900 0.269, ~1.5 SE).
The rise begins ~snap0700–0800, coinciding with epsilon annealing to its floor at ~trial 800
(`EPSILON_DECAY_FRACTION 0.8`): once exploration cools, the greedy policy refines just past
the prior best. **Net: 1000 trials beats the 300-trial best, but only modestly (~+3 pts) and
only in late training.** Plot: `models/run1782765573/progression.png`.

**Takeaways:** (1) the dual yardstick works — vs-random for "did it learn / how fast" (fast),
vs-frozen-best for fine ranking that vs-random saturates away; (2) the frozen *external* best
has no moving-reference artifact and resolves a real ~3-pt gain at 1200 games; (3) diminishing
returns — most of the 1000-trial budget buys little over 300; the gain is concentrated after
epsilon bottoms out. A natural next lever is a slower/longer epsilon floor or more
post-anneal trials, since that late window is where improvement actually happens.

## 2026-06-29 (later) — Draw-stall investigation: artifact, not a stall

Instrumented every decision in 200 games (the acting agent's own legal options logged).
**Corrects the "~10% draw-stall" claim in the entry below — it was a measurement artifact.**

**The trained model rarely draws when it can play.** Share of an agent's draws that had a
legal non-DRAW move available:

| Agent | Draws (% of its turns) | …of those, % with a legal play available |
|---|---|---|
| Trained DQN (vs random) | 37% | **4%** |
| Trained DQN (4× greedy self-play) | 30% | **3%** |
| RandomAgent | 44% | **64%** |

So ~96% of the DQN's draws are **forced** (no matching card, or a `+2`/draw-two state where
drawing is the only legal move). The agent that "draws with cards in hand" is `RandomAgent`
(uniform choice over legal moves), by design — not a learned pathology.

**The undecided games were long, not stalled.** 4× greedy *trained* models: **0% undecided**,
~75 turns/game, 65% plays. Mode-A capped games still had a healthy deck (68–89 cards) and
ongoing plays at turn 400 — just long. Raising `TURN_CAP` 400→3000: **0% undecided**, median
136 turns, max **476**, only 6/100 games needed >400 turns.

**Genuine non-termination happens only with untrained nets.** `snap0000` (random weights)
picks DRAW ~half the time even with plays available → all seats lock into drawing → the
discard never grows → the deck exhausts → never ends. That control result was wrongly
generalized into "greedy play stalls."

**Corrected win rate vs random** (cap 2000, all 500 games decided, 0 undecided):
**0.866 (433/500)** vs 0.25 baseline — slightly *higher* than the 0.847 below, which dropped
the long games from its denominator.

**Implications:** no draw-penalty / reward-shaping is warranted (it would solve a non-problem);
the only eval fix is a higher `TURN_CAP` (~2000). An optional no-progress termination rule
matters only when evaluating near-random nets.

## 2026-06-29 — Post-tuning baseline + CPU speedup

**Code:** `b736083` (RL tuning) as the trained config; eval/snapshot tooling `e5feaa8`;
DQN speedups `265cd86`. Hardware: WSL2, RTX 4070 present but **TF 2.4.1 is CPU-only**
(`is_built_with_cuda() == False`) — all runs are CPU.

**Train config:** 4 players (1 learner + 3 synced opponents), `--trials 300`,
`trial_len 300`, lr `1e-3`, replay buffer `20000`, batch `64`, ε anneals 1.0→0.1 over ~80%
of trials, `OPPONENT_SYNC_EVERY 5`. **Eval config:** greedy (ε=0), seeded games,
`TURN_CAP 400`; mode A 500 games, mode B 40 games/snapshot.

### Speed (300 trials, same config)
| Path | Wall time | Per trial | Speedup |
|---|---|---|---|
| Before (`model.predict` + `model.fit`, default TF threads) | **963 s** (~16 min) | ~3.2 s | 1× |
| After (thread caps + direct `model()` calls + `@tf.function` step) | **~28 s** | ~0.07 s | **~35×** |

GPU was never the bottleneck: tiny net, batch-1 / batch-64 ops, latency-bound on per-call
overhead. Speedup is entirely CPU-side and the learned policy is unchanged (see below).

### Win rate vs random — Q: does it beat `1/num_players` (= 0.25)?
| Model | Win rate | Decided | Verdict |
|---|---|---|---|
| Baseline checkpoint (slow-path run) | **0.847** (389/459) | 459/500 | 3.4× chance |
| Fast-path re-run (speedup validation) | **0.826** (371/449) | 449/500 | equivalent learning |

The ~0.02 gap is run-to-run variance (training is unseeded) — confirms the speedup changes
are numerically equivalent, not just faster. **Corrected (cap 2000, all games decided):
0.866 (433/500)** — the rates above are over *decided* games only; see the investigation above.

### Undecided ("draw") rate due to `TURN_CAP = 400`
- **Mode A (DQN vs 3 random, 500 games):** slow-path run **41/500 = 8.2%**;
  fast-path run **51/500 = 10.2%** undecided. **[Corrected — see investigation above: these
  are naturally long games hitting the 400 cap, NOT stalls; the trained model is not looping
  on drawing (only ~4% of its draws are avoidable). With `TURN_CAP` 2000 the rate is 0%.]**
- **Mode B (snapshot vs 3× untrained `snap0000`, 40 games each):** **6–19 undecided per
  matchup** (~15–48%) — much higher, because the untrained opponents stall constantly.
  The `snap0000` vs `snap0000` control is fully degenerate (all capped → 0 decided),
  confirming untrained greedy self-play never finishes.

### Convergence (Q: how fast?)
Reward-per-episode trend rises ~−230 → ~−50 over 300 episodes, still mildly climbing at the
end (**not fully plateaued**). Training-time win rate ~22.7% (68/300) ≈ `1/num_players` — the
expected self-play-symmetry artifact, *not* a skill measure. Plot:
`models/run<ts>/../training<ts>.png`.

### Progression (Q: latest vs earlier?)
Every trained snapshot beats untrained `snap0000` at **0.91–1.00**; curve saturates by
`snap0025` (beating a pathological opponent is trivial), so it confirms "trained ≫ untrained"
but does not finely rank trained snapshots. See `models/run<ts>/progression.png`.

### Open items / next
- ~~Raise eval `TURN_CAP` 400→~2000~~ **DONE** (now 2000; 0 undecided for all trained
  snapshots — see the 1000-trial entry on top). No draw-penalty / reward-shaping.
- ~~Graded progression curve (fixed mid-snapshot / round-robin)~~ **DONE** — replaced the
  vs-untrained curve with vs-random **+ vs a frozen external best**; the frozen-best yardstick
  resolves fine ranking that vs-random saturates away (no moving-reference artifact).
- ~~Reward trend not plateaued → try 600–1000 trials~~ **DONE** (1000-trial run; reward
  trend still gently rising but greedy skill saturates by ~snap0100).
- *From the 1000-trial result:* the only real gain over the 300-run best comes **after epsilon
  bottoms out (~trial 800)**. Worth trying a slower epsilon decay / longer post-anneal tail (or
  more trials past 1000) to see if that late window keeps yielding improvement.
- *From the win-only collapse (2026-06-30):* fully-sparse reward is a dead end here. To revisit
  the draw-penalty concern, try **gentle/potential-based shaping** (penalize only *net* hand
  growth, or reward progress toward emptying) rather than removing per-step signal; and when
  changing the objective, raise exploration + hold a **fixed strong opponent set** (don't sync
  opponents to a possibly-collapsing learner).
- **Shaped self-play keeps improving until ~snap300k, then saturates at ~0.90 vs random**
  (1M-trial run, 2026-07-01). The earlier "~0.86 ceiling" was undertrained: with enough trials
  vs-random reaches ~0.90 and the head-to-head chain `shaped-snap300000` > `snap10000` >
  `snap3000` > `1000-best` holds. Past ~snap300k more trials are wasted. Current best:
  **`checkpoint_shaped_snap300000`** (0.305 / ~7 SE vs the prior best; 0.901 vs random).
- **Win-reward: harmful (confirmed).** The anneal's win-dominated tail regresses ~3 SE below the
  1000-best. Don't pursue win-only/near-win-only; if revisiting the draw-penalty, use gentle
  potential-based shaping (see [PLAN.md](PLAN.md)).
- **Eval precision:** these models differ by only ~2–3 pts, so rank snapshots at **≥3000 games**
  (SE ±0.0079); 1200 games (±0.0125) is too noisy and produced a false "plateau" read.
- *Next lever for bigger gains* (vs the slow shaped creep): structural — larger network / richer
  observation features, stronger/more diverse opponents, or n-step/MC returns + reward
  normalization for more stable, multi-state updates (see [PLAN.md](PLAN.md)).
