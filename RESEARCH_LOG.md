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

## 2026-07-05 — DQN-hygiene Phase 2: ablation (10k) + full-bundle 100k A/B. Verdict: NOT the plateau-breaker; only the scale pair does anything, and champion-beating is the color-sym effect

Autonomous follow-up to the weak 10k bundle result below. Two stages, all on `exp-dqn-hygiene`,
thread-capped (`OMP=1`) + `nice -19 ionice`, seeds 0/1/2, `--color-sym` in every arm, 3000-game
CRN evals (parity 0.25, per-arm SE ≈ 0.0079, pooled-over-3-seeds SE ≈ 0.0046). Runs/logs under
`models/_hygiene_ab/` (ablation + long100k subdirs, `RESULTS.txt`).

### #2 — 10k ablation: which half of the bundle carries the signal?
Bundle split into its two coherent pairs vs the Phase-1 color-sym control (head-to-head, parity 0.25):
- **scale pair** (`--loss huber --reward-scale 0.1`): 0.266 / 0.269 / 0.284 → **pooled 0.273, +5.1 SE, 3/3 positive.**
- **stability pair** (`--double-dqn --target-sync-mode steps 2000`): 0.246 / 0.254 / 0.256 → pooled 0.252, **+0.4 SE (flat).**

So the entire (weak) Phase-1 head-to-head signal is the **scale pair** (Huber + reward↓); Double DQN
+ slow target do **nothing** at 10k. The full bundle's bad Phase-1 seed-1 (0.239) was worse than
either half alone → the two pairs interact slightly *negatively*.

### #1 — 100k full-bundle A/B (control = color-sym; treatment = full bundle)
| metric | seed0 | seed1 | seed2 | pooled |
|---|---|---|---|---|
| control vs random | .911 | .918 | .909 | **.913** |
| treat vs random | .926 | .930 | .928 | **.928** (+1.5 pt, 3/3) |
| treat vs control (h2h) | .256 | .274 | .240 | .256 (**+1.4 SE**, seed-inconsistent) |
| treat vs champion `snap300000` | .276 | .277 | .271 | **.275 (+5.4 SE)** |
| control vs champion `snap300000` | .279 | .266 | .264 | **.270 (+5.2 SE)** |

**The decisive comparison is the last two rows: treat-vs-champ (.275) ≈ control-vs-champ (.270).**
Both the plain color-sym run and the full bundle beat the current best (`checkpoint_shaped_snap300000`,
a 300k-snapshot of the 1M-trial run) by the *same* ~+5 SE margin. So **"beats the champion" is the
color-sym + fresh-100k effect, NOT the hygiene levers.** Over its own control the full bundle is only
marginal: head-to-head ~parity (+1.4 SE, one seed below), with a small but consistent +1.5 pt
vs-random edge (.928 vs .913, 3/3 seeds) — a mild robustness bump, not a strategic gain.

### Takeaways
1. **DQN-hygiene is not the plateau-breaker.** At 100k it adds at most ~+1.5 pt vs-random over
   color-sym and no clear head-to-head edge. Double DQN + slow target are inert at both 10k and 100k
   here (the near-on-policy target A1 flagged wasn't actually costing much); only Huber + reward-scaling
   move the needle, and only slightly.
2. **Headline is color-sym, again:** a **100k color-sym run beats the 1M-trial champion** head-to-head
   (~.270, +5 SE) and hits ~0.91–0.93 vs random — corroborating the color-sym A/B. The candidate new
   best is a color-sym 100k snapshot, not a hygiene one (promotion deferred to the color-sym A/B owner).
3. If squeezing the hygiene levers further: keep **Huber + reward-scaling** (the only live part), drop
   Double DQN + slow target, and note the negative scale×stability interaction. But the larger lever
   remains structural (network capacity / richer observation), per the plateau analysis. Nothing promoted.

## 2026-07-04 — DQN-hygiene package A/B at 10k trials (branch `exp-dqn-hygiene`): weak, seed-inconsistent; does NOT clear the adopt bar

**Setup.** 3 seeded pairs (seeds 0/1/2), 10k trials, `--reward shaped --color-sym` in **both**
arms. Treatment adds the full package: `--double-dqn --loss huber --reward-scale 0.1
--target-sync-mode steps --target-sync-every 2000` (slow lagged target that drops the
episode-end hard copy, Double DQN, Huber, reward÷10). Thread-capped (OMP/OPENBLAS=1) + `nice
-19 ionice`, run concurrently with the other session's 100k color-sym A/B. Trainings 6/6 OK in
22 min; run dirs + logs under `models/_hygiene_ab/`. Evals: 3000 games, seed 0 (CRN).

**Results** (head-to-head = treatment as 1 seat vs 3 control-copy seats; parity 0.25, SE ≈ 0.0079):

| seed | control vs rand | treat vs rand | head-to-head | vs parity |
|---|---|---|---|---|
| 0 | 0.895 | 0.911 | 0.289 | +4.9 SE |
| 1 | 0.904 | 0.900 | 0.239 | −1.4 SE (below parity) |
| 2 | 0.911 | 0.907 | 0.267 | +2.2 SE |
| pooled | 0.903 | 0.906 | 0.265 | +3.2 SE |

**Verdict: does not meet the pre-registered adopt bar** (needed ≥2/3 seeds at ≥3 SE, consistent
sign; got 1/3, with seed 1 *below* parity). vs-random is unchanged (treat 0.906 vs control 0.903)
— the package neither hurts nor helps the random baseline. Pooled head-to-head is +3.2 SE (~+1.5
pts over parity) but driven almost entirely by seed 0; high between-seed variance vs a small
effect → **not robust**. Contrast color-sym's decisive +7.3 pts / z≈8.6 at the same scale.

**Caveats / why this isn't a clean "reject".** (1) The treatment bundles 4 levers — a mixed result
can't attribute; the slow-target/Double-DQN benefit is a *stability* effect expected to compound
over LONG horizons, and 10k trials is exactly where it's least visible (unlike color-sym, which
pays off immediately, so the 10k screen was predictive there and may under-read here). (2) Possible
internal cancellation (e.g. reward÷10 + Huber δ=1 interaction, or slow target hurting at short
horizon). **Next options (Phase 2 gated on review):** component ablation (which lever moves it),
the `--target-sync-mode polyak` variant, or — despite the weak screen — a 100k A/B of the full
bundle since these levers target the long-run plateau, not 10k skill. **Not adopted; nothing
promoted.**

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
