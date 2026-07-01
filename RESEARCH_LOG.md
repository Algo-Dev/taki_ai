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
