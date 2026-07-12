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
