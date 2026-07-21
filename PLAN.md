# Taki AI — Review Findings & Fix Plan

## THE TARGET (read first — it reranks everything below)

**The goal is not the strongest possible Taki player.** It is to find the best moves available
**under a human information set**, and then to **interrogate the resulting policy's strategy** —
to discover what optimal play looks like *for a human*, in concrete scenarios a human can actually
reason about.

Two consequences, and most of this file predates them:

1. **The observation is a deliberate model of what a human sees and remembers** — it is a design
   constraint, not a limitation to be engineered away. A7's removal of the discard-pile histogram
   was **on purpose**: humans do not perfectly card-count a played pile. Any proposal that widens
   the information set beyond human reach (full discard histogram, exact deck composition, a
   63-slot unseen-count vector, opponents' hands) is **OFF-GOAL and must be rejected** — it buys
   win rate by granting superhuman memory, which defeats the point of the project. See CLAUDE.md.
2. **The win-rate plateau is not a wall — it is plausibly the finish line.** A10/A11 (capacity),
   A12 (a dedicated best-response cannot exploit A8), and R1 (an exact 9!-symmetry augmentation
   moved nothing) are converging evidence that the agent is at **the ceiling of its information
   set**. That is the design working. The right question has shifted from *"is it stronger yet?"*
   to **"is it strong enough to interrogate — and what does it say?"**

So the strength-chasing items below are now **means, not ends**: worth doing only insofar as they
make the policy a more trustworthy oracle to question. The **behavioural probes (B-series) are the
actual deliverable.**

---

Findings from a full review of the project (rules engine, DQN agent, training loop,
demo harness). Items are grouped by category. **Correctness, observability, rules
fidelity, and code hygiene are being fixed now. The "RL / training design issues"
section is intentionally deferred — we will tune those later.**

-- (future) On-the-fly reward re-labeling for curriculum/reward changes: instead of
   freezing each transition's reward at collection time, store the raw scalars needed to
   recompute it (learner_hand_size, sum_opp, min_opp, won) in remember() [dqn.py] and have
   replay() recompute reward under the *current* schedule when sampling. This removes the
   stale-reward-scale lag (the buffer always reflects the current reward) and makes reward
   annealing exact. Note: not possible from what's stored today — the observation keeps only
   the min-opponent and next-player hand sizes, not sum_opp, so the original end bonus
   (sum of opponents' cards) can't be reconstructed from new_state alone.



-- applied for finetuning: the reward, always rewards decrease in the hand number of cards. But there is a possibility that sometimes it is better to take a card rather than drop one. Let's finetune the model only with the winning reward. I clipped at 4 the minimal opponent hand. the reward change failed even when applied slowly. Should consider more stable updates, that include info from many states. discuss with claude.


-- (scoping, 2026-06-30) The "more stable / multi-state updates" ideas — reward/return
   normalization (e.g. PopArt) and n-step / Monte-Carlo returns — are tools **for the
   reward-change / sparse-reward direction** (win-only, draw-penalty), NOT levers for the
   fixed shaped reward. Reward-scale normalization only matters when the reward (scale)
   changes mid-training; with a fixed dense shaped reward there's no scale shock, and n-step/MC
   mainly pay off for sparse rewards. For pushing the **shaped** model past its slow creep, the
   real levers are structural: bigger network / richer observation features, stronger or more
   diverse opponents. Use normalization + n-step/MC only if/when we revisit changing the reward.

## Infrastructure / evaluation optimizations (future)

-- Parallelize eval.py's game loop across processes (games are independent, deterministically
   seeded) — cut eval wall-clock ~3-4x. Worth it for Mode B / screening.

## RL design notes (future levers, from the 2026-07-02 review)

-- Replay buffer horizon: 20000 transitions at ~20-35 learner decisions/trial is only ~600-1000
   trials of history. During long runs (100k-1M trials) the buffer is a tiny recent window —
   reasonable, but keep it in mind when interpreting late-run flatness (no long-term memory of
   earlier play styles).

-- Dead knob: AIAgent's constructor default epsilon_decay=0.995 is always overwritten by
   train.py's run-length-scaled decay; the default is noise, not a real setting.


## Full project review (2026-07-03) — software + algorithm findings and action items

### Algorithm findings

-- A5. **Self-play opponents are a ≤5-trial-old mirror.** `OPPONENT_SYNC_EVERY=5` means the learner
   only ever faces (almost) itself — prone to strategy cycling, and was the amplifier in the
   win-only collapse (a collapsing learner got copied into its own opponents, locking the whole
   table). Try an opponent *pool* (past snapshots + occasionally a RandomAgent seat) — standard
   fictitious-self-play fix, likely the best lever to break the ~0.90 plateau since "more of the
   same self-play" is already established as exhausted. notice that if we increase OPPONENT_SYNC_EVERY, at some point it will be to big for us to learn also from the opponents seats.
   Tension with A8 (learn from all seats): a stale/weak pool member (esp. a RandomAgent seat)
   would feed A8's buffer with off-distribution transitions — not a correctness problem for
   Q-learning, but a dilution risk. Mitigation if implemented: keep the pool recent (last
   10-20 sync intervals, not full history) and skip A8 collection for RandomAgent seats.

-- A9. **Reward design, if ever revisited:** the gentler middle ground PLAN.md's "penalize only net
   hand growth" idea gropes toward is **potential-based shaping** — `r' = r_win + gamma*phi(s') -
   phi(s)` with `phi(s) = -len(hand)`. Provably policy-invariant w.r.t. the win objective, keeps a
   dense signal (no sparse-reward trap), stops punishing strategically-correct draws beyond their
   true cost. Use this formulation (not another step-coefficient schedule) if the draw-penalty
   concern comes back up — with high epsilon and a fixed opponent set during the switch, per the
   win-only collapse post-mortem.

### Action items, ranked by expected value / cost

1. Opponent pool of past snapshots (+ occasional random seat) (A5) — **demoted by A12** (see
   below): a best response trained specifically against frozen A8 never reached parity with it,
   so self-play is not obviously leaving an exploit on the table. Still the best of the
   training-scheme levers, but a training-scheme lever is now the less likely fix.
2. ~~Dueling head (A11)~~ — **DONE, NEGATIVE. Closed.** See RESEARCH_LOG.md 2026-07-11 and branch
   `a11-dueling-head` (code kept there; it invalidates checkpoints via a two-input obs+mask
   contract, so it is deliberately not on master). `Q = V + (A - mean_legal(A))`, advantage
   centred over legal actions only, heads split off the existing trunk so capacity is unchanged.
   The 450k rerun (epsilon floor at 360k) *did* train ~90k post-floor trials — the one caveat
   that made the first attempt inconclusive — and the verdict held: best swap-controlled edge
   **+1.00 pts/seat** vs a **+2.0** bar, with the top post-floor snapshot replicating across
   3 seeds at **mean +0.25** once seat-swapped. **Both halves of the architecture lever (A10
   capacity, A11 dueling) are now closed and negative.**
   (Worth remembering: A11's screening *looked* promising — post-floor mean 0.273 vs 0.25 parity,
   same snapshot topping all 3 seeds — and did not survive the gate. **Not** a seat-0 artifact, as
   an earlier draft of the log wrongly claimed: **R7 below is correct, `play_match` shuffles seats
   and Mode B is fair in expectation.** Mode B is simply a 1v3 win rate while the gate is a 2v2
   per-seat rate. A live instance of R8's winner's curse — top-of-53 overstates.)
3. n-step returns (n=3-5): shortens the bootstrap chain; large contributor in Rainbow ablations
   even with dense rewards. Cheap to implement in the buffer.
4. Lower epsilon floor late in training (0.1 -> 0.02-0.05, or decay to floor by ~50% of trials) —
   the 1000-trial run showed gains concentrated *after* epsilon bottomed out.
5. LR decay for long runs (1e-3 -> 1e-4 cosine/step) — the faint 100k-1M creep is consistent with
   bouncing around a minimum at too-large a step size.
6. Prioritized replay: medium effort, real but smaller expected gain here; do after 1-3.
7. If revisiting reward: potential-based shaping (A9), high epsilon + fixed opponent set during
   the switch.

**The A12 exploitability probe (2026-07-11) reorders this list.** A best response trained for
200k trials against 3 *frozen* A8 seats peaked at **0.236 vs A8 — never reaching the 0.25 parity
line** (RESEARCH_LOG.md 2026-07-11). A8 is therefore near-unexploitable within this function
class: the ~0.91 plateau looks like the ceiling of the current representation, not a self-play
equilibrium trap that a better opponent distribution would escape. That is evidence *against*
the whole training-scheme family (#1 opponent pool, and by extension #3-#6, which all search
harder within the same function class) and *for* changing the function class itself — i.e.
**representation** (belief/memory features; note the A7 observation dropped the full discard
histogram, so the net has no memory of what has been played) and **search**. Rank-symmetry
augmentation (R1 below) also survives this argument, since it is a data lever of the kind that
has actually worked here (color-sym, A8), not a training-scheme variation.

### Tuning guide: replay frequency, buffer size, and related knobs

Framing: the **replay ratio** — since A8 (all-seats collection) ~100 transitions/trial are
collected (~25 x 4 seats), `replay()` runs at learner-steps 0,4,8,... plus once at episode end
(~8 calls x batch 64 ~= 512 samples/trial), so each transition is now trained on ~5x before
eviction (was ~20x under seat-0-only; Atari reference point is ~8). Replay ratio, learning
rate, and target-sync period must move together: more updates per datum -> lower lr and/or slower
target sync.

| Knob | Current | Sweep | What to watch |
|---|---|---|---|
| Replay frequency | every 4 learner steps + episode end (train.py) | every 1 / 2 / 4 / 8 steps | Q divergence & loss spikes at high ratio; slow learning at low |
| Buffer size | 80k (dqn.py:39, A8) | 80k / 200k / 500k | oscillation/forgetting vs earlier snapshots if too small; sluggish early adaptation if too large |
| Target sync | ~2x/episode (effective) | hard: 500 / 2000 / 10000 learner steps; or Polyak tau 0.01 / 0.001 | loss sawtooth at each hard sync; head-to-head vs current best |
| Learning rate | 1e-3 | 1e-3 / 3e-4 / 1e-4, +/- decay schedule | late-run creep vs plateau; gradient norms |
| Batch size | 64 | 64 / 256 (scale lr with it) | wall-clock per trial; interacts with replay ratio |
| epsilon floor / decay | 0.1, decay over 80% of trials | floor 0.1 / 0.03 / 0.01; fraction 0.4 / 0.8 | vs-random early (too little exploration) vs late head-to-head gains |
| gamma | 0.99 | 0.99 / 0.995 | endgame credit: at ~30 decisions/episode the win bonus reaches the opening at ~0.74 discount — mild, so low priority |
| Opponent sync | every 5 trials | 5 / 25 / 100, then pool | robustness vs random AND vs frozen bests simultaneously |

Buffer-size context: since A8 (~100 transitions/trial), 80k transitions ~= 800 trials of history
— restores the horizon 20k gave under seat-0-only collection; scale with run length (200k-500k
for long runs). Store observations as float32 (they are float64 now) to halve buffer memory.

### Recommended process (matches how the color-sym experiment was actually run)

2. Screen each change at 10k trials, 2-3 seeded pairs, changing **one knob** (or one declared
   package) at a time.
3. Judge on the three readouts that caught color-sym: 3000-game vs-random, head-to-head vs the
   control run's snapshot (the decisive one), and Mode-B progression vs
   `checkpoint_shaped_snap300000`.
4. Confirm winners at 100k trials before promotion.
5. Add cheap training-time diagnostics so stability is visible without running full evals: mean
   |TD error|, mean max-Q on a frozen probe set of ~1k states (drift up = overestimation),
   gradient norms. Microseconds of cost; would have
   diagnosed the win-only collapse in minutes instead of a 7700-second run.

**Single highest-leverage next experiment:** an opponent pool of past snapshots (#1) — the
DQN-hygiene package has already been run (marginal, see RESULT above), so the plateau-breaker is
now expected to be structural (richer observation / bigger network) or a stronger, more diverse
opponent set rather than more of the same self-play.

---

## Full project review (2026-07-10) — algorithm-level, all branches and results

Review of the whole project (all branches, RESEARCH_LOG history, train/eval/agent code), focused
on high-level and algorithmic questions rather than software ones. Suggestions only; nothing was
changed.

### Where the project actually stands

The result history is unusually clean, and it tells one coherent story. Every lever that worked was
a **data lever**; every lever that didn't was a **network/update-rule lever**:

| Lever class | Experiments | Outcome |
|---|---|---|
| More effective data | color-sym (24x augmentation), A8 (4x all-seats collection), A7 (richer obs) | All three wins; color-sym alone bought 3.3x sample efficiency and turned out to be the stabilizer |
| Reward design | win-only, anneal | Collapse / no gain |
| Update-rule hygiene | Double DQN, Huber, slow target | Inert once color-sym is on |
| Capacity/architecture | A10 (2.7x params), A11 (dueling) | Negative / parity |

Meanwhile the headline metric is saturated (0.91 vs random since the 1M-trial run) and head-to-head
promotion margins are shrinking: +3.7 (A8), then two consecutive failures to clear +2.0. The project
is at a genuine decision point, and the central unanswered question is: **is the plateau a self-play
equilibrium trap, or the ceiling of this function class / this information set?** Nothing run so far
distinguishes these two hypotheses — and they call for completely different next moves. Most of the
suggestions below are organized around resolving that.

### New suggestions (not previously in this file)

**R1. Rank-symmetry augmentation — DONE (2026-07-11), NEGATIVE. The highest-prior idea on the
board was inert.**
Shipped as `--rank-sym` (branch `r1-rank-sym`; RESEARCH_LOG.md 2026-07-11). The symmetry is real
and exact — 9! rank relabelings composing with the 24 color perms, ~8.7M in total, verified
against the rules engine — but a 500k-trial run at A8's exact config landed at **parity with A8**
(mean edge -0.14 pts across the 5 screened candidates in seat-swap-controlled head-to-heads; best
+1.6, which is a max-of-5 and does not clear the +2.0 bar A10/A11 already failed). Kept in the code,
off by default.

**The important consequence: "data-side symmetries pay" is dead as a guiding heuristic.** This was
its cleanest possible test — an exact symmetry, a 15,000x larger group than color-sym, the same
machinery — and it moved nothing. So color-sym's win was not augmentation-as-such; it most likely
acted as a *stabilizer* (its ablation diverges on long runs). Everything below that was motivated by
"add more exact relabelings / more augmentation" should be **down-weighted accordingly**, and the
open question shifts to the **information set** (see R3/observation items): what the agent can *see*
is now the only untested lever with obvious headroom — notably the discard histogram A7 removed.

**R2. Exploitability probe (best-response training) — DONE (2026-07-11), the second branch fired.**
Ran as A12 (`train.py --freeze-opponents`, merged to master; RESEARCH_LOG.md 2026-07-11). A fresh
learner trained 200k trials against 3 frozen `checkpoint_a8_snap455000` seats **peaked at 0.236 vs
A8 and never reached the 0.25 parity line** (ended 0.213; flat from ~trial 70k). It could not even
match the champion, let alone exploit it.

**Verdict: A8 is near-unexploitable within this function class** — the second of the two branches
this experiment was designed to separate. Effort should go to **representation** (belief/memory
features) and **search**, not to training-scheme variations; see the reordered action items above.
Consistent with the flat parity bands in A10/A11 (different architectures, same ~0.21-0.30 wall).
Caveats: single seed, 200k trials (vs A8's 450-500k protocol), though the long flat plateau makes a
late breakout unlikely.

**R3. Estimate the skill ceiling with an oracle and a heuristic baseline.**
- **The heuristic baseline is DONE** (2026-07-12, `agents/heuristic.py`; RESEARCH_LOG). It scores
  **0.849 vs 3 random** (A8: 0.912) and A8 beats it at only **0.343 vs 3 heuristics** (parity 0.25).
  Conclusions: **vs-random is retired as a ranking metric** — hand-written rules recover most of it,
  so it cannot resolve differences up here; use `eval.py --opponent heuristic` when discrimination
  matters. A8's edge over a non-lineage opponent is real but modest, so the plateau is the ceiling of
  *what vs-random can see*, not of play. Follow-ups: tune the heuristic's hand-set weights; add it to
  the training opponent pool (`train.py`) to see whether a non-lineage sparring partner moves A8.
- **Still open: a full-information oracle** (an agent that sees all hands — trained the same way, or a
  greedy full-info heuristic) to upper-bound what any policy could do. The gap between it and A8
  measures how much the hidden information is worth, i.e. how much belief-state features (R4) can
  possibly buy. Cheap version: reuse `HeuristicAgent`'s scoring with the opponent model replaced by
  the true hands.
- **Also worth doing: a hybrid probe agent** — A8 wrapped with heuristic *overrides* (e.g. force the
  colored-TAKI hoard, force the near-winner block). If forcing a rule *raises* win rate, A8 never
  learned that behaviour; if it lowers it, A8 already knows better. This is a direct instrument for
  the behavioural questions in CLAUDE.md, and it can reuse the rule code as-is.

**R4. Belief-state features — DONE (2026-07-21), on branch `r4-color-void`. The policy learned
color denial; win rate did not move.** The on-goal half shipped: per-opponent color-void beliefs
(obs 150 -> 162), retrained mixed-count from scratch x3 seeds. Behaviourally a clean win — the
probe (`probes/r4_color_void.md`) shows every seed steers onto the color the next player lacks,
with M1s3 as a perfect S=0 control. On win rate it is a **tie** with M1s3 (small edge only at two
seats), so it fails the promotion standard and is NOT merged; the champion stays M1s3. This is the
info-set-ceiling thesis again — a legitimate human-reachable feature adds behaviour, not win rate.
The off-goal half stays rejected. See RESEARCH_LOG 2026-07-21. *(Original item below.)*

**R4. Belief-state features — SPLIT by the target. One half is off-goal, one half is on-goal.**
Written before the project target was recorded; the two bullets it proposed are now on opposite
sides of the line.
- ~~The full **63-slot unseen-count vector**~~ — **REJECTED, OFF-GOAL.** It is exactly perfect card
  counting: "total - my hand - everything I have seen played". No human tracks 63 running counters.
  It would raise the win rate by handing the agent a memory no human has, which is the one thing the
  project exists *not* to do. Do not implement. (This is precisely the trap the target section warns
  about — the idea is attractive on strength grounds and wrong on goal grounds.)
- **Per-opponent color-void flags — KEEP, and it is now arguably the most on-goal feature idea in
  the file.** When a player draws (or plays off-color) with color X showing, they likely lack X.
  This is *the* core human-level inference in UNO-family games — a human absolutely does notice
  "Dana hasn't played green all game" — and it is currently invisible to the net because it requires
  history. 3x4 "hasn't shown color X since last draw-on-X" bits, ~zero cost. This does not widen the
  information set beyond human reach; it *closes a gap* where the net is currently **sub**-human, and
  is therefore legitimate. Invalidates checkpoints — batch with the next contract break.

**B-series. Behavioural probes — THE DELIVERABLE.**
Interrogate a trained policy's strategy in constructed positions: build the position directly, dump
Q-values over the legal moves, and compare against the strategically correct line. Needs no new
training — runnable against A8 today. Requires a small harness (construct a `Game` at an arbitrary
state, ask an agent for its Q-vector, pretty-print the ranking); `gametest.py`'s hand-built states
and `main.py`'s greedy demo already show every piece needed.

- **B1. The colored-TAKI hoard — HALF RETRACTED (2026-07-12), REDO after the retrain.**
  **Sequencing: RETRACTED — the finishing rule was a bug.** The engine only let a hand end on a
  number or the King; the real rule is **any card except PLUS**. So B1a's "the run must end on the
  red 5" was false — the STOP and the +2 are finishers too, every ordering wins, and the scenario
  had no trap in it. A8's deferral of the 5 was real behaviour, but it was skill at a game that
  isn't Taki, so the "plans the run backwards from its finisher" claim (DiD +15.6, worth +0.40 win
  rate) is **withdrawn, not merely re-measured**. Rebuilding it needs a **PLUS in hand** — that is
  now the only card that constrains a run's last play, which makes "sequencing skill" a much
  thinner concept than B1 assumed. *That thinness is itself a finding: in real Taki there is
  almost nothing to sequence.*
  **Hoarding: FAIL — and this half SURVIVES the rule fix** (b1c never touched finishing: no hand
  in the sweep can empty, so no finisher is ever chosen). Still must be **re-run against the
  retrained champion**, since A8 itself learned the wrong game. As measured: it correctly keeps a
  weakly-backed TAKI (sheds at ≤1 backers) but dumps from 2 backers up — while the rollouts say **dumping never beats keeping at
  any backing level, against either opponent pool**; at 4 backers a five-card turn wins *less* often
  than a one-card turn (−0.033 ± 0.011). **This file's own premise — "hoarding is nearly free" — is
  confirmed by measurement, and the policy is what doesn't fully believe it.** Cause: the shaped
  reward pays `-len(hand)` per step, so the training signal massively overvalues a big discharge the
  win rate is indifferent to → the strongest evidence yet for **B2** and **R6**, and the motivation
  for **B4** below. See `probes/b1_colored_taki_hoard.md`; harness `probe.py` / `probetest.py`.
- **B2. Holding cards back — DONE (2026-07-13). The premise was false; the answer is worth 13 points.**
  **"Holding back" is two behaviours with opposite signs**, a distinction nothing here had drawn:
  **REFUSAL** (decline to play at all — DRAW, or CLOSE a TAKI/King continuation, while a legal play
  exists) is **catastrophic**; **PREFERENCE** (play a *different* card) is **valuable**. Measured by
  ablating R3's heuristic — the only agent in the repo that holds back on purpose, and in terms we
  own — rather than by asking the DQN's own value function, which would be circular.
  **A 13-point cliff sits exactly at its `SCORE_DRAW = -5.0` threshold**: holding the King is +2.0
  pts at `p_king=5` and **-10.4 pts at `p_king=6`**, where it starts *drawing* rather than playing it.
  **The champion already has this right** (census: **1.8%** refusal, **20.4%** preference hold-back —
  it is *not* a greedy shedder), and **the shaped reward is why**: `-len(hand)` punishes drawing, so
  it teaches the single most valuable rule in the game. **On shedding it is aligned, not biased.**
  **BUT the second half of B2 — weapon timing — fails, and there the reward IS the culprit.** With the
  next player one card from winning the champion **declines to block**, which rollout prices at
  **+0.050 +/- 0.017** for blocking; and its Q values the *identical hand* **3.4 higher** when an
  opponent is about to win, because **a loss pays nothing** and an imminent defeat truncates the
  `-len(hand)` stream. **So B2 kills R6's hold-back justification and proves its defensive one.**
  *This also resolves B1's apparent contradiction* — B1 only ever compared play-vs-play (the
  preference regime), where its "keeping beats dumping" finding is confirmed exactly.
  See `probes/b2_holding_back.md`; harness `holdback.py`.
  > **Statement about Taki: never draw to protect a plan.** Keeping a card by playing something else
  > is free or better; keeping it by passing costs more than the card is ever worth.

- **H-series (new, 2026-07-13, spun out of B2). Rebuild the heuristic around the refusal/preference
  distinction.** B2 measured that *how* you hold a card back matters more than *whether* you do:
  **refusal** (decline to play at all) is worth **-13 pts**, **preference** (play a different card) is
  worth **+2..+7**. The agent has no way to express the second without risking the first, which makes
  every one of its hold weights a latent cliff. These items follow directly.

  - **H1. DONE 2026-07-14 — and it was worth +0.183, with ZERO weights changed.** `refusal_mode=
    'structural'` ships; versions `h1` (R3's weights + structure) and `h1b2` (B2's weights +
    structure) are frozen in `agents/heuristic.py`. The pre-registered invariant holds as a property
    test (no weight assignment, up to holds of 500, can produce a voluntary draw) and in the census
    (draw refusals: R3 1293 -> H1 **0**). Vs the R6 champion the heuristic closes from **-0.263 to
    -0.065**. **What it settled:** the ~18 points were the cliff *entire* — retuning (B2) and
    restructuring (H1) recover the same points and **do not stack** (`h1` vs `b2` = -0.007, a tie).
    H1 is still the one to keep, because it makes the fix a *property* rather than a tuning outcome —
    which is precisely what makes H8 safe to run. **Also found:** B2's retuned point *still refused*
    (via `w_nofin`, untouched at 8.0 — H5 was real), and the costliest refusal in the agent was
    `_play_draw_two` eating a +2 pile to keep the King, which lives outside the `max()` entirely.
    See RESEARCH_LOG 2026-07-14. *(Original item below, for the record.)*

  - ~~**H1. Make "never refuse to play" a STRUCTURAL INVARIANT, not a tuning accident.**~~ *(The headline.)*
    Today `SCORE_DRAW = -5.0` is a finite score competing in the same `max` as the plays, so any hold
    penalty above 5.0 silently converts "I would rather keep this" into "I would rather not play at
    all" — a **13-point cliff** that `p_king = 6.0` (the shipped default!) falls straight off. Split
    the decision in two:
      1. **Refuse?** Only when a *rule* says so: no legal play, or every play is forbidden by the
         finishing rule (a lone PLUS). Never because a card is precious.
      2. **Which play?** Rank by preference. Hold penalties live *here*, where they can reorder plays
         but can never outvote playing at all.
    Then hold weights become **unbounded-safe** — you can say "I really want to keep the King" without
    that ever meaning "so I will draw instead". Pin it with a test: *no weight assignment can produce a
    voluntary draw.* B2's census (`holdback.py`) already measures exactly this (the ablated agent reads
    0.00% refusal), so the invariant is checkable, not just asserted.

  > **STEP 4 IS DONE (2026-07-14) AND IT WENT BADLY FOR THIS LIST.** Every pre-registered prediction
  > below failed; H5 and H6 **reversed sign**. The cause is one thing, and it is worth internalising
  > before writing any further H item: **B2's hold-back ablations were measuring the REFUSAL CLIFF,
  > not the hold-back.** Any weight B2 priced at or above 5.0 was priced against that confound. See
  > RESEARCH_LOG 2026-07-14 (later) and `probes/h6_cliff_confound.log`, which sweeps `w_reserve` on
  > both structures and shows LEGACY's "catastrophe" tracking the refusal count exactly while
  > STRUCTURAL merely goes flat.

  - **H2. REFUTED 2026-07-14 — threshold 1 is WORSE (-0.0112 +/- 0.0024, ~5 SE).** The shipped
    threshold of 2 is at or near the optimum; widening to 3 is free (+0.0010, tie). *The caveat B2
    wrote down was the whole story:* its rollout priced the block with the **DQN as the acting seat**
    and noted "they should transfer, but confirm with the heuristic as the actor" — **they did not
    transfer.** A per-decision rollout price (k=1 -> +0.02..+0.045) did not survive being wired into
    an agent that then has to live with the consequences. Treat that gap as the standing lesson: a
    priced *decision* is not a priced *policy*. ~~Original item below.~~
    ~~The blocker fires too late AND too widely — `BLOCK_HAND_THRESHOLD = 2` looks wrong. Predicted
    fix: threshold 1, worth ~2-4 points.~~

  - **H3. DONE 2026-07-14. The spend side EARNS its keep: `w_block=0` costs -0.0223 +/- 0.0026.**
    So blocking is neither mis-aimed nor merely weak — it is worth **+2.2 points**, the largest
    single behaviour measured in the agent. The hole in B2's coverage is filled. **New finding:**
    `w_chdir_block` is **dead weight** (-0.0010 +/- 0.0015, tie) — a candidate for deletion on
    simplification grounds, not performance.

  - **H4. CONFIRMED dead 2026-07-14 — delete the branch, do NOT zero the weight.** `score_decline_king`
    fires **0 times in 200 games in every version**, and `-100` changes nothing (+0.0001 +/- 0.0001),
    so removing it is free. But setting it to **0** measures **-0.0004 +/- 0.0002**: at 0 the option
    starts *winning* the `max()` against negative-scoring plays and the phantom behaviour wakes up —
    badly. That is the general lesson for retiring dead code here: **remove the branch, don't
    neutralize the weight.**

  - **H5. REFUTED 2026-07-14 — `w_nofin = 8.0` EARNS +1.1 points** (dropping it to 0 costs
    -0.0107 +/- 0.0008; to 2, -0.0058). The premise was that it "guards a nearly-vacuous rule"; it
    does not. What *was* real is the other half: at 8.0 it sat above the draw threshold and could
    make the agent refuse — and B2's retune **missed it**, so B2's published point still drew 4 times
    in 40 games. **H1 already fixed that half by construction.** Keep the weight, keep the rule.

  - **H6. REFUTED 2026-07-14 — the hoard is worth +2.2 to +2.9 points. B6 STAYS.** `w_reserve=0`
    costs -0.0224 +/- 0.0029 and the full `-hoard` ablation costs -0.0289 +/- 0.0030. Both halves of
    the original claim were wrong: the payoff is real, and the "large downside" (catastrophic at
    8/10) was **the refusal cliff, not the hoard** — under H1 those same values are ties (-0.003 at
    `w_reserve=20`, with zero voluntary draws). B2 even named the mechanism in its own parenthesis —
    *"(where it starts drawing)"* — and did not follow it. ~~Original item: "the hoard buys nothing —
    consider deleting B6 entirely ... complexity with no measurable payoff and a large downside".~~

  - **H7. IMPLEMENTED AND REFUTED 2026-07-14 — racing LOSES, monotonically.** The behaviour ships
    (`race`, `race_hand_threshold`, `race_hold_scale`) but is **off**, and should stay off. Vs `h1b2`,
    20k games x 2: dropping the holds harder makes it *worse* (`race_hold_scale` 0.75 -> -0.003,
    0.5 -> -0.014, 0.25 -> -0.023, **0.0 -> -0.035**) and racing more often makes it *worse*
    (`race_hand_threshold` 1 -> -0.015, 2 -> -0.035, 3 -> -0.054, **4 -> -0.066**). Both knobs say the
    same thing: **every increment of racing costs points; the optimum is not to race.** (Re-measured
    after fixing a real bug in the first cut — racing forced the hoard open without checking the run
    could legally be *ended*. The fix changed the number by 0.0001, so the result is the game, not the
    bug.) **Why it is interesting:** this is H5/H6 from the other side. The hold-backs are worth
    +1.1 (`w_nofin`), +2.9 (hoard) and +2.2 (blocker-spend), and they are worth **most in the
    endgame** — which is precisely what racing discards. *Statement about Taki: when someone is about
    to win, do not empty your hands faster; that is when your kept cards are worth the most.*
    ~~Original item: "There is no RACE behaviour ... a human speeds up. This is the natural
    counterpart to H2 and is currently absent."~~

  - **H8. DONE 2026-07-14 — and the freed space was EMPTY.** Version `h8` ships. 2 passes x 44
    candidates x 12k games from `h1b2` vs the R6 champion. **Not one hold-back weight moved**
    (`p_king` 5.0, `w_reserve` 4.0, `w_nofin` 8.0, `w_open_hoard` 15.0 — all unchanged); every gain
    came from ordinary terms (`w_deny` 3->1.5, `w_block` 4->6, `w_chdir_block` 2->0, `w_plus_tempo`
    1->2, `w_save_blocker` 0.7->0). H8's whole rationale was that H1 lets the search go **above the
    old cap of 5.0** — it went, and came back with nothing, because up there the holds **saturate**
    (`p_king` at 6/9/14 returns margins identical to four decimals: past a threshold the hold is
    already decisive and more cannot move the `argmax`). **The cap was never costing anything; the
    cliff was.** H1 was still necessary — it is what makes the search *safe* — but the points were
    not hiding up there. **Winner's curse, observed as pre-registered:** in-search -0.0472 did not
    replicate; held out, `h8` beats `h1b2` by a real but smaller **+0.0118 +/- 0.0035** (20k common
    decks) and reaches **0.906** vs random. Two independent confirmations of H3 fell out — the tuner
    zeroed `w_chdir_block` (dead weight) and `w_save_blocker` (a tax on the agent's best behaviour).

- **B5. DONE 2026-07-14 — PROMOTED. `REFERENCE` is now `h8`, and the champion got smaller.**
  Mode A, 1-vs-3, parity 0.250 (the `r3` column reproduces the published numbers exactly, which is
  the check that the harness is faithful):

  | | vs `r3` (old) | vs **`h8`** (new) |
  |---|---|---|
  | **R6 champion** | **0.378** | **0.297** |
  | A9-rules | 0.348 | 0.271 |

  **The champion's headline was inflated by ~8 points by a yardstick with a 13-point bug in it.**
  R6 beats a *competent* heuristic by **+4.7 points over chance, not +12.8**. It is still genuinely
  ahead — the DQN does beat the best hand-written rules we have — but by less than half what the
  record claimed. **(b) from the original item is answered, and it cuts the other way from what B2
  guessed:** against a competent opponent the champion is *not* at parity with hand-written rules;
  it is ahead, but modestly. **(c) is answered too:** a real tuner (H8) went further and the
  heuristic still does not beat the DQN. **Every "vs heuristic" number published before 2026-07-14
  refers to `r3`** and is not comparable to one measured after; `heuristic:r3` still runs.

- **B3. Scenario battery + human-readable output.** Generalize B1/B2 into a small suite of named
  positions with an expected/interesting line each, and report the policy's Q-ranking per scenario.
  This is what "testing optimal scenarios for human players" ultimately produces: not a win rate, but
  a set of **statements about how to play Taki well**, each backed by the model's own valuation.
  *B1 built the machinery:* `probe.py`'s `Scenario` registry means a new probe is **data, not code** —
  add a hand, a top card, the expected-legal/illegal moves and (optionally) named forced lines.

- **B4 (new, spun out of B1). Measure the shaped return Ĝ, and split "wrong move" from "wrong
  reward".** Q is not a win probability — it is the discounted shaped return `train.py` pays
  (`-len(hand)` per step, γ=0.99 per *decision*, and a TAKI run is many decisions). So when the
  policy's Q-ranking and the rollout win rate disagree, today we cannot say *why*. Measuring the
  empirical Ĝ alongside the win rate in `mc_line` splits it cleanly: **Q ≠ Ĝ** means the value
  function mis-estimated its own objective; **Ĝ ≠ win rate** means the objective itself is pointed
  the wrong way. The second is exactly the evidence B2 and R6 need, and the rollouts already exist —
  it is bookkeeping (faithfully replicating `seat_reward`'s per-decision discounting), not new
  machinery. Deliberately cut from B1 because B1's correct line is provable from the rules and
  needed no value-calibration argument.

Open question worth settling early: **is A8 strong enough to be a trustworthy oracle?** R3's
heuristic/oracle yardsticks are the honest way to find out, and they matter more now — a probe is
only as credible as the policy it interrogates.

**R5. Use the simulator we own — search as measurement, then maybe as a method.**
The whole line so far is model-free, yet `game.py` is a perfect, fast simulator. The cheapest
version isn't a commitment to MCTS — it's a *measurement*: wrap the current net in a **1-ply
rollout / determinized lookahead** at eval time (for each legal move, sample a few determinizations
of hidden hands consistent with the unseen counts, roll out with the frozen net, pick the best) and
play it against raw A8. The gap is the policy-improvement headroom of the current value function. If
it's large, the long-term path is expert iteration (distill the search-improved policy back into the
net, AlphaZero-style with determinization); if it's small, the value function is already consistent
with its own improvement operator and the bottleneck is elsewhere. Nothing in this file previously
touched this direction.

**R17 (new, 2026-07-13, spun out of R6's failure). Potential-based shaping — the step term is a
LENGTH TAX, and that is what actually blocks defence.**
R6 (a terminal loss penalty) **fixed the valuation and not the policy** (RESEARCH_LOG). It confirmed
B2's causal story — a near-winning opponent flipped from good news (Q -6.56) to bad news (-20.07) —
but the champion still would not block. The reason is a tension B2 missed:

> **One reward term is doing two jobs, and they conflict.** `-len(hand)` per step teaches *"never
> refuse to play"* (worth ~13 pts, B2 Arm 1 — the champion's best habit). It **also charges ~3 per
> extra turn**, and *every defensive move prolongs the game*. So the term that makes the agent good
> at shedding is the term that makes it bad at defence. A one-off terminal penalty cannot outweigh a
> per-turn tax.

**The fix is to shape on the CHANGE in hand size, not its level.** Replace the level penalty with a
potential-based term `F(s,s') = gamma*PHI(s') - PHI(s)`, `PHI(s) = -c*len(hand(s))`:
- Shedding a card pays ~`+c`; drawing one pays ~`-c`. **So "never refuse to play" is preserved** —
  which the B2 census can verify directly (the refusal rate must stay ~2%).
- **Standing still pays ~`c*(1-gamma)*len(hand)` ~ 0.01c, i.e. nothing.** The length tax disappears,
  and defence stops being punished for prolonging the game.
- Ng et al.: potential-based shaping is **policy-invariant** w.r.t. the base reward — so the base
  reward must then carry the real objective, i.e. **win bonus AND loss penalty** (R6's term, which we
  now know is necessary but not sufficient).

**Two things to get right, both easy to get wrong:**
1. **`PHI(terminal) = 0` is required for the invariance theorem** — but taken literally it pays a
   *losing* seat `+c*len(hand)` on its terminal transition (a bonus for losing with a big hand). That
   is only sound because the base reward's loss penalty dominates it; size `L` against `c*8` and
   check the sign, or the fix reintroduces the very pathology it is meant to remove.
2. `train.py` currently stores `pending[seat] = (state, action)` and computes the reward at *close*
   time from `len(hand)` then. PBRS needs `len(hand)` at **both** ends, so `pending` must also carry
   the hand size at action time (a one-line change at train.py:336).

**The evidence for R17 is now much stronger than when it was written.** R6 was run to completion at
two magnitudes and the pattern is unambiguous: **L=20 -> delta(k=1) = -3.23; L=60 -> -0.87.** The
curve is being *shifted*, not *reshaped* — a terminal penalty moves the intercept a little and cannot
cross zero, because it is fighting a **per-turn** tax with a **one-off** payment. Meanwhile the
*slope* is already correct at L=60, i.e. **the agent knows the threat is there and still will not
spend a card on it.** That is exactly a length-tax signature. Removing the tax (shape on the CHANGE
in hand size) is the only lever left that addresses the mechanism rather than shouting over it.

**Acceptance test: `r6_accept.py` already exists and is pre-registered.** `delta(k=1)` must flip sign
AND the refusal rate must not rise. **R6 passed the second and failed the first at BOTH L=20 and
L=60; R17 must pass both.** Note R6 also shipped a new champion while failing this test — so R17 must
be judged on the *test*, not on win rate alone, or it will be mis-graded the same way.

**R6. The loss signal is missing — DONE (2026-07-13). Shipped a NEW CHAMPION (+8.5 pts), and STILL
FAILED its behavioural test. Both halves matter.**
`train.py --loss-penalty` (default 0.0; the champion is trained with **60**). B2 found the reward pays
`-len(hand)` every step and a bonus only on a win — **a loss pays nothing**, so ending the game is
itself a reward and an imminent defeat is a relief.
- **STRENGTH: a decisive win.** `checkpoint_r6L60_snap100000` beats the previous champion by
  **+0.085 +/- 0.011 per-seat** (seat-swapped 2v2, three disjoint deck blocks, ~8 SE) and scores
  **0.378 vs 3 heuristic** (was 0.348) — in **100k trials, not 500k**. Strength is **monotone in the
  penalty** (L=0 -> 0.348, L=20 -> 0.366, L=60 -> 0.378), so one more run at a larger L is cheap and
  probably still worth it.
- **BEHAVIOUR: the pre-registered test FAILED, and this is the finding.** `delta(k=1) = -0.87` —
  much closer to zero than the old champion's -4.01, but **still negative: it does not block a
  one-card opponent.** What changed is the **slope**: the old champion wanted the +2 *least* when the
  threat was greatest (backwards); R6 wants it *most* then (correct). **R6 fixed the sign of the
  derivative, not the intercept.** It learned *that a near-winner is dangerous* without learning *to
  spend a card on one*.
- **So strength and the behavioural fix came apart.** The +8.5 points did **not** come from blocking —
  it still does not block. They come from removing the "imminent loss is a relief" pathology
  *everywhere*. **Had we only measured win rate we would have declared the defect fixed.** That
  dissociation is precisely what the pre-registered test existed to reveal, and it is the strongest
  argument in this file for keeping behavioural probes alongside win rates.
- **Guard-rail passed** (census refusal 2.92% vs 1.75%): "never refuse to play" survived.
- **Remaining defect -> R17.** Playing the +2 still *prolongs the game*, and `-len(hand)` still charges
  ~3 per extra turn. A terminal penalty shifts the whole `delta` curve but cannot pay off a *per-turn
  tax* on defence — the level term is an **intercept** problem. Out-shouting it does not work.

Original note follows (its diagnosis was right).

**R6. The loss signal is missing — losers get no terminal penalty.**
In `train.py`'s `seat_reward`, a losing seat's terminal transition is just `-len(hand)` — there is
no explicit loss event. The winner gets a bonus, but a seat that lets the player to its left win off
a 1-card hand receives exactly the same reward as one that fought it. Defensive play (holding a
+2/STOP for the near-winner — the unseen counts and hand sizes are *in* the observation since A7,
but nothing rewards using them) is under-incentivized. A terminal loss penalty (e.g. minus the
winner's margin, or a constant, ideally inside a potential-based formulation per A9) is a one-line
reward change and one of the few reward experiments *not* yet run — the reward lessons learned so
far are all about the win-side/step-side terms.

### Reconsiderations of existing results and plans

**R7. The eval.py seat-0-bias claim is wrong — CONFIRMED 2026-07-11, re-derived against the source.**
The 2026-07-06 RESEARCH_LOG entry states `play_match` "always seats the test agent at seat 0", and
the "Infrastructure / evaluation optimizations" section above used to carry a "fix seat-0 bias in
eval.py" item. But `play_match` has had per-game seeded seat shuffling since the *original* eval
harness commit (`e5feaa8`) — [eval.py:81](eval.py#L81) `seat_rng.shuffle(order)` — so the test agent
occupies the advantaged opening seat in ~25% of games, exactly its parity share. **Verified directly
in the source. R7 is correct:** (a) Mode-B `--baseline` numbers are *fair in expectation*, not
inflated as the log caveats claim; (b) the "fix seat-0 bias" item targeted a non-bug and **has now
been deleted** from the section above. The seat-swap protocol for `eval_headtohead.py` remains fully
valid (that script really does take explicit fixed seats), and no ranking conclusion changes.

This bit for real during A11: the 2026-07-11 entry's first draft blamed a "seat-0 artifact" for the
gap between screening (0.273) and the gate (+1.00 pts/seat). That explanation was false and has been
corrected in the log. The true reason is mundane — **Mode B is a 1v3 win rate, the gate is a 2v2
per-seat rate**; different quantities, no confound needed.

**R7 is now fully CLOSED (2026-07-11).** Established via `git log -S`: the shuffle landed in
`e5feaa8` (2026-06-29), the *original* eval-harness commit — a week **before** the 2026-07-06 entry
that denies it. So the claim was **wrong when written**, not merely stale. All three carriers are
fixed: the non-bug item is deleted from the section above; the 2026-07-06 RESEARCH_LOG entry now
carries a prominent correction (original text preserved); and CLAUDE.md now states positively that
Mode A/B seating is shuffled and its numbers are fair, plus that Mode B (1-vs-N win rate) and
`eval_headtohead.py` (per-seat rate) are different quantities that should not be compared directly.

**R8. The promotion pipeline has a winner's-curse problem, and the +2.0 bar is thin.**
A10 screened 66 snapshots, A11 screened 48, each taking the max of a ~+/-2 pt noise band into a
single confirmation whose own SE is ~1 pt/seat. Picking the best of ~50 noisy readings and then
testing it once biases the confirmation upward too (the log correctly flags this for A11 but the
process is unchanged). Suggested tightening: any candidate that clears the bar gets **one fresh
replication at a different eval seed** before promotion, and the bar becomes "clears +2.0 twice"
rather than once. Equivalently: the +2.0 bar's only justification is "smallest margin ever promoted
on" — with SE ~= 1, that's a 2-sigma bar applied post-selection, which is closer to a coin flip than
it looks.

**R9. Pairwise promotion chains assume transitivity — run one round-robin.**
Every "current best" was crowned by beating exactly one predecessor. Self-play lineages are
notorious for non-transitive cycles (A beats B beats C beats A). A single round-robin tournament
among the promoted checkpoints (`shaped_snap300000`-era models excluded by the encoding break, but
`a4a7_snap550000`, `a8_snap455000`, top A10/A11 candidates, plus a heuristic agent from R3) with
seat-swap controls would either confirm the linear ordering or reveal cycling — and cycling would
itself be evidence for the equilibrium-trap hypothesis and the opponent pool. The log's own
mixed-table caveat (the swap control only covers homogeneous tables) points the same direction and
has never been followed up.

**R10. Opponent pool (#1 above) — still the right lead among the existing items, with two design notes.**
Its ranking stands, but run R2 (exploitability probe) first — it's cheaper and tells us how much the
pool can possibly buy. Design notes beyond what's recorded above: (a) keep opponents' epsilon in the
sweep — today the learner *never* trains against greedy play (opponents fixed at eps=0.1) yet is
always evaluated greedy; a pool of frozen snapshots played at eps=0 both diversifies and closes that
train/eval mismatch. (b) The A8-dilution tension noted in A5 resolves cleanly with the probe result
in hand: if exploitability is high, diverse data matters more than on-policy purity.

**R11. Epsilon floor / exploration (#4 above) — the evidence for it is stronger than its ranking.**
Three separate runs now show gains concentrated *after* the epsilon floor (the 1000-trial run, A8's
snap455000 peak sitting past the 440k floor, and A11's verdict being voided by never reaching it).
That's the most consistent training-dynamics signal in the whole log, and it suggests the tail
matters more than the schedule shape. A lower floor (0.02-0.05) and/or reaching the floor at 50% of
trials is a one-flag experiment; run it *before* n-step or LR decay, and arguably before finishing
A11 — cheaper and better-evidenced.

**R12. A11 completion at 450k trials — fine, but rank it honestly.**
It's cheap and settles an open verdict, but both architecture experiments to date came back empty,
and the screening band was flat across all of training. Expected value is low; treat it as background
work behind #1/R2/R11, not a gate for anything.

**R13. A10's capacity caveat — only retest capacity jointly with LR.**
The single-knob methodology has served the project well, but capacity x learning-rate is a known
non-separable pair (bigger nets at lr 1e-3 with MSE on rewards of magnitude ~50 can easily just be
less stable). If capacity is ever revisited, do it as a declared package (width + lr 3e-4 + maybe
Huber); otherwise consider capacity closed and stop paying for it.

**R14. Training-time diagnostics (recommended-process item 5 above) — do it before the next long run, and add policy churn.**
The byte-identical converged-policy discovery (control run, 2026-07-04) was found by accident through
eval. A probe set of ~1k fixed states with mean max-Q, mean |TD error|, and **fraction of probe states
whose argmax changed since the last snapshot** costs microseconds and would have caught the win-only
collapse, the vanilla divergence, *and* the policy freeze in-run. Three WSL-wedge-killed runs in A11
alone make cheap in-run observability worth more than another experiment slot.

**R15. The unresolved vanilla-divergence discrepancy — deprioritize, don't bisect.**
The open puzzle (pre-S1-S8 control trains fine at 300k; post-S1-S8 control diverges 2/2 at 100k) is
intellectually interesting but practically moot: color-sym is default and stabilizes everything, and
the hygiene package is confirmed inert on top of it. A bisect would cost several multi-hour runs to
explain a configuration we never intend to run again. Treat as closed-unless-relevant.

**R16. If the DQN line truly exhausts: the principled families are policy-based.**
For completeness — 4-player imperfect-info Taki has no convergence guarantee under any self-play
Q-learning scheme. If the exploitability probe says headroom exists but the pool doesn't capture it,
the established next tier is NFSP / PPO-with-league-play (population-based, a la AlphaStar-lite),
with Deep CFR almost certainly overkill at this scale. This is a rewrite, not a lever; nothing in the
current evidence forces it yet.

### Suggested priority order — SUPERSEDED (2026-07-11), kept for the record

The original list is below, struck through. Its top two items are **done** (R2 and R1, both 2026-07-11,
both negative), and it was written before the project target was recorded — so it ranks strength levers
above the actual deliverable. Items 3-8 remain live but are now *means*, not ends.

~~1. Exploitability probe (R2) — DONE, negative (A8 near-unexploitable).~~
~~2. Rank-symmetry augmentation (R1) — DONE, negative (parity).~~
~~3-8. epsilon floor / opponent pool / belief features / round-robin / rollout / background.~~

### Priority order (2026-07-11, target-aware)

1. **B-series behavioural probes (B1 the TAKI hoard, B2 holding cards back)** — **the deliverable.**
   No new training, runnable against A8 today, and directly answers the question the project exists to
   ask. Everything below is only worth doing if it makes this more trustworthy.
2. **Heuristic/oracle yardsticks** (R3) — how good *is* A8 in absolute terms? A probe is only as
   credible as the policy it interrogates, so this now gates the interpretation of every B-result.
3. **Color-void flags + loss signal** (R4-second-bullet + R6, one contract break) — the two remaining
   changes that are *fidelity* fixes rather than strength hacks: one closes a sub-human gap in what the
   net can infer, the other fixes a reward that never punishes losing. Note this invalidates checkpoints
   and requires a retrain, so it is the one strength-adjacent item worth its cost.
4. **Rollout-improvement measurement** (R5) — reframed: not "is search the path to a stronger bot" but
   **"is A8's value function self-consistent enough to trust as an oracle?"** A large improvement gap
   would mean its Q-values are a shaky basis for the B-series verdicts.
5. Background / opportunistic: epsilon floor (R11), opponent pool (R10), round-robin (R9), training
   diagnostics (R14).

**No longer needed:** the promotion-bar replication rule (old item 8) — **done 2026-07-11**. sigma is
measured at **0.5-0.7 pts** (3 seeds, pre-committed final snapshot), so A8's +3.7 was ~5.7 sigma and the
+2.0 bar is ~3 sigma. Both sound; nothing in the log needs retracting. The live hazard is **snapshot
argmax**, not seed luck (within-run snapshot SD 1.25 pts vs seed-to-seed 0.51) — see R8, now partially
answered: pre-commit the snapshot before the confirming head-to-head, and never report the max of a
selection set as the edge.


-- I am training a model for 2p game against the heuristic agent. Then refine the heu. agent if possible also separately for 2p and 3p game. rerunning the trainings with one hot encoding of the number of players. Only then consider next step.

-- possibly remove all the co-authored by: claude
-- possibly try a rewrite in pytorch. First as if there is advantage
   and train + test vs. random.

-- create a new github repo which is mine with all the commits and same dates. the project has strayed significantly from the original.

