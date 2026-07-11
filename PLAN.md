# Taki AI — Review Findings & Fix Plan

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

-- Fix seat-0 bias in eval.py's play_match (Mode A/B) — see RESEARCH_LOG.md 2026-07-06
   "seat-0 first-mover advantage" entry for the finding and CLAUDE.md's Evaluation
   discipline section for the current workaround (eval_headtohead.py).

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

1. Opponent pool of past snapshots (+ occasional random seat) (A5) — likely the best lever to
   break the ~0.90 plateau.
2. ~~Dueling head (A11)~~ — **DONE, NEGATIVE. Closed.** See RESEARCH_LOG.md 2026-07-11 and
   branch `a11-dueling-head`. `Q = V + (A - mean_legal(A))` at parity with
   `checkpoint_a8_snap455000`: best swap-controlled edge **+1.00 pts/seat** (bar is +2.0),
   and the top post-floor snapshot replicated across 3 seeds gave **mean +0.25** once seat-swapped.
   The 450k rerun *did* train ~90k post-floor trials (the one caveat of the earlier inconclusive
   result), so this is now a clean negative. **Both halves of the architecture lever are now
   closed** — neither width (A10) nor dueling beats A8 with hyperparameters held fixed.
   Watch-out recorded there: Mode B screening seats the test agent at seat 0, so its vs-baseline
   numbers above 0.25 are confounded and inflated *uniformly* — they looked like a replicating
   post-floor effect and were not. Rank with Mode B; decide only with the swap-controlled gate.
3. n-step returns (n=3-5): shortens the bootstrap chain; large contributor in Rainbow ablations
   even with dense rewards. Cheap to implement in the buffer.
4. Lower epsilon floor late in training (0.1 -> 0.02-0.05, or decay to floor by ~50% of trials) —
   the 1000-trial run showed gains concentrated *after* epsilon bottomed out.
5. LR decay for long runs (1e-3 -> 1e-4 cosine/step) — the faint 100k-1M creep is consistent with
   bouncing around a minimum at too-large a step size.
6. Prioritized replay: medium effort, real but smaller expected gain here; do after 1-3.
7. If revisiting reward: potential-based shaping (A9), high epsilon + fixed opponent set during
   the switch.

The opponent pool (#1) is now clearly the lead. The architecture lever is exhausted: capacity
(A10) and dueling (A11) are both done and both negative, with hyperparameters held fixed. That
makes the remaining candidates *algorithmic* (n-step, epsilon floor, LR decay, prioritized
replay) or *opponent-distributional* (#1) — and the standing evidence is that "more of the same
self-play" is the binding constraint, which is exactly what #1 attacks.

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

**R1. Rank-symmetry augmentation — the direct sequel to color-sym.**
Verified in `game.py`: the nine number ranks (ONE..NINE) are *fully interchangeable*. The deck has
2 copies of each per color, matching is "same color or same type", the opening rule (must open on a
number) and finishing rule (numbers + King) are rank-set-invariant, and no rule branches on a
specific rank. So any global relabeling of the 9 ranks is an exact symmetry of the dynamics — a
group of 9! = 362,880 permutations, composable with the 24 color perms (~8.7M total relabelings).
Same trick as color-sym, implementable with the same `OBS_PERMS`/`ACT_PERMS` machinery (permute the
9 rank slots within each color block + shown-card block; the unseen-count features track only
+2/King/CHCOL, which are rank-invariant). Given that color-sym is the single largest gain the
project has ever produced, and that the one strong empirical regularity is "data-side symmetries
pay", this is the highest-prior cheap experiment on the board. Ranks do carry a strategic role
(same-rank cross-color chaining), but relabeling preserves it exactly.

**R2. Exploitability probe (best-response training) — the highest information-per-cost experiment.**
Train a fresh learner against **3 frozen `checkpoint_a8_snap455000` seats** — no opponent sync, no
self-play. This directly measures how exploitable the champion is:
- If the best-response climbs well above parity against A8, there is real headroom in the current
  architecture, self-play is the bottleneck, and the opponent-pool work (#1 above) is strongly
  justified.
- If it can't beat A8 by more than a couple of points, A8 is near-unexploitable within this function
  class, and further effort should go to representation (belief features) or search, not to training
  scheme variations.

Reuses the existing loop almost verbatim (disable `OPPONENT_SYNC_EVERY`; skip A8-style collection
from the frozen seats to keep the buffer on-policy-ish). The flat parity bands in A10/A11 screening
— different architectures, different seeds, all landing at 0.23-0.30 against A8 — are consistent
with *either* hypothesis; this experiment separates them.

**R3. Estimate the skill ceiling with an oracle and a heuristic baseline.**
The project currently has only two external yardsticks: random opponents and its own lineage. Two
cheap agents would recalibrate everything:
- **A full-information oracle** (an agent that sees all hands — trained the same way, or even a
  greedy full-info heuristic) upper-bounds what any policy could do; the gap between it and A8
  measures how much the hidden information is worth, i.e. how much belief-state features (R4) can
  possibly buy.
- **A hand-crafted heuristic agent** (hold wilds, dump colors you're rich in, block the near-winner
  with +2/STOP) as an independent eval opponent and pool member. Right now a lineage-specific blind
  spot would be invisible: every checkpoint has only ever been ranked against its own ancestors.
  Taki vs 3 random at 0.91 may already be near the luck-imposed ceiling — an oracle run would tell
  us whether the remaining 0.09 is winnable at all, which determines whether vs-random should be
  retired as a metric entirely.

**R4. Belief-state features — reconsider what A7 removed.**
Taki is a POMDP and the DQN is a reactive policy on a single frame. A7 *removed* the discard-pile
histogram, deliberately trading card memory for hand-size/deck features — which worked, but it
deleted the agent's ability to count cards, leaving only 3 coarse unseen counts. The principled
version of "richer observation" is the **belief state**, and two cheap approximations exist:
- The full **63-slot unseen-count vector** (total - my hand - discard = what opponents+deck could
  hold). This is the sufficient statistic for the hidden deal, it's a few lines in `observation()`,
  and it is strictly more informative than the current 3 counts. It is symmetric under both color
  and rank perms, so it composes with R1.
- **Per-opponent color-void flags**: when a player draws (or plays off-color) with color X showing,
  they likely lack X — the core human-level inference in UNO-family games, currently invisible to
  the net because it requires history. The env can maintain 3x4 "hasn't shown color X since last
  draw-on-X" bits at ~zero cost.

Both invalidate checkpoints, so batch them with whatever the next contract-breaking change is.

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

**R7. The eval.py seat-0-bias claim looks wrong — re-derive before acting on it.**
The 2026-07-06 RESEARCH_LOG entry states `play_match` "always seats the test agent at seat 0", and
the "Infrastructure / evaluation optimizations" section above carries a "fix seat-0 bias in eval.py"
item. But `play_match` has had per-game seeded seat shuffling since the *original* eval harness
commit (`e5feaa8`) — [eval.py:81](eval.py#L81) shuffles seating, so the test agent occupies the
advantaged opening seat in ~25% of games, exactly its parity share. If that's right: (a) Mode-B
`--baseline` numbers are *fair in expectation*, not inflated as the log caveats claim; (b) the
"fix seat-0 bias" item targets a non-bug. The seat-swap protocol for `eval_headtohead.py` remains
fully valid (that script really does take explicit fixed seats), and no ranking conclusion changes —
but the standing "read Mode-B head-to-heads as inflated" guidance in the log and CLAUDE.md deserves
a re-check.

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

### Suggested priority order (from this review)

1. **Exploitability probe** (R2) — cheapest, resolves the central question, reuses the loop.
2. **Rank-symmetry augmentation** (R1) — same class as the project's biggest win, near-free.
3. **Lower/earlier epsilon floor** (R11) — one flag, three runs' worth of supporting evidence.
4. **Opponent pool** (R10 / #1 above) — shaped by the probe's result.
5. **Belief features package** (R4 + R6, batched as one contract break) — unseen-count vector,
   color-void flags, loss signal.
6. **Round-robin + heuristic/oracle yardsticks** (R3, R9) — recalibrates the whole eval frame.
7. **Rollout-improvement measurement** (R5) — decides whether search is the long-term path.
8. Background: A11 450k completion, diagnostics before the next long run, promotion-bar replication
   rule.
