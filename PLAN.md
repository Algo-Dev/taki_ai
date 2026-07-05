# Taki AI — Review Findings & Fix Plan

Findings from a full review of the project (rules engine, DQN agent, training loop,
demo harness). Items are grouped by category. **Correctness, observability, rules
fidelity, and code hygiene are being fixed now. The "RL / training design issues"
section is intentionally deferred — we will tune those later.**


-- Check out the game replays, maybe more are needed? understand the replay frequency, its part vs. updates from current game. Understand when exploration epsilon is reset.

-- review again the observation features

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


## RL design notes (future levers, from the 2026-07-02 review)

-- Replay buffer horizon: 20000 transitions at ~20-35 learner decisions/trial is only ~600-1000
   trials of history. During long runs (100k-1M trials) the buffer is a tiny recent window —
   reasonable, but keep it in mind when interpreting late-run flatness (no long-term memory of
   earlier play styles).

-- Dead knob: AIAgent's constructor default epsilon_decay=0.995 is always overwritten by
   train.py's run-length-scaled decay; the default is noise, not a real setting.


## Rules fidelity notes

Moved out of PLAN.md: the game-rules behaviour and deliberate house-rule interpretations now
live in **RULES.md**. Check (and update) that file for rules questions.


## Full project review (2026-07-03) — software + algorithm findings and action items

### Software findings

-- S1–S8 (2026-07-03 review) — **DONE** on branch `review-fixes-s1-s8` (2026-07-04). All eight
   implemented; the only behavioral change was S3 (`valid_moves` deduplication), which shifted the
   vs-random baseline to **0.907** (post-dedup). Details in RESEARCH_LOG 2026-07-04.

### Algorithm findings

-- A4. **Learner always seat 0, always opens.** `Game.reset()` sets `curr = 0` and the learner is
   always seat 0 in train.py — 100% of training games have the learner going first, while eval
   shuffles seating per game. First-mover advantage in a shedding game means a train/eval
   distribution shift. Randomize (or rotate) the starting seat per trial.

-- A5. **Self-play opponents are a ≤5-trial-old mirror.** `OPPONENT_SYNC_EVERY=5` means the learner
   only ever faces (almost) itself — prone to strategy cycling, and was the amplifier in the
   win-only collapse (a collapsing learner got copied into its own opponents, locking the whole
   table). Try an opponent *pool* (past snapshots + occasionally a RandomAgent seat) — standard
   fictitious-self-play fix, likely the best lever to break the ~0.90 plateau since "more of the
   same self-play" is already established as exhausted.

-- A6. Semi-MDP handling (one transition per learner decision, opponents fast-forwarded, TAKI
   chains, truncation with done=False) reviewed and found correct — no action item.

-- A7. **Observation gaps.** The `dir` feature is useless without per-opponent hand sizes in turn
   order — add all 3 opponent hand sizes ordered from the current player in play direction (dir
   then becomes meaningful); add deck size; consider an explicit "unseen cards" 62-vector (deck
   minus hand minus discard — derivable, but saves the tiny net from learning subtraction).
   Invalidates checkpoints — batch with a network-capacity change (A8).

-- A8. **Only seat 0's experience is used**; the other 3 seats play the same policy and their
   transitions are discarded. Storing all 4 seats' transitions is ~4x data per trial at near-zero
   cost (DQN is off-policy; the epsilon mismatch is negligible). Recompute the replay ratio if
   adopted (see tuning table below).

-- A9. **Reward design, if ever revisited:** the gentler middle ground PLAN.md's "penalize only net
   hand growth" idea gropes toward is **potential-based shaping** — `r' = r_win + gamma*phi(s') -
   phi(s)` with `phi(s) = -len(hand)`. Provably policy-invariant w.r.t. the win objective, keeps a
   dense signal (no sparse-reward trap), stops punishing strategically-correct draws beyond their
   true cost. Use this formulation (not another step-coefficient schedule) if the draw-penalty
   concern comes back up — with high epsilon and a fixed opponent set during the switch, per the
   win-only collapse post-mortem.

### Action items, ranked by expected value / cost

> **RESULT (2026-07-05, exp-dqn-hygiene): the DQN-hygiene package (Double DQN + slow/Polyak
> target + Huber + reward÷10) has been run and is NOT the plateau-breaker — do not re-run it
> blind.** A/B'd at 10k and 100k with color-sym on in every arm, it adds at most ~+1.5 pt
> vs-random over plain color-sym and no clear head-to-head edge. Ablation: Double DQN + slow
> target inert; only Huber + reward-scale moves the needle, marginally. "Beats the 1M-trial
> champion" is the **color-sym** effect, not these levers. **The lead lever is now structural
> (#4 richer observation / #5 bigger-dueling network), plus #1 opponent pool.** Details in
> RESEARCH_LOG 2026-07-05.

1. Opponent pool of past snapshots (+ occasional random seat) (A5) — likely the best lever to
   break the ~0.90 plateau.
2. Randomize starting seat in training (A4) — trivial, removes a train/eval mismatch.
3. Learn from all four seats (A8) — ~4x data per trial for free.
4. Richer observation (A7): opponent hand sizes in turn order, deck size, unseen-cards vector.
   Invalidates checkpoints — batch with #5.
5. Bigger/dueling network: 201->124->64->64 is tiny; try 256-256, and a dueling head (state-value +
   advantage) — well-suited since most of the 64 actions are illegal in any given state. Same
   checkpoint-invalidation caveat as #4.
6. n-step returns (n=3-5): shortens the bootstrap chain; large contributor in Rainbow ablations
   even with dense rewards. Cheap to implement in the buffer.
7. Lower epsilon floor late in training (0.1 -> 0.02-0.05, or decay to floor by ~50% of trials) —
   the 1000-trial run showed gains concentrated *after* epsilon bottomed out.
8. LR decay for long runs (1e-3 -> 1e-4 cosine/step) — the faint 100k-1M creep is consistent with
   bouncing around a minimum at too-large a step size.
9. Prioritized replay: medium effort, real but smaller expected gain here; do after 1-3.
10. If revisiting reward: potential-based shaping (A9), high epsilon + fixed opponent set during
    the switch.

The structural work (#4-5) and the opponent pool (#1) are now the lead, no longer gated behind
the (completed, marginal) DQN-hygiene package.

### Tuning guide: replay frequency, buffer size, and related knobs

Framing: the **replay ratio** — currently ~25 learner transitions/trial collected, `replay()` runs
at steps 0,4,8,... plus once at episode end (~8 calls x batch 64 ~= 512 samples/trial), so each
transition is trained on ~20x before eviction (Atari reference point is ~8). Replay ratio, learning
rate, and target-sync period must move together: more updates per datum -> lower lr and/or slower
target sync.

| Knob | Current | Sweep | What to watch |
|---|---|---|---|
| Replay frequency | every 4 steps + episode end (train.py:202) | every 1 / 2 / 4 / 8 steps | Q divergence & loss spikes at high ratio; slow learning at low |
| Buffer size | 20k (dqn.py:39) | 20k / 100k / 500k | oscillation/forgetting vs earlier snapshots if too small; sluggish early adaptation if too large |
| Target sync | ~2x/episode (effective) | hard: 500 / 2000 / 10000 learner steps; or Polyak tau 0.01 / 0.001 | loss sawtooth at each hard sync; head-to-head vs current best |
| Learning rate | 1e-3 | 1e-3 / 3e-4 / 1e-4, +/- decay schedule | late-run creep vs plateau; gradient norms |
| Batch size | 64 | 64 / 256 (scale lr with it) | wall-clock per trial; interacts with replay ratio |
| epsilon floor / decay | 0.1, decay over 80% of trials | floor 0.1 / 0.03 / 0.01; fraction 0.4 / 0.8 | vs-random early (too little exploration) vs late head-to-head gains |
| gamma | 0.99 | 0.99 / 0.995 | endgame credit: at ~30 decisions/episode the win bonus reaches the opening at ~0.74 discount — mild, so low priority |
| Opponent sync | every 5 trials | 5 / 25 / 100, then pool | robustness vs random AND vs frozen bests simultaneously |

Buffer-size context: 20k transitions ~= 600-1000 trials of history — 6-10% of a 10k run but ~0.1%
of a 1M run; scale with run length (100k-500k for long runs). Store observations as float32 (they
are float64 now) to halve buffer memory.

### Recommended process (matches how the color-sym experiment was actually run)

1. ~~Add `--seed` to train.py; promote hardcoded knobs (S7) to recorded CLI flags.~~ **DONE** —
   `--seed`, `--trial-len`, `--target-sync-every` all present (branch `double-dqn-huber`).
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


-- once color-sym is proven to be good, set is as default behaviour and remove the flag, or set the flag is no-color-sym

-- once i finish all the fixes above. Change the observation memory. From the discard pile, I want it to remember only the number of +2 cards, the color changes, and the king cards (were they added to this game simulation?)