# Taki AI — Review Findings & Fix Plan

Findings from a full review of the project (rules engine, DQN agent, training loop,
demo harness). Items are grouped by category. **Correctness, observability, rules
fidelity, and code hygiene are being fixed now. The "RL / training design issues"
section is intentionally deferred — we will tune those later.**


-- Check out the game replays, maybe more are needed? understand the replay frequency, its part vs. updates from current game. Understand when exploration epsilon is reset.

-- review again the observation features

-- the direction is not really relevant unless I provide the opponents number of card by their turn order. For now let's keep it but reconsider in the future.

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

-- Double DQN: replay() [dqn.py] takes max over the *target* net's Q-values — vanilla DQN's
   overestimation-prone update. Double DQN (pick argmax with the online net, evaluate it with
   the target net; keep the legality mask on both) is a ~2-line change in replay() and the
   cheapest untried structural lever. Worth trying before the bigger network / richer
   observation work.

-- Target-network cadence is aggressive: a hard copy every 100 steps AND at every episode end
   [train.py]. Episodes average well under 100 learner steps, so the episode-end copy dominates
   and the target is nearly on-policy. A slower hard update or Polyak (soft) averaging is a
   cheap stability lever — potentially relevant to the reward-change instability.

-- Replay buffer horizon: 20000 transitions at ~20-35 learner decisions/trial is only ~600-1000
   trials of history. During long runs (100k-1M trials) the buffer is a tiny recent window —
   reasonable, but keep it in mind when interpreting late-run flatness (no long-term memory of
   earlier play styles).

-- Dead knob: AIAgent's constructor default epsilon_decay=0.995 is always overwritten by
   train.py's run-length-scaled decay; the default is noise, not a real setting.


## Rules fidelity notes (house-rule interpretations, from the 2026-07-02 review)

-- A colorless Change Color is playable inside an open TAKI (it passes the color filter as a
   wild). Defensible reading; official rules are ambiguous here.

-- Playing a Change Color inside an open TAKI does NOT change taki_color for the rest of the
   TAKI — the chosen color takes effect only after the TAKI closes (the shown card then carries
   the new color). Also a deliberate interpretation, documented in game.py.

-- 2-player CHDIR is a pure no-op ((curr+1) % 2 == (curr-1) % 2), whereas official Taki treats
   change-direction as a stop/extra-turn with two players. Irrelevant to the 4-player
   experiments; fix only if 2-player play ever matters.


## Full project review (2026-07-03) — software + algorithm findings and action items

### Software findings

-- S1. game.py:415-416 — `if card not in self.hands[agent]: print(...)` on an impossible state
   just prints and then falls through to `remove()`, which raises `ValueError` anyway. Should be
   an explicit `raise` (or removed).

-- S2. game.py:330-336 — deck construction uses `[Card(t, color)] * 2` / `* 4`, aliasing the same
   Card object twice/four times in the deck. Currently safe (the only in-place mutation, the CHCOL
   color reset in draw_card, is a no-op for hand copies) but a landmine; gametest.py already has to
   work around it when deep-copying for the color-symmetry tests. Construct distinct objects.

-- S3. game.py:486-512 (valid_moves) — duplicate hand cards produce duplicate move-list entries,
   and each CHCOL expands to 4 entries. RandomAgent and epsilon-exploration are therefore uniform
   over move *instances*, not distinct moves — duplicated cards and CHCOL are picked disproportion-
   ately often. Not a bug, but means the "vs random" baseline is a specific non-uniform policy;
   worth knowing when interpreting win-rate numbers.

-- S4. Action scalar 60 (colorless CHCOL) is unreachable — valid_moves always expands CHCOL into
   the 4 colored plays, so scalar 60 is never legal/trained. Harmless dead output unit; worth a
   comment at game.py:178 so it isn't "fixed" by routing through it later.

-- S5. Dead code / unused observation dims: Card.amount() (game.py:98-105, deck construction
   hardcodes counts instead); State.STOP and State.FINISHED one-hot slots are never seen by a
   deciding agent.

-- S6. eval.py defaults (`--games 500`, `--games-b 150`, eval.py:137-140) contradict the
   documented "≥3000 games to rank near-equal snapshots" rule — running with defaults reproduces
   the earlier 1200-game false-plateau mistake. Consider raising the defaults or printing a warning
   below the precision bar.

-- S7. train.py `__main__` hardcodes `trial_len = 300` and `update_target_network = 100`
   (train.py:101-102) instead of CLI flags recorded in config.txt, unlike `--reward` etc. Promote
   to flags once any of the target-cadence/trial-length tuning below is attempted.

-- S8. No training seed — runs are unseeded, and the log attributes ~0.02 vs-random run-to-run
   spread partly to this. Add `--seed` to train.py so future A/B replicates are controlled (eval.py
   already does this correctly).

-- Minor: opponent AIAgents in training carry unused 20k replay buffers/target nets; main.py loads
   the same checkpoint 3x for the demo; agents/human.py uses `from game import *`; per-trial
   `print` is log spam at 1M-trial scale.

### Algorithm findings

-- A1. **Target network is effectively disabled.** `target_train()` fires at step 0 of every
   episode (`step % 100 == 0` triggers at step 0, train.py:204-205) AND again at every episode end
   (train.py:213). Episodes are ~20-35 learner steps, so the target net is a near-copy of the
   online net essentially always — the mechanism meant to stabilize bootstrapping isn't doing
   anything. Plausibly implicated in both the reward-change collapse and the 100k-1M plateau.
   **Top of the queue.**
   **TESTED (2026-07-05, exp-dqn-hygiene, with color-sym on):** slow/stepped target
   (`--target-sync-mode steps 2000`) is **inert** at both 10k and 100k — the near-on-policy
   target wasn't actually costing much. The real long-run stabilizer turned out to be
   **color-sym itself** (vanilla replay diverges at ~20k trials; color-sym does not — see
   RESEARCH_LOG 2026-07-05), which subsumes what A1 was meant to fix.

-- A2. **Vanilla max-Q bootstrap (overestimation-prone).** dqn.py:120 takes max over the *target*
   net. Double DQN (argmax via the online net over `next_valid`, evaluate via the target net) is a
   ~2-line change; pair with A1 (a slower target only helps once max-bias is also addressed).
   **TESTED (2026-07-05):** Double DQN + slow target (the "stability pair") is **inert** —
   pooled head-to-head 0.252, +0.4 SE over 3 seeds at 10k, no edge at 100k either. Not worth
   adopting on current evidence.

-- A3. **Loss/reward scale is hot for plain MSE + Adam(1e-3).** Shaped per-step rewards run to
   ~±30, the win bonus is `sum(opp)` (can exceed 50), episode returns are -50..-230, so Q-targets
   live in the ±100s with no clipping (dqn.py:63, dqn.py:78-82). Switch to Huber loss (+ maybe
   gradient-norm clipping and/or scale rewards ~÷10) — also the cheapest stability lever for any
   future reward-schedule change (this is exactly what would have damped the win-only collapse).
   Note the full-64-action MSE means non-taken actions have zero error, so effective per-action lr
   is ~1/64 of nominal.
   **TESTED (2026-07-05):** the "scale pair" (`--loss huber --reward-scale 0.1`) is the **only
   live part** of the hygiene bundle — pooled head-to-head 0.273, +5.1 SE, 3/3 positive at 10k,
   but only ~+1.5 pt vs-random over color-sym at 100k. Mild robustness bump, not a strategic
   gain. If pursuing hygiene at all, keep this pair and drop A1/A2. Note a slight negative
   scale×stability interaction (bundling all four was worse than either pair alone).

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

> **RESULT (2026-07-05, exp-dqn-hygiene): items #1–3 have been run and are NOT the
> plateau-breaker.** The DQN-hygiene bundle (Double DQN + slow target + Huber + reward÷10),
> A/B'd at 10k and 100k with color-sym on in every arm, adds at most ~+1.5 pt vs-random over
> plain color-sym and no clear head-to-head edge. Ablation: #2/#1 (Double DQN + slow target)
> inert; only #3 (Huber + reward-scale) moves the needle, marginally. "Beats the 1M-trial
> champion" is the **color-sym** effect, not these levers. **The lead lever is now structural
> (#7 richer observation / #8 bigger-dueling network), plus #4 opponent pool.** Do not re-run
> #1–3 blind. Details in RESEARCH_LOG 2026-07-05.

1. ~~Slow the target network (A1)~~ — **DONE, inert.** remove per-episode syncs; hard update
   every 1000-5000 *learner* steps, or Polyak tau ~= 0.005 per replay call.
2. ~~Double DQN (A2)~~ — **DONE, inert.** pairs with #1; keep the `next_valid` mask on the argmax.
3. ~~Huber loss + gradient-norm clipping, and/or scale rewards ~÷10 (A3)~~ — **DONE, only
   marginal (+1.5 pt vs-random).** the one live part of the bundle.
4. Opponent pool of past snapshots (+ occasional random seat) (A5) — likely the best lever to
   break the ~0.90 plateau.
5. Randomize starting seat in training (A4) — trivial, removes a train/eval mismatch.
6. Learn from all four seats (A8) — ~4x data per trial for free.
7. Richer observation (A7): opponent hand sizes in turn order, deck size, unseen-cards vector.
   Invalidates checkpoints — batch with #8.
8. Bigger/dueling network: 201->124->64->64 is tiny; try 256-256, and a dueling head (state-value +
   advantage) — well-suited since most of the 64 actions are illegal in any given state. Same
   checkpoint-invalidation caveat as #7.
9. n-step returns (n=3-5): shortens the bootstrap chain; large contributor in Rainbow ablations
   even with dense rewards. Cheap to implement in the buffer.
10. Lower epsilon floor late in training (0.1 -> 0.02-0.05, or decay to floor by ~50% of trials) —
    the 1000-trial run showed gains concentrated *after* epsilon bottomed out.
11. LR decay for long runs (1e-3 -> 1e-4 cosine/step) — the faint 100k-1M creep is consistent with
    bouncing around a minimum at too-large a step size.
12. Prioritized replay: medium effort, real but smaller expected gain here; do after 1-6.
13. If revisiting reward: potential-based shaping (A9), high epsilon + fixed opponent set during
    the switch.

Items 1-3 were one package (make DQN actually DQN) and *were* A/B'd together before the
structural work — result above: marginal at best, not the plateau-breaker. Structural work
(7-8) and the opponent pool (4) are now the lead, no longer gated behind the hygiene package.

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

1. Add `--seed` to train.py so replicates are controlled; promote hardcoded knobs (S7) to recorded
   CLI flags.
2. Screen each change at 10k trials, 2-3 seeded pairs, changing **one knob** (or one declared
   package like Double-DQN+slow-target+Huber) at a time.
3. Judge on the three readouts that caught color-sym: 3000-game vs-random, head-to-head vs the
   control run's snapshot (the decisive one), and Mode-B progression vs
   `checkpoint_shaped_snap300000`.
4. Confirm winners at 100k trials before promotion.
5. Add cheap training-time diagnostics so stability is visible without running full evals: mean
   |TD error|, mean max-Q on a frozen probe set of ~1k states (drift up = overestimation — would
   directly show whether A1/A2 are biting), gradient norms. Microseconds of cost; would have
   diagnosed the win-only collapse in minutes instead of a 7700-second run.

**Single highest-leverage next experiment:** the DQN-hygiene package (slow target + Double DQN +
Huber) A/B'd at 10k trials, then rerun the 100k-300k long-run question — the current plateau was
measured under a setup where the stabilizers were effectively off.


-- once color-sym is proven to be good, set is as default behaviour and remove the flag, or set the flag is no-color-sym

-- once i finish all the fixes above. Change the observation memory. From the discard pile, I want it to remember only the number of +2 cards, the color changes, and the king cards (were they added to this game simulation?)