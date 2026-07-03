# Taki AI — Review Findings & Fix Plan

Findings from a full review of the project (rules engine, DQN agent, training loop,
demo harness). Items are grouped by category. **Correctness, observability, rules
fidelity, and code hygiene are being fixed now. The "RL / training design issues"
section is intentionally deferred — we will tune those later.**


-- because the acutal colors do not matter, per each state we could shuffle the colors, and adapt all the observations including: own hand, discard pile, open  taki color, top card and it should results in exactly the same Q value. there are 4*3*2 ways to shuffle four colors.

-- Check out the game replays, maybe more are needed?

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

