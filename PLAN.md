# Taki AI — Review Findings & Fix Plan

Findings from a full review of the project (rules engine, DQN agent, training loop,
demo harness). Items are grouped by category. **Correctness, observability, rules
fidelity, and code hygiene are being fixed now. The "RL / training design issues"
section is intentionally deferred — we will tune those later.**

Legend: `[x]` done · `[ ]` deferred

---

## Correctness bugs (FIX NOW)

- [x] **#1 — `replay()` bootstraps from invalid actions.** `agents/dqn.py`
  The TD target used `np.max(next_q[i])` over **all 64 outputs**, including actions
  that are illegal in the next state. The policy (`act()`) is masked but the target
  was not, causing systematic Q-overestimation. **Fix:** store the next state's valid
  action scalars in the replay buffer and take the max only over them.

- [x] **#2 — "Self-play" opponents never improve.** `train.py`
  Opponents were fixed at their initial (random) weights for the entire run; the
  learner never faced a stronger version of itself. **Fix:** periodically sync the
  learner's weights into the opponents (`OPPONENT_SYNC_EVERY` trials). The interval
  is a one-line constant so tuning stays deferred to the RL pass.

- [x] **#3 — Deck-exhaustion crash in `draw_card`.** `game.py`
  When the deck is empty and the discard holds only the top card, `deck.pop()` raised
  `IndexError`. **Fix:** guard the reshuffle — if there is nothing to recycle, stop
  drawing instead of crashing.

## Observability gap (FIX NOW)

- [x] **Observation omits direction and opponent hand sizes.** `game.py`
  The reward rewards reducing opponents' card counts, but the agent couldn't see them,
  nor the turn direction — so it can't learn STOP / +2 / reverse tactics. **Fix:** add
  three fixed-size features to the observation: turn direction, the next player's hand
  size, and the minimum opponent hand size. (Raw, unnormalised — normalisation is a
  deferred RL item.)

## Rules fidelity (FIX NOW)

- [x] **Could win on an action card.** `game.py`
  Win was detected purely by an empty hand. Standard Taki forbids ending on an action
  card. **Fix:** if a player empties their hand by playing a non-number card, they draw
  one penalty card and play continues (only a number card — or an exhausted deck —
  finishes the game).

- [x] **Initial discard card's effect ignored.** `game.py`
  A starting +2 / STOP / TAKI / change-color was treated as inert. **Fix:** re-draw the
  opening card until it is a plain number card (standard rule), so no special effect is
  silently dropped.

- [ ] New cards (3+, breaker, King) excluded — **intentional**, per the README. No change.

## Code hygiene (FIX NOW)

- [x] **`MODEL_PATH` hardcoded to `None` in both entry points.** `train.py`, `main.py`
  Out of the box there was no way to load a trained model. **Fix:** `--model` CLI flag.

- [x] **`plot_rewards` ends in a blocking `plt.show()`.** `train.py`
  Stalls headless runs. **Fix:** always save the figure to a PNG; only `show()` behind
  a `--show` flag.

- [x] **No game-flow tests.** `gametest.py`
  Only the encoding helpers were covered. **Fix:** add tests for turn advancement, STOP
  skip, +2 stacking, TAKI chaining, win detection, and the new rules.

- [x] **Stale, incompatible model blobs committed under `models/`.**
  They cannot load against the current network (and even less so after the observation
  change). **Fix:** remove them (recoverable from git history).

---

## RL / training design issues (DEFERRED — handle later)

- [ ] Adam learning rate `0.01` is ~10× high for DQN; consider `1e-3` or lower.
- [ ] Epsilon decays per step (`0.995`) and bottoms out at `0.1` in ~460 steps; consider
      per-episode decay or a slower schedule.
- [ ] `predict`/`fit` per replay retraces in TF2 eager and is slow; move to a
      `@tf.function` train step if scaling up.
- [ ] No input normalisation — the summed discard vector and raw counts grow unbounded.
- [ ] Replay memory is small (`maxlen=2000`); revisit alongside batch size and update
      cadence.
