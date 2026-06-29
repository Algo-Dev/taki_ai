# Taki AI — Review Findings & Fix Plan

Findings from a full review of the project (rules engine, DQN agent, training loop,
demo harness). Items are grouped by category. **Correctness, observability, rules
fidelity, and code hygiene are being fixed now. The "RL / training design issues"
section is intentionally deferred — we will tune those later.**


## RL / training design issues

- [x] Adam learning rate `0.01` is ~10× high for DQN; consider `1e-3` or lower.
      → lowered to `1e-3`.
- [x] Epsilon decays per step (`0.995`) and bottoms out at `0.1` in ~460 steps; consider
      per-episode decay or a slower schedule.
      → now decays once per episode, reaching `epsilon_min` at ~80% of trials.
- [ ] `predict`/`fit` per replay retraces in TF2 eager and is slow; move to a
      `@tf.function` train step if scaling up. **(still deferred)**
- [x] No input normalisation — the summed discard vector and raw counts grow unbounded.
      → count features rescaled by fixed constants in `observation()`.
- [x] Replay memory is small (`maxlen=2000`); revisit alongside batch size and update
      cadence.
      → buffer `20000`, batch `64`; replay/target cadence kept.
