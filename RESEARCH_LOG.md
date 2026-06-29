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
- Raise eval `TURN_CAP` 400→~2000 so naturally long vs-random games are decided
  (undecided → 0%). The "draw-stall" framing was wrong (see investigation above), so **no
  draw-penalty / reward-shaping** — the trained model already draws only when forced.
- For a *graded* progression curve, compare snapshots against a fixed mid-training snapshot
  (or round-robin), not the untrained one.
- Reward trend not plateaued → try 600–1000 trials (now ~1–2 min on the fast path).
