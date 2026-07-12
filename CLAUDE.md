# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal (read this before proposing any change to the observation)

The goal is **not** the strongest possible Taki player. It is to find the best moves **using the
information a human actually has** — and then to *interrogate the resulting policy's strategy*.

1. **The observation is a deliberate model of a human information set.** It encodes what an
   intelligent, reasonable human would see and remember. The full discard-pile histogram was
   **removed on purpose** in A7 (only the shown top card remains, plus coarse unseen
   +2/King/CHCOL counts) because a human does not perfectly card-count a played pile. It is a
   design constraint, **not an oversight and not an ablation to undo**. Never propose widening
   the information set (full discard histogram, exact deck composition, opponents' hands) — it
   would buy win rate by granting superhuman memory, which defeats the purpose.
2. **The payoff is behavioural analysis, not the win-rate number.** Once the model is good
   enough, the point is to probe *what it learned* in concrete scenarios and see whether it
   discovered real Taki strategy. Open questions of this kind:
   - Does it ever **hold cards back** — playing fewer cards now for a better end-game win chance
     — rather than greedily dumping the most cards each turn?
   - With a **colored TAKI plus several cards of that color**, does it learn to *keep* them to the
     end? A colored TAKI opens a run in which that whole color group discharges in a **single
     turn** (RULES.md), so hoarding it costs little and guarantees a fast finish. Does the policy
     see that, or does it dump the TAKI greedily the moment it is playable? (There is *almost* no
     sequencing wrinkle to go with it: a hand may end on **any card except PLUS**
     (`FINISHING_TYPE_VALUES`, RULES.md), so a run only has to be planned around a last card when
     it contains a PLUS. The engine formerly restricted finishing to numbers and the King — that
     was a **bug**, fixed 2026-07-12, and it invalidated the sequencing half of B1.)
   This reframes the plateau: A10/A11 (capacity), A12 (near-unexploitable), and R1 (rank-sym,
   parity) are converging evidence the agent is at the **ceiling of its information set** — i.e.
   the project working as designed, not a wall to break through.

## Project

DQN agent learning to play Taki (an UNO-like card game) via self-play, using Keras/TensorFlow.

> **STALE CHAMPION (2026-07-12).** The finishing rule was wrong until today: the engine let a hand
> end only on a number or the King, when the real rule is **any card except PLUS** (RULES.md).
> Every checkpoint below — A8 included — was trained on that other game, so its numbers describe a
> game the engine no longer plays. The observation/action contract is unchanged, so they still
> *load*; they are simply no longer trustworthy. A8 is being retrained; until a new champion is
> screened and promoted, treat the win rates in this section as historical.

Current best model: `models/checkpoint_a8_snap455000` (A8, the 500k-run seed1/snap455000 — **0.912** vs 3 random opponents at 3000 games; beats the previous best `checkpoint_a4a7_snap550000` head-to-head by **+3.7 points per-seat** in a seat-swap-controlled comparison (0.278 vs 0.241), the largest and most consistent edge among the top-5 candidate snapshots screened from the run). A8 = self-play training collects transitions from all four seats, not just the learner's, via `train.py`'s turn-by-turn loop (~4x data/trial; replay buffer 20k->80k). Same 147-float observation as A4+A7 (opener randomized per trial, A4; opponent hand sizes/deck size/unseen counts, A7) — checkpoints remain cross-loadable with `checkpoint_a4a7_snap550000`. Color-sym and rank-sym replay augmentation are both on by default (opt out with `--no-color-sym` / `--no-rank-sym`). **Rank-sym is on by principle, not evidence:** the nine number cards are an exact symmetry of the rules, so relabeling them is free bias removal — but it measured at exact *parity* with A8 (R1, RESEARCH_LOG.md), so the default is not a claim that it helps, and A8 (trained color-sym-only) remains the champion. Earlier checkpoints (`checkpoint_colorsym_snap180000` with 205-float obs, `checkpoint_shaped_snap300000` with 201-float obs) are **not loadable** — observation encoding changed. Chance baseline 0.25.

## Environment & Commands

Python runs from the conda env `tensorflow_env` (Python 3.9 — TF is pinned `>=2.4,<2.11` and numpy `<1.24`, so Python must be ≤ 3.10). CPU-only is fine.

```bash
python -m unittest gametest                                  # all rules-engine tests
python -m unittest gametest.GameFlowTest.test_stop_skips_next_player   # single test

python train.py --trials 3000 --reward shaped                # train (self-play)
python train.py --model <ckpt> --epsilon-start 0.1           # warm-start, near-greedy continue

python eval.py --model <ckpt> --games 1000                   # Mode A: win rate vs random
python eval.py --run-dir models/run<ts> --baseline <ckpt> \
    --snap-stride 100 --games-b 3000                         # Mode B: snapshot progression

python eval_headtohead.py <cand> <champ> \
    --team1-seats 0,2 --seat-swap --games 3000               # standard promotion test (see below)

python main.py --model <ckpt>                                # seeded demo game (greedy)

python probe.py --model <ckpt> --scenario all --controls     # B-series behavioural probes (Q-ranking)
python probe.py --model <ckpt> --scenario b1c_k2 --mc 1500   # + rollout adjudication of a line
python -m unittest probetest                                 # probe-harness consistency tests
```

Training/eval runs can take hours (1M trials ≈ 13 h); run long jobs in the background.

## Architecture

Three layers with a strict encoding contract between them:

- **`game.py`** — the rules engine, agent-agnostic. Owns the encodings everything else depends on:
  - card vector (63 slots: 60 colored, Change Color, Super TAKI, King), action scalars (65: 0–59 colored plays, 60 CHCOL, 61 Super TAKI, 62 King, 63 DRAW, 64 CLOSE_TAKI), and the 147-float observation (`OBSERVATION_SIZE`), all defined here and imported by `agents/dqn.py`. The colored-block stride is `TYPES_PER_COLOR` (15), deliberately decoupled from `len(Type)` because the King is a colorless wild (never in the colored block).
  - `valid_moves()` is the legality source of truth; agents only ever choose among legal moves.
  - Changing `OBSERVATION_SIZE`/`ACTION_SIZE` or the network shape **invalidates every saved checkpoint** — loads fail hard by design (no silent cold-start). **The King card (this branch) did exactly that:** every pre-King checkpoint (incl. `checkpoint_colorsym_snap180000`) is unloadable here. The last commit on `master` with the old 62/64/201 contract that still loads those models is **`344535a`** — check it out to use the pre-King models. **The A7 observation-reshape (this branch) did it again:** the observation went 205→147 — added the 3 opponent hand sizes in turn order, deck size, and unseen +2/King/CHCOL counts, and **removed the full discard-pile histogram** (only the shown top card remains, so the net has a coarse card-count sense, not a full memory of what's been played). That invalidates every pre-A7 checkpoint too, so this branch trains from scratch. Also on this branch (A4): training randomizes the opener per trial (the learner stays list index 0 but no longer opens 100% of games), matching eval's shuffled seating.
- **`agents/`** — anything with `play(game) -> (Action, Card)`: `dqn.py` (DQN with replay buffer; masks illegal actions in both action selection and the bootstrap target), `random.py` (uniform over legal moves, with `reseed()` for eval), `human.py`.
- **`train.py` / `eval.py` / `main.py`** — orchestration. The training loop, not the agent, computes rewards (`--reward {shaped,win,anneal}`) and drives replay/target-sync cadence. Self-play = learner at seat 0 + 3 opponents whose weights sync to the learner every 5 trials.

## Evaluation discipline (do not break)

- **Training win rate is meaningless** (self-play is symmetric → sits near 1/N regardless of skill); always measure with `eval.py`.
- `eval.py` implements common random numbers: per-game deck seed (`seed+g`), deterministic seating, and per-game opponent reseed (`RandomAgent.reseed(f'{seed}:{g}:opp')`). Any (model, seed, games) result is bit-reproducible and independent of sweep composition. Preserve this invariant when touching eval code.
- **`eval.py`'s Mode A/B seating is SHUFFLED, not fixed — its numbers are fair, not inflated.** "Deterministic seating" above means a *seeded shuffle*, not seat 0: `play_match` calls `seat_rng.shuffle(order)` ([eval.py:81](eval.py#L81)) and has since the original eval-harness commit, so the test agent occupies the advantaged opening seat in ~25% of games — exactly its parity share. **Mode-B `--baseline` win rates are therefore fair in expectation; a value above `1/N` is a real (if small) edge, NOT a seat artifact.** Do not "correct" for a seat-0 bias here — there isn't one. (The 2026-07-06 RESEARCH_LOG entry claims otherwise and is wrong; it carries a correction. Believing it led to a real misdiagnosis in A11 — see RESEARCH_LOG 2026-07-11 and PLAN.md R7.) Mode B's real limits are ordinary ones: noise at low `--games-b`, and winner's curse when you take the max over many screened snapshots.
- Ranking near-equal snapshots requires **≥3000 games** (SE ≈ ±0.008); 1200 games is too noisy for the typical 2–3 pt gaps.
- **Mode B and `eval_headtohead.py` measure different quantities — don't compare their numbers directly.** Mode B `--baseline` is a **1-vs-N** win rate (one test model against N−1 copies of the baseline; parity `1/N`). `eval_headtohead.py` reports a **per-seat** rate for an arbitrary team split (e.g. 2v2). A modest edge does not map between them one-for-one; a gap between the two is expected, not evidence of a bug or a confound.
- **Head-to-head evals need seat-swap controls** — this applies to `eval_headtohead.py`, which takes **explicit fixed seats**, and *not* to `eval.py`'s shuffled Mode A/B (see the bullet above). `Game.reset()` defaults `start_seat=0`, and seat 0 (the first to act) wins measurably more regardless of which model occupies it — confirmed across 14 seat configurations (every seat-position, every 2v2 partition) comparing `checkpoint_a4a7_snap550000` vs an A8-trained checkpoint: seat 0 was the top-scoring seat in nearly every run *independent of occupant*. A raw "model A at seat 0 vs model B at seats 1-3" result is therefore confounded and not a valid skill comparison.
- **The standard head-to-head is the alternating 2v2 seat swap: `--team1-seats 0,2 --seat-swap`.** `--seat-swap` runs *both* occupancies of the partition (model1 at `0,2`, then model2 at `0,2`) over the **same decks** and reports the seat-balanced, paired comparison — so it is a promotion decision in one command. Two reasons this is the default rather than the old 1v3 (`--team1-seats 0`):
  - **Alternating seats (`A,B,A,B`) are the only 2v2 layout with no friendly fire.** STOP/+2/CHDIR hit your *neighbour*; in a contiguous 2v2 (`A,A,B,B`) half of each team's aggression lands on its own teammate, and CHDIR makes that asymmetry direction-dependent. Interleaved, every neighbour of an A is a B and vice versa.
  - **Seat balance is exact, and every seat is informative.** Across the two runs each model occupies each seat exactly once, so the seat-0 edge cancels by construction (not by averaging); and all four seats contribute to the team indicator (parity 0.5) instead of one Bernoulli per game (parity 0.25). Report the **paired same-seat margin** (parity 0.0, with a paired SE over common decks) — that is the headline number.
  - Keep a **1v3 run (`--team1-seats 0 --seat-swap`) as a secondary check**, because 1v3 is the same shape as Mode B (one model in a homogeneous field) and is what the champion's headline number means. If the two disagree, that is real information — it says the candidate's edge depends on field composition — not a bug.
  - Pairing requires that both runs see the same decks, which is why `play_match` reseeds the deck RNG to `seed + g` before **every** game (as `eval.py` does). Without that, mid-game reshuffles make game *g*'s deal depend on how earlier games played out, and the two swap runs silently diverge. Preserve this.

## Conventions & gotchas

- **`RESEARCH_LOG.md`** is the experiment history (newest on top; setup, headline metrics, caveats). Record training/eval experiments there. **`RULES.md`** documents the rules-engine behaviour and deliberate house-rule interpretations in `game.py` — check it before "fixing" rules behavior (e.g. the King, Change Color inside an open TAKI, 2-player CHDIR). **`PLAN.md`** is a pool of ideas / open items only (deferred RL levers, TODOs) — not documentation; don't put reference material there.
- `models/` is gitignored (checkpoints are regenerable). Snapshots land in `models/run<timestamp>/snap<NNNN>/`; promoted best models are kept as `models/checkpoint_*` and named in RESEARCH_LOG.md. Old `checkpoint<float-timestamp>` dirs are from a previous architecture and will not load.
- TF thread pools are deliberately shrunk (env vars in `eval.py` before the TF import; `tf.config.threading` in `agents/dqn.py`) — the tiny network thrashes on default pools. Keep any TF-touching entry point consistent with this.
- Reward lessons already learned (don't re-run blind): `shaped` trained the best model; switching a trained model to `win`-only collapsed it; `anneal`'s win-dominated tail measurably hurts. Details in RESEARCH_LOG.md.
- `AIAgent`'s `epsilon_decay` constructor default is dead — `train.py` always overwrites it with a run-length-scaled value.
