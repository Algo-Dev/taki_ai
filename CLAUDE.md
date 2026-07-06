# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DQN agent learning to play Taki (an UNO-like card game) via self-play, using Keras/TensorFlow.
Current best model: `models/checkpoint_a8_snap455000` (A8, the 500k-run seed1/snap455000 — **0.912** vs 3 random opponents at 3000 games; beats the previous best `checkpoint_a4a7_snap550000` head-to-head by **+3.7 points per-seat** in a seat-swap-controlled comparison (0.278 vs 0.241), the largest and most consistent edge among the top-5 candidate snapshots screened from the run). A8 = self-play training collects transitions from all four seats, not just the learner's, via `train.py`'s turn-by-turn loop (~4x data/trial; replay buffer 20k->80k). Same 147-float observation as A4+A7 (opener randomized per trial, A4; opponent hand sizes/deck size/unseen counts, A7) — checkpoints remain cross-loadable with `checkpoint_a4a7_snap550000`. Color-sym augmentation on by default (opt out with `--no-color-sym`). Earlier checkpoints (`checkpoint_colorsym_snap180000` with 205-float obs, `checkpoint_shaped_snap300000` with 201-float obs) are **not loadable** — observation encoding changed. Chance baseline 0.25.

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

python main.py --model <ckpt>                                # seeded demo game (greedy)
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
- Ranking near-equal snapshots requires **≥3000 games** (SE ≈ ±0.008); 1200 games is too noisy for the typical 2–3 pt gaps.
- **Head-to-head evals need seat-swap controls.** `Game.reset()` defaults `start_seat=0`, and seat 0 (the first to act) wins measurably more regardless of which model occupies it — confirmed across 14 seat configurations (every seat-position, every 2v2 partition) comparing `checkpoint_a4a7_snap550000` vs an A8-trained checkpoint: seat 0 was the top-scoring seat in nearly every run *independent of occupant*. A raw "model A at seat 0 vs model B at seats 1-3" result is therefore confounded and not a valid skill comparison. To compare two checkpoints head-to-head: run **both seat assignments** (A at seats X, then B at seats X, same partition) and compare **same-seat, swapped-occupant** win rates, or average across a full seat-swap set — see `eval_headtohead.py` for the pattern (`--team1-seats` takes explicit seat indices so any assignment/swap can be scripted).

## Conventions & gotchas

- **`RESEARCH_LOG.md`** is the experiment history (newest on top; setup, headline metrics, caveats). Record training/eval experiments there. **`RULES.md`** documents the rules-engine behaviour and deliberate house-rule interpretations in `game.py` — check it before "fixing" rules behavior (e.g. the King, Change Color inside an open TAKI, 2-player CHDIR). **`PLAN.md`** is a pool of ideas / open items only (deferred RL levers, TODOs) — not documentation; don't put reference material there.
- `models/` is gitignored (checkpoints are regenerable). Snapshots land in `models/run<timestamp>/snap<NNNN>/`; promoted best models are kept as `models/checkpoint_*` and named in RESEARCH_LOG.md. Old `checkpoint<float-timestamp>` dirs are from a previous architecture and will not load.
- TF thread pools are deliberately shrunk (env vars in `eval.py` before the TF import; `tf.config.threading` in `agents/dqn.py`) — the tiny network thrashes on default pools. Keep any TF-touching entry point consistent with this.
- Reward lessons already learned (don't re-run blind): `shaped` trained the best model; switching a trained model to `win`-only collapsed it; `anneal`'s win-dominated tail measurably hurts. Details in RESEARCH_LOG.md.
- `AIAgent`'s `epsilon_decay` constructor default is dead — `train.py` always overwrites it with a run-length-scaled value.
