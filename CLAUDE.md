# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DQN agent learning to play Taki (an UNO-like card game) via self-play, using Keras/TensorFlow.
Current best model: `models/checkpoint_colorsym_snap180000` (color-sym, the 300k-run snap180000 — **0.926** vs 3 random opponents at 3000 games, and beats the previous best `checkpoint_shaped_snap300000` head-to-head 0.292 > 0.25 parity). Color-sym augmentation is on by default in training (opt out with `--no-color-sym`). Previous best `checkpoint_shaped_snap300000` (~0.91 vs random) kept for reference. Chance is 0.25.

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
  - card vector (62 slots: 60 colored, Change Color, Super TAKI), action scalars (64: 0–59 colored plays, 60 CHCOL, 61 Super TAKI, 62 DRAW, 63 CLOSE_TAKI), and the 201-float observation (`OBSERVATION_SIZE`), all defined here and imported by `agents/dqn.py`.
  - `valid_moves()` is the legality source of truth; agents only ever choose among legal moves.
  - Changing `OBSERVATION_SIZE`/`ACTION_SIZE` or the network shape **invalidates every saved checkpoint** — loads fail hard by design (no silent cold-start).
- **`agents/`** — anything with `play(game) -> (Action, Card)`: `dqn.py` (DQN with replay buffer; masks illegal actions in both action selection and the bootstrap target), `random.py` (uniform over legal moves, with `reseed()` for eval), `human.py`.
- **`train.py` / `eval.py` / `main.py`** — orchestration. The training loop, not the agent, computes rewards (`--reward {shaped,win,anneal}`) and drives replay/target-sync cadence. Self-play = learner at seat 0 + 3 opponents whose weights sync to the learner every 5 trials.

## Evaluation discipline (do not break)

- **Training win rate is meaningless** (self-play is symmetric → sits near 1/N regardless of skill); always measure with `eval.py`.
- `eval.py` implements common random numbers: per-game deck seed (`seed+g`), deterministic seating, and per-game opponent reseed (`RandomAgent.reseed(f'{seed}:{g}:opp')`). Any (model, seed, games) result is bit-reproducible and independent of sweep composition. Preserve this invariant when touching eval code.
- Ranking near-equal snapshots requires **≥3000 games** (SE ≈ ±0.008); 1200 games is too noisy for the typical 2–3 pt gaps.

## Conventions & gotchas

- **`RESEARCH_LOG.md`** is the experiment history (newest on top; setup, headline metrics, caveats). Record training/eval experiments there. **`PLAN.md`** holds open items, deferred RL levers, and deliberate house-rule interpretations in `game.py` — check it before "fixing" rules behavior (e.g. Change Color inside an open TAKI, 2-player CHDIR).
- `models/` is gitignored (checkpoints are regenerable). Snapshots land in `models/run<timestamp>/snap<NNNN>/`; promoted best models are kept as `models/checkpoint_*` and named in RESEARCH_LOG.md. Old `checkpoint<float-timestamp>` dirs are from a previous architecture and will not load.
- TF thread pools are deliberately shrunk (env vars in `eval.py` before the TF import; `tf.config.threading` in `agents/dqn.py`) — the tiny network thrashes on default pools. Keep any TF-touching entry point consistent with this.
- Reward lessons already learned (don't re-run blind): `shaped` trained the best model; switching a trained model to `win`-only collapsed it; `anneal`'s win-dominated tail measurably hurts. Details in RESEARCH_LOG.md.
- `AIAgent`'s `epsilon_decay` constructor default is dead — `train.py` always overwrites it with a run-length-scaled value.
