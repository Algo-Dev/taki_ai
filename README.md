# Taki AI
This is an experiment of teaching an AI to play [Taki](https://www.takigame.com/), a game which I own no rights to.

This project uses Keras and Tensorflow for Neural Networks.

The game does not use the 3+, 3+ breaker cards and the king card (new Taki cards).

This project taught me a lot about reinforcement learning and AI and I hope to continue making silly robots that learn!

See [RESEARCH_LOG.md](RESEARCH_LOG.md) for the full experiment history (training runs, evals,
negative results, and which checkpoint is the current best), and [PLAN.md](PLAN.md) for open
items and future levers.

# Algorithm
Uses DQN with experience replay, trained by self-play: 1 learner + 3 opponents whose weights
are periodically synced to the learner's, so it keeps facing a stronger version of itself.

Illegal actions are masked throughout: the agent only ever picks among legal moves, and the
replay bootstrap target only maxes over actions that are legal in the next state.

## Rewards
Three reward schemes are available via `train.py --reward`:
* `shaped` (default): minus the number of cards in the learner's hand after each of its turns,
  plus the sum of the opponents' cards on a win. This trained the current best model
  (~0.90 win rate vs 3 random opponents; chance is 0.25) and taught the AI to use 2+ against
  enemy players.
* `win`: win-only — 0 every step, and on a win the fewest cards any opponent still holds
  (clipped to 4). Warning: finetuning with this collapsed the model (see RESEARCH_LOG.md).
* `anneal`: a curriculum drifting from `shaped` to (almost) `win` over training. Stable, but
  the win-dominated tail measurably hurts — win-based objectives have not beaten `shaped`.

## Card Vector
Cards are a vector with dimension of the number of distinct cards (62: 60 colored slots, plus
the colorless Change Color and the Super TAKI), with one at the respective index.

A deck is a vector sum of card vectors.

## Action Space
64 actions: scalars 0–59 play a colored card, 60 plays a colorless Change Color (the chosen
color is encoded in the colored slots when played), 61 plays a Super TAKI, 62 draws, and 63
closes an open TAKI.

## State Space
The observation is a 201-float concatenation of the following blocks (count features are
normalised to a bounded scale):
* The hand of the player (62, counts / 4)
* The discard pile (62, counts / 4)
* The game state one-hot (7: normal, draw-two, taki, super-taki, finished, plus, stop)
* The amount of 2+ stacked (1 scalar, / 8)
* The active color of an open (Super) TAKI, one-hot (4)
* The card shown on the table (62, one-hot)
* Extra scalars (3): turn direction, next player's hand size / 8, minimum opponent hand size / 8

# Using the project
Requires Python ≤ 3.10 (TensorFlow is pinned `>=2.4,<2.11`, see `requirements.txt`); training
and eval run fine on CPU.

`train.py` trains the AI via self-play. Useful flags: `--trials`, `--model` (checkpoint to
warm-start from; incompatible checkpoints raise a load error), `--epsilon-start` (lower it to
continue a trained model near-greedily), `--snapshot-every`, and `--reward {shaped,win,anneal}`.
Snapshots land in `models/run<timestamp>/snap<NNNN>/` and the final model in
`models/checkpoint<timestamp>`.

`eval.py` measures trained models (training-time win rate can't: self-play is symmetric, so it
sits near 1/num_players regardless of skill):
* Mode A — `python eval.py --model <checkpoint>`: win rate vs 3 random opponents
  (chance = 0.25).
* Mode B — `python eval.py --run-dir models/run<ts> --baseline <best-checkpoint>
  --snap-stride 100 --games-b 3000`: skill progression across snapshots, each scored vs random
  and vs a frozen reference model. Decks, seating and the random opponents' choice streams are
  all seeded per game, so snapshots are compared on identical games. Rank near-equal snapshots
  at ≥3000 games (SE ≈ ±0.008); 1200 games is too noisy for the ~2–3 pt gaps between them.

`main.py` runs a seeded demo game of DQN agents: fully greedy when `--model` is given,
uniform-random otherwise. Add a `HumanAgent()` to the agents list to play along. Random
policies are pretty easy to beat since they are likely to draw cards when they are low on cards.

`gametest.py` unit tests for the rules engine (`python -m unittest gametest`).

`game.py` defines all the classes needed to run the game.

`agents/` playable agents: `dqn.py` (the DQN agent), `human.py` (interactive), `random.py`
(uniform over legal moves).

`models/` training outputs — gitignored except `.gitkeep` (checkpoints are regenerable);
promoted best models are kept as `models/checkpoint_*` and named in RESEARCH_LOG.md.

# License

```
Copyright (c) 2020 Oded "Dondish" Shapira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
