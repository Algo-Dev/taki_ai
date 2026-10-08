# Taki AI
A reinforcement-learning agent that plays [Taki](https://www.takigame.com/) — a card game I own
no rights to — using Keras/TensorFlow, trained entirely by self-play.

The goal is **not** the strongest possible player. It is to find the best moves available under a
**human information set** — the agent sees only what an attentive human could see and remember, so
the full discard pile is deliberately withheld — and then to interrogate what the resulting policy
actually learned about Taki strategy. See [CLAUDE.md](CLAUDE.md) for that design constraint in full.

The game does not use the 3+ and 3+ breaker cards (newer Taki cards). The King card **is**
implemented — see [RULES.md](RULES.md) for how it and the other special cards behave.

Started from [dondish/taki-ai](https://github.com/dondish/taki-ai), which supplied the original
rules engine and the first DQN scaffold. Everything since — the current agent, the observation
design, the self-play training loop, the evaluation harness (paired seat-swap orbits, common random
numbers), the frozen heuristic yardstick, the behavioural probes and the research log — is new work,
and the project has diverged substantially from its starting point.

See [RESEARCH_LOG.md](RESEARCH_LOG.md) for the full experiment history (training runs, evals,
negative results, and which checkpoint is the current best), [RULES.md](RULES.md) for the
rules-engine behaviour and house-rule interpretations, and [PLAN.md](PLAN.md) for open items
and future levers.

# Results

Current best model: `models/checkpoint_M1s3_mixed_snap300000` — **one network that plays 2, 3 and
4 seats**, trained from scratch by mixed-count self-play (`--num-players 2,3,4 --trials 300000
--reward shaped --loss-penalty 60`). The seat counts are genuinely *different games*, not
different table sizes: a STOP is a free extra turn at two seats, and a Change Direction is a no-op
there but reverses the cycle at three or more.

`eval.py` Mode A, one test agent against N−1 identical opponents, 3000 games, seed 0:

| 1-vs-(N−1), 3000 games | 2 seats | 3 seats | 4 seats |
|---|---|---|---|
| parity (`1/N`) | 0.500 | 0.333 | 0.250 |
| **vs random opponents** | **0.968** | **0.952** | **0.912** |
| **vs the heuristic yardstick** (`heuristic:h9`) | **0.566** | **0.373** | **0.307** |
| *the heuristic itself, vs random* | *0.966* | *0.925* | *0.906* |

**Read the second row, not the first.** vs-random is **retired as a ranking metric**: it is
saturated and cannot separate the top of this project. The last row is the proof — a few hundred
lines of hand-written rules score **0.906** vs random at four seats against the network's
**0.912**, a gap inside noise. Against that same heuristic the network wins **0.307** at parity
0.250, i.e. +5.7 points. One metric calls them the same agent; the other is decisive. vs-random
survives only as a smoke test that a run has not collapsed.

The yardstick is a **frozen, versioned** hand-written agent (`agents/heuristic.py`), not a moving
target — `h9` is the current reference, and every historical number names the version it was
measured against. It is bound by the same information set as the network and is deliberately
strong: its weights were tuned by coordinate descent *against* a trained champion, so beating it
is a real result rather than a formality.

Promotions are not decided on the table above. They use `eval_headtohead.py`'s paired rotation
orbit (both occupancies of a seat partition over the same decks, so the first-mover edge cancels
by construction rather than by averaging), and both the balanced and solo arms must pass. On that
standard this model beats the previous champion at **every** seat count, four seats included,
which was the previous champion's own specialty. See [RESEARCH_LOG.md](RESEARCH_LOG.md).

## Trained models

Checkpoints are **not in the repository** — `models/` is gitignored, because git stores every
revision of a binary in full and cannot prune it later. The promoted champions are published as
assets on the
[`champion-M1s3` release](https://github.com/Algo-Dev/taki_ai/releases/tag/champion-M1s3) instead:

| asset | what it is |
|---|---|
| `checkpoint_M1s3_mixed_snap300000.tar.gz` | the **current champion** — plays 2, 3 and 4 seats (150-float observation) |
| `checkpoint_r6L60_snap100000.tar.gz` | previous champion, four-seat only; the reference the current one was promoted against (147-float) |
| `checkpoint_a9rules_snap500000.tar.gz` | the champion before that; first model trained on the corrected finishing rule (147-float) |
| `SHA256SUMS` | checksums for the three archives |

```bash
tar -xzf checkpoint_M1s3_mixed_snap300000.tar.gz -C models/
python eval.py --model models/checkpoint_M1s3_mixed_snap300000 --opponent heuristic --games 3000
```

The two older nets predate the seat-count one-hot, so they cannot be warm-started for training,
but `eval.py` and `eval_headtohead.py` play them on the leading 147 floats they were trained on,
which is exact. Everything older than `a9rules` was lost on 2026-07-12 and was trained on a
finishing-rule bug in any case; those results survive in the research log as history only.

# Algorithm
Uses DQN with experience replay, trained by self-play: 1 learner + (N−1) opponents whose weights
are periodically synced to the learner's, so it keeps facing a stronger version of itself.
Transitions are collected from **every** seat, not just the learner's — the observation is
egocentric and every seat runs the learner's own recently-synced policy, so all of them are valid
training data.

Replay is augmented with two **exact symmetries** of Taki's dynamics: the four colours are
interchangeable (24 permutations) and the nine number ranks are interchangeable (9!). Colour
symmetry is the single largest win in this project's history — 3.3x sample efficiency, and it
stabilised training outright. Rank symmetry is on by principle rather than evidence: it measured
at exact parity, but relabelling ranks is free bias removal.

Illegal actions are masked throughout: the agent only ever picks among legal moves, and the
replay bootstrap target only maxes over actions that are legal in the next state.

## Rewards
Three reward schemes are available via `train.py --reward`:
* `shaped` (default): minus the number of cards in the learner's hand after each of its turns,
  plus the sum of the opponents' cards on a win. This trained every champion so far, and taught
  the agent to use +2 against enemy players. **Known limitation:** that dense step term is most
  of the episode return, so it doubles as a *length tax* — and every defensive move prolongs the
  game. The agent is consequently excellent at shedding and reluctant to block a near-winner.
  See PLAN.md R17.
* `win`: win-only — 0 every step, and on a win the fewest cards any opponent still holds
  (clipped to 4). Warning: finetuning with this collapsed the model (see RESEARCH_LOG.md).
* `anneal`: a curriculum drifting from `shaped` to (almost) `win` over training. Stable, but
  the win-dominated tail measurably hurts — win-based objectives have not beaten `shaped`.

`--loss-penalty L` is separate from the three schemes above and **defaults to 0.0, but every
current champion was trained with `60`.** Without it a loss pays *nothing* — the stream of
negative step rewards simply stops — so ending the game is itself rewarded and an imminent defeat
reads as relief. Any run meant to continue the current lineage must pass `--loss-penalty 60`.

## Card Vector
Cards are a vector with dimension of the number of distinct cards (63: 60 colored slots, plus
the colorless Change Color, Super TAKI, and King), with one at the respective index.

A deck is a vector sum of card vectors.

## Action Space
65 actions: scalars 0–59 play a colored card, 60 plays a colorless Change Color (the chosen
color is encoded in the colored slots when played), 61 plays a Super TAKI, 62 plays a King, 63
draws, and 64 closes an open TAKI.

## State Space
The observation is a 150-float concatenation of the following blocks (count features are
normalised to a bounded scale). The full discard pile is deliberately **not** exposed as a
histogram — only the shown top card is — so the agent gets a coarse card-count sense from the
unseen-count scalars below rather than a perfect memory of everything played:
* The hand of the player (63, counts / 4)
* The game state one-hot (8: normal, draw-two, taki, super-taki, finished, plus, stop, king)
* The amount of 2+ stacked (1 scalar, / 8)
* The active color of an open (Super) TAKI, one-hot (4)
* The card shown on the table (63, one-hot)
* Extra scalars (11): turn direction; the 3 opponents' hand sizes in turn order / 8 (zero-padded
  when fewer opponents); deck size / 120; unseen +2 / 8, unseen King / 2, unseen Change-Color / 4
  (unseen = total copies minus those in this hand and the discard pile); and a one-hot of the
  number of seats (2 / 3 / 4-or-more, saturating) so one net can play every player count and tell
  them apart — the count changes the *game*, not just the table size (STOP is a free extra turn at
  2 seats; CHDIR is a no-op there). That one-hot is constant, hence inert, in a single-count run;
  it earns its place only under mixed-count training (`--num-players 2,3,4`).

# Using the project
Requires Python ≤ 3.10 (TensorFlow is pinned `>=2.4,<2.11`, see `requirements.txt`); training
and eval run fine on CPU.

`train.py` trains the AI via self-play. Useful flags:

| flag | what it does |
|---|---|
| `--trials N` | episodes to run (1M ≈ 13 h on CPU) |
| `--num-players 4` / `2,3,4` | one seat count, or a set sampled uniformly per trial (mixed-count) |
| `--reward {shaped,win,anneal}` | reward scheme; see above |
| `--loss-penalty 60` | terminal penalty on the losing seats. **Off by default, on in every champion** |
| `--model <ckpt>` | warm-start; an incompatible checkpoint raises rather than silently cold-starting |
| `--epsilon-start` | lower it to continue a trained model near-greedily |
| `--no-color-sym` / `--no-rank-sym` | opt out of the replay symmetry augmentations (both on by default) |
| `--wide` | the 256→128→64 net instead of the narrow 124→64. Measured to buy ~nothing; kept for reproducibility |
| `--seed` | makes a run bit-reproducible (decks, epsilon draws, weight init, augmentation) |
| `--snapshot-every` | snapshot cadence for the progression eval |

Snapshots land in `models/run<timestamp>/snap<NNNN>/` and the final model in
`models/checkpoint<timestamp>`; promoted models are renamed to `models/checkpoint_<tag>`.

`eval.py` measures trained models (training-time win rate can't: self-play is symmetric, so it
sits near 1/num_players regardless of skill):
* Mode A — `python eval.py --model <checkpoint> --opponent heuristic --games 3000`: win rate in a
  homogeneous field. **`--opponent` defaults to `random`, which is the retired metric** — pass
  `heuristic` for a number with any resolution, or `heuristic:<version>` (`r3`, `b2`, `h1`,
  `h1b2`, `h8`, `h9`, `h10`, `greedy`, …) to pin a specific frozen version. `--num-players N`
  selects the seat count.
* Mode B — `python eval.py --run-dir models/run<ts> --baseline <best-checkpoint>
  --snap-stride 100 --games-b 3000`: skill progression across snapshots, each scored vs random
  and vs a frozen reference model. Decks, seating and the random opponents' choice streams are
  all seeded per game, so snapshots are compared on identical games. Rank near-equal snapshots
  at ≥3000 games (SE ≈ ±0.008); 1200 games is too noisy for the ~2–3 pt gaps between them.

`eval_headtohead.py` is **the promotion standard** — two agents, paired on shared decks, with seat
effects cancelled by construction instead of averaged away:

```bash
python eval_headtohead.py <cand> <champ> --team1-seats 0,2 --seat-swap --games 3000   # balanced arm
python eval_headtohead.py <cand> <champ> --team1-seats 0   --seat-swap --games 3000   # solo arm
```

`--seat-swap` runs the rotation orbit of the partition; the headline number is the paired
same-seat margin at parity 0.0. Both arms are required, and they are different field
compositions rather than different sample sizes.

`probe.py` runs the behavioural probes — the project's actual deliverable. It builds a Taki
position by hand and dumps the policy's ranking over the legal moves (`--scenario all --controls`),
optionally adjudicating a line by determinized rollout (`--mc 1500`).

`holdback.py` censuses hold-back behaviour over real games (does the agent ever decline to shed?).
`tune_heuristic.py` coordinate-descends the heuristic's weights against a trained net.

`main.py` runs a seeded demo game of DQN agents: fully greedy when `--model` is given,
uniform-random otherwise. Add a `HumanAgent()` to the agents list to play along. Random
policies are pretty easy to beat since they are likely to draw cards when they are low on cards.

Tests (~160, all fast — no training required):

```bash
python -m unittest gametest dqntest traintest agenttest h2htest probetest holdbacktest
```

`gametest` covers the rules engine and both symmetry tables; `agenttest` holds the **heuristic
freeze guards** (weight pins plus move-sequence fingerprints) that protect every published "vs
heuristic" number; `h2htest` pins the rotation-orbit construction; `dqntest` covers the model
load path and the observation-contract adapter; `traintest` covers seat-count sampling.

`game.py` defines all the classes needed to run the game, and owns the encoding contract
(card vector, action scalars, observation) that everything else imports.

`agents/` playable agents: `dqn.py` (the DQN agent), `heuristic.py` (the hand-written, frozen,
versioned yardstick — no network), `human.py` (interactive), `random.py` (uniform over legal
moves).

`models/` training outputs — gitignored except `.gitkeep` (checkpoints are regenerable);
promoted best models are kept as `models/checkpoint_*` and named in RESEARCH_LOG.md.

# License

MIT — see [LICENSE](LICENSE). The original engine and DQN scaffold are
© 2020 Oded "Dondish" Shapira; the reinforcement-learning work layered on top is © 2026 Ori Honigman.

```
Copyright (c) 2020 Oded "Dondish" Shapira
Copyright (c) 2026 Ori Honigman

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
