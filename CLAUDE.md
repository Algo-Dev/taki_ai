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

Current best model: **`models/checkpoint_r6L60_snap100000`** (R6 — shaped reward **plus a terminal
loss penalty**: `train.py --trials 100000 --seed 1 --reward shaped --loss-penalty 60`, color-sym +
rank-sym). Beats the previous champion by **+0.085 ± 0.011 per-seat** on the seat-swapped 2v2
promotion standard, replicated on three disjoint deck blocks (~8 SE). It reached that in **100k
trials, not 500k**. Chance baseline 0.25.

> **Its "0.378 vs 3 heuristic opponents" is a RETIRED number — it was measured against `r3`, the
> yardstick with a 13-point bug in it.** Against the current reference (`h8`) R6 scores **0.297**
> (parity 0.250), i.e. it beats a *competent* heuristic by **+4.7 points over chance, not +12.8**.
> Still genuinely ahead, but less than half the claimed margin. See B5 below and RESEARCH_LOG
> 2026-07-14. Anything you compare against must be measured against the SAME version.

> **AND IT IS A FOUR-SEAT NUMBER. Every "beats the heuristic" claim in this project is.** The
> engine and the observation support 2..10 seats (`OPP_HAND_SLOTS` zero-pads, so the 147-float
> contract holds at every count and checkpoints load), and `eval.py --num-players N` has always
> run. Measured there (P1, RESEARCH_LOG 2026-07-17), R6 vs `h8` is **0.413 at two seats (parity
> 0.500 — it LOSES, 9.7 SE below)**, **0.342 at three (a tie)**, 0.297 at four. `h8` is not
> flailing off its turf: it is at its *strongest* vs random at two seats (0.963) and still beats
> its own `greedy` control there. **vs-random is blind to all of it** (0.91-0.96 in every cell).
> Do not restate a champion's win rate without saying how many seats it means.
>
> **The counts are different games, not different sizes.** STOP is a free extra turn at two
> seats; CHDIR is a no-op at two and *identical to STOP* at three (RULES.md). Training at four
> seats measurably degrades two-seat play (Arm B: 0.413 -> 0.365, 3.8 SE). Warm-starting R6 at
> two seats gains **+0.170 (13.4 SE)** in 100k trials — so the A10/A11/A12 "ceiling" is
> **per-count**, not global: four seats is plateaued (another 100k buys 0.4 SE), two seats had 17
> points sitting in it.

> **`--loss-penalty` is not the default (it defaults to 0.0), but the champion was trained WITH it.**
> Any run meant to continue this lineage must pass `--loss-penalty 60`. Why it matters: without it a
> **loss pays nothing** — the stream of `-len(hand)` step penalties simply stops — so *ending* the
> game is itself a reward and an imminent defeat looks like a relief. B2 measured the champion
> valuing an identical hand **3.4 higher** when the next player held 1 card than when they held 7.
> See RESEARCH_LOG 2026-07-13 and `probes/b2_holding_back.md`.
>
> **It still does not block a near-winner.** R6's pre-registered acceptance test (`r6_accept.py`)
> **failed**: `delta(k=1) = -0.87`, still negative. R6 fixed the *slope* (it now wants a +2 more as
> the threat grows) but not the *intercept*. The +8.5 points did **not** come from blocking. Do not
> assume the defect is fixed — the open fix is **PLAN R17** (potential-based shaping).
>
> **But check R17's PREMISE before spending a run on it (P2, RESEARCH_LOG 2026-07-17).** Blocking
> is **learnable without it** — warm-starting R6 at two seats flips `delta(k=1)` to **+2.24** and
> passes the acceptance test, in the existing architecture, observation and reward. And **blocking
> does not predict four-seat strength**: three nets score 0.308 (blocks), 0.302 (does not), 0.311
> (does not) — the blocker is in the middle. It may be *correct* not to block at four seats (you
> spend a card to delay one of three threats; at two seats you delay the only one), which would
> make R6's "defect" an adaptation. R17 may fix the behaviour and buy no win rate. Unresolved
> against this: `h8` blocks (`w_block` tuned UP 4.0->6.0) and beats its own `greedy` control at
> four seats — so blocking pays in the *heuristic's* economy. Do not treat either side as settled.

Previous champion: `models/checkpoint_a9rules_snap500000` (A9-rules — the first model trained on the
**corrected finishing rule**; 500k trials, shaped, no loss penalty). 0.348 vs `heuristic:r3` (0.271
vs the current `h8` reference), 0.906 vs random. Kept as the promotion baseline. These two are the **only** checkpoints that exist — see the
warnings below.

> **ALL PRE-2026-07-13 CHECKPOINTS ARE GONE FROM DISK.** `checkpoint_a8_snap455000` and
> `checkpoint_a4a7_snap550000` were destroyed in the models-symlink incident (commits `0f96882` /
> `62a7433`); `models/` is gitignored, so nothing was recoverable. Their recorded win rates survive
> in RESEARCH_LOG.md as *history only* — no head-to-head against them can ever be run again. Do not
> write instructions that assume they exist.
>
> **AND THEY WOULD HAVE BEEN INVALID ANYWAY.** The finishing rule was wrong until 2026-07-13 (the
> engine let a hand end only on a number or the King; the real rule is **any card except PLUS** —
> RULES.md). Every pre-A9 checkpoint learned that other game. Cross-rule win rates are not
> comparable in either direction.

**Rank on the heuristic, not on random.** `--opponent heuristic` is the yardstick; vs-random is
saturated and misleading — A9-rules reaches 0.902 vs random by **trial 25,000** and then does not
improve for the remaining 475k trials, while the same run's heuristic score is only 0.348 (vs `r3`).
A high vs-random number means almost nothing (A8 scored 0.912 vs random but 0.343 vs heuristic). Note
the heuristic itself now reaches **0.906 vs random** — *exactly matching A9-rules' 0.906* — while
losing to that same DQN on the discriminating metric. A few hundred lines of rules and a 500k-trial
network are indistinguishable on vs-random; that is the clearest possible statement that it has no
resolution left at the top.

## The heuristic is a VERSIONED, FROZEN yardstick (H-series, 2026-07-14)

Because it is the ranking metric, changing `HeuristicAgent` in place would silently rewrite the
project's history. So `agents/heuristic.py` owns a **registry of named, frozen versions**, and one
spec grammar (`resolve_weights`) shared by every harness — `eval.py`, `eval_headtohead.py`,
`holdback.py`, `tune_heuristic.py`. Spec forms: `heuristic:<version>`, `heuristic:-<ablation>`,
`heuristic:k=v,k=v`, and `heuristic:<version>,k=v` (**the leading version matters** — without it an
override is based on `REFERENCE`, i.e. the *legacy* structure).

| version | what it is | vs R6 champion | vs random |
|---|---|---|---|
| `r3` | the **retired** yardstick — as shipped, 13-point bug and all. Every "vs heuristic" number published **before 2026-07-14** refers to THIS | -0.263 | 0.850 |
| `b2` | B2's retuned weights (legacy structure) | -0.065 | 0.899 |
| `h1` | **R3's exact weights** + H1's structure (the +0.183 result) | -0.079 | 0.896 |
| `h1b2` | B2's weights + H1's structure | -0.065 | 0.897 |
| **`h8`** | **the REFERENCE** — H8's tuned point on H1's structure | **-0.073** | **0.906** |
| `h7` | `h1b2` + the RACE behaviour — **refuted, off** (racing loses monotonically) | — | — |
| `greedy` | every hold-back off (the B2 control) | — | — |

**`REFERENCE = 'h8'` — promoted from `r3` on 2026-07-14 (B5).** Changing it silently redefines every
"vs heuristic" number the project reports, so it is a deliberate act that requires re-running the
champions against the new reference and recording *both* numbers. Do not change it as a side effect.
`heuristic:r3` still runs, and still means exactly what it always meant — cite the old numbers by
that name.

**The freeze is enforced by TWO guards, and you need both** (`agenttest.FrozenVersionTest`). The
**weight pin** asserts each version's `Weights` values; the **move-sequence fingerprint** hashes every
decision it makes across 60 seeded games. They cover each other's blind spots: a structural edit
changes behaviour while the weights read identical (H1: +0.183, zero weights touched — the pin is
blind), and a saturated-hold edit changes the data while behaviour is identical (`p_king` 5.0→6.0
flips no argmax — the fingerprint is blind). **If a fingerprint fails, do not paste in the new hash**
— a frozen version drifting means the numbers published against it no longer describe the agent in
the tree. Real behaviour changes get a **new version**; the old one is never edited.

**The facts to carry forward** (RESEARCH_LOG 2026-07-14):
1. **The heuristic's 18-point defect was STRUCTURAL, not a bad weight.** `score_draw = -5.0` competed
   in the same `max()` as the plays, so any hold above 5.0 bought a *refusal* (-13 pts) instead of a
   *preference*. H1 (`refusal_mode='structural'`) makes refusing rules-only, so **no weight
   assignment can produce a voluntary draw** — pinned as a property test. `h1` carries R3's exact
   weights and beats it by **+0.183**.
2. **Therefore every hold-back ablation B2 ran was confounded.** Any weight B2 priced at or above 5.0
   was priced against the cliff. When H1 removes it, B2's conclusions **invert**: the hold-backs
   *earn* points (`w_nofin` +1.1, the hoard +2.9, spending the blocker +2.2) rather than buying
   nothing, and big holds are not "catastrophic" but merely *saturating*. Do not cite B2's ablation
   numbers without re-measuring under `structural`.
3. **The hold-backs are the endgame skill — do not "simplify" them away.** PLAN wanted B6 (the hoard)
   deleted; it is worth **+2.9**. H7 added the natural-looking RACE behaviour (when a near-winner
   cannot be blocked, dump maximally); it **loses monotonically**, because racing discards exactly
   the holds that pay most at the end. *Statement about Taki: when an opponent is about to win, do
   not empty your hand faster — that is when your kept cards are worth the most.*
4. **The H-series' predictions failed far more often than they succeeded, and the failures taught
   more than the wins.** H2, H5, H6, H7 all refuted; H8's freed search space turned out empty. What
   actually paid was the one *structural* change (H1). Be suspicious of a weight-level fix for a
   structure-level bug — and of a per-decision rollout price (H2's) surviving contact with a policy
   that has to live with the consequences.

A8's architecture is still what trains: self-play collects transitions from all four seats via
`train.py`'s turn-by-turn loop (~4x data/trial; replay buffer 20k->80k), on the 147-float A4+A7
observation (opener randomized per trial; opponent hand sizes/deck size/unseen counts; no discard
histogram). Color-sym and rank-sym replay augmentation are both on by default (opt out with
`--no-color-sym` / `--no-rank-sym`). **Rank-sym is on by principle, not evidence:** the nine number
cards are an exact symmetry of the rules, so relabeling them is free bias removal — but it measured
at exact *parity* (R1, RESEARCH_LOG.md), so the default is not a claim that it helps.

## Environment & Commands

Python runs from the conda env `tensorflow_env` (Python 3.9 — TF is pinned `>=2.4,<2.11` and numpy `<1.24`, so Python must be ≤ 3.10). CPU-only is fine.

```bash
python -m unittest gametest                                  # all rules-engine tests
python -m unittest gametest.GameFlowTest.test_stop_skips_next_player   # single test
python -m unittest dqntest                                   # DQN load path / obs-contract adapter
python -m unittest traintest                                 # seat-count sampling; single-count runs unperturbed

python train.py --trials 3000 --reward shaped                # train (self-play)
python train.py --model <ckpt> --epsilon-start 0.1           # warm-start, near-greedy continue
python train.py --num-players 2 --trials 100000              # self-play at 2 or 3 seats (default 4)
python train.py --num-players 2,3,4 --trials 300000          # MIXED-COUNT: count sampled uniformly per trial.
                                                             # The only mode where the seat-count one-hot does
                                                             # anything (it is constant, hence inert, at one count)

python eval.py --model <ckpt> --games 1000                   # Mode A: win rate vs random
python eval.py --run-dir models/run<ts> --baseline <ckpt> \
    --snap-stride 100 --games-b 3000                         # Mode B: snapshot progression
python eval.py --model <ckpt> --opponent heuristic \
    --games 3000                                             # vs the REFERENCE heuristic (r3) — the discriminating opponent
python eval.py --model <ckpt> --opponent heuristic:h1b2 \
    --games 3000                                             # vs a specific frozen version (r3/b2/h1/h1b2/greedy)
python eval.py --model heuristic --games 3000                # the heuristic itself, vs random
python eval.py --model <ckpt> --opponent <ckpt2> \
    --num-players 2 --games 3000                             # 1-vs-N against a MODEL (the only way at 2/3 seats;
                                                             # eval_headtohead.py is 4-seat-only)
python holdback.py heuristic:h1 --games 200                  # refusal/hold-back census (draw refusals must be 0 under H1)
python tune_heuristic.py --games 3000 --passes 2             # H8 coordinate descent vs the champion (base h1b2)

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
  - card vector (63 slots: 60 colored, Change Color, Super TAKI, King), action scalars (65: 0–59 colored plays, 60 CHCOL, 61 Super TAKI, 62 King, 63 DRAW, 64 CLOSE_TAKI), and the 150-float observation (`OBSERVATION_SIZE`), all defined here and imported by `agents/dqn.py`. The colored-block stride is `TYPES_PER_COLOR` (15), deliberately decoupled from `len(Type)` because the King is a colorless wild (never in the colored block).
  - `valid_moves()` is the legality source of truth; agents only ever choose among legal moves.
  - Changing `OBSERVATION_SIZE`/`ACTION_SIZE` or the network shape **invalidates every saved checkpoint** — loads fail hard by design (no silent cold-start). **The King card (this branch) did exactly that:** every pre-King checkpoint (incl. `checkpoint_colorsym_snap180000`) is unloadable here. The last commit on `master` with the old 62/64/201 contract that still loads those models is **`344535a`** — check it out to use the pre-King models. **The A7 observation-reshape (this branch) did it again:** the observation went 205→147 — added the 3 opponent hand sizes in turn order, deck size, and unseen +2/King/CHCOL counts, and **removed the full discard-pile histogram** (only the shown top card remains, so the net has a coarse card-count sense, not a full memory of what's been played). That invalidates every pre-A7 checkpoint too, so this branch trains from scratch. Also on this branch (A4): training randomizes the opener per trial (the learner stays list index 0 but no longer opens 100% of games), matching eval's shuffled seating.
  - **The seat-count one-hot (this branch, `nplayers-onehot`) did it a THIRD time: 147→150.** The tail now carries a one-hot of the number of seats (2 / 3 / 4-or-more, saturating), so a policy can condition on a count that changes the *game* and not just the table size — STOP is a free extra turn at 2 seats, CHDIR is a no-op at 2 and identical to STOP at 3. Every net on this branch is trained from scratch: **you cannot warm-start from `checkpoint_r6L60_snap100000` or `checkpoint_a9rules_snap500000`**, and `train.py` refuses rather than trying. The last commit with the 147-float contract is **`43ef04d`**.
  - **The one-hot is INERT in a single-count run — do not "re-run the lineage with it" and expect anything.** Within a run at one count the feature never varies, and a constant input folds into the next layer's bias, so such a run is equivalent to one without the feature (verified: 40 seeded trials at `--num-players 4` give **bit-identical** weights across the commit that added mixed counts). It buys information only when the count *varies between trials* — i.e. under `--num-players 2,3,4`. The one-hot is the **prerequisite** for mixed-count training, not a change to single-count training.
  - **But an obs change does NOT make old champions unmeasurable — do not assume it does.** Every change so far APPENDED features, so today's vector is a strict superset and `obs[:147]` is bit-identical to the old contract (pinned by `gametest`, `dqntest`). `eval.py` / `eval_headtohead.py` pass `allow_obs_truncation=True`, which plays an older net on exactly the leading floats it was trained on — **faithful, not approximate**. R6 vs `heuristic:h8` reproduces its published **0.297** to the digit under the 150-float build, and R6 can sit at the same table as a one-hot net in a head-to-head. What R6 cannot see is the seat count: free at 4 seats (the one-hot is constant there), and the thing under test at 2 and 3. The adapter is **eval-only and opt-in** — `replay()` raises on a truncated net, so it can never quietly fit an old-contract net while the appended features go nowhere.
  - **Correction to "loads fail hard by design" above: a mismatched load does not fail at construction.** `AIAgent.__init__` catches the `set_weights` `ValueError` and adopts the checkpoint's own saved architecture. Without `allow_obs_truncation` that now raises a `RuntimeError` naming the contract gap; the deliberate eval path is the truncation above.
- **`agents/`** — anything with `play(game) -> (Action, Card)`: `dqn.py` (DQN with replay buffer; masks illegal actions in both action selection and the bootstrap target), `random.py` (uniform over legal moves, with `reseed()` for eval), `human.py`, `heuristic.py` (hand-crafted rules, no network — the independent non-lineage eval opponent, and a **registry of frozen versions** since the H-series: see "The heuristic is a VERSIONED, FROZEN yardstick" above. Consumes `game.history`, the public table-event log, and is bound by the same human information set as the DQN — it must never read opponents' hand *contents*).
- **`train.py` / `eval.py` / `main.py`** — orchestration. The training loop, not the agent, computes rewards (`--reward {shaped,win,anneal}`) and drives replay/target-sync cadence. Self-play = learner at seat 0 + 3 opponents whose weights sync to the learner every 5 trials.
  - **`--num-players` takes one count (`4`) or a set (`2,3,4`, sampled uniformly per trial).** Under a set the learner keeps index 0 and the first `n-1` opponents of the pool fill the table; the pool is sized to the largest count and *all* of it is synced, so an opponent that sat out a trial is still current. **One `Game` instance is reused and reseated via `Game.reset(agents=...)` — do not "clean this up" by constructing a Game per trial.** The deck RNG lives on the instance, so a fresh Game per trial restarts the stream and deals **identical cards every trial** (pinned by `traintest.py`). The count sampler has its own RNG and is drawn from *only* when more than one count is given, which is what keeps single-count runs bit-identical to the pre-mixed-count code.

## Evaluation discipline (do not break)

- **Training win rate is meaningless** (self-play is symmetric → sits near 1/N regardless of skill); always measure with `eval.py`.
- **vs-random is retired as a *ranking* metric (R3, 2026-07-12).** The hand-crafted `HeuristicAgent` scores 0.849 vs 3 random against A8's 0.912 — a few hundred lines of rules recover most of the headline number, so it has almost no resolution at the top. Keep it only as a smoke test that a run hasn't collapsed. **For discriminating power use `eval.py --opponent heuristic`** (A8 scores 0.343 there, parity 0.25), or `eval_headtohead.py` for model-vs-model.
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
- **NEVER symlink `models/` into a worktree, and never `git add -A` if you have.** `.gitignore` has `/models/*` (contents) but `models/.gitkeep` is *tracked*, so a **symlink at the path `models` is not ignored** — `git add -A` stages it as a tracked symlink, and the next checkout/merge of that commit **deletes the real `models/` directory**, ignored contents and all. This destroyed every checkpoint in `models/` once (2026-07-12), champion included. To use checkpoints from a worktree, pass an **absolute path** (`--model /home/orih/taki-ai/models/<ckpt>`) instead.
- `models/` is gitignored (checkpoints are regenerable). Snapshots land in `models/run<timestamp>/snap<NNNN>/`; promoted best models are kept as `models/checkpoint_*` and named in RESEARCH_LOG.md. Old `checkpoint<float-timestamp>` dirs are from a previous architecture and will not load.
- TF thread pools are deliberately shrunk (env vars in `eval.py` before the TF import; `tf.config.threading` in `agents/dqn.py`) — the tiny network thrashes on default pools. Keep any TF-touching entry point consistent with this.
- Reward lessons already learned (don't re-run blind): `shaped` trained the best model; switching a trained model to `win`-only collapsed it; `anneal`'s win-dominated tail measurably hurts. Details in RESEARCH_LOG.md.
- **Architecture lessons already learned (don't re-run blind):** the wide network (`--wide`, 256->128->64, 2.72x params) buys **nothing robust** — at 300k it is identical to narrow across three seeds vs `h8`, and its one apparent win (blocking, M2 seed 1) did **not** replicate (RESEARCH_LOG 2026-07-18). This matches A10 (width inert at four seats). The flag exists for reproducibility of that result; default narrow. Also note the `delta(k)` blocking probe is a **transient** — its sign swings across ~[-2.5,+2.5] over training with no stable tie to seed or architecture, so never call blocking from one snapshot of one run.
- `AIAgent`'s `epsilon_decay` constructor default is dead — `train.py` always overwrites it with a run-length-scaled value.
