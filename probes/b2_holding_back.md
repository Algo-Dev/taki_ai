# B2 — Holding cards back: the question was wrong, and the answer is worth 13 points

**Policy probed:** `models/checkpoint_a9rules_snap500000` (the champion on the corrected
finishing rule; 0.906 vs 3 random, 0.348 vs 3 heuristic).
**Harness:** `holdback.py` (census), `agents/heuristic.py` (`Weights`/`GREEDY`/`ABLATIONS`),
`eval_headtohead.py --seat-swap`. **Tests:** `holdbacktest.py` (14) + `agenttest.py` (+5).

PLAN.md asked: *the shaped reward pays `-len(hand)` every step, so the policy is trained to be
greedy about hand size. Does it ever play **fewer** cards now for a better end-game? If it never
does, that is a real finding about the shaped reward's bias, and the strongest argument yet for R6.*

## Verdict

**The premise is false, and R6 is unmotivated by it.**

1. **"Holding back" is not one behaviour. It is two, and they have opposite signs.**
   - **REFUSAL** — decline to put a card down at all (DRAW, or CLOSE a TAKI/King continuation)
     while a legal play exists. You keep the card *and* lose the tempo. **Catastrophic.**
   - **PREFERENCE** — play a *different* card instead. You keep the card, you spend the turn.
     **Valuable.**
2. **The champion already has this exactly right**: it refuses on **1.8%** of its free decisions
   (essentially never) while holding back by preference on **20.4%** of them. It is *not* a greedy
   hand-size minimiser, and it is not making the mistake B2 predicted.
3. **The shaped reward is the reason it gets this right, not a bias to be fixed.** `-len(hand)` per
   step punishes drawing — drawing *grows* the hand — so the training signal directly encodes the
   single most valuable discipline in the game. It is **aligned** with the dominant strategic truth,
   not misaligned. **R6 (a terminal loss penalty, as a fidelity fix for hold-back) should not be
   built on B2's evidence.**
4. **Bonus, and it is a big one: R3's heuristic is badly tuned.** Fixing only its hold-back
   discipline takes it from **0.850 → 0.899** vs random and to **statistical parity with the
   500k-trial DQN champion**. The project's non-lineage yardstick was weak because of a tuning bug.

---

## How this was measured (and why not with the DQN's own rollouts)

B2's question — *does holding back pay?* — cannot be honestly answered by asking the DQN's value
function, because the value function is the thing under suspicion. So we asked an agent whose
strategy we **own**.

`agents/heuristic.py` (R3) is a hand-coded agent that holds cards back **on purpose**, and its
hold-back lives in named constants. Critically, `SCORE_DRAW = -5.0` is a *finite score competing in
the same `max`*: **any hold penalty above 5.0 makes the agent voluntarily draw rather than play a
legal card.** That is the refusal/preference line, and it is a *mechanism*, not an interpretation.

So: refactor the constants into a `Weights` dataclass (defaults byte-identical — pinned by
`agenttest`), switch each hold-back behaviour off, and play the variants against each other with
`eval_headtohead.py --team1-seats 0,2 --seat-swap` (the promotion standard: both occupancies of an
alternating 2v2 over the same decks, paired). No network, no rollouts, seconds per run.

---

## Result 1 — Every hold-back term, priced

Margin = full agent − ablated agent, 3000 games × 2 occupancies, seed 0.
**Positive ⇒ the behaviour is worth having.**

| behaviour ablated | margin | kind | verdict |
|---|---|---|---|
| **hoard** (`w_reserve` 10→0) | **−0.0623 ± 0.0105** | refusal (draws to protect the group) | **harmful** |
| **wilds** (`p_king`/`p_chcol`/`p_super`, in-run) | **−0.0450 ± 0.0116** | mixed | **harmful net** |
| **king_cancel** (eat +2 to keep the King) | **−0.0353 ± 0.0054** | refusal (draws 2) | **harmful** |
| blocker (save STOP/+2 with no threat) | +0.0110 ± 0.0064 | preference | mildly good (1.7 SE) |
| finisher (keep a legal last card) | +0.0083 ± 0.0035 | rules-driven | good (2.4 SE) |
| king_follow (decline the free card) | **+0.0000 ± 0.0000** | — | **never fires** |

`king_follow` reading *exactly* ±0.0000 is not noise — the census shows the full agent declines a
King's follow-up **0 times in 200 games**. The code path never binds. (A nice consistency check on
both instruments at once.)

## Result 2 — The mechanism: a 13-point cliff at exactly the draw threshold

The `wilds` row above is a mix, so decompose it. Sweep `p_king` alone, everything else held at the
full agent's values. `SCORE_DRAW = -5.0` predicts a behavioural flip between 5 and 6:

| `p_king` | margin vs `p_king=0` | behaviour |
|---|---|---|
| 2 | +0.0160 ± 0.0044 | prefers other plays; **still plays the King** |
| 4 | +0.0190 ± 0.0051 | " |
| **5** | **+0.0200 ± 0.0051** | " ← peak: holding the King is worth **+2.0 pts** |
| **6** | **−0.1043 ± 0.0093** | **draws rather than play it** ← **−10.4 pts** |
| 8 | −0.1043 ± 0.0093 | identical to 6 (same policy — the effect is the *flip*, not the size) |

**A 12.4-point swing across a one-unit parameter change, located exactly where the mechanism says
it must be.** Directly: `p_king=6` vs `p_king=5` = **−0.130 ± 0.009**, and it replicates on three
disjoint deck blocks (−0.130 / −0.129 / −0.132).

The same shape appears in the hoard. Sweeping `w_reserve` across the same threshold:

| `w_reserve` | 0 | 1 | 2 | 3 | 4 | 6 | **8** | **10** |
|---|---|---|---|---|---|---|---|---|
| margin | −0.013 | −0.002 | −0.003 | −0.002 | +0.003 | +0.004 | **−0.026** | **−0.069** |

Flat (≈0) all the way up to 6 — *hoarding costs nothing and is free to do* — then it collapses once
the reserve is strong enough to make the agent **draw** to protect the group.

> **Statement about Taki #1: never draw to protect a plan.** Keeping a card by *playing something
> else* is free or better. Keeping it by *passing* costs more than the card is ever worth. Every
> harmful hold-back we found is a refusal; every valuable one is a preference.

## Result 3 — This resolves B1's apparent contradiction

B1 concluded *"dumping the TAKI never beats keeping it, at any backing level"* and used that as
evidence the policy was too greedy. The ablation says hoarding is **−6.2 points**. Both are right:

**B1 only ever compared play-vs-play** (dump the TAKI, or shed an off-colour card — both plays).
That is the preference regime, and there B1's finding is confirmed exactly: `w_reserve` 0→6 is flat.
The R3 heuristic's `W_RESERVE = 10.0` goes *further* than anything B1 tested — it **draws** rather
than break the group. The 6.2 points are the drawing, not the hoarding. No contradiction, and B1's
half-retracted hoarding result survives intact — with a sharp boundary now drawn around it.

## Result 4 — Where the champion actually sits (the census)

`holdback.py` walks real self-play games and, at every decision, asks whether the agent chose a move
that provably sheds fewer cards **this turn** than a legal alternative. "Sheds fewer" is a rules-level
search (`max_shed`), not a rule of thumb, because action identity lies: closing a TAKI on a PLUS or a
King **keeps the turn** (game.py:584-598), so `CLOSE_TAKI` is not always a refusal at all.

200 games, self-play (all 4 seats recorded), seed 0:

| agent | REFUSAL (% of free decisions) | PREFERENCE hold-back | vs 3× random |
|---|---|---|---|
| `random` | 35.1% | 47.1% | 0.250 |
| `heuristic` (R3, as shipped) | **14.5%** | 34.6% | 0.850 |
| **DQN champion** | **1.8%** | **20.4%** | **0.906** |
| `heuristic` (retuned) | 0.3% | 14.4% | 0.899 |
| `heuristic:greedy` (control) | **0.0%** | 13.7% | 0.879 |

The controls behave exactly as they must: the ablated agent refuses **0** times (the classifier is
measuring what we think), random refuses constantly.

**The champion is in the right regime and nowhere near the pathology B2 predicted.** It holds cards
back one time in five — it is not a greedy shedder — and it does so almost entirely by preference.

## Result 5 — Retuning the heuristic reaches DQN parity

Keep every hold-back that is a *preference*; delete every one that is a *refusal*
(`p_king=5, p_chcol=4.5, p_super_taki=4.5, w_reserve=4, king_cancel_min_penalty=0,
hold_wilds_in_run=false`):

| | vs 3× random | vs DQN champion (seat-swapped 2v2, 3 disjoint seeds) |
|---|---|---|
| heuristic (R3, shipped) | 0.850 | −0.208 ± 0.012 |
| heuristic (fully greedy) | 0.879 | — |
| **heuristic (retuned)** | **0.899** | **+0.018 / −0.018 / +0.005 → parity** |
| DQN champion | 0.906 | — |

**A few hundred lines of rules, retuned on nothing but hold-back discipline, match a 500k-trial DQN.**
R3's headline ("A8 is genuinely better, 0.343 vs 3 heuristics") was measured against a *crippled*
opponent: the yardstick had a 17-point tuning bug in it.

---

## What this means for the plan

- **R6 is not motivated by B2.** The shaped reward's `-len(hand)` term teaches "never refuse to play",
  which is the most valuable single rule we have found. Do not build a terminal loss penalty on the
  strength of the hold-back argument. (R6 may still be worth doing for *defensive* play — but that is
  a different claim, and B2 does not support it.)
- **Promote the retuned heuristic** as the project's eval yardstick, and re-run anything that used the
  R3 one to rank a model (R3's own conclusions included).
- **The 0.906-vs-0.348 gap is smaller than it looked.** Against a competent opponent the champion is
  at parity with hand-written rules — which is a much sharper statement of the plateau than vs-random
  ever gave.

## Caveats

- The retuned weights were swept on seed 0, then **confirmed on disjoint deck blocks** (the parity
  claim rests on 3 independent seeds); the exact weight *values* still carry some selection risk. They
  are not optimised, only un-broken — a real tuner would likely go further.
- `eval_headtohead` seeds deck *g* with `seed + g`, so seeds must be ≥ `games` apart to be independent.
  Seeds 0/1/2 share 2999 of 3000 decks and are **not** a replication (this bit us once; noted here so
  it does not again).
- The census is 200 self-play games per agent; the refusal rates rest on ~6k–80k free decisions each,
  so their standard errors are small, but they are one distribution (self-play), not all of them.
- **Not done: the Ĝ / B4 arm.** Arm 1 answered "does holding back pay" by a stronger, non-circular
  route, so measuring the empirical shaped return was not needed to reach the verdict. It remains the
  way to ask a *different* question: whether the champion's 20.4% preference hold-backs are
  well-*calibrated*, as opposed to merely present.
