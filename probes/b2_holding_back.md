# B2 — Holding cards back: the shaped reward is the champion's best habit *and* its worst blind spot

**Policy probed:** `models/checkpoint_a9rules_snap500000` (the champion on the corrected
finishing rule; 0.906 vs 3 random, 0.348 vs 3 heuristic).
**Harness:** `holdback.py` (census), `scenarios_b2.py` (weapon timing), `agents/heuristic.py`
(`Weights`/`GREEDY`/`ABLATIONS`), `eval_headtohead.py --seat-swap`, `probe.mc_line` (rollouts).
**Tests:** `holdbacktest.py` (17) + `agenttest.py` (+5). Full suite 100 green.

PLAN.md asked: *the shaped reward pays `-len(hand)` every step, so the policy is trained to be
greedy about hand size. Does it ever play **fewer** cards now for a better end-game? If it never
does, that is a real finding about the shaped reward's bias, and the strongest argument yet for R6.*

## Verdict

**B2 asked one question that turned out to be two, and they have opposite answers.** The shaped
reward is simultaneously the source of the champion's best discipline and of its worst blind spot.

**Half 1 — "does it shed greedily?" The premise is FALSE. The reward is ALIGNED.**

1. **"Holding back" is not one behaviour. It is two, and they have opposite signs.**
   - **REFUSAL** — decline to put a card down at all (DRAW, or CLOSE a TAKI/King continuation)
     while a legal play exists. You keep the card *and* lose the tempo. **Catastrophic (−13 pts).**
   - **PREFERENCE** — play a *different* card instead. You keep the card, you spend the turn.
     **Valuable (+2 to +7 pts).**
2. **The champion already has this exactly right**: it refuses on **1.8%** of its free decisions
   (essentially never) while holding back by preference on **20.4%** of them. It is *not* a greedy
   hand-size minimiser, and it is not making the mistake B2 predicted.
3. **The shaped reward is why it gets this right.** `-len(hand)` per step punishes drawing — drawing
   *grows* the hand — so the training signal directly encodes the most valuable rule in the game.

**Half 2 — "does it hold a weapon for a near-winner?" It FAILS, and the shaped reward is directly,
mechanically the cause. R6 is PROVEN — on defence, not on shedding.**

4. With the next player one card from winning, the champion **declines to block** — and blocking is
   worth **+0.050 ± 0.017** by rollout. Worse, its preference for the wrong move is *strongest*
   exactly there. **It is most confidently wrong where the stakes are highest.**
5. **The mechanism is visible in the Q-values and then in the reward code.** `train.py`'s
   `seat_reward` is `-len(hand)` every step, plus a bonus **only on a win** — **a loss pays nothing,
   the penalty stream simply stops.** Since every step is negative, *an opponent about to win is good
   news*: it ends the stream sooner. The champion values the identical hand at **−6.56 when the next
   player holds 1 card** and **−9.93 when they hold 7**. Blocking prolongs the game, so it looks
   *worse*. **The value function is not mis-fit — it is faithfully optimising an objective that is
   indifferent to who wins.** That is exactly PLAN.md R6, now with a complete causal chain.

**Bonus, and a big one: R3's heuristic is badly tuned.** Fixing only its hold-back discipline takes
it from **0.850 → 0.899** vs random and to **statistical parity with the 500k-trial DQN champion**.
The project's non-lineage yardstick was weak because of a tuning bug.

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

## Result 6 — Weapon timing: the champion is most confidently wrong where it matters most

*This is the sharpest probe in the B-series, and it is where R6 actually lives.* `scenarios_b2.py`.

A red +2 and a red 5, both legal on a red 7, **both shedding exactly one card**. Because the shed is
identical, the shaped reward is **exactly indifferent** between them — so whatever preference the net
shows here **was not paid for**, and this isolates the *value function* from the *reward*. The only
thing that varies across the sweep is **k**, the number of cards in the **next player's** hand (the
legal set, deck size and unseen counts are constant by construction — the `b1c` matched-contrast
pattern). Strategy says: fire the +2 when k is small.

```
  k (next player's cards)    Q(+2)   Q(red 5)    delta = Q(+2) - Q(5)    picks
  1                         -10.58      -6.56                  -4.01     red 5
  2                         -10.72      -7.98                  -2.74     red 5
  3                         -11.36      -9.00                  -2.36     red 5
  5                         -11.74      -9.66                  -2.08     red 5
  7                         -11.40      -9.93                  -1.47     red 5
```

**It does use the feature** — the spread over k is **2.55**, against an untrained-net noise floor of
**0.01–0.06** (three seeds). This is not indifference.

**But the sign is backwards.** `delta` should *rise* as k falls. It *falls*: the +2 is valued least,
relative to the number, exactly when the next player is one card from winning.

### Rollout adjudication (1200 determinizations, paired, 3× heuristic opponents)

| k | force the block (+2) | force the number (5) | block − number | champion plays |
|---|---|---|---|---|
| **1** | **0.310** | **0.260** | **+0.050 ± 0.017** | the number ❌ **wrong** |
| 2 | 0.319 | 0.369 | −0.050 ± 0.017 | the number ✓ |
| 3 | 0.342 | 0.378 | −0.036 ± 0.017 | the number ✓ |
| 5 | 0.358 | 0.394 | −0.036 ± 0.018 | the number ✓ |

**It is right at every k except the one that matters, and it is most confident precisely there.**

### Why — and it is not a mis-fit network

Look at `Q(red 5)`: **−6.56 at k=1**, **−9.93 at k=7**. The *identical* hand is valued **3.4 higher
when an opponent is about to win.** That is not a bug in the net; it is the net correctly predicting
its own objective. From [train.py:288-290](../train.py#L288-L290):

```python
r = -len(game.hands[seat])                                    # every step, always negative
if won:
    r += sum(len(h) for i, h in enumerate(game.hands) if i != seat)   # bonus only on a WIN
```

**A loss pays nothing — the stream of negative rewards simply stops.** So an imminent loss is
*rewarded*: it truncates the penalty. And blocking the near-winner **prolongs** the game, which the
objective punishes. The agent is trained to let a near-winner go out.

> **Statement about Taki #2: block the player who is about to go out.** Spending a +2 on a
> one-card opponent is worth ~5 points; spending it on anyone else is worth *less* than a plain
> number. The champion has the second half and not the first.

**This is R6, with a complete causal chain**: the reward code → the Q-values → the behaviour → a
measured 5.2-point cost. Note what B2 kills and what it proves. R6's *hold-back* justification
("the shaped reward makes it greedily shed") is **dead** — that reward term is doing the most useful
work in the model. R6's *defensive* justification ("losers get no terminal penalty, so defence is
under-incentivised") is **exactly right**, and this is the first direct evidence for it.

---

## What this means for the plan

- **R6 is motivated — but for the opposite reason from the one PLAN.md gave.** Do **not** build it to
  stop greedy shedding (the `-len(hand)` term is what teaches "never refuse to play", the single most
  valuable rule we found). Build it because **a loss is currently free**, which makes an imminent
  defeat look *good* and defence look *bad*. The fix is a terminal loss penalty; the pre-registered
  prediction is that `delta(k)` in Result 6 flips sign at k=1 and the champion starts blocking.
- **Promote the retuned heuristic** as the project's eval yardstick, and re-run anything that used the
  R3 one to rank a model (R3's own conclusions included).
- **The 0.906-vs-0.348 gap is smaller than it looked.** Against a competent opponent the champion is
  at parity with hand-written rules — a much sharper statement of the plateau than vs-random ever gave.

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
