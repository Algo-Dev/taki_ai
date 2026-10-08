# B1 — The colored-TAKI hoard: what does A8 actually know?

> ## ⚠️ PARTIAL RETRACTION (2026-07-12) — read before believing anything below
>
> **The engine's finishing rule was wrong when this was run.** `game.py` let a hand end only on a
> number or the King. The real Taki rule (now implemented): **you may finish on any card except
> PLUS** — PLUS alone obliges you to play another card, which an empty hand cannot do. See RULES.md.
>
> **What that kills — Result 1, the sequencing half, and "Statement about Taki #1".** Result 1's
> position (`red TAKI, red STOP, red +2, red 5`) rests on "the only finisher in hand is the red 5,
> so the run must end on it." Under the real rule the STOP and the +2 are finishers too: **every
> ordering of that run wins**, there is no trap, and the scenario is degenerate. The measured
> behaviour was real — A8 genuinely defers the 5, and the DiD control genuinely separates it from
> random-init nets — but it is skill at a game that isn't Taki. *Statement about Taki #1 is
> withdrawn.* Result 2 (the minimal pair) goes with it, for the same reason. Pinned in
> `probetest.py::test_b1a_has_no_sequencing_trap_every_order_wins`.
>
> **What survives — Result 3, the hoarding half (b1c).** The matched-TAKI sweep never touches the
> finishing rule: no hand in it can empty, so no finisher is ever chosen. Its finding — the policy
> dumps a well-backed TAKI that the rollouts say it should keep — stands as *reasoning*, but the
> numbers were produced by a policy trained on the buggy game and must be **re-measured against the
> retrained champion**.
>
> **Bigger point.** With PLUS the only card that cannot end a hand, "sequencing skill" in Taki is far
> thinner than this report assumed: a run needs planning only when it contains a PLUS. That absence
> is itself a finding, and it is what a redone B1 should test.

**Policy probed:** `models/checkpoint_a8_snap455000` — the champion *at the time*, 0.912 vs 3
random opponents. **That checkpoint no longer exists:** it was destroyed in the models-symlink
incident (2026-07-12, commits `0f96882`/`62a7433`) and `models/` is gitignored, so nothing was
recoverable. Nothing below can ever be *re-run* — only **redone** against a current champion,
which is what the retraction above asks for. It was also trained on the buggy finishing rule.
**Harness:** `probe.py`. **Consistency tests:** `probetest.py` (14 tests, all green).
**Raw output:** `b1_qarm.log` (Q arm), `b1_mc.log` (rollout arm) — run results are no longer
tracked (2026-07-19), so these name the runs that produced the numbers; a fresh clone will not
have the files.

The question, from PLAN.md: *with a colored TAKI plus several cards of that color, does the policy
**keep** them — the whole group discharges in a single turn — or **dump** the TAKI the moment it is
playable? And can it **sequence** the run to close on a legal finisher?*

## Verdict

**A8 splits the two halves of B1.**

- **Sequencing — it passes, decisively.** It plans a TAKI run backwards from its finisher, and it does
  so *specifically because a win is available*, not as a reflex (the control rejects that at ~1500×
  the null). Getting this right is worth **+0.40 win rate**.
- **Hoarding — it fails, at exactly the point PLAN.md predicted.** It correctly *keeps* a
  weakly-backed TAKI, but from two backers up it dumps — and the rollouts say **dumping never beats
  keeping, at any backing level, against either opponent pool.** At four backers, a five-card turn
  wins *less* often than a one-card turn (−0.033 ± 0.011).
- **The cause is legible:** the shaped reward pays `-len(hand)` per step, so the training signal
  massively overvalues a big discharge that the win rate does not care about. This is the strongest
  evidence yet for PLAN.md's **B2/R6**, and it motivates the new **B4**.

So PLAN.md's premise — *"hoarding is nearly free"* — is **confirmed by measurement**, and the policy
is the thing that doesn't fully believe it.

---

## How to read a Q-value here (this matters)

The numbers below are **not win probabilities**. `train.py` trains Q against a *shaped return*:
roughly `-len(hand)` at every step, discounted at γ=0.99 **per decision** — and each card of a TAKI
run is its own decision. So a Q of `-45` is not "bad odds", it is "I expect to be carrying cards for
a while". **Only the ordering within a single position is interpretable**, which is why every claim
below is a within-position comparison and never a cross-position one.

## Two engine facts the probes rest on (pinned by tests, not by assertion)

1. **A TAKI run is a *sequence* of decisions.** Playing a colored TAKI keeps the turn
   (`Game.next_turn` in [game.py](../game.py) only advances on `NORMAL`/`DRAW_TWO`/`STOP`), so the
   policy's *ordering* of the run is directly observable — one `play()` call per card.
2. **The penalty draw does not end the turn.** Emptying your hand on a non-finisher
   (the `FINISHING_TYPE_VALUES` branch of `Game.next_turn`) costs one card and you **keep playing, still inside the
   open TAKI**. So mis-sequencing forfeits a *guaranteed* win, not the win outright — the card you
   draw may itself be a red number and win anyway. Claims are phrased accordingly.
   (`probetest.EngineFactsTest` pins both, so this report cannot quietly drift from the engine.)

---

## Result 1 — It sequences the run correctly

Position: hand `red TAKI, red STOP, red +2, red 5`, showing a `red 3`. All four cards are legal. The
only finisher in hand is the red 5, so the run must **end** on it.

```
  move                            Q       dQ
  play red taki               -7.31     0.00   <- picks this
  play red stop              -13.96    -6.65
  play red 2+                -15.46    -8.15
  play red 5                 -23.39   -16.07
  draw                       -25.66   -18.34

  the policy actually plays:
  1. play red taki            hand 4 -> 3
  2. play red 2+              hand 3 -> 2
  3. play red stop            hand 2 -> 1
  4. play red 5               hand 1 -> 0   *** WINS ***
```

It opens the run, **defers the only finisher to last**, and wins outright in a single turn. Note the
red 5 is ranked *fourth of five* — below even the cards that shed nothing useful. A greedy
hand-size-minimiser has no reason to rank a playable number that low.

## Result 2 — …and it is *detecting the win*, not just disliking low numbers

Result 1 has an obvious alternative explanation: **a policy that merely dislikes playing low number
cards produces the identical ranking, for entirely the wrong reason.** So we take the same choice and
kill the win:

| | position | Q(red STOP) − Q(red 5) |
|---|---|---|
| **A — trap live** | in an open red TAKI, hand `red STOP, red 5`. STOP-then-5 **wins**. | **+18.54** |
| **B — trap dead** | identical **plus an unplayable blue 9**. No win available this turn. | **+2.91** |

**Difference-in-differences: +15.63.**

The preference for deferring the 5 is **six times stronger when the win is actually on the table**.
The generic-reflex explanation predicts ≈0 here. It is decisively rejected:

| policy | diff-in-diff |
|---|---|
| **A8 (champion)** | **+15.63** |
| `checkpoint_a4a7_snap550000` (different trained lineage, same encoding) | **+7.79** |
| random-init net × 5 seeds | −0.00, +0.01, −0.00, +0.01, −0.01 |
| `run…_bestresponse/snap0000` (untrained, real load path) | −0.00 |

The effect is absent in every untrained net, **replicates in an independently trained lineage**, and
is ~1500× the control's magnitude in A8.

> **Statement about Taki #1:** *Plan a TAKI run backwards from its last card.* The run must close on
> a number or the King, so the finisher is the one card you must not spend early. A8 knows this, and
> it knows it **specifically when the win is live** — not as a blanket habit.

## Result 3 — It does not dump the TAKI on sight; it prices it by its backing

Showing a **blue TAKI**, a red TAKI is legal *by type* while red numbers stay illegal outside the
run. That pins the legal set at exactly `{red TAKI, green TAKI, blue 1, DRAW}` for every variant, so
the *only* thing that changes is **k**, the number of red cards backing the red TAKI. The hand also
holds a **green TAKI backed by nothing** — a matched control card that differs only in the color of
the run it opens.

`Δ(k) = Q(red TAKI) − Q(green TAKI)` is therefore a within-position, matched-card contrast.
**At k=0 the observation is *exactly* invariant under swapping red↔green** (pinned in
`probetest.ColorSymmetryTest`), and that swap maps the red-TAKI action onto the green-TAKI one — so a
perfectly color-equivariant net **must** give Δ(0)=0. The measured Δ(0) is the net's residual
asymmetry: a free, calibrated noise floor.

| k (red backers) | Δ(k) = Q(red TAKI) − Q(green TAKI) | A8 actually picks |
|---|---|---|
| 0 | +0.41  ← noise floor | **play blue 1** (sheds; keeps the TAKI) |
| 1 | +0.65 | **play blue 1** (sheds; keeps the TAKI) |
| 2 | +1.57 | **play red TAKI** |
| 3 | **+2.05** | **play red TAKI** |
| 4 | +1.86 | **play red TAKI** |

Δ grows to ~5× its own noise floor as the backing grows, and **the behaviour flips at k=2**: with one
or zero red cards A8 *declines* the TAKI and sheds an off-color card instead; from two backers up it
opens the run. Random-init controls show Δ(k) **flat** across k (e.g. `+0.24, +0.25, +0.25, +0.22,
+0.22`) — they have a color bias but it is *insensitive to the hand*, which is the point.

> **Statement about Taki #2:** *A colored TAKI is worth what its backing is worth.* It is not a card
> to spend on sight — with nothing behind it, it sheds exactly one card and hands the turn on in a
> color you do not hold. A8 has learned the contingency and the crossover, not a reflex.

---

## Result 4 — Rollout adjudication: is the crossover in the *right place*?

Results 1–3 say what A8 *believes*. Result 3's crossover, unlike Result 1's sequencing, is **not
provable from the rules** — PLAN.md's "hoarding is nearly free" is an intuition, and in a 4-player
game dumping three cards now may simply beat saving them. So here we do not assert; we measure.

Each candidate line is **forced**, the policy then finishes the game, and the position is played out
repeatedly with the hidden cards resampled. The learner's hand, the discard, the state and the
opponents' hand *sizes* are held fixed; which cards the opponents actually hold and the deck order are
resampled. That is exactly the learner's belief state — sound because the observation is *bit-identical*
across resamples (it sees only sizes and coarse counts; pinned in `probetest.MonteCarloValidityTest`).
Every line sees the **same** deals (common random numbers), so differences are tested **paired**.

1500 games per line, two opponent pools: **copies of A8 itself at ε=0.1** (the self-play distribution
Q was trained under — the honest test of the policy's own belief) and **`RandomAgent`** (the pool
behind the headline 0.912). Parity vs three copies of itself is 0.25.

### 4a. Sequencing: A8 is right, and it is worth a great deal

| b1a line | vs A8 copies | vs random |
|---|---|---|
| `correct` (TAKI, +2, STOP, **5 last**) | **1.000** | **1.000** |
| `open_taki` (force only the TAKI, then let A8 choose) | **1.000** | **1.000** |
| `trap` (TAKI, then **5 early**) | 0.596 ± 0.013 | 0.923 ± 0.007 |
| `no_taki` (never open the run) | 0.357 ± 0.012 | 0.893 ± 0.008 |
| paired **correct − trap** | **+0.404 ± 0.013** | +0.077 ± 0.007 |
| paired **correct − open_taki** | **+0.000 ± 0.000** | +0.000 ± 0.000 |

Three things fall out. The `correct` line returning exactly **1.000** is the harness self-check — it
must, and it does. `open_taki` also returning **1.000** means that **once A8 opens the run it finishes
it correctly on its own, every single time** — Result 1 was not a lucky position. And mis-sequencing
is genuinely expensive: **+0.404** win rate. The trap's `win-this-turn` of 0.359 is exactly the engine
fact in action — you eat the penalty draw, keep the turn, and still win outright about a third of the
time when the drawn card happens to be red-and-a-number.

### 4b. Hoarding: **A8 is wrong, and the shaped reward is the reason**

`dump` = force the red TAKI (and discharge the run). `shed` = force the blue 1 instead, keeping the
group intact. Paired difference, so a negative number means **shedding beat dumping**:

| k | cards shed this turn (dump / shed) | **dump − shed** vs A8 copies | **dump − shed** vs random |
|---|---|---|---|
| 0 | 1 / 1 | −0.019 ± 0.012 | −0.024 ± 0.013 |
| 1 | 2 / 1 | −0.007 ± 0.013 | −0.023 ± 0.012 |
| 2 | 3 / 1 | −0.005 ± 0.014 | −0.019 ± 0.013 |
| 3 | 4 / 1 | +0.019 ± 0.015 | −0.001 ± 0.012 |
| 4 | **5 / 1** | −0.017 ± 0.017 | **−0.033 ± 0.011** |

**Dumping the TAKI never beats keeping it — at any k, against either pool.** Every entry is negative
or within noise of zero, and the largest, most significant one is at **k=4**, where dumping is
*worse*.

Look at what that row means. At k=4, dumping discharges **five cards in a single turn** (hand 7 → 2);
shedding plays **one** (hand 7 → 6). And the five-card turn **wins less often** (0.862 vs 0.895 against
random, a 3-SE gap). Shedding five times as many cards buys nothing, because the TAKI group was never
the problem — it was the guaranteed escape hatch, and cashing it early spends it for nothing while
leaving behind exactly the cards (a green TAKI, a blue 1) that may not be playable when the turn comes
back around.

**So PLAN.md's intuition is confirmed by measurement — and A8 only half-follows it.** Its behaviour
(Result 3) sheds at k ≤ 1 and dumps from k = 2 up. The first half is right. The second half is not:
the rollouts say it should be indifferent at worst and should still be *keeping* the group at k=4.

> **Statement about Taki #3:** *Cashing a colored TAKI is never urgent.* Shedding five cards this turn
> wins no more often than shedding one and keeping the group — the group is an asset precisely because
> it is **unspent**. Hold it and play your awkward off-color cards while you still have the tempo.

### 4c. Why it gets this wrong — and what to do about it

This is not a mysterious failure; it is the shaped reward doing exactly what it was told. `train.py`
pays roughly **`-len(hand)` every step**, so a five-card turn is enormously rewarded *in the training
signal* while being worth nothing *in win rate*. A8's Q-ranking prefers the dump from k=2 up
(Result 3) precisely where the win rate says not to. **Q and the win rate disagree, and the shaped
reward is the prime suspect.**

But note carefully what this probe **cannot** settle: whether the network mis-estimated its own
objective, or estimated it correctly and the objective is what's wrong. Separating those needs the
empirical shaped return Ĝ measured alongside the win rate — which is now **PLAN.md B4**, and which is
mostly bookkeeping on top of the rollouts that already exist here. This is also the strongest evidence
yet for **B2** (the shaped reward biases toward greedy hand-size reduction) and **R6**.

---

## Caveats

- **Q is a shaped return, not a win probability** (see above). Only within-position orderings are used.
- The rollouts assume a **uniform prior** over the unseen cards. That is the right prior given the
  observation, but a human would sometimes know better from the play history.
- The positions are hand-built. They are guaranteed *reachable* (`make_position` asserts the deck is
  conserved and that the discard is large enough for a real game to have reached the position), but
  they are still hand-picked, not sampled from play.
- `RandomAgent` has no Q-values; it is a control for the behaviour arm only, never the Q arm.
