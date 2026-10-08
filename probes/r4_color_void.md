# R4 probe — does the policy USE the per-opponent color-void feature?

Harness: `probe_void.py` (Q arm), scenarios `void_{blue,none,green}` in `probe.py`.
Nets: R4 candidate `seed2 snap300000` (162 floats); control `checkpoint_M1s3_mixed_snap300000`
(150 floats — cannot see the void block).

**Where the nets are.** `models/` is gitignored, so the three R4 seeds are published as assets on
the [`r4-color-void-models` release](https://github.com/Algo-Dev/taki_ai/releases/tag/r4-color-void-models):
`checkpoint_r4_seed{1,2,3}_snap300000.tar.gz` (the nets this probe measured; seed 2 is the
promotion candidate `runs/run_r4_promote.sh` evaluates) and `r4_full_runs_all_snapshots.tar.gz`
(all three runs, every 10k-trial snapshot). They load only on this branch: the observation here
is 162 floats. Training runs were `runs/run_r4_train.sh`; in the original worktree seed 1 was
`models/run1784638148.283497`, seed 2 `models/run1784651224.783525`, seed 3
`models/run1784625099.535345`.

## The feature

Since R4 the observation carries, for each of the 3 opponents in turn order, a "lacks color c"
belief in [0,1] for each of the 4 colors (`game.py`, `_OpponentModel`, appended at obs[150:162]).
A draw while color c was active sets it (~0.77 after the drawn card's own decay); it decays as
that seat draws; a genuinely-colored play of c clears it. This is the heuristic's own B8 model,
now shared with the DQN observation.

## The probe

One fixed 4-seat position: learner (seat 0) on a red 5, holding blue-5, green-5 and two dead
yellows. The only real choice is **which color to leave the next player (seat 1)** — play blue-5
(leave blue active) or green-5 (leave green). Correct denial: hand them a color they *lack*, so
they must draw.

Three variants differ ONLY in seat 1's injected history:
- `void_blue`: seat 1 drew while blue was active → lacks blue.
- `void_none`: no revealed void.
- `void_green`: seat 1 drew while green was active → lacks green.

Because only `history` changes, the deck / discard / hands are byte-identical across variants, so
any shift in the policy's preference isolates the void feature. Read:

    D = Q(play blue-5) − Q(play green-5)
    S = D(void_blue) − D(void_green)     # the void-usage signal

A void-using net raises D when seat 1 lacks blue and lowers it when it lacks green → `S > 0`.
A pre-R4 net cannot see the void block, so its D is identical across variants → `S = 0` by
construction (built-in negative control). Each net is fed obs[:obs_size], so the control is
genuinely blind to the appended floats.

## Result

| net | D(lacks blue) | D(none) | D(lacks green) | S |
|---|---|---|---|---|
| R4 seed 2 snap300000 | +0.86 | +0.30 | −0.95 | **+1.81** |
| R4 seed 3 snap300000 | +0.92 | +0.21 | +0.07 | +0.85 |
| R4 seed 1 snap300000 | +2.46 | +1.56 | +0.20 | +2.27 |
| M1s3 (control, 150f) | −1.376 | −1.376 | −1.376 | **0.0000** |

Every R4 seed gives `S > 0`; for seed 2 the sign of D flips with the void (+0.86 → −0.95). The
control is flat to four decimals across all three variants. **The policy learned to steer the
game onto the color the next player cannot follow** — the core human-level denial inference, which
the pre-R4 net structurally could not perform.

## Caveat

Q is a shaped return, not a win probability, so only the *ordering* / *shift* is interpretable
(that is why S is a within-position difference, and why the control pins the null at exactly 0).
The behaviour is real; whether it pays in win rate is a separate question — and the head-to-head
says it is a wash except at two seats (RESEARCH_LOG 2026-07-21). Per-seed sd is 0 because the
constructed observation is deck-seed-invariant here (numbers-only filler pins the unseen counts).
