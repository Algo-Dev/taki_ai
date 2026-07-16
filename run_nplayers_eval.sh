#!/usr/bin/env bash
# Evals for the seat-count-one-hot curriculum. Run after run_nplayers.sh, which writes
# snapshots.env with the five checkpoint paths.
#
# R6 IS PLAYABLE HERE, and that is what makes these evals strong. The one-hot was APPENDED,
# so the 150-float observation's leading 147 are bit-identical to the old contract (pinned in
# gametest + dqntest); eval loads R6 with allow_obs_truncation and feeds it exactly the vector
# it was trained on. Verified end to end: R6 vs h8 reproduces its published 0.297 to the digit.
# So the champion can sit at the same table as a new net, and comparisons here are direct
# head-to-heads rather than inferences chained through the heuristic.
#
# What R6 cannot see is the seat count. At 4 seats that costs it nothing -- the one-hot is a
# constant there -- so a 4p match against it is a fair fight. At 2/3 seats it is genuinely
# blind to the table size, which is not a handicap to apologise for but the very thing being
# measured: it is the old champion played at a table it cannot perceive.
set -u
cd /home/orih/taki-ai-nplayers
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate tensorflow_env
source ./snapshots.env

R6=/home/orih/taki-ai/models/checkpoint_r6L60_snap100000
E="nice -n 19 ionice -c2 -n7 python eval.py"
H="nice -n 19 ionice -c2 -n7 python eval_headtohead.py"
G=3000   # >=3000: SE ~ +/-0.008, needed for the 2-3 pt gaps this lineage produces
         # (3000 games runs in ~13s, so there is no reason to skimp)

# --- E0: the adapter still reproduces R6's published number. Cheap, and everything below
# is only meaningful if it holds. Must print 0.297.
echo "=== E0: adapter check -- R6 vs h8 @4p must be 0.297 ==="
$E --model "$R6" --opponent heuristic:h8 --num-players 4 --games $G 2>&1 | tee eval_E0_R6_h8_4p.log

# --- E1: is the one-hot a no-op at 4 seats? THE NULL. At 4 seats the new block is a constant
# input, so N0 is R6's exact recipe plus three dead inputs -- it should land on R6's 0.297 up
# to seed noise, and tie R6 head-to-head. If it does not, the finding is about the observation
# change (or run-to-run variance) and nothing downstream is interpretable yet.
echo "=== E1: N0 vs h8 @4p  (expect ~0.297; parity 0.250) ==="
$E --model "$N0" --opponent heuristic:h8 --num-players 4 --games $G 2>&1 | tee eval_E1_N0_h8_4p.log
echo "=== E1b: N0 vs R6 head-to-head @4p  (expect a tie; parity 0.0) ==="
$H "$N0" "$R6" --team1-seats 0,2 --seat-swap --games $G 2>&1 | tee eval_E1b_N0_vs_R6.log

# --- E2: the curriculum question. Both arms are 200k at identical settings; the only
# difference is that ARM A spent its first 100k at 2 seats. Standard promotion test:
# alternating 2v2, seat-swapped, paired over common decks. Headline = paired same-seat margin.
echo "=== E2: ARM A vs ARM B, 2v2 seat-swap @4p  (paired margin, parity 0.0) ==="
$H "$ARM_A" "$ARM_B" --team1-seats 0,2 --seat-swap --games $G 2>&1 | tee eval_E2_armA_vs_armB.log
# 1v3 secondary: same shape as Mode B and as the champion's headline. Disagreement with E2 is
# real information -- it says the arm's edge depends on field composition -- not a bug.
echo "=== E2b: ARM A vs ARM B, 1v3 seat-swap @4p (secondary) ==="
$H "$ARM_A" "$ARM_B" --team1-seats 0 --seat-swap --games $G 2>&1 | tee eval_E2b_armA_vs_armB_1v3.log

# --- E3: both arms against the actual champion and against the yardstick, so each lands on
# the same scale as every published number. This is the promotion decision.
for arm in ARM_A ARM_B; do
    echo "=== E3: $arm vs R6, 2v2 seat-swap @4p  (parity 0.0) ==="
    $H "${!arm}" "$R6" --team1-seats 0,2 --seat-swap --games $G 2>&1 | tee "eval_E3_${arm}_vs_R6.log"
    echo "=== E3: $arm vs h8 @4p  (R6 = 0.297; parity 0.250) ==="
    $E --model "${!arm}" --opponent heuristic:h8 --num-players 4 --games $G 2>&1 \
        | tee "eval_E3_${arm}_h8_4p.log"
done

# --- E4: the specialists at their own table sizes, against BOTH the yardstick and R6.
# The R6 column is the real question of this whole branch: does a net that trained at 2 (or 3)
# seats and can see the seat count beat the 4p champion at that table? Parity is 1/num_players.
echo "=== E4: P2 vs h8 @2p  (parity 0.500) ==="
$E --model "$P2" --opponent heuristic:h8 --num-players 2 --games $G 2>&1 | tee eval_E4_P2_h8_2p.log
echo "=== E4: P2 vs R6 @2p  (1v1; parity 0.500) ==="
$E --model "$P2" --opponent "$R6" --num-players 2 --games $G 2>&1 | tee eval_E4_P2_vs_R6_2p.log
echo "=== E4: R6 vs h8 @2p  (the baseline P2 must beat; parity 0.500) ==="
$E --model "$R6" --opponent heuristic:h8 --num-players 2 --games $G 2>&1 | tee eval_E4_R6_h8_2p.log

echo "=== E4: 3P vs h8 @3p  (parity 0.333) ==="
$E --model "$P3" --opponent heuristic:h8 --num-players 3 --games $G 2>&1 | tee eval_E4_3P_h8_3p.log
echo "=== E4: 3P vs R6 @3p  (1v2; parity 0.333) ==="
$E --model "$P3" --opponent "$R6" --num-players 3 --games $G 2>&1 | tee eval_E4_3P_vs_R6_3p.log
echo "=== E4: R6 vs h8 @3p  (the baseline 3P must beat; parity 0.333) ==="
$E --model "$R6" --opponent heuristic:h8 --num-players 3 --games $G 2>&1 | tee eval_E4_R6_h8_3p.log

# --- E5: vs-random, smoke ONLY. Retired as a ranking metric (R3); h8 itself scores 0.906.
# A collapsed run shows up here; nothing else does.
echo "=== E5: smoke, vs random ==="
$E --model "$N0" --num-players 4 --games 1000 2>&1 | tail -3 | tee eval_E5_N0_random.log
$E --model "$P2" --num-players 2 --games 1000 2>&1 | tail -3 | tee eval_E5_P2_random.log

echo "ALL EVALS COMPLETE $(date +%H:%M)"
