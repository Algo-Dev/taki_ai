#!/usr/bin/env bash
# (a) SE-estimator diagnostic: measure the actual correlation between orbit pairs on a shared
#     deck, to check the claim that pooling understates the SE (I asserted ~sqrt(3), which is
#     only the rho=1 worst case).
# (b) The 4-seat 1v3 orbit: A,B,B,B and B,A,A,A with the solo seat ROTATING through all four
#     positions -- 4 pairs = 8 runs, vs the 2 runs the old hardcoded harness could do.
#     The old published 1v3 secondary (+0.0500 +/- 0.0098) used the solo seat at position 0
#     ONLY; it should reappear here as pair 1.
set -u
cd /home/orih/taki-ai
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate tensorflow_env

M=/home/orih/taki-ai/models
CHAMP=$M/checkpoint_M1s3_mixed_snap300000
R6=$M/checkpoint_r6L60_snap100000
G=3000
mkdir -p results
OUT=results/orbit_1v3_results.txt; : > $OUT
say() { echo "$@" | tee -a $OUT; }

say "######## (a) SE estimator diagnostic (3 seats, vs h8, block 0) ########"
nice -n 19 ionice -c2 -n7 python runs/se_check.py 2>&1 \
  | grep -vE "^  game |^--- " | tee -a $OUT

say ""
say "######## (b) 4-seat 1v3 orbit: solo seat rotating (4 pairs = 8 runs) ########"
for OPP in "$R6" "heuristic:h8"; do
    for blk in 0 777777; do
        say ""
        say "=== M1s3 vs $(basename "$OPP") | 4 seats | 1v3 rotating | deck block $blk ==="
        nice -n 19 ionice -c2 -n7 python eval_headtohead.py "$CHAMP" "$OPP" \
             --num-players 4 --team1-seats 0 --seat-swap --games $G --seed "$blk" 2>&1 \
          | grep -E "complementary pair|pair [0-9] \(|Seat-balanced|^    checkpoint|^    heuristic|Paired margin|\+/-|Verdict" \
          | tee -a $OUT
    done
done

say ""
say "1v3 ORBIT COMPLETE $(date +%H:%M)"
