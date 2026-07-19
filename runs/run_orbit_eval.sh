#!/usr/bin/env bash
# Re-evaluate the champion under the ROTATION-ORBIT head-to-head at every seat count.
#
# WHY: until now eval_headtohead.py was hardcoded to 4 seats, so the promotion's 2- and 3-seat
# claims rested on eval.py Mode A (1-vs-N, seeded seat SHUFFLE, mirrored composition never run).
# That is a weaker grade of evidence than the 4-seat paired swap. The orbit puts all three counts
# on the same footing.
#
# REGRESSION GATE: the 4-seat cell must reproduce the published +0.0417 +/- 0.0112 (2 blocks).
# If it does not, the generalization changed the 4-seat design and every published margin is
# invalidated -- stop and investigate rather than adopting the new number.
set -u
cd /home/orih/taki-ai
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate tensorflow_env

M=/home/orih/taki-ai/models
CHAMP=$M/checkpoint_M1s3_mixed_snap300000
R6=$M/checkpoint_r6L60_snap100000
G=3000
mkdir -p results
OUT=results/orbit_results.txt; : > $OUT
say() { echo "$@" | tee -a $OUT; }

# Two deck blocks, as the promotion used -- absolute values move up to 1.7 pts between blocks.
BLOCKS="0 777777"

cell() {   # cell <opponent> <label>
    local OPP="$1" LABEL="$2"
    for n in 2 3 4; do
        local base=0; [ "$n" -eq 4 ] && base=0,2
        for blk in $BLOCKS; do
            say ""
            say "=== $LABEL | ${n} seats | base $base | deck block $blk ==="
            nice -n 19 ionice -c2 -n7 python eval_headtohead.py "$CHAMP" "$OPP" \
                 --num-players "$n" --team1-seats "$base" --seat-swap \
                 --games $G --seed "$blk" 2>&1 \
              | grep -E "complementary pair|Seat-balanced|^    checkpoint|^    heuristic|Paired margin|\+/-|Verdict" \
              | tee -a $OUT
        done
    done
}

say "############ M1s3 vs R6 (the promotion claim: better at EVERY count) ############"
cell "$R6" "M1s3 vs R6"

say ""
say "############ M1s3 vs heuristic:h8 (the reference yardstick) ############"
cell "heuristic:h8" "M1s3 vs h8"

say ""
say "ORBIT EVAL COMPLETE $(date +%H:%M)"
