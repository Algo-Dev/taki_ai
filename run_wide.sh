#!/usr/bin/env bash
# M2: the mixed-count CAPACITY test. Identical to M1 in every way except the network width
# (124->64  ->  256->128->64, A10's 2.72x). Mixed 2,3,4, 300k, seed 1, from scratch.
#
# PRE-REGISTERED PREDICTION (before seeing the result): M1 sits ~1 SE below the per-count best
# at every count (2p 0.570 vs P1 0.583; 3p 0.372 vs C 0.382; 4p 0.290 vs C 0.311) -- a uniform
# shortfall that reads as multi-task interference. If capacity is the binding constraint, M2
# CLOSES those gaps: one net reaching the per-count ceilings simultaneously. If the mixed
# setting is information-bound like A10 found four-seat single-count to be, M2 TIES M1 and the
# small deficits were noise. M2 is NOT expected to push any count PAST its per-count ceiling.
set -u
cd /home/orih/taki-ai-mixwide
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate tensorflow_env

echo "=== M2: wide mixed-count 2,3,4 -- 300k from scratch, starting $(date +%H:%M) ==="
nice -n 19 ionice -c2 -n7 python train.py --num-players 2,3,4 --trials 300000 --seed 1 \
     --reward shaped --loss-penalty 60 --snapshot-every 5000 2>&1 | tee train_M2_wide.log
echo "=== M2 EXIT=${PIPESTATUS[0]}  $(date +%H:%M) ==="

M2_DIR=$(grep -oP '(?<=^run_dir: )\S+' train_M2_wide.log | head -1)
M2=$M2_DIR/snap300000
[ -d "$M2" ] || { echo "ABORTING: M2 produced no $M2"; exit 1; }
cp -r "$M2" models/checkpoint_M2_widemixed_snap300000

cat > snapshots.env <<EOF
# Written by run_wide.sh
M2_DIR=/home/orih/taki-ai-mixwide/${M2_DIR#./}
M2=/home/orih/taki-ai-mixwide/${M2#./}
M2_CKPT=/home/orih/taki-ai-mixwide/models/checkpoint_M2_widemixed_snap300000
EOF
echo "M2 COMPLETE $(date +%H:%M)"
cat snapshots.env
