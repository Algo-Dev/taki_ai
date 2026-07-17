#!/usr/bin/env bash
# M1: mixed-count training, and then P2's open item (C -> 4p). Sequential -- concurrent TF
# trainings are >10x slower each on this box.
#
# THE EXPERIMENT. C (RESEARCH_LOG P2, the best all-round net) reached 0.553/0.382/0.311 across
# 2/3/4 seats by spending its 300k trials SEQUENTIALLY: 100k @4 (as R6), then 100k @2 (as P1),
# then 100k @3. M1 spends the same 300k, and -- since counts are sampled uniformly -- roughly
# the same ~100k per count, but INTERLEAVED. Same total budget, same per-count budget; the only
# variable is interleaved vs sequential. That isolates the one thing P2 could not test: whether
# forgetting is a consequence of *sequencing* rather than of the counts themselves.
#
# M1 must train from scratch (the one-hot moved the observation to 150, so no 147-float net can
# be warm-started), which also means it carries no R6 inheritance. If it matches C it did so
# with strictly less: no curriculum, no champion to start from.
set -u

NP=/home/orih/taki-ai-nplayers
TWOP=/home/orih/taki-ai-2p
# From the P2 entry (config.txt verified): C = P1->3p, the best all-round net on record.
C=$TWOP/models/run1784225951.30801/snap100000

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate tensorflow_env

# --- M1: 300k mixed-count, from scratch. Epsilon starts at the 1.0 default (real exploration).
cd "$NP"
echo "=== M1: mixed-count 2,3,4 -- 300k from scratch, starting $(date +%H:%M) ==="
nice -n 19 ionice -c2 -n7 python train.py --num-players 2,3,4 --trials 300000 --seed 1 \
     --reward shaped --loss-penalty 60 --snapshot-every 5000 2>&1 | tee train_M1_mixed.log
echo "=== M1 EXIT=${PIPESTATUS[0]}  $(date +%H:%M) ==="
M1_DIR=$(grep -oP '(?<=^run_dir: )\S+' train_M1_mixed.log | head -1)
M1=$M1_DIR/snap300000
[ -d "$M1" ] || { echo "ABORTING: M1 produced no $M1"; exit 1; }
echo "M1 = $M1"

# --- C -> 4p: P2's first open item. Does the 2p->3p->4p curriculum survive a final 4-seat
# stage? Arm A (P1->4p, which destroyed the 2-seat skill) predicts no. Runs in the 2p worktree
# on the 147-float contract, because C is a 147 net and cannot be warm-started under 150.
cd "$TWOP"
echo "=== C->4p: 100k, starting $(date +%H:%M) ==="
nice -n 19 ionice -c2 -n7 python train.py --num-players 4 --trials 100000 --seed 1 \
     --reward shaped --loss-penalty 60 --epsilon-start 0.1 --snapshot-every 2500 \
     --model "$C" 2>&1 | tee train_C_to_4p.log
echo "=== C->4p EXIT=${PIPESTATUS[0]}  $(date +%H:%M) ==="
C4_DIR=$(grep -oP '(?<=^run_dir: )\S+' train_C_to_4p.log | head -1)
C4=$TWOP/${C4_DIR#./}/snap100000
[ -d "$C4" ] || { echo "ABORTING: C->4p produced no $C4"; exit 1; }
echo "C4 = $C4"

cat > "$NP/snapshots.env" <<EOF
# Written by run_mixed.sh
M1_DIR=$NP/${M1_DIR#./}
M1=$NP/${M1#./}
C4=$C4
EOF
echo "ALL TRAINING COMPLETE $(date +%H:%M)"
cat "$NP/snapshots.env"
