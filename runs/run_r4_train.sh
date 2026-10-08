#!/usr/bin/env bash
# R4 full retrain: mixed-count (2,3,4) 300k trials from scratch, 3 seeds sequentially.
# Mirrors M1s3's config (narrow net, color+rank sym on by default, shaped + loss-penalty 60).
# Writes each run to its own ./models/run<ts> dir; progress mirrored to results/train_seed<N>.log.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tensorflow_env
cd /home/orih/taki-ai-r4-color-void
mkdir -p results
for SEED in 3 1 2; do
  echo "=== $(date '+%F %T')  starting seed $SEED ==="
  nice -n 19 ionice -c2 -n7 python train.py \
      --num-players 2,3,4 --trials 300000 --reward shaped --loss-penalty 60 \
      --seed "$SEED" --snapshot-every 10000 > "results/train_seed${SEED}.log" 2>&1
  echo "=== $(date '+%F %T')  finished seed $SEED ==="
done
echo "=== ALL SEEDS DONE $(date '+%F %T') ==="
