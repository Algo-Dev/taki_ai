#!/usr/bin/env bash
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh; conda activate tensorflow_env
cd /home/orih/taki-ai-r4-color-void
CAND=/home/orih/taki-ai-r4-color-void/models/run1784651224.783525/snap300000
CHAMP=/home/orih/taki-ai/models/checkpoint_M1s3_mixed_snap300000
G=3000
run() {  # label, extra-args
  local label="$1"; shift
  for SEED in 0 777777; do
    echo "######## $label  (deck block seed=$SEED) ########"
    python eval_headtohead.py "$CAND" "$CHAMP" --games $G --seed $SEED "$@" 2>&1 \
      | grep -vE "tensorflow/|MLIR|CPU Freq|Sets are not|NOTE:|Instructions"
  done
}
echo "=== R4 candidate (seed2 snap300000) vs M1s3 champion ==="
run "2 seats (orbit)"        --num-players 2 --team1-seats 0 --seat-swap
run "3 seats (orbit)"        --num-players 3 --team1-seats 0 --seat-swap
run "4 seats BALANCED 2v2"   --num-players 4 --team1-seats 0,2 --seat-swap
run "4 seats SOLO 1v3"       --num-players 4 --team1-seats 0 --seat-swap
echo "=== PROMOTION EVAL DONE ==="
