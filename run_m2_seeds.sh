#!/usr/bin/env bash
# M2 replication: two more seeds of the WIDE mixed-count net, to firm up the n=1 findings in
# RESEARCH_LOG 2026-07-18 -- specifically (a) does the 200k peak that reached the per-count
# ceilings replicate, and (b) does the wide net reliably BLOCK (M2 seed1: delta(k=1)=+2.54;
# M1 narrow: -0.36). Seeds 2 and 3, else identical to M2 (--wide, mixed 2,3,4, 300k, shaped,
# loss-penalty 60). Sequential: concurrent TF trainings are >10x slower each on this box.
# ~4h per seed, so ~8h total.
set -u
cd /home/orih/taki-ai
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate tensorflow_env

CKPT=/home/orih/taki-ai/models
E="nice -n 19 ionice -c2 -n7 python eval.py"
G=3000
OUT=m2_seeds_results.txt; : > $OUT
say() { echo "$@" | tee -a $OUT; }
score() { $E --model "$1" --opponent "$2" --num-players "$3" --games $G 2>&1 \
            | grep -oP '\d+/\d+ = \K[\d.]+' | head -1; }

say "M2 replication -- seeds 2,3 vs seed1 (M2: 0.569/0.375/0.295 @300k, peak@200k 0.587/0.383/0.307, blocks +2.54)"
say "narrow control M1: 0.570/0.372/0.290, does NOT block (-0.36)"

for SEED in 2 3; do
    say ""
    say "############### SEED $SEED ###############"
    echo "=== seed $SEED training start $(date +%H:%M) ==="
    nice -n 19 ionice -c2 -n7 python train.py --wide --num-players 2,3,4 --trials 300000 \
         --seed $SEED --reward shaped --loss-penalty 60 --snapshot-every 5000 \
         2>&1 | tee train_M2seed${SEED}.log
    rc=${PIPESTATUS[0]}
    echo "=== seed $SEED EXIT=$rc  $(date +%H:%M) ==="
    [ "$rc" -eq 0 ] || { say "seed $SEED ABORTED (exit $rc)"; exit 1; }

    DIR=$(grep -oP '(?<=^run_dir: )\S+' train_M2seed${SEED}.log | head -1)
    [ -d "$DIR/snap300000" ] || { say "seed $SEED produced no snap300000"; exit 1; }
    # Preserve the peak and the final, named by seed.
    cp -r "$DIR/snap200000" "$CKPT/checkpoint_M2seed${SEED}_widemixed_snap200000"
    cp -r "$DIR/snap300000" "$CKPT/checkpoint_M2seed${SEED}_widemixed_snap300000"

    # (a) progression vs h8 -- does the 200k peak replicate?
    say ""
    say "--- seed $SEED: progression vs h8 (peak should sit near 0.583/0.382/0.311 ceilings) ---"
    for snap in 100000 200000 300000; do
        [ -d "$DIR/snap$snap" ] || continue
        printf "  snap%-7s " "$snap" | tee -a $OUT
        for n in 2 3 4; do printf "%dp=%s  " "$n" "$(score "$DIR/snap$snap" heuristic:h8 $n)" | tee -a $OUT; done
        echo | tee -a $OUT
    done

    # (b) blocking probe on the peak (200k) and final (300k) -- does wide reliably block?
    for snap in 200000 300000; do
        say ""
        say "--- seed $SEED snap$snap: delta(k) blocking probe ---"
        python - "$DIR/snap$snap" 2>&1 <<'PYEOF' | tee -a $OUT
import sys
from r6_accept import q_sweep
d = q_sweep(sys.argv[1])
print(f"  delta(k=1..7) = {' '.join('%+.2f' % x for x in d)}")
print(f"  delta(k=1) = {d[0]:+.2f} -> {'BLOCKS' if d[0] > 0 else 'does NOT block'}   spread {max(d)-min(d):.2f}")
PYEOF
    done
done

say ""
say "ALL SEEDS COMPLETE $(date +%H:%M)"
