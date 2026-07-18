#!/usr/bin/env bash
# M1 (NARROW) mixed-count replication, seeds 2 and 3 -- paired with the wide seeds 2,3.
#
# TWO PURPOSES, and the first is the reason this is worth 8h:
#   1. PROMOTION. We intend to promote a narrow mixed net as the champion (a single model for
#      2/3/4 seats). M1 is one seed, and only its snap300000 survived a worktree cleanup -- its
#      deleted earlier snapshots were BETTER at 3p/4p (snap100000: 0.381/0.306 vs 300k's
#      0.372/0.290). So this gives n=3 and, with a denser snapshot grid, lets us promote the
#      genuinely best checkpoint instead of assuming the last one.
#   2. SYMMETRY. The capacity comparison ran 3 wide seeds against 1 narrow. This squares it.
#
# Identical to M2's seeds except NO --wide (so the historical 124->64 trunk, M1's arch).
# ~4h per seed, sequential (concurrent TF trainings are >10x slower each here) => ~8h.
#
# Reference numbers (vs h8; parity 0.500 / 0.333 / 0.250):
#   M1 seed1 snap300000: 0.570 / 0.372 / 0.290   <- the current promotion candidate
#   M1 seed1 snap100000: 0.570 / 0.381 / 0.306   <- DELETED, but was better at 3p/4p
#   specialists:         P1 0.583 | C 0.382 | C 0.311
set -u
cd /home/orih/taki-ai
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate tensorflow_env

CKPT=/home/orih/taki-ai/models
E="nice -n 19 ionice -c2 -n7 python eval.py"
G=3000
OUT=m1_seeds_results.txt; : > $OUT
say() { echo "$@" | tee -a $OUT; }
score() { $E --model "$1" --opponent "$2" --num-players "$3" --games $G 2>&1 \
            | grep -oP '\d+/\d+ = \K[\d.]+' | head -1; }

say "M1 NARROW mixed replication -- seeds 2,3 (promotion candidates)"
say "reference: M1 seed1 @300k = 0.570/0.372/0.290 ; specialists P1 0.583 | C 0.382 | C 0.311"

for SEED in 2 3; do
    say ""
    say "############### NARROW SEED $SEED ###############"
    echo "=== narrow seed $SEED training start $(date +%H:%M) ==="
    nice -n 19 ionice -c2 -n7 python train.py --num-players 2,3,4 --trials 300000 \
         --seed $SEED --reward shaped --loss-penalty 60 --snapshot-every 5000 \
         2>&1 | tee train_M1seed${SEED}.log
    rc=${PIPESTATUS[0]}
    echo "=== narrow seed $SEED EXIT=$rc  $(date +%H:%M) ==="
    [ "$rc" -eq 0 ] || { say "narrow seed $SEED ABORTED (exit $rc)"; exit 1; }

    DIR=$(grep -oP '(?<=^run_dir: )\S+' train_M1seed${SEED}.log | head -1)
    [ -d "$DIR/snap300000" ] || { say "narrow seed $SEED produced no snap300000"; exit 1; }
    say "  run_dir: $DIR   (ALL snapshots kept here -- do not delete, promotion picks from them)"

    # Dense progression: M1's best cells were at 100k, so do not assume the last snapshot wins.
    say ""
    say "--- narrow seed $SEED: progression vs h8 (2p / 3p / 4p) ---"
    for snap in 50000 100000 150000 200000 250000 300000; do
        [ -d "$DIR/snap$snap" ] || continue
        printf "  snap%-7s " "$snap" | tee -a $OUT
        for n in 2 3 4; do printf "%dp=%s  " "$n" "$(score "$DIR/snap$snap" heuristic:h8 $n)" | tee -a $OUT; done
        echo | tee -a $OUT
    done

    # Blocking probe at the same two snapshots the wide seeds used, for the symmetric record.
    for snap in 200000 300000; do
        say ""
        say "--- narrow seed $SEED snap$snap: delta(k) blocking probe ---"
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
say "ALL NARROW SEEDS COMPLETE $(date +%H:%M)"
say "Next: pick the best snapshot across seeds 1-3 and promote it (update CLAUDE.md/README)."
