#!/usr/bin/env bash
# Evals for M1 (mixed-count, 300k, from scratch) and C4 (C -> 4p). Waits for the training
# session, so it can be launched immediately and left alone.
#
# Everything runs from THIS worktree even though C/C4/R6 are 147-float nets: eval loads an
# older-contract checkpoint on the observation prefix it was trained on, which is exact (the
# one-hot was appended). Verified by R6 reproducing its published 0.297. That is what makes a
# cross-contract comparison -- M1 (150) against C (147) -- possible at all.
#
# THE BAR, from RESEARCH_LOG P2 (Mode A vs h8, 3000 games/cell):
#   net    | total | 2 seats (0.500) | 3 seats (0.333) | 4 seats (0.250)
#   R6     | 100k  | 0.413           | 0.342           | 0.297
#   P1     | 200k  | 0.583           | 0.364           | 0.284
#   Arm A  | 300k  | 0.422           | 0.356           | 0.308
#   Arm B  | 200k  | 0.365           | 0.338           | 0.302
#   C      | 300k  | 0.553           | 0.382           | 0.311   <- best all-round
set -u
cd /home/orih/taki-ai-nplayers

# Wait on the training-complete SIGNAL, not on the tmux session: a session can outlive the
# script that ran in it (a lingering pane keeps it alive), and watching it wedged this eval for
# 18h once. run_mixed.sh writes snapshots.env only after every stage succeeded, so its presence
# is exactly "training done" -- and no train.py still running guards a half-written file.
while [ ! -f ./snapshots.env ] || pgrep -f 'train\.py' >/dev/null; do sleep 60; done

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate tensorflow_env
source ./snapshots.env    # M1_DIR, M1, C4 -- written by run_mixed.sh

TWOP=/home/orih/taki-ai-2p
C=$TWOP/models/run1784225951.30801/snap100000
R6=/home/orih/taki-ai/models/checkpoint_r6L60_snap100000
E="nice -n 19 ionice -c2 -n7 python eval.py"
G=3000    # SE ~ +/-0.008; ~15-30s per cell, so there is no reason to skimp

OUT=eval_results.txt
: > $OUT
say() { echo "$@" | tee -a $OUT; }

# --- E0: the adapter still reproduces R6's published number. Everything below is only
# meaningful if this prints 0.297.
say "=== E0: adapter check -- R6 vs h8 @4p must be 0.297 ==="
$E --model "$R6" --opponent heuristic:h8 --num-players 4 --games $G 2>&1 | grep -E "vs heuristic" | tee -a $OUT

# --- E1: the headline. M1 across every count, against the frozen yardstick.
say ""
say "=== E1: M1 (mixed 300k, from scratch) vs h8 -- compare to C's 0.553 / 0.382 / 0.311 ==="
for n in 2 3 4; do
    printf "  %dp: " $n | tee -a $OUT
    $E --model "$M1" --opponent heuristic:h8 --num-players $n --games $G 2>&1 \
        | grep -oP '\d+/\d+ = [\d.]+.*' | head -1 | tee -a $OUT
done

# --- E2: M1 against C directly, at every count. Same 300k total and ~the same per-count
# budget; the only difference is interleaved vs sequential. Parity is 1/n.
say ""
say "=== E2: M1 vs C head-to-head (1-vs-N) -- interleaved vs sequential, matched budget ==="
for n in 2 3 4; do
    printf "  %dp (parity %s): " $n "$(python3 -c "print(f'{1/$n:.3f}')")" | tee -a $OUT
    $E --model "$M1" --opponent "$C" --num-players $n --games $G 2>&1 \
        | grep -oP '\d+/\d+ = [\d.]+.*' | head -1 | tee -a $OUT
done

# --- E3: C -> 4p. P2's open item: does the 2p->3p->4p curriculum survive its final 4-seat
# stage? C was 0.553 / 0.382 / 0.311. Arm A (P1->4p: 0.583 -> 0.422 at two seats) predicts the
# 2-seat skill is destroyed.
say ""
say "=== E3: C4 (= C -> 4p, 400k total) vs h8 -- the forgetting test. C was 0.553/0.382/0.311 ==="
for n in 2 3 4; do
    printf "  %dp: " $n | tee -a $OUT
    $E --model "$C4" --opponent heuristic:h8 --num-players $n --games $G 2>&1 \
        | grep -oP '\d+/\d+ = [\d.]+.*' | head -1 | tee -a $OUT
done

# --- E4: M1's trajectory. Does mixed training still improve at 300k, and does it hold every
# count at once as it goes, or trade them off against each other?
say ""
say "=== E4: M1 progression vs h8 (is 300k enough? are the counts traded off?) ==="
for snap in 100000 200000 300000; do
    [ -d "$M1_DIR/snap$snap" ] || continue
    printf "  snap%-7s " "$snap" | tee -a $OUT
    for n in 2 3 4; do
        # Match the SCORE line ("1234/3000 = 0.428"), not eval.py's earlier
        # "Baseline (1/num_players) = 0.500" -- a bare (?<== ) grep grabs the baseline first.
        r=$($E --model "$M1_DIR/snap$snap" --opponent heuristic:h8 --num-players $n \
              --games $G 2>&1 | grep -oP '\d+/\d+ = \K[\d.]+' | head -1)
        printf "%dp=%s  " "$n" "$r" | tee -a $OUT
    done
    echo | tee -a $OUT
done

# --- E5: the blocking probe on M1. P2 showed blocking is doubly dissociated from strength, so
# this is a behavioural datum, NOT a strength prediction. R6 = -0.87 (fail), P1 = +2.24 (pass),
# C = -3.65 (fail) -- and C is the strongest. Scenario is a 4-player position.
say ""
say "=== E5: delta(k) blocking probe on M1 (R6 -0.87 FAIL, P1 +2.24 PASS, C -3.65 FAIL) ==="
python - "$M1" 2>&1 <<'EOF' | tee -a $OUT
import sys
from r6_accept import q_sweep
d = q_sweep(sys.argv[1])          # one delta per k = 1..7
print(f"  delta(k=1..7) = {' '.join('%+.2f' % x for x in d)}")
print(f"  delta(k=1) = {d[0]:+.2f} -> {'BLOCKS' if d[0] > 0 else 'does NOT block'}")
print(f"  spread over k = {max(d) - min(d):.2f}   (R6 1.95, P1 5.88)")
EOF

# --- E6: vs-random smoke ONLY. Retired as a ranking metric (R3); h8 itself scores 0.906.
say ""
say "=== E6: smoke, M1 vs random (a collapsed run shows up here; nothing else does) ==="
for n in 2 3 4; do
    printf "  %dp: " $n | tee -a $OUT
    $E --model "$M1" --num-players $n --games 1000 2>&1 \
        | grep -oP '\d+/\d+ = [\d.]+.*' | head -1 | tee -a $OUT
done

say ""
say "ALL EVALS COMPLETE $(date +%H:%M)"
