#!/usr/bin/env bash
# Evals for M2 (wide mixed-count, 300k). The capacity test: does 2.72x width let one net reach
# the per-count ceilings simultaneously, closing M1's uniform ~1-SE interference gaps?
#
# Waits on the COMPLETION SIGNAL (snapshots.env written + no train.py running), never a tmux
# session -- a session can outlive its script and wedged an eval 18h once.
set -u
cd /home/orih/taki-ai-mixwide
while [ ! -f ./snapshots.env ] || pgrep -f 'train\.py' >/dev/null; do sleep 60; done
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate tensorflow_env
source ./snapshots.env    # M2_DIR, M2, M2_CKPT

TWOP=/home/orih/taki-ai-2p
M1=/home/orih/taki-ai-nplayers/models/checkpoint_M1_mixed_snap300000
P1=$TWOP/models/run1784210486.600344/snap100000     # 2-seat specialist, 0.583
C=$TWOP/models/run1784225951.30801/snap100000       # best all-round (2p->3p): 0.553/0.382/0.311
R6=/home/orih/taki-ai/models/checkpoint_r6L60_snap100000
E="nice -n 19 ionice -c2 -n7 python eval.py"
G=3000

OUT=eval_results.txt; : > $OUT
say() { echo "$@" | tee -a $OUT; }
score() { $E --model "$1" --opponent "$2" --num-players "$3" --games $G 2>&1 \
            | grep -oP '\d+/\d+ = \K[\d.]+' | head -1; }

# E0: adapter check -- R6 vs h8 @4p must be 0.297, else nothing below is trustworthy.
say "=== E0: adapter check -- R6 vs h8 @4p must be 0.297 ==="
say "  $(score "$R6" heuristic:h8 4)"

# E1: THE HEADLINE. M2 vs h8 at each count, against M1 and the per-count ceilings.
say ""
say "=== E1: M2 (wide mixed) vs h8   [M1: 0.570/0.372/0.290 | ceilings: P1 0.583, C 0.382, C 0.311] ==="
for n in 2 3 4; do say "  ${n}p: $(score "$M2" heuristic:h8 $n)"; done

# E2: M2 vs M1 head-to-head -- the DIRECT capacity signal (same recipe, only width differs).
say ""
say "=== E2: M2 vs M1 head-to-head (1-vs-N; parity 1/n). >parity = width helped ==="
for n in 2 3 4; do
    say "  ${n}p (parity $(python3 -c "print(f'{1/$n:.3f}')")): $(score "$M2" "$M1" $n)"
done

# E3: M2 vs the per-count SPECIALISTS -- the 'one net = three specialists?' test.
say ""
say "=== E3: M2 vs the per-count specialist at its own count (parity 1/n) ==="
say "  2p vs P1: $(score "$M2" "$P1" 2)   (P1 is the 2-seat specialist)"
say "  3p vs C : $(score "$M2" "$C" 3)   (C is best at 3)"
say "  4p vs C : $(score "$M2" "$C" 4)   (C is best at 4)"

# E4: progression -- does the WIDE net keep improving PAST where M1 plateaued (100k)? A late
# gain is the clean capacity signal; a plateau at 100k like M1 means width bought no headroom.
say ""
say "=== E4: M2 progression vs h8 (M1 plateaued at 100k -- does width extend it?) ==="
for snap in 100000 200000 300000; do
    [ -d "$M2_DIR/snap$snap" ] || continue
    printf "  snap%-7s " "$snap" | tee -a $OUT
    for n in 2 3 4; do printf "%dp=%s  " "$n" "$(score "$M2_DIR/snap$snap" heuristic:h8 $n)" | tee -a $OUT; done
    echo | tee -a $OUT
done

# E5: blocking probe -- behavioural datum, not a strength claim (R6 -0.87, P1 +2.24, C -3.65,
# M1 -0.36; all strong nets except P1 fail). Does more capacity change it?
say ""
say "=== E5: delta(k) blocking probe on M2 (M1 was -0.36, does NOT block) ==="
python - "$M2" 2>&1 <<'PYEOF' | tee -a $OUT
import sys
from r6_accept import q_sweep
d = q_sweep(sys.argv[1])
print(f"  delta(k=1..7) = {' '.join('%+.2f' % x for x in d)}")
print(f"  delta(k=1) = {d[0]:+.2f} -> {'BLOCKS' if d[0] > 0 else 'does NOT block'}   spread {max(d)-min(d):.2f}")
PYEOF

# E6: vs-random smoke.
say ""
say "=== E6: smoke, M2 vs random ==="
for n in 2 3 4; do say "  ${n}p: $($E --model "$M2" --num-players $n --games 1000 2>&1 | grep -oP '\d+/\d+ = \K[\d.]+' | head -1)"; done

say ""
say "ALL EVALS COMPLETE $(date +%H:%M)"
