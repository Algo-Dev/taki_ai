#!/usr/bin/env bash
# The 2p/3p/4p curriculum, re-run under the 150-float observation (seat-count one-hot).
#
# WHY EVERY STAGE IS FROM SCRATCH: the one-hot moved the observation 147 -> 150, so R6's net
# (input dim 147) cannot be warm-started from -- it raises on the first act(). The old chain
# anchored every stage on R6; this one has to grow its own base. That is a cost, but it also
# buys the fix below.
#
# WHAT THIS FIXES vs run_4p_arms.sh: that script's comment claimed "both arms end at 200k
# trials, so the only difference is what the first 100k were spent on". It was not true --
# its 2p stage ALSO warm-started from R6 (models/run1784210486.600344/config.txt), so ARM A
# ran 100k(R6 @4p) + 100k @2p + 100k @4p = 300k against ARM B's 200k. The curriculum arm had
# 50% more training than its own control, which is exactly the confound the control existed
# to remove. Here both arms are honestly 200k and differ only in what the first 100k bought.
#
# Stages (SEQUENTIAL: concurrent TF trainings are >10x slower each on this box):
#   N0    4p 100k from scratch          -- R6's exact recipe under the new observation.
#                                          Doubles as ARM B's first half.
#   P2    2p 100k from scratch          -- the 2p specialist.
#   ARM A P2 -> 4p 100k    (200k total) -- curriculum: 100k @2 seats, then 100k @4.
#   ARM B N0 -> 4p 100k    (200k total) -- control:    100k @4 seats, then 100k @4.
#   3P    P2 -> 3p 100k    (200k total) -- curriculum stage at 3 seats.
set -u
cd /home/orih/taki-ai-nplayers

# The old chain still owns the CPU; wait it out rather than halving both.
while tmux has-session -t taki4p 2>/dev/null || tmux has-session -t taki3p 2>/dev/null; do
    sleep 60
done

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate tensorflow_env

RUN="nice -n 19 ionice -c2 -n7 python train.py"
# Every published number in this lineage was trained at these settings; only --num-players,
# --model and --epsilon-start vary below.
COMMON="--trials 100000 --seed 1 --reward shaped --loss-penalty 60 --snapshot-every 2500"

# Each stage sets STAGE_SNAP to the snapshot it produced. train.py mints its own
# models/run<timestamp>/ and prints it, so the consuming stage cannot hardcode the path --
# we read it back out of the stage's own log rather than guessing by mtime, which would
# silently hand back the PREVIOUS stage's snapshot if this one produced none.
STAGE_SNAP=
stage() {  # stage <name> <logfile> <args...>
    local name=$1 log=$2; shift 2
    echo "=== $name starting $(date +%H:%M) ==="
    $RUN "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    echo "=== $name EXIT=$rc  $(date +%H:%M) ==="
    [ "$rc" -eq 0 ] || { echo "ABORTING: $name failed (exit $rc)"; exit 1; }

    local run_dir
    run_dir=$(grep -oP '(?<=^run_dir: )\S+' "$log" | head -1)
    STAGE_SNAP="$run_dir/snap100000"
    [ -n "$run_dir" ] && [ -d "$STAGE_SNAP" ] || {
        echo "ABORTING: $name exited 0 but produced no $STAGE_SNAP"; exit 1; }
    echo "$name -> $STAGE_SNAP"
}

# --- N0: the base. From scratch, so epsilon starts at the 1.0 default (real exploration).
stage "N0  4p 100k from scratch" train_N0_4p.log $COMMON --num-players 4
N0=$STAGE_SNAP

# --- P2: the 2p specialist. Also from scratch -- warm-starting it from N0 would put the
# curriculum arm ahead of its control again, which is the bug this script exists to fix.
stage "P2  2p 100k from scratch" train_P2_2p.log $COMMON --num-players 2
P2=$STAGE_SNAP

# --- The two arms. Both continue a 100k net near-greedily, so both use --epsilon-start 0.1.
stage "ARM A  2p -> 4p curriculum" train_armA_2p_to_4p.log \
      $COMMON --num-players 4 --epsilon-start 0.1 --model "$P2"
ARM_A=$STAGE_SNAP

stage "ARM B  4p -> 4p control" train_armB_4p_cont.log \
      $COMMON --num-players 4 --epsilon-start 0.1 --model "$N0"
ARM_B=$STAGE_SNAP

# --- 3P: the curriculum stage at 3 seats.
stage "3P  2p -> 3p curriculum" train_3p.log \
      $COMMON --num-players 3 --epsilon-start 0.1 --model "$P2"
P3=$STAGE_SNAP

cat <<EOF | tee snapshots.env
# Produced by run_nplayers.sh -- source this before run_nplayers_eval.sh
N0=$N0
P2=$P2
ARM_A=$ARM_A
ARM_B=$ARM_B
P3=$P3
EOF
echo "ALL STAGES COMPLETE $(date +%H:%M)"
