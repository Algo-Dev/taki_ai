#!/usr/bin/env bash
# Evals for the seat-count-one-hot curriculum. Run after run_nplayers.sh, which writes
# snapshots.env with the five checkpoint paths.
#
# THE ONE THING TO UNDERSTAND HERE: the new nets take 150 floats and R6 takes 147, so no
# DQN trained before this branch can ever meet one of these in eval_headtohead -- the load
# raises on the first act(). The frozen `h8` heuristic reads game.history, not the
# observation, so it is unaffected by the change and is the ONLY bridge back to the
# published numbers. That is what E1 below is for. Cite h8 numbers, never r3 ones.
set -u
cd /home/orih/taki-ai-nplayers
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate tensorflow_env
source ./snapshots.env

E="nice -n 19 ionice -c2 -n7 python eval.py"
H="nice -n 19 ionice -c2 -n7 python eval_headtohead.py"
G=3000   # >=3000: SE ~ +/-0.008, needed for the 2-3 pt gaps this lineage produces

# --- E1: is the one-hot itself a no-op at 4 seats? THE NULL, and run it first.
# At 4 seats the new block is a CONSTANT input ([0,0,1] every step), so N0 is R6's exact
# recipe plus three dead inputs. It should land on R6's 0.297 vs h8 (parity 0.250) up to
# seed noise. If it does NOT, the finding is about the observation change, not the
# curriculum, and the rest of this script is not interpretable yet.
echo "=== E1: N0 vs h8 @4p  (R6 scored 0.297; parity 0.250) ==="
$E --model "$N0" --opponent heuristic:h8 --num-players 4 --games $G 2>&1 | tee eval_E1_N0_h8_4p.log

# --- E2: the curriculum question. Both arms are 200k @ the same settings; the only
# difference is that ARM A spent its first 100k at 2 seats. Standard promotion test:
# alternating 2v2, seat-swapped, paired over common decks. Headline = the paired
# same-seat margin (parity 0.0).
echo "=== E2: ARM A vs ARM B, 2v2 seat-swap @4p  (paired margin, parity 0.0) ==="
$H "$ARM_A" "$ARM_B" --team1-seats 0,2 --seat-swap --games $G 2>&1 | tee eval_E2_armA_vs_armB.log

# 1v3 secondary check: same shape as Mode B and as the champion's headline number. If it
# disagrees with E2 that is real information -- the arm's edge depends on field composition.
echo "=== E2b: ARM A vs ARM B, 1v3 seat-swap @4p (secondary) ==="
$H "$ARM_A" "$ARM_B" --team1-seats 0 --seat-swap --games $G 2>&1 | tee eval_E2b_armA_vs_armB_1v3.log

# --- E3: each arm on the yardstick, so both are on the same scale as every published number.
for arm in ARM_A ARM_B; do
    echo "=== E3: $arm vs h8 @4p  (parity 0.250) ==="
    $E --model "${!arm}" --opponent heuristic:h8 --num-players 4 --games $G 2>&1 \
        | tee "eval_E3_${arm}_h8_4p.log"
done

# --- E4: the specialists at their own table sizes. No published baseline exists for either
# (the project has only ever trained at 4 seats), so these establish the numbers rather than
# beat one. Parity is 1/num_players: 0.500 at 2 seats, 0.333 at 3.
echo "=== E4: P2 vs h8 @2p  (parity 0.500) ==="
$E --model "$P2" --opponent heuristic:h8 --num-players 2 --games $G 2>&1 | tee eval_E4_P2_h8_2p.log
echo "=== E4: 3P vs h8 @3p  (parity 0.333) ==="
$E --model "$P3" --opponent heuristic:h8 --num-players 3 --games $G 2>&1 | tee eval_E4_3P_h8_3p.log

# --- E5: vs-random, smoke test ONLY -- it is retired as a ranking metric (R3) and h8 itself
# scores 0.906 there. A collapsed run shows up here; nothing else does.
echo "=== E5: smoke, vs random ==="
$E --model "$N0" --num-players 4 --games 1000 2>&1 | tail -5 | tee eval_E5_N0_random.log
$E --model "$P2" --num-players 2 --games 1000 2>&1 | tail -5 | tee eval_E5_P2_random.log

echo "ALL EVALS COMPLETE $(date +%H:%M)"
