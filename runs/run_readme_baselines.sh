#!/usr/bin/env bash
# Mode A baselines for the current champion at every seat count, vs BOTH opponents -- the
# numbers published in README.md's Results table and RESEARCH_LOG 2026-10-08.
#
# Why this exists: M1s3 was promoted (2026-07-19) AFTER vs-random was retired as a ranking
# metric (R3, 2026-07-12), so the champion had no vs-random number at all. The pair of rows is
# worth having precisely because it shows the retirement was right -- at four seats the
# hand-written heuristic scores 0.906 and the network 0.912 (a gap inside noise), while against
# that same heuristic the network wins 0.307 at parity 0.250.
#
# These are Mode A (1-vs-N, homogeneous field, ONE composition) and are NOT the promotion
# standard -- use eval_headtohead.py's rotation orbit for that. The 3-seat cell is ~1 pt
# optimistic for the usual composition-balance reason (see CLAUDE.md).
#
# Output goes to results/ (gitignored). If a number matters, transcribe it into RESEARCH_LOG.
set -u

CK=${1:-/home/orih/taki-ai/models/checkpoint_M1s3_mixed_snap300000}
GAMES=${GAMES:-3000}
SEED=${SEED:-0}
cd "$(dirname "$0")/.." || exit 1
mkdir -p results
OUT=results/readme_baselines.txt
: > "$OUT"

run() {   # run <label> <args...>
  echo "### $1" | tee -a "$OUT"
  nice -n 19 ionice -c2 -n7 python eval.py --games "$GAMES" --seed "$SEED" "${@:2}" \
    2>/dev/null | grep -E 'vs 1/N' | tee -a "$OUT"
}

for n in 2 3 4; do
  for opp in random heuristic; do
    run "champion n=$n vs $opp" --model "$CK" --opponent "$opp" --num-players "$n"
  done
done

# The yardstick's own vs-random score, which is the control that makes the point above.
for n in 2 3 4; do
  run "heuristic n=$n vs random" --model heuristic --num-players "$n"
done

echo "### DONE" | tee -a "$OUT"
