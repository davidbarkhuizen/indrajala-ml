#!/bin/bash
# R1's probe (docs/one-code-one-state-workplan.md, stage 3; a probe, it claims nothing): the answers' own branch
# with every factor on the shared state branch (--answer-branch), seeds 0 and 1, on the three backdrops:
# division p 13 (L3's cell: --factor 32 38 twice, reading the output, the fork, 600 epochs) and round 2's S5 and
# move-to-front-5 cell (two factors, --codes-ratio 32/11). 6 runs at once on stone, after B3 p 11 ends.
# Expectations, written before its data:
# - division: the state stays exact (pairs per code at 1.0-1.5 at the best steps, as L3's 1.0) and the answers do
#   better than L3's (worst answer at best above L3's 0.38-0.99 in its seeds 0, 1), certifying in either seed -> C1
#   goes ahead as registered; pairs per code well above 1.5 -> the answers' branch disturbs the state too.
# - S5 and move-to-front-5: descriptive (pairs per code against B2's same seeds: S5 120 and 66; move-to-front-5
#   failure A in both).
# The algebras' half holds its own lock directory (AB_RUNS_ROOT=$LABOUR/ab-runs-r1), beside the division half's.
# Control checks first, each control's log against this commit without the flag (every row but the budget stop,
# seconds aside); a failed check stops the probe before its runs.
source ~/labour/env.sh
L=~/labour/data/window-branches/b3/launch.log
until grep -q "B3 p11 ended" $L 2>/dev/null; do sleep 60; done
cd $LABOUR/code/indrajala-ml
CODE=$LABOUR/code/wt-r1
OUT=$LABOUR/data/one-code-one-state/r1-probe
C=$LABOUR/data/one-code-one-state/controls
mkdir -p $OUT
PROBE="env PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 $CODE/../wt-branches/.venv/bin/python $CODE/scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
same() {
  local n=$(( $(wc -l < "$1") - 1 ))
  [ "$n" -gt 2 ] && diff <(head -n $n "$1" | sed -E 's/, "seconds": [0-9.]+//; s/, "machine": \{[^}]*\}//') \
    <(head -n $n "$2" | sed -E 's/, "seconds": [0-9.]+//') > "$1.diff"
}
log "R1 probe: control checks start at $(git -C $CODE rev-parse --short HEAD)"
K=$OUT/bits-check; rm -rf $K; mkdir -p $K/division $K/algebras
$PROBE --p 13 --arm family --factor 32 38 --factor 32 38 --reads output --fork --budget-epochs 4 --budget-minutes 20 \
  --first-seed 0 --seeds 1 --workers 1 --out $K/division > $K/division.out 2>&1
$PROBE --runs word-problem-s5:0 move-to-front-5:0 --factors 2 --codes-ratio 32/11 --reads output --fork \
  --budget-epochs 8 --budget-minutes 20 --workers 2 --out $K/algebras > $K/algebras.out 2>&1
for pair in "division/seed0 $LABOUR/data/division-ladder/l3/seed0/log.jsonl" \
            "algebras/word-problem-s5/seed0 $C/round2-word-problem-s5-seed0.log.jsonl" \
            "algebras/move-to-front-5/seed0 $C/round2-move-to-front-5-seed0.log.jsonl"; do
  set -- $pair
  if same $K/$1/log.jsonl $2; then log "control check: $1 repeated by bits"
  else log "control check FAILED on $1: see $K/$1/log.jsonl.diff; probe stopped"; exit 1; fi
done
log "R1 probe starts"
$PROBE --p 13 --arm family --factor 32 38 --factor 32 38 --reads output --fork --answer-branch --budget-epochs 600 \
  --budget-minutes 900 --first-seed 0 --seeds 2 --workers 2 --out $OUT/division >> $OUT/division.out 2>&1 &
AB_RUNS_ROOT=$LABOUR/ab-runs-r1 $PROBE --task word-problem-s5 move-to-front-5 --factors 2 --codes-ratio 32/11 --reads output --fork --answer-branch \
  --budget-epochs 600 --budget-minutes 900 --first-seed 0 --seeds 2 --workers 4 --out $OUT/algebras >> $OUT/algebras.out 2>&1 &
wait
log "R1 probe ended"
