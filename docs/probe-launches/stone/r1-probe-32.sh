#!/bin/bash
# R1's probe at width 32 (docs/one-code-one-state-workplan.md, D3'; exploratory, it claims nothing): as
# r1-probe.sh (--answer-branch, seeds 0 and 1, division 13 in L3's cell, S5 and move-to-front-5 in round 2's) with
# --window-width 32, after the 16-wide probe ends. Read against W1 (division at 32: 5 of 6) and C0's same seeds
# (S5, move-to-front-5 at 32 without branches). Expectation, before its data: at 32 the answers' own branch does no
# worse than width alone on any backdrop (pairs per code at the best steps no higher than C0's same seeds), and if
# it certifies where C0's same seed doesn't, separation adds something width alone doesn't.
source ~/labour/env.sh
until grep -q "R1 probe ended" $LABOUR/data/one-code-one-state/r1-probe/launch.log 2>/dev/null; do sleep 60; done
cd $LABOUR/code/indrajala-ml
CODE=$LABOUR/code/wt-r1
OUT=$LABOUR/data/one-code-one-state/r1-probe-32
mkdir -p $OUT
PROBE="env PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 $CODE/../wt-branches/.venv/bin/python $CODE/scripts/modular_family_probe.py run"
echo "R1 probe at 32 starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch.log
$PROBE --p 13 --arm family --factor 32 38 --factor 32 38 --reads output --fork --answer-branch --window-width 32 \
  --budget-epochs 600 --budget-minutes 900 --first-seed 0 --seeds 2 --workers 2 --out $OUT/division >> $OUT/division.out 2>&1 &
AB_RUNS_ROOT=$LABOUR/ab-runs-r1 $PROBE --task word-problem-s5 move-to-front-5 --factors 2 --codes-ratio 32/11 --reads output \
  --fork --answer-branch --window-width 32 --budget-epochs 600 --budget-minutes 900 --first-seed 0 --seeds 2 --workers 4 \
  --out $OUT/algebras >> $OUT/algebras.out 2>&1 &
wait
echo "R1 probe at 32 ended $(date)" >> $OUT/launch.log
