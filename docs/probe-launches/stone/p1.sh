#!/bin/bash
# P1 (docs/code-consolidation-workplan.md, stage 2c, registered in #847): A with the certificate guard,
# --consolidate --guard both, word-problem-s4 and move-to-front-4, seeds 0-5, round 1's cell, 600 epochs;
# 24 runs, 6 at once, from wt-p1 (bab7690, crate 7df828d: A's code, its own venv). Starts when L2 ends.
# On stone, in the sandbox; the probe holds the lock. Before 07:00 Monday if unfinished: the same
# command with --resume at 18:00.
source ~/labour/env.sh
while pgrep -f "division-ladder/l2" > /dev/null; do sleep 60; done
cd $LABOUR/code/indrajala-ml
CODE=$LABOUR/code/wt-p1
OUT=$LABOUR/data/consolidation/p1
mkdir -p $OUT
echo "P1 starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch.log
PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 $CODE/.venv/bin/python $CODE/scripts/modular_family_probe.py run \
  --task word-problem-s4 move-to-front-4 --factors 2 --codes-ratio 32/11 --reads output --fork --consolidate \
  --guard both --budget-epochs 600 --budget-minutes 900 --first-seed 0 --seeds 6 --workers 6 \
  --out $OUT "$@" >> $OUT/p1.out 2>&1
rc=$?
echo "P1 ended $(date) exit $rc" >> $OUT/launch.log
