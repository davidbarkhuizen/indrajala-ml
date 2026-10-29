#!/bin/bash
# The division ladder's L2 (docs/division-ladder-workplan.md, registered in #824): p 13 at 1,800 epochs a run,
# everything else as round 2 (seeds 0-5, commit 440a680). On stone, in the sandbox; the probe holds the lock.
source ~/labour/env.sh
cd $LABOUR/code/indrajala-ml
CODE=$LABOUR/code/wt-ladder
OUT=$LABOUR/data/division-ladder/l2
mkdir -p $OUT
echo "L2 starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch.log
PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 .venv/bin/python $CODE/scripts/modular_family_probe.py run --p 13 \
  --arm family --factors 2 --codes-ratio 32/11 --reads output --fork --budget-epochs 1800 --budget-minutes 1200 \
  --first-seed 0 --seeds 6 --workers 6 --out $OUT >> $OUT/l2.out 2>&1
rc=$?
echo "L2 ended $(date) exit $rc" >> $OUT/launch.log
