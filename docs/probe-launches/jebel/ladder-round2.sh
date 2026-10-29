#!/bin/bash
# The division ladder, round 2 (docs/division-ladder-workplan.md, L1 registered in #817): round 1 failed
# at p 17 and 23 (0 of 6 each), so D4 gives p 13; seeds 0-5, 6 at once. Code ~/code/wt-ladder (pinned
# 440a680, #818); the probe holds the machine lock.
cd ~/code/indrajala-ml
CODE=~/code/wt-ladder
OUT=data/window-recurrent/division-ladder/round2
mkdir -p $OUT
echo "round 2 starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch.log
PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 .venv/bin/python $CODE/scripts/modular_family_probe.py run --p 13 \
  --arm family --factors 2 --codes-ratio 32/11 --reads output --fork --budget-epochs 600 --budget-minutes 900 \
  --first-seed 0 --seeds 6 --workers 6 --out $OUT >> $OUT/round2.out 2>&1
rc=$?
echo "round 2 ended $(date) exit $rc" >> $OUT/launch.log
