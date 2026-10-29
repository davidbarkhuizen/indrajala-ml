#!/bin/bash
# The division ladder, round 1 (docs/division-ladder-workplan.md, L1 registered in #817): p 17 and 23,
# seeds 0-5, 8 at once. Code ~/code/wt-ladder (pinned 440a680, #818); the probe holds the machine lock.
cd ~/code/indrajala-ml
CODE=~/code/wt-ladder
OUT=data/window-recurrent/division-ladder/round1
mkdir -p $OUT
echo "round 1 starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch.log
PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 .venv/bin/python $CODE/scripts/modular_family_probe.py run --p 17 23 \
  --arm family --factors 2 --codes-ratio 32/11 --reads output --fork --budget-epochs 600 --budget-minutes 900 \
  --first-seed 0 --seeds 6 --workers 8 --out $OUT >> $OUT/round1.out 2>&1
echo "round 1 ended $(date) exit $?" >> $OUT/launch.log
