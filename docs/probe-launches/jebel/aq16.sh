#!/bin/bash
# Remedy A on random-q16-0 (docs/code-consolidation-workplan.md, stage 2b, registered in a-on-q16
# 4335177): --consolidate, seeds 0-5, 600 epochs, 6 at once on jebel, from the same worktree and
# commit as B and its control (~/code/wt-bq16, 0b4a6b9, its own venv); the probe holds the lock.
cd ~/code/indrajala-ml
CODE=~/code/wt-bq16
OUT=data/window-recurrent/consolidation/b-q16
mkdir -p $OUT/a
echo "A on q16 starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch.log
PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 $CODE/.venv/bin/python $CODE/scripts/modular_family_probe.py run \
  --task random-q16-0 --factors 2 --codes-ratio 32/11 --reads output --fork --consolidate \
  --budget-epochs 600 --budget-minutes 900 --first-seed 0 --seeds 6 --workers 6 \
  --out $OUT/a >> $OUT/a.out 2>&1
echo "A on q16 ended $(date) exit $?" >> $OUT/launch.log
