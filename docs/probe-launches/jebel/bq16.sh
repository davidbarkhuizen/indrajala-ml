#!/bin/bash
# Remedy B on random-q16-0 (docs/code-consolidation-workplan.md, stage 2b, #838): the control (2b's
# cell) then B (--long-tail 4), seeds 0-5 each, 600 epochs, 6 at once on jebel. Code ~/code/wt-bq16
# (pinned 0b4a6b9, its own venv with crate 7df828d); each probe holds the machine lock.
cd ~/code/indrajala-ml
CODE=~/code/wt-bq16
OUT=data/window-recurrent/consolidation/b-q16
mkdir -p $OUT/control $OUT/b
echo "B on q16 starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch.log
for arm in control b; do
  extra=""; [ $arm = b ] && extra="--long-tail 4"
  PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 $CODE/.venv/bin/python $CODE/scripts/modular_family_probe.py run \
    --task random-q16-0 --factors 2 --codes-ratio 32/11 --reads output --fork $extra \
    --budget-epochs 600 --budget-minutes 900 --first-seed 0 --seeds 6 --workers 6 \
    --out $OUT/$arm >> $OUT/$arm.out 2>&1
  echo "$arm ended $(date) exit $?" >> $OUT/launch.log
done
echo "B on q16 ended $(date)" >> $OUT/launch.log
