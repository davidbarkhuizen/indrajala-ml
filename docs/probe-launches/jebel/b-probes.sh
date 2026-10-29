#!/bin/bash
# B's origins, probes (docs/code-consolidation-workplan.md, stage 2b, #832): random-perm-4-0 (every
# token a bijection) and random-q8-1, random-q12-0, random-q16-0 (random-4's tokens and classes at
# 8, 12, 16 states), seeds 0-2, round 1's cell, 6 at once on jebel. Code ~/code/wt-bprobes (pinned
# 1496113); the probe holds the machine lock.
cd ~/code/indrajala-ml
CODE=~/code/wt-bprobes
OUT=data/window-recurrent/consolidation/b-probes
mkdir -p $OUT
echo "B probes start $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch.log
PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 .venv/bin/python $CODE/scripts/modular_family_probe.py run \
  --task random-perm-4-0 random-q16-0 random-q12-0 random-q8-1 \
  --factors 2 --codes-ratio 32/11 --reads output --fork --budget-epochs 600 --budget-minutes 900 \
  --first-seed 0 --seeds 3 --workers 6 --out $OUT >> $OUT/b-probes.out 2>&1
echo "B probes ended $(date) exit $?" >> $OUT/launch.log
