#!/bin/bash
# P2 on random-q16-0 (docs/code-consolidation-workplan.md, stage 2c, registered in #847; built in
# #849): --consolidate --restore-best, seeds 0-5, 600 epochs, 6 at once on jebel, from ~/code/wt-p2
# (b5f9a05, its own venv); its control is 2b's A arm (A's first 15 epochs repeat by bits on b5f9a05).
cd ~/code/indrajala-ml
CODE=~/code/wt-p2
OUT=data/window-recurrent/consolidation/b-q16
mkdir -p $OUT/p2
echo "P2 on q16 starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch.log
PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 $CODE/.venv/bin/python $CODE/scripts/modular_family_probe.py run \
  --task random-q16-0 --factors 2 --codes-ratio 32/11 --reads output --fork --consolidate --restore-best \
  --budget-epochs 600 --budget-minutes 900 --first-seed 0 --seeds 6 --workers 6 \
  --out $OUT/p2 >> $OUT/p2.out 2>&1
echo "P2 on q16 ended $(date) exit $?" >> $OUT/launch.log
