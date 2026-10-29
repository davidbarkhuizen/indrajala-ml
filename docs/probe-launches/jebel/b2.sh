#!/bin/bash
# B2 (docs/window-branches-workplan.md, registered in #863 before its data): round 2 of the general algebras'
# cells (word-problem-s5, move-to-front-5; two factors, --codes-ratio 32/11, reading the output, the fork, 600
# epochs, seeds 0-5) with window branches (--branches). From ~/code/wt-b2 at 393b3b1 (#868's merge; its own
# venv, crate afa7204). Without --branches this commit repeats round 2 by bits (8 epochs of seed 0, each task:
# data/window-recurrent/window-branches/b2-bits-check). 12 runs, 6 at once; the probe holds the lock.
# Resume: b2.sh --resume.
cd ~/code/indrajala-ml
CODE=~/code/wt-b2
OUT=data/window-recurrent/window-branches/b2
mkdir -p $OUT
echo "B2 starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch.log
PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 $CODE/.venv/bin/python $CODE/scripts/modular_family_probe.py run \
  --task word-problem-s5 move-to-front-5 --branches \
  --factors 2 --codes-ratio 32/11 --reads output --fork --budget-epochs 600 --budget-minutes 900 \
  --first-seed 0 --seeds 6 --workers 6 --out $OUT "$@" >> $OUT/b2.out 2>&1
rc=$?
echo "B2 ended $(date) exit $rc" >> $OUT/launch.log
