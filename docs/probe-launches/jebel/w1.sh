#!/bin/bash
# W1 (docs/window-branches-workplan.md, registered in #863 before its data): L3's cell (division p 13, --factor 32 38
# twice, reading the output, the fork, 600 epochs, seeds 0-5) with the window 32 wide (--window-width 32). From
# ~/code/wt-branches at 85a9e44 (its .venv links wt-p2's: crate 40a2761, main's pin). 6 at once; the probe holds the
# lock. Its default width repeats L3 by bits on this commit (checked: 4 epochs of seed 0). Resume: w1.sh --resume.
cd ~/code/indrajala-ml
CODE=~/code/wt-branches
OUT=data/window-recurrent/window-branches/w1
mkdir -p $OUT
echo "W1 starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch.log
PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 $CODE/.venv/bin/python $CODE/scripts/modular_family_probe.py run --p 13 \
  --arm family --factor 32 38 --factor 32 38 --reads output --fork --window-width 32 --budget-epochs 600 \
  --budget-minutes 900 --first-seed 0 --seeds 6 --workers 6 --out $OUT "$@" >> $OUT/w1.out 2>&1
rc=$?
echo "W1 ended $(date) exit $rc" >> $OUT/launch.log
