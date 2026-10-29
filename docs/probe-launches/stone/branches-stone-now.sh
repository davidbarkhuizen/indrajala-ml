#!/bin/bash
# The window branches workplan's B1, then B3's division p 11 cell (docs/window-branches-workplan.md, registered in #863
# before their data), on stone, started early at the owner's say (2026-10-19, about 10:00): from ~/labour/code/wt-branches at 6d808a5 (stage 5; crate
# afa7204, built offline). Checked there: the branch tests pass, and without --branches L3's cell repeats L3 by bits.
# B1: L3's cell (p 13, --factor 32 38 twice, reading the output, the fork, 600 epochs, seeds 0-5) with --branches.
# B3 (p 11): the p11 ladder's cell (two factors, --codes-ratio 32/11) with --branches. 6 at once each; the probe holds
# the lock. If 07:00 comes first: the owner stops it; then this script with RESUME=--resume.
source ~/labour/env.sh
cd $LABOUR/code/indrajala-ml
CODE=$LABOUR/code/wt-branches
OUT=$LABOUR/data/window-branches
mkdir -p $OUT/b1 $OUT/b3
run () { PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 $CODE/.venv/bin/python $CODE/scripts/modular_family_probe.py run "$@" $RESUME; }
echo "B1 starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/b1/launch.log
run --p 13 --arm family --factor 32 38 --factor 32 38 --reads output --fork --branches --budget-epochs 600 \
  --budget-minutes 900 --first-seed 0 --seeds 6 --workers 6 --out $OUT/b1 >> $OUT/b1/b1.out 2>&1
echo "B1 ended $(date) exit $?" >> $OUT/b1/launch.log
echo "B3 p11 starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/b3/launch.log
run --p 11 --arm family --factors 2 --codes-ratio 32/11 --reads output --fork --branches --budget-epochs 600 \
  --budget-minutes 900 --first-seed 0 --seeds 6 --workers 6 --out $OUT/b3/p11 >> $OUT/b3/b3-p11.out 2>&1
echo "B3 p11 ended $(date) exit $?" >> $OUT/b3/launch.log
