#!/bin/bash
# L3 (docs/division-ladder-workplan.md, registered in #859 before its data): p 13 with each factor 32 wide
# (--factor 32 38 x2), else L1 round 2: the family, reading the output, the fork, 600 epochs, seeds 0-5,
# 6 at once, from wt-ladder (440a680). No --resume at this commit: must fit one window (about 1.5-2.5 h).
source ~/labour/env.sh
cd $LABOUR/code/indrajala-ml
CODE=$LABOUR/code/wt-ladder
OUT=$LABOUR/data/division-ladder/l3
mkdir -p $OUT
echo "L3 starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch.log
PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 .venv/bin/python $CODE/scripts/modular_family_probe.py run --p 13 \
  --arm family --factor 32 38 --factor 32 38 --reads output --fork --budget-epochs 600 --budget-minutes 300 \
  --first-seed 0 --seeds 6 --workers 6 --out $OUT >> $OUT/l3.out 2>&1
rc=$?
echo "L3 ended $(date) exit $rc" >> $OUT/launch.log
