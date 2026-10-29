#!/bin/bash
# The approximation claim (docs/approximation-claim-workplan.md), pyramidon's share: seeds 214-219, every arm, 3 workers.
cd ~/code/indrajala-ml
OUT=data/window-recurrent/approximation-claim
CODE=~/code/wt-approx
mkdir -p $OUT
echo "pyramidon started $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch-pyramidon.txt
for arm in quotient sum family product; do
  PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 .venv/bin/python $CODE/scripts/modular_family_probe.py run --p 5 --arm $arm \
    --factors 2 --reads output --vocabulary sum difference product --budget-epochs 400 --budget-minutes 180 \
    --first-seed 214 --seeds 6 --workers 3 --out $OUT/$arm >> $OUT/$arm-pyramidon.out 2>&1
  echo "$arm done $(date) exit $?" >> $OUT/launch-pyramidon.txt
done
echo "pyramidon ended $(date)" >> $OUT/launch-pyramidon.txt
