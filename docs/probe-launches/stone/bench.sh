#!/bin/bash
# Throughput on stone: the ladder's p 13 cell (jebel's round 2, same seeds, bit-identical work), 30 epochs a
# seed, at 4, 8 and 12 workers in turn.
source ~/labour/env.sh
cd $LABOUR/code/indrajala-ml
for w in 4 8 12; do
  OUT=$LABOUR/bench/p13-w$w
  mkdir -p $OUT
  echo "w$w starts $(date)" >> $LABOUR/bench/bench.log
  OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/modular_family_probe.py run --p 13 --arm family --factors 2 \
    --codes-ratio 32/11 --reads output --fork --budget-epochs 30 --budget-minutes 30 --first-seed 0 --seeds $w \
    --workers $w --out $OUT > $OUT/run.out 2>&1
  rc=$?
  echo "w$w ended $(date) exit $rc" >> $LABOUR/bench/bench.log
done
