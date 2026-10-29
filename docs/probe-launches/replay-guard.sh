#!/bin/bash
# The certificate guard probe (2026-10-17): P1's 10 replays (E2b' seed 6's step-7 network, 2 epochs of
# step 8, generators snap-wall-p1-0..9, numpy, wall off), each update that makes a right machine wrong
# undone and its batch skipped. Without the guard 9 of 10 lost the proof (machine-diff/).
# Registered before the data: the guard works if 8 or more of 10 end certified in both verdicts.
cd ~/code/indrajala-ml
OUT=data/window-recurrent/replay-guard
NET=data/window-recurrent/snap-wall-p1/start/step-7.network.json
CODE=~/code/wt-guard
echo "started $(date) at $(git -C $CODE rev-parse --short HEAD) (+ uncommitted guard)" >> $OUT/guard.log
run_one() {
  i=$1
  PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 .venv/bin/python $CODE/scripts/window_recurrent_study.py replay "$NET" \
    --config $CODE/scripts/window_recurrent_e2b_prime.json --run curriculum-linear-k1-s16-m16-seed6 --step 8 \
    --epochs 2 --generator snap-wall-p1-$i --numpy --guard --json $OUT/guard-$i.json > $OUT/guard-$i.log 2>&1
  echo "done $i exit $?" >> $OUT/guard.log
}
export -f run_one; export OUT NET CODE
seq 0 9 | xargs -P 3 -I{} bash -c 'run_one {}'
echo "ended $(date)" >> $OUT/guard.log
