#!/bin/bash
# P1 of docs/snap-wall-workplan.md: 10 replays x (wall off, wall on), 2 at a time, from E2b' seed 6's step 7.
# Code from ~/code/wt-p1 (pinned), data in the main checkout. Log: data/window-recurrent/snap-wall-p1/p1.log
cd ~/code/indrajala-ml
OUT=data/window-recurrent/snap-wall-p1
NET=$OUT/start/step-7.network.json
CODE=~/code/wt-p1
echo "started $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/p1.log
jobs_list() { for i in $(seq 0 9); do echo "off $i 0"; echo "on $i 1"; done; }
run_one() {
  arm=$1; i=$2; wall=$3
  PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 .venv/bin/python $CODE/scripts/window_recurrent_study.py replay "$NET" \
    --config $CODE/scripts/window_recurrent_e2b_prime.json --run curriculum-linear-k1-s16-m16-seed6 --step 8 \
    --epochs 2 --generator snap-wall-p1-$i --numpy --wall $wall --json $OUT/$arm-$i.json 2>&1 | sed "s/^/$arm /"
}
export -f run_one; export OUT NET CODE
jobs_list | xargs -P 2 -L 1 bash -c 'run_one $0 $1 $2' >> $OUT/p1.log
echo "ended $(date) exit $?" >> $OUT/p1.log
