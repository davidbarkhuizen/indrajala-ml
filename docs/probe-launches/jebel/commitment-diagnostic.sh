#!/bin/bash
# The commitment-control diagnostic (docs/commitment-control-workplan.md, registered in #851, corrected in
# #852 before its data): round 1's six failure-A runs with --commitment-from 6, round 1's cell, 600
# epochs, 6 at once on jebel, from ~/code/wt-cc (6e5d746; its .venv links wt-p2's, crate 40a2761). Its
# control is round 1's runs (by bits on 15a24d6). Starts when round 2 ends; the probe holds the lock.

while pgrep -f algebras-round2.sh > /dev/null; do sleep 60; done
cd ~/code/indrajala-ml
CODE=~/code/wt-cc
OUT=data/window-recurrent/commitment-control/diagnostic
mkdir -p $OUT
echo "diagnostic starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch.log
PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 $CODE/.venv/bin/python $CODE/scripts/modular_family_probe.py run \
  --runs word-problem-s3:0 word-problem-s3:1 word-problem-s3:3 word-problem-s3:4 word-problem-s3:5 move-to-front-4:3 \
  --factors 2 --codes-ratio 32/11 --reads output --fork --budget-epochs 600 --budget-minutes 900 \
  --commitment-from 6 --workers 6 --out $OUT >> $OUT/diagnostic.out 2>&1
rc=$?
echo "diagnostic ended $(date) exit $rc" >> $OUT/launch.log
