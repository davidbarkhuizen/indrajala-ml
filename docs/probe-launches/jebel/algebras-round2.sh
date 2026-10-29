#!/bin/bash
# Round 2 of the general algebras (docs/general-algebras-workplan.md, stage 4; the random class set aside
# by the owner, registered in #850 before its data): word-problem-s5 and move-to-front-5, seeds 0-5,
# round 1's cell, 600 epochs; 12 runs, 6 at once on jebel, from ~/code/wt-round2 (ff2b2f2; its .venv
# links wt-p2's, the same crate 40a2761). Starts when P2 ends; the probe holds the lock.
while pgrep -f "consolidation/b-q16/p2" > /dev/null; do sleep 60; done
cd ~/code/indrajala-ml
CODE=~/code/wt-round2
OUT=data/window-recurrent/general-algebras/round2
mkdir -p $OUT
echo "round 2 starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch.log
PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 $CODE/.venv/bin/python $CODE/scripts/modular_family_probe.py run \
  --task word-problem-s5 move-to-front-5 \
  --factors 2 --codes-ratio 32/11 --reads output --fork --budget-epochs 600 --budget-minutes 900 \
  --first-seed 0 --seeds 6 --workers 6 --out $OUT >> $OUT/round2.out 2>&1
rc=$?
echo "round 2 ended $(date) exit $rc" >> $OUT/launch.log
