#!/bin/bash
# After E2b (pid 90703) ends: the p 11 ladder (docs/p11-trigger-workplan.md, stage 3), then jebel's share of the
# approximation claim (docs/approximation-claim-workplan.md, seeds 200-213). Code ~/code/wt-p11 (pinned 7618dfe).
cd ~/code/indrajala-ml
CODE=~/code/wt-p11
LOG=data/window-recurrent/jebel-chain.log
echo "chain waiting for E2b (90703) $(date)" >> $LOG
while kill -0 90703 2>/dev/null; do sleep 60; done
echo "E2b ended $(date); p 11 ladder starts at $(git -C $CODE rev-parse --short HEAD)" >> $LOG
P11=data/window-recurrent/p11-ladder
mkdir -p $P11
PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 .venv/bin/python $CODE/scripts/modular_family_probe.py run --p 11 --arm family \
  --factor 16 32 --factor 16 32 --reads output --fork --budget-epochs 600 --budget-minutes 900 \
  --first-seed 0 --seeds 6 --workers 6 --out $P11/family >> $P11/family.out 2>&1
echo "p 11 ladder ended $(date) exit $?" >> $LOG
OUT=data/window-recurrent/approximation-claim
mkdir -p $OUT
run_arm() {
  PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 .venv/bin/python $CODE/scripts/modular_family_probe.py run --p 5 --arm $1 \
    --factors 2 --reads output --vocabulary sum difference product --budget-epochs 400 --budget-minutes 180 \
    --first-seed 200 --seeds 14 --workers 3 --out $OUT/$1 >> $OUT/$1-jebel.out 2>&1
  echo "$1 done $(date) exit $?" >> $LOG
}
echo "approximation claim (jebel) starts $(date)" >> $LOG
run_arm quotient & run_arm sum & wait
run_arm family & run_arm product & wait
echo "chain ended $(date)" >> $LOG
