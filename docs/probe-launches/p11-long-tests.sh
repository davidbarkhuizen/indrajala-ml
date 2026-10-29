#!/bin/bash
# Long tests (division D4: U1 and S1 at 128, 256, 1,024) of the p 11 ladder's 4 certified networks
# (p11-trigger stage 3), one at a time. Code: ~/code/wt-dev2 (origin/main c4a91e7); don't develop there.
cd ~/code/indrajala-ml
D=data/window-recurrent/p11-ladder
L=$D/long-tests.log
echo "started $(date) at $(git -C ~/code/wt-dev2 rev-parse --short HEAD)" > $L
export PYTHONPATH=$HOME/code/wt-dev2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
for n in $D/family/seed*/certified-step-*.network.json; do
  seed=$(basename "$(dirname "$n")")
  echo "start $seed $(date)" >> $L
  .venv/bin/python ~/code/wt-dev2/scripts/modular_family_probe.py long-test "$n" --task modular-division-11 --arm family \
    --out $D/long-tests/$seed >> $L 2>&1
  echo "done $seed $(date) exit $?" >> $L
done
echo "ended $(date)" >> $L
