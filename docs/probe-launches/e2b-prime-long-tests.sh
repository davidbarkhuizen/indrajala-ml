#!/bin/bash
# E2b''s long tests (claims workplan E2b', registered): B1 at 256 and 1,024 on every run whose break
# is 64 (all 20), the last step's network (step 18). Code: ~/code/wt-dev2 (origin/main c4a91e7);
# don't develop in wt-dev2 until this ends.
cd ~/code/indrajala-ml
L=data/window-recurrent/e2b-prime-long-tests.log
echo "started $(date) at $(git -C ~/code/wt-dev2 rev-parse --short HEAD)" > $L
export PYTHONPATH=$HOME/code/wt-dev2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
.venv/bin/python ~/code/wt-dev2/scripts/window_recurrent_study.py long-test $(cat ~/code/e2b-prime-long-nets-20.txt) \
  --n 256 1024 --properties B1 --workers 2 --out data/window-recurrent/e2b-prime-long-tests >> $L 2>&1
echo "ended $(date) exit $?" >> $L
