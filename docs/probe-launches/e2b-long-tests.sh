#!/bin/bash
# E2b's long tests (claims workplan E2b): B1 at 256 and 1,024 on every run whose break is 64,
# both arms (20 snapped, continuous seeds 2 and 9). Code: ~/code/wt-dev (origin/main).
cd ~/code/indrajala-ml
L=data/window-recurrent/e2b-long-tests.log
echo "started $(date) at $(git -C ~/code/wt-dev rev-parse --short HEAD)" > $L
export PYTHONPATH=$HOME/code/wt-dev OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
.venv/bin/python ~/code/wt-dev/scripts/window_recurrent_study.py long-test $(cat ~/code/e2b-long-nets-22.txt) \
  --n 256 1024 --properties B1 --workers 2 --out data/window-recurrent/e2b-long-tests >> $L 2>&1
echo "ended $(date) exit $?" >> $L
