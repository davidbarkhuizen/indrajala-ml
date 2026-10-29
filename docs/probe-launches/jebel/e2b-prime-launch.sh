#!/bin/bash
# E2b' (claims workplan D39): start when stage 5's long-test workers on jebel (79045 79047 79049) end.
cd ~/code/indrajala-ml
while kill -0 79045 2>/dev/null || kill -0 79047 2>/dev/null || kill -0 79049 2>/dev/null; do sleep 60; done
echo "E2b' launched $(date) from ~/code/wt-e2bp at $(git -C ~/code/wt-e2bp rev-parse --short HEAD), 6 workers" > data/window-recurrent/e2b-prime-launch.txt
PYTHONPATH=$HOME/code/wt-e2bp OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 exec .venv/bin/python ~/code/wt-e2bp/scripts/window_recurrent_study.py run ~/code/wt-e2bp/scripts/window_recurrent_e2b_prime.json > data/window-recurrent/e2b-prime.out 2>&1
