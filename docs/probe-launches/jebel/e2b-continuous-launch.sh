#!/bin/bash
# E2b continuous arm (claims workplan D38): start when E2 (74145: its launcher, exec-ed into the study) ends.
cd ~/code/indrajala-ml
while kill -0 74145 2>/dev/null; do sleep 60; done
echo "E2b continuous launched $(date) from ~/code/wt-e2b at $(git -C ~/code/wt-e2b rev-parse --short HEAD), 6 workers" > data/window-recurrent/e2b-continuous-launch.txt
PYTHONPATH=$HOME/code/wt-e2b OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 exec .venv/bin/python ~/code/wt-e2b/scripts/window_recurrent_study.py run ~/code/wt-e2b/scripts/window_recurrent_e2b_continuous.json
