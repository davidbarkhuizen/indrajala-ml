#!/bin/bash
# E2b snapped arm (claims workplan D38): start when division stage 5's pyramidon arms (3778260, 3778261) end.
cd ~/code/indrajala-ml
while kill -0 3778260 2>/dev/null || kill -0 3778261 2>/dev/null; do sleep 60; done
echo "E2b snapped launched $(date) from ~/code/wt-e2b at $(git -C ~/code/wt-e2b rev-parse --short HEAD), 3 workers" > data/window-recurrent/e2b-snapped-launch.txt
PYTHONPATH=$HOME/code/wt-e2b OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 exec .venv/bin/python ~/code/wt-e2b/scripts/window_recurrent_study.py run ~/code/wt-e2b/scripts/window_recurrent_e2b_snapped.json --workers 3
