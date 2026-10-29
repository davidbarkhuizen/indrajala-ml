#!/bin/bash
# E2 (claims workplan, D36): start when division stage 5 jebel arms (73666, 73667) end.
# Code from the wt-e2 worktree (ce4015d), so main can be pulled meanwhile; data paths relative to the main checkout.
cd ~/code/indrajala-ml
while kill -0 73666 2>/dev/null || kill -0 73667 2>/dev/null; do sleep 60; done
echo "E2 launched $(date) from ~/code/wt-e2 at $(git -C ~/code/wt-e2 rev-parse --short HEAD)" > data/window-recurrent/e2-launch.txt
PYTHONPATH=$HOME/code/wt-e2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 exec .venv/bin/python ~/code/wt-e2/scripts/window_recurrent_study.py run ~/code/wt-e2/scripts/window_recurrent_e2.json
