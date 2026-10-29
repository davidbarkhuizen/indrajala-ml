#!/bin/bash
# Waits until jebel has two free study slots (at most 5 study runs), then moves pyramidon's D67
# move-to-front-4 34 and 35 there (move-d67-mtf4-to-jebel.sh). Detached, so it outlives a session.
until n=$(ssh -o ConnectTimeout=20 jebel 'pgrep -f "^[^ ]*/python[0-9.]* scripts/[m]odular_family_probe.py run" | wc -l' 2>/dev/null) && [ -n "$n" ] && [ "$n" -le 5 ]; do sleep 120; done
echo "jebel has $n study runs at $(date): moving"
bash ~/code/move-d67-mtf4-to-jebel.sh
echo "move done $(date)"
