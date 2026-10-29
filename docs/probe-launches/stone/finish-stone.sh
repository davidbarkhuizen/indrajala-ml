#!/bin/bash
# Finish stone runs one at a time: wait for D59 S3 seed 35 to end, then resume D55 mtf-4 seed 25 (paused by its 06:55 stop) alone.
source ~/labour/env.sh
O=$LABOUR/data/state-formation/option1
while pgrep -f "probe-d59/s3-seed35" > /dev/null; do sleep 60; done
while pgrep -f "probe-d55/mtf4-seed25" > /dev/null; do sleep 60; done
echo "finish-stone: resuming mtf4:25 alone $(date)" >> $O/probe-d55/launch.log
cd ~/labour && PY="env PYTHONPATH=$LABOUR/site-straight:$LABOUR/code/wt-d54 $LABOUR/code/wt-branches/.venv/bin/python" \
  ~/labour/probe-d55.sh $LABOUR/code/wt-d54 $O/probe-d55 $LABOUR/ab-runs-d55 1 "mtf4:25" --resume
