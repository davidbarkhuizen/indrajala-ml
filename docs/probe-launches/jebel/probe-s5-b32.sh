#!/bin/bash
# A quick probe (the owner, 2026-10-19: "first get it working, then refine"; claims nothing): S5 with the window
# branches AND the window 32 wide (--branches --window-width 32), round 2's cell otherwise (two factors,
# --codes-ratio 32/11, reading the output, the fork, 600 epochs), seeds 0 and 1 (both failed in round 2 and B2).
# Its expectations, written before its data: either certifies -> the combination is S5's candidate cell, and a
# registered round (6 seeds, with the width-only control) follows; neither, but pairs per code (breaks, d 2) at
# their best steps fall well below B2's same seeds' (120 and 66) -> width helps the codes mean one state, not
# enough alone; neither and no fall -> width adds nothing on S5's confusion.
# From ~/code/wt-b2 at 393b3b1, 2 workers on jebel's 2 free cores beside B2/W2, under its own lock directory
# (AB_RUNS_ROOT=~/code/ab-runs-probe) so the study's lock doesn't refuse it; no timing runs on jebel meanwhile.
cd ~/code/indrajala-ml
CODE=~/code/wt-b2
OUT=data/window-recurrent/window-branches/probe-s5-b32
mkdir -p $OUT ~/code/ab-runs-probe
echo "probe starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch.log
AB_RUNS_ROOT=~/code/ab-runs-probe PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 $CODE/.venv/bin/python \
  $CODE/scripts/modular_family_probe.py run --task word-problem-s5 --branches --window-width 32 \
  --factors 2 --codes-ratio 32/11 --reads output --fork --budget-epochs 600 --budget-minutes 900 \
  --first-seed 0 --seeds 2 --workers 2 --out $OUT "$@" >> $OUT/probe.out 2>&1
echo "probe ended $(date) exit $?" >> $OUT/launch.log
