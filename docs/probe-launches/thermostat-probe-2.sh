#!/bin/bash
# The thermostat's second probe (docs/commitment-control-workplan.md, D2''; a probe, it claims nothing): the
# diagnostic's six failure-A runs (word-problem-s3 seeds 0, 1, 3, 4, 5; move-to-front-4 seed 3; round 1's cell) with
# --thermostat --thermostat-screen 0.95 (off below 3 effective codes; from off, back once the step's best screen
# reaches 0.95, ramped over 20 epochs). On pyramidon from main, 6 at once. Controls: round 1 (all six failure A),
# the diagnostic (off for steps 1-5: 3 of 6 formed, S3 seed 5 certified) and the first probe (D2': 0 of 6, 2 formed).
# Expectations, written before its data: 3 or more of 6 form a state (share >= 0.5) and at least one certifies ->
# D2'' does what the diagnostic's schedule did, adaptively: C3 goes ahead with D2''. Fewer than the diagnostic's
# 3 -> the screen gate isn't enough: the thermostat is retired (the owner's choice, 2026-10-19), failure A to
# next-steps.md with the branches and width 32 as its leads.
cd ~/code/indrajala-ml
OUT=data/window-recurrent/commitment-control/thermostat-probe-2
mkdir -p $OUT
echo "thermostat probe 2 starts $(date) at $(git rev-parse --short HEAD)" >> $OUT/launch.log
OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 .venv/bin/python scripts/modular_family_probe.py run \
  --runs word-problem-s3:0 word-problem-s3:1 word-problem-s3:3 word-problem-s3:4 word-problem-s3:5 move-to-front-4:3 \
  --factors 2 --codes-ratio 32/11 --reads output --fork --thermostat --thermostat-screen 0.95 --budget-epochs 600 \
  --budget-minutes 900 --workers 6 --out $OUT >> $OUT/probe.out 2>&1
echo "thermostat probe 2 ended $(date) exit $?" >> $OUT/launch.log
