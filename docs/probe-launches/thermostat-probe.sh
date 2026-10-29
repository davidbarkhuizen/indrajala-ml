#!/bin/bash
# The thermostat's probe (docs/commitment-control-workplan.md, "Stage 2, re-aimed at failure A", D5'; a probe, it
# claims nothing): the diagnostic's six failure-A runs (word-problem-s3 seeds 0, 1, 3, 4, 5; move-to-front-4 seed
# 3; round 1's cell: two factors, --codes-ratio 32/11, reading the output, the fork, 600 epochs) with --thermostat
# (D2': off below 3 effective codes, back at half the codebook over 20 epochs). On pyramidon from main, 6 at once.
# Its controls: round 1 (all six failure A; the default repeats round 1 by bits on the thermostat's commit) and the
# diagnostic (the commitment off for steps 1-5: 3 of 6 formed a state, S3 seed 5 certified, the states eroded when
# it came back).
# Expectations, written before its data: 3 or more of 6 form a state (share >= 0.5) and keep it (no fall of 0.3 or
# more after the ramp ends), and at least one certifies -> the thermostat does what the diagnostic's phase schedule
# couldn't; C3 goes ahead as registered. States form but erode as the ramp ends -> the ramp or the restore level
# is wrong: probe a longer ramp or a higher restore. None forms -> the thermostat's signal (code use) misses failure
# A where the phase schedule found it: read each run's effective codes against the diagnostic's.
cd ~/code/indrajala-ml
OUT=data/window-recurrent/commitment-control/thermostat-probe
mkdir -p $OUT
echo "thermostat probe starts $(date) at $(git rev-parse --short HEAD)" >> $OUT/launch.log
OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 .venv/bin/python scripts/modular_family_probe.py run \
  --runs word-problem-s3:0 word-problem-s3:1 word-problem-s3:3 word-problem-s3:4 word-problem-s3:5 move-to-front-4:3 \
  --factors 2 --codes-ratio 32/11 --reads output --fork --thermostat --budget-epochs 600 --budget-minutes 900 \
  --workers 6 --out $OUT >> $OUT/probe.out 2>&1
echo "thermostat probe ended $(date) exit $?" >> $OUT/launch.log
