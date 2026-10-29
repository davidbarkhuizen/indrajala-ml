#!/bin/bash
# The commitment never returns, with R3 and R4 (a probe, it claims nothing; the toolbox only, a combination:
# `--commitment-from 1000 --keep-machine --counterexamples 0.25`). Move-to-front-5 seeds 0 and 1, round 2's cell
# (two factors, --codes-ratio 32/11, reading the output, the fork, 600 epochs) at --window-width 32.
# Why: the never probe's arm 3 (the same cell, the commitment off alone) formed a state in both seeds (shares
# 0.79-0.88 from step 6, held 13-17 steps) at 4.6-7 pairs per code (bottleneck 2, against round 2's 98-120), then
# lost it (seed 0 to 2 codes at step 27; seed 1 to 0.39-0.61); 0 of 2 certified. R3 (keep the machine with the
# fewest pairs per code, restore it at a plateau's end) and R4 (a quarter of each epoch from the walk's
# counterexamples) certified S5 seed 1 at width 16 (their probe), a seed that failed in round 2 and B2.
# Controls: the never probe's arm 3 (one change: R3 and R4 added), C0's same seeds (failure A), annealing arm 2
# (failure A). No by-bits check: the flags change step 1 (R4's examples); arm 3's control covers the commit.
# Feasibility (docs/information-flow.md): move-to-front-5 q 120, 32 codes a factor, m 1,024, cap 1,143, width 32.
# Expectations, written before its data:
#  - either certifies -> the commitment off (bottleneck 1) with R3 and R4 (2, and retention) is the first candidate
#    general method: read next on S3's failure-A seeds, S4 and S5, and division 13 at 32, 2 seeds each.
#  - neither, but pairs per code lower than arm 3's 4.6 or a state held longer than 17 steps -> R3/R4 help on top;
#    read where it stops.
#  - neither, and no better than arm 3 -> R3/R4 add nothing here; retention needs another remedy.
# On stone beside R1's probe at 32 (2 workers), its own lock directory.
set -u
source ~/labour/env.sh
export AB_RUNS_ROOT=$LABOUR/ab-runs-never-r34
CODE=$LABOUR/code/wt-anneal
OUT=$LABOUR/data/commitment-control/never-r34
mkdir -p $OUT $AB_RUNS_ROOT
cd $LABOUR/code/indrajala-ml
PROBE="env PYTHONPATH=$LABOUR/site-soft-codes:$CODE OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 $CODE/../wt-branches/.venv/bin/python $CODE/scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "never + R3/R4 starts at $(git -C $CODE rev-parse --short HEAD)"
$PROBE --runs move-to-front-5:0 move-to-front-5:1 --factors 2 --codes-ratio 32/11 --reads output --fork \
  --window-width 32 --commitment-from 1000 --keep-machine --counterexamples 0.25 --budget-epochs 600 \
  --budget-minutes 900 --workers 2 --out $OUT/runs >> $OUT/runs.out 2>&1
log "never + R3/R4 ended exit $?"
