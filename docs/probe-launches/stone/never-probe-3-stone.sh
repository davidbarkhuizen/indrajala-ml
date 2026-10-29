#!/bin/bash
# The never probe's third arm (docs/commitment-control-workplan.md; a probe, it claims nothing; the toolbox only).
# The first arm: word-problem-s3 seeds 0 and 3 with `--commitment-from 1000` (the commitment never returns) both
# certified (steps 6 and 8, recertified on pyramidon). Here the larger failure-A backdrop at the base width:
# move-to-front-5 seeds 0 and 1, round 2's cell (two factors, --codes-ratio 32/11, reading the output, the fork,
# 600 epochs) at --window-width 32, the commitment off from the start. Controls: C0's same seeds (failure A: 2-3
# codes, state share 0.152 and 0.208, provisional) and annealing arm 2 (failure A: 15 and 9 codes, 0.465 and 0.363,
# frozen from step 6). Without the flag this commit (8e88c9d) repeats C0's seed 0 by bits (arm 2's control check);
# with it the runs differ from step 1, so there is no by-bits check of their own.
# Feasibility (docs/information-flow.md): move-to-front-5 q 120, 32 codes a factor, m 1,024, cap 1,143; width 32
# certified this task before (B2's seed 5 at 16; C0 runs the cell).
# Expectations, written before its data (a run forms a state if its share reaches 0.5 at its best):
#  - either forms a state -> the commitment off moves failure A on the larger cell too, where annealing didn't.
#  - either certifies -> the strongest evidence yet for the commitment off as the base's training.
#  - neither forms -> failure A on move-to-front-5 has another cause besides the commitment (read its codes in use).
# On stone beside R1's probes and annealing arm 2 (2 workers), its own lock directory.
set -u
source ~/labour/env.sh
export AB_RUNS_ROOT=$LABOUR/ab-runs-never3
CODE=$LABOUR/code/wt-anneal
OUT=$LABOUR/data/commitment-control/never-probe/arm3
mkdir -p $OUT $AB_RUNS_ROOT
cd $LABOUR/code/indrajala-ml
PROBE="env PYTHONPATH=$LABOUR/site-soft-codes:$CODE OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 $CODE/../wt-branches/.venv/bin/python $CODE/scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "never probe arm 3 starts at $(git -C $CODE rev-parse --short HEAD)"
$PROBE --runs move-to-front-5:0 move-to-front-5:1 --factors 2 --codes-ratio 32/11 --reads output --fork \
  --window-width 32 --commitment-from 1000 --budget-epochs 600 --budget-minutes 900 --workers 2 --out $OUT/runs \
  >> $OUT/runs.out 2>&1
log "never probe arm 3 ended exit $?"
