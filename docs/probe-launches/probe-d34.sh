#!/bin/bash
# The state formation workplan's D34, the bounded handover's probe (expectations written before its data).
# The D33 probe's adaptive crucible with --cool-rise 0.02: each factor's adaptive step no colder than
# raises its coupling by 0.02 (inside the coupling window; elsewhere D33's step). Why: the D33 probe's
# in-flight reading (the workplan): S3 seeds 1, 3, 5 formed the state and lost it as the coupling rose
# in jumps of 0.25-0.65 an epoch below the cascade; the quasi-static crucible's never passed 0.02.
# Population: word-problem-s3 seeds 0, 1, 3, 4, 5, 6, 7; seed 2 carried over from the D33 probe (it
# certified in the heat, step 13, before any cool: the same run under D34).
# Feasibility (docs/information-flow.md): S3 q 6, 8 codes a factor, m 64, cap 69; width 32; certified
# before on these seeds. Up to 1,600 epochs or 690 minutes a run, at most JOBS at once (pyramidon 4).
# Expectations (read offline: population on each log; the coupling's largest rise an epoch from the
# logs; heat-capacity for the cascade's pacing):
#  - mechanics: no coupling rise above 0.021 an epoch in any run; the steps through each cascade as
#    D33's (the bound inactive there);
#  - no loss: no run that formed a state (best share >= 0.7) loses it (population's lost -);
#  - existence: seeds 1, 3, 5 at least 2 of 3 certified; S3 at least 6 of 8 with seed 2.
# Refuted: a loss with the coupling's rises bounded (the handover's speed isn't the mechanism: where
# the loss came, from the logs); a stall (the clock at one temperature for 100 epochs).
set -u
CODE=$1; OUT=$2; RUNS_ROOT=$3; JOBS=${4:-5}
EXTRA=${5:-}  # --resume: each run goes on from its checkpoint
mkdir -p $OUT
cd $CODE
# PY: the machine's Python with the crate (stone: PYTHONPATH to its wheel and a shared venv)
PROBE="env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ${PY:-.venv/bin/python} scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "D34 probe starts at $(git rev-parse --short HEAD), $JOBS at once $EXTRA"
BASE="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 1600 --budget-minutes 690 \
  --crucible --anneal --anneal-tail 40 --anneal-never-commit --warm-up --handover --melt-hot 2 --melt-kappa 1e-2 \
  --melt-hold 20 --workers 1 --warm-up-screen 1.01 --melt-gate 0.99 --handover-octaves 3"
ADAPTIVE="--melt-shape adaptive --cool-pace 0.03 --cool-settle 10 --cool-within 0.75 --cool-rise 0.02"
CONVEX="--melt-shape convex --melt-cool 600 --melt-octaves 20 --melt-halving 30"
launch() {  # name, task:seed, shape's flags; at most JOBS at once
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  log "launch $1"
  AB_RUNS_ROOT=$RUNS_ROOT-$1 $PROBE --runs $2 $BASE $3 --out $OUT/$1 $EXTRA >> $OUT/$1.out 2>&1 &
}
for SEED in 0 1 3 4 5 6 7; do
  launch s3-seed$SEED word-problem-s3:$SEED "$ADAPTIVE"
done
wait
log "probe ended"
