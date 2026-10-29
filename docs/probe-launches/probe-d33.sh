#!/bin/bash
# The state formation workplan's D32 and D33, the adaptive cool's existence probe (expectations written
# before its data). The quasi-static crucible of 4a'''' (D26-D30) with its convex cool replaced by the
# adaptive schedule (--melt-shape adaptive, D33: each factor's temperature steps once an equilibrium test on
# its soft assignment's energy passes, or after --cool-settle 10 epochs at it, by T exp(-pace / sqrt(C)),
# pace 0.03, at most an octave; D29's gate and D25's coupling kept). Population: word-problem-s3 seeds 0-7
# (4a'''''s 8 of 8 the convex control), move-to-front-4 seed 3 adaptive and, as its control (never run in
# the crucible), convex.
# Feasibility (docs/information-flow.md): S3 q 6, 8 codes a factor, m 64, cap 69; mtf-4 q 24, 15 codes a
# factor, m 225, cap 248; width 32; each cell certified before on this machine. Up to 1,600 epochs or 690
# minutes a run, on stone, at most JOBS at once (about half its cores by day).
# Expectations (read offline: heat-capacity and population on each log; approximations on any certified
# network):
#  - mechanics: in each run whose factor passes a cascade (two or more splits), its temperature falls
#    fewer octaves an epoch from the cascade's first split to its last than over the cool's other epochs
#    (the step slows where the capacity peaks), in at least 3 of 4 such factors; and fewer than half the
#    cascade's temperatures leave at the cap (the equilibrium test, not the cap, steps it);
#  - existence: S3 at least 4 of 8 certified (Wilson 95% beside 4a'''''s 8 of 8); mtf-4 seed 3 adaptive
#    certified.
# Refuted: no S3 certificate (the adaptive cool loses what the convex kept: where in the profile, from the
# logs); a stall (a factor's cascade temperatures all left at the cap, or the clock at one temperature
# for 100 epochs: the test's form doesn't fit training's moving landscape); steps no slower in the cascade
# (the capacity isn't steering the cool: D33's cost wrong).
set -u
CODE=$1; OUT=$2; RUNS_ROOT=$3; JOBS=${4:-5}
EXTRA=${5:-}  # --resume: each run goes on from its checkpoint
mkdir -p $OUT
cd $CODE
# PY: the machine's Python with the crate (stone: PYTHONPATH to its wheel and a shared venv)
PROBE="env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ${PY:-.venv/bin/python} scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "D33 existence probe starts at $(git rev-parse --short HEAD), $JOBS at once $EXTRA"
BASE="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 1600 --budget-minutes 690 \
  --crucible --anneal --anneal-tail 40 --anneal-never-commit --warm-up --handover --melt-hot 2 --melt-kappa 1e-2 \
  --melt-hold 20 --workers 1 --warm-up-screen 1.01 --melt-gate 0.99 --handover-octaves 3"
ADAPTIVE="--melt-shape adaptive --cool-pace 0.03 --cool-settle 10 --cool-within 0.75"
CONVEX="--melt-shape convex --melt-cool 600 --melt-octaves 20 --melt-halving 30"
launch() {  # name, task:seed, shape's flags; at most JOBS at once
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  log "launch $1"
  AB_RUNS_ROOT=$RUNS_ROOT-$1 $PROBE --runs $2 $BASE $3 --out $OUT/$1 $EXTRA >> $OUT/$1.out 2>&1 &
}
launch mtf4-adaptive move-to-front-4:3 "$ADAPTIVE"
launch mtf4-convex move-to-front-4:3 "$CONVEX"
for SEED in 0 1 2 3 4 5 6 7; do
  launch s3-seed$SEED word-problem-s3:$SEED "$ADAPTIVE"
done
wait
log "probe ended"
