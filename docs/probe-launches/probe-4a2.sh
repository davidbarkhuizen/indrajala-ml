#!/bin/bash
# The state formation workplan's stage 4a', the shapes probe (D22-D24; expectations written before its
# data). Option 1 with no curriculum (--whole: every epoch over every length to the cap, warm-up
# included), the full melt (T_hot 2 T_c), held 20 epochs, thermal noise kappa 1e-2 (agitation: the owner's
# thermal picture; readable now D20's distance sits above the noise), the commitment never back, the cool
# falling 20 octaves in 600 epochs as each shape (concave, convex, s), then halved every 30, the hardness
# test read only once the shaped cool has run (D24: each shape's whole curve trains soft);
# round 1's and 2's cell (two factors, --codes-ratio 32/11, reading the output) at --window-width 32; no
# fork (D18). Backdrops: word-problem-s3 seed 0 and move-to-front-4 seed 3, the failure-A seeds that
# certified with the commitment never on (not S4 seed 0: a harm seed, certified only with the commitment).
# Up to 1,600 epochs a run (a warm-up up to 8 blocks, about 480 epochs, then 626 to the cool's end, the
# tail after; a whole epoch 18.6 s on S3, about 61 s at cap 248: up to about 9 h and 27 h; multi-day
# allowed where justified: the owner, 2026-10-20).
# Feasibility (docs/information-flow.md): S3 q 6, 8 codes a factor, m 64, cap 69; mtf-4 q 24, 15, 225, 248;
# width 32 certified each (C0); every epoch trains lengths 1..cap, so the exclusion bound holds throughout.
# Expectations:
#  - risk 1 (D22): the continuous warm-up learns the whole distribution: its screen rises above chance
#    on the long lengths; if it never reaches 0.95, the warm-up ends at its cap (8 blocks), recorded;
#  - the warm-up ends with each factor's distinct states at least 4 times its codes (long words give many);
#  - the melt: at the hold's end each factor's codes in 1-2 groups (20 epochs, D20's distance above noise);
#  - the splits: in each shape the groups rise as the cool passes each factor's melting point (D21);
#  - states: at least one run reaches a state share >= 0.5; at least one certifies (both seeds certified
#    with the commitment never on, round 1, with the curriculum).
# Existence only (one seed a shape: no ranking of shapes). Decision: every shape that certifies or forms a
# state (share >= 0.5) goes forward to the panel (4b), its commitment's return then tested; none forming
# a state: if risk 1 hit (the warm-up unlearned), D22 (c), uniform over a shorter range; else the readings
# (melt, splits, where codes were lost) decide the next design.
set -u
RUN=$1; CODE=$2; OUT=$3; RUNS_ROOT=$4  # RUN: task:seed
mkdir -p $OUT
cd $CODE
# PY: the machine's Python with the crate (stone: PYTHONPATH to its wheel and a shared venv)
PROBE="env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ${PY:-.venv/bin/python} scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "stage 4a' ($RUN) starts at $(git rev-parse --short HEAD)"
BASE="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 1600 --budget-minutes 2880 \
  --whole --anneal --anneal-tail 40 --anneal-never-commit --warm-up --melt-hot 2 --melt-kappa 1e-2 --melt-hold 20 \
  --melt-cool 600 --melt-octaves 20 --melt-halving 30 --workers 1"
for SHAPE in concave convex s; do
  AB_RUNS_ROOT=$RUNS_ROOT-$SHAPE $PROBE --runs $RUN $BASE --melt-shape $SHAPE --out $OUT/$SHAPE > $OUT/$SHAPE.out 2>&1 &
done
wait
log "probe ended"
