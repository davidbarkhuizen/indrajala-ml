#!/bin/bash
# The state formation workplan's D54, the continuous crystallisation's existence probe (expectations
# written before its data, 2026-10-21 evening). The owner: "make everything we can, that still makes
# sense to do so, continuous. all at once - the whole point of the crystallization is to be
# quasi-static so that we can smoothly transition down the energy profile. then let's just test that."
# The method (D48, --continuous): the crucible's warm-up (D26, D28) and k-means codes as 4a''''; then
# the heat and the hold (D15), an exponential cool halving every 30 epochs whose clock advances by the
# rate g from the screen against the hold's level (D49: 1 at 0.99, 0 at 0.95); each factor's coupling
# from its codes' sharpness, from 0 at the cool's start (D50); soft to the end with a straight-through
# gradient, each factor's temperature held where its codes are first hard to 0.1% (D51); growth from
# the crucible's length to the cap in ln L at 0.02 g c (D52); a step an epoch, with its readings (D53).
# The commitment never back; D35's stop at the exact state bound (S3 6, mtf-4 24), the certificate
# judging it; D41's success (the stop's network certified at both verdicts).
# Population: word-problem-s3 seeds 0-7 (the crucible's design seeds; 4a'''' certified 8 of 8) and
# move-to-front-4 seeds 0-3 (the crucible never certified mtf-4 seed 3, D33's control; --whole did).
# Feasibility (docs/information-flow.md): S3 q 6, 8 codes a factor, m 64, cap 69 (the exclusion bound,
# reached by growth), width 32; mtf-4 q 24, 15 codes a factor, m 225, cap 248; row 15: S3 depth 2 + k 1,
# mtf-4 3 + 2, within the crucible's 8. Up to 1,600 epochs or 690 minutes a run.
# Expectations (read with population on each log, its per-epoch readings, and the stops' networks):
#  - mechanics: no coupling before the cool; every logged coupling's rise in an epoch below 0.05 (the
#    quasi-static path: the quasi-static crucible's largest was 0.021, the adaptive cool's losses came
#    at 0.03-0.65); the length grows only once coupled; the temperature never None;
#  - existence: S3 at least 6 of 8 succeed (D41); move-to-front-4 at least 1 of 4 (the crucible's first).
# Refuted: S3 at 4 or fewer (the continuous path loses what the quasi-static crucible's switches had);
# move-to-front-4 0 of 4 is read, not a refutation (existence on a new backdrop, 4 seeds); any false
# discovery (a stop the certificate rejects at both verdicts in a run that never certifies).
# Analysis: each failure classified by kind (failure-kinds.md) from the per-epoch log: the
# certificates' wrong states against wrong outputs, the coupling and sharpness at the loss, the
# length; the stop's trained length (D42).
#
# Usage: probe-d54.sh CODE OUT RUNS_ROOT "S3 SEEDS" "MTF4 SEEDS" [--resume]
set -u
CODE=$1; OUT=$2; RUNS_ROOT=$3; S3=${4:-0 1 2 3 4 5 6 7}; MTF4=${5:-0 1 2 3}
EXTRA=${6:-}  # --resume: each run goes on from its checkpoint
mkdir -p $OUT
cd $CODE
# PY: the machine's Python with the crate (stone: PYTHONPATH to its build)
PROBE="env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ${PY:-.venv/bin/python} scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "D54 probe (S3 seeds $S3; mtf-4 seeds $MTF4) starts at $(git rev-parse --short HEAD) $EXTRA"
CELL="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 1600 --budget-minutes 690 \
  --anneal --anneal-tail 40 --anneal-never-commit --warm-up --melt-hot 2 --melt-kappa 1e-2 --melt-hold 20 \
  --melt-halving 30 --melt-shape exponential --crucible --warm-up-screen 1.01 --continuous --workers 1"
for SEED in $S3; do
  log "launch s3-seed$SEED"
  AB_RUNS_ROOT=$RUNS_ROOT-s3-$SEED $PROBE --runs word-problem-s3:$SEED $CELL --state-bound 6 \
    --out $OUT/s3-seed$SEED $EXTRA >> $OUT/s3-seed$SEED.out 2>&1 &
done
for SEED in $MTF4; do
  log "launch mtf4-seed$SEED"
  AB_RUNS_ROOT=$RUNS_ROOT-mtf4-$SEED $PROBE --runs move-to-front-4:$SEED $CELL --state-bound 24 \
    --out $OUT/mtf4-seed$SEED $EXTRA >> $OUT/mtf4-seed$SEED.out 2>&1 &
done
wait
log "probe ended"
