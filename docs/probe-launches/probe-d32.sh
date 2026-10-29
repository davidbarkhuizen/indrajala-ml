#!/bin/bash
# The state formation workplan's D32, its understanding step (expectations written before its data): does
# the soft assignment's heat capacity (thermals: d<D>/dT, Rose 1998's specific heat, logged each profiled
# epoch as `capacity`, with `energy`) peak where the codes split, so that Huang, Romeo and
# Sangiovanni-Vincentelli's variance-sized step (Romeo 1989, Ch. 7) has something to read? Three certified
# runs rerun with the logging only: word-problem-s3 seeds 4 (dips and held epochs) and 7 (a cover) as
# 4a'''' (probe-d29.sh), and move-to-front-4 seed 3 convex as 4a'' (probe-4a3.sh, no curriculum).
# Feasibility: the cells as before (S3 q 6, 8 codes a factor, m 64, cap 69; mtf-4 q 24, 15 codes, m 225,
# cap 248; width 32, each certified on this machine).
# Expectations (read offline from each log's `anneal` rows):
#  - passive: each run certifies at its original step (S3 seed 4 at 123, seed 7 at 67; mtf-4 convex at
#    104), the logging touching neither the training nor its randomness (the golden run bit-identical at
#    364a09e);
#  - the capacity peaks at the splits: in each factor's cool, the epochs where its groups rise (code_groups,
#    D20) carry its capacity's local maxima, the largest within 2 epochs of the largest split (the melting
#    point's nucleation) in at least 2 of the 3 runs.
# Refuted: a run certifying at another step (the logging isn't passive: fix before anything else); no peak
# at the splits (the capacity is flat or monotone through them: the soft assignment's energy isn't the
# schedule's cost here; next, the answers' screen or the batch loss's spread at a fixed temperature).
set -u
S3=$1; MTF=$2; OUT=$3; RUNS_ROOT=$4
mkdir -p $OUT
PROBE="env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ${PY:-.venv/bin/python} scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
cd $S3
log "D32 understanding step starts at $(git rev-parse --short HEAD)"
D29="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 1600 --budget-minutes 690 \
  --crucible --anneal --anneal-tail 40 --anneal-never-commit --warm-up --handover --melt-hot 2 --melt-kappa 1e-2 \
  --melt-hold 20 --melt-cool 600 --melt-octaves 20 --melt-halving 30 --melt-shape convex --workers 1 \
  --warm-up-screen 1.01 --melt-gate 0.99 --handover-octaves 3"
for SEED in 4 7; do
  AB_RUNS_ROOT=$RUNS_ROOT-s3-$SEED $PROBE --runs word-problem-s3:$SEED $D29 --out $OUT/s3-seed$SEED \
    >> $OUT/s3-seed$SEED.out 2>&1 &
done
cd $MTF
A4="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 1600 --budget-minutes 2880 \
  --whole --anneal --anneal-tail 40 --anneal-never-commit --warm-up --handover --melt-hot 2 --melt-kappa 1e-2 \
  --melt-hold 20 --melt-cool 600 --melt-octaves 20 --melt-halving 30 --workers 1 --melt-shape convex"
AB_RUNS_ROOT=$RUNS_ROOT-mtf4 $PROBE --runs move-to-front-4:3 $A4 --out $OUT/mtf4-convex >> $OUT/mtf4-convex.out 2>&1 &
wait
log "probe ended"
