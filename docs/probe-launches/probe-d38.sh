#!/bin/bash
# The state formation workplan's D38, the coupling probe (expectations written before its data). The
# quasi-static crucible (4a'''', D26-D30) with every factor's coupling capped at 0 (--coupling-cap 0):
# the codes anneal on the cells' outputs and never enter the forward path, in every phase (growth's
# too); the certificate reads the snapped network as ever, and the run stops at its first (D35's stop
# is the panel's: one change here). Why (the reasoning review's re-reading, the workplan's findings):
# every certificate of the quasi-static crucible came at couplings of 0.06-0.90, its codes at their
# final groups at 0.00-0.50, so the handover's completion isn't what conversion needs; if the coupling
# isn't needed at all, the method loses a schedule.
# Population: word-problem-s3 seeds 0-7 in the crucible (4a''''s cell exactly, plus the cap), and
# move-to-front-4 seed 3 under --whole (4a'''s convex cell, which certified at step 104, plus the cap).
# Feasibility (docs/information-flow.md): S3 q 6, 8 codes a factor, m 64, cap 69, width 32; mtf-4 q 24,
# 15 codes a factor, m 225, cap 248; row 15: S3 depth 2 + k 1, mtf-4 3 + 2, within the crucible's 8.
# Up to 1,600 epochs or 690 minutes a run, all at once (stone from 18:00: 9 runs and the D33 probe's
# seed 7 on its 12 cores).
# Expectations (read offline with `population` and `answers-stop --snapped` on each log; the certified
# networks' `approximations`):
#  - mechanics: every logged coupling 0; the codes' groups split in the cool as 4a''''s (the codes
#    anneal on the liquid as they did at coupling 0-0.5);
#  - existence: S3 at least 6 of 8 certified (4a'''': 8 of 8); move-to-front-4 seed 3 read, not counted.
# Refuted: S3 at 4 or fewer (the coupling matters: where the snapped screen (answers-stop --snapped)
# and the coupled one part says why); runs whose codes split but whose snapped machine never answers
# right (the codes track the liquid's clusters, not its dynamics: a coupling, or another signal, is
# what makes the codes' transitions the machine's).
# Decision: holds, the panel (4b) runs D36's arms at coupling cap 0 and the handover (D25, D30) goes
# to the shelf; refuted, the panel keeps the handover.
set -u
CODE=$1; OUT=$2; RUNS_ROOT=$3; SEEDS=${4:-0 1 2 3 4 5 6 7}
EXTRA=${5:-}  # --resume: each run goes on from its checkpoint
# Which arms this machine runs (a schedule split only, 2026-10-21 16:50: S3 on jebel, mtf-4 on stone)
ARMS=${6:-s3 mtf4}
mkdir -p $OUT
cd $CODE
# PY: the machine's Python with the crate (stone: PYTHONPATH to its wheel and a shared venv)
PROBE="env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ${PY:-.venv/bin/python} scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "D38 probe (arms $ARMS; S3 seeds $SEEDS, mtf-4 seed 3) starts at $(git rev-parse --short HEAD) $EXTRA"
CELL="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 1600 --budget-minutes 690 \
  --anneal --anneal-tail 40 --anneal-never-commit --warm-up --handover --melt-hot 2 --melt-kappa 1e-2 \
  --melt-hold 20 --melt-cool 600 --melt-octaves 20 --melt-halving 30 --melt-shape convex --workers 1 \
  --coupling-cap 0"
CRUCIBLE="--crucible --warm-up-screen 1.01 --melt-gate 0.99 --handover-octaves 3"
[[ " $ARMS " == *" s3 "* ]] && for SEED in $SEEDS; do
  log "launch s3-seed$SEED"
  AB_RUNS_ROOT=$RUNS_ROOT-s3-$SEED $PROBE --runs word-problem-s3:$SEED $CELL $CRUCIBLE --out $OUT/s3-seed$SEED $EXTRA \
    >> $OUT/s3-seed$SEED.out 2>&1 &
done
if [[ " $ARMS " == *" mtf4 "* ]]; then
  log "launch mtf4-seed3"
  AB_RUNS_ROOT=$RUNS_ROOT-mtf4-3 $PROBE --runs move-to-front-4:3 $CELL --whole --out $OUT/mtf4-seed3 $EXTRA \
    >> $OUT/mtf4-seed3.out 2>&1 &
fi
wait
log "probe ended"
