#!/bin/bash
# The state formation workplan's D56, the continuous crystallisation on an unstructured machine
# (expectations written before its data, 2026-10-22, 00:40). D54's cell unchanged (D48's method,
# --continuous), on random-q16-0 seeds 0-3: the panel's random cells (D37: seeds 0 and 1, 0 of 6 at
# round 1's cell, by extra codes and lost near solutions; the consolidation plan's control, B and A
# each 0 of 6), discovery's likely target; on jebel's four night cores once D38 freed them.
# Feasibility (docs/information-flow.md): q 16, 12 codes a factor, m 144 (9.0 q), cap 159 (the
# exclusion bound, reached by growth), width 32; row 15: depth 4 + k 1, within the crucible's 8.
# Up to 1,600 epochs or 690 minutes a run; the state bound 16 (D35's stop).
# Expectations (read as D54's: population, the per-epoch readings, the stops' networks):
#  - existence: at least 1 of 4 succeed (D41), the first certificate of random-q16-0 in any arm;
#    0 of 4 is read, not a refutation: each failure classified by kind (extra codes against near
#    ties, failure-kinds.md), which says what the method lacks on unstructured machines;
#  - any false discovery (a stop the certificate rejects at both verdicts in a run that never
#    certifies) refutes D35's stop on this backdrop;
#  - the mechanics read as D54's (the coupling's rises, the trained length).
#
# Usage: probe-d56.sh CODE OUT RUNS_ROOT SLOTS "TASK:SEED ..." [--resume]   (TASK rq16)
set -u
CODE=$1; OUT=$2; RUNS_ROOT=$3; SLOTS=$4; JOBS=$5; EXTRA=${6:-}
mkdir -p $OUT
cd $CODE
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "D56 probe ($JOBS; $SLOTS at once) starts at $(git rev-parse --short HEAD) $EXTRA"
run() {
  TASK=${1%:*}; SEED=${1#*:}
  CELL="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 1600 \
    --budget-minutes 690 --anneal --anneal-tail 40 --anneal-never-commit --warm-up --melt-hot 2 \
    --melt-kappa 1e-2 --melt-hold 20 --melt-halving 30 --melt-shape exponential --crucible \
    --warm-up-screen 1.01 --continuous --workers 1"
  case $TASK in
    s3) NAME=word-problem-s3; BOUND=6 ;;
    mtf4) NAME=move-to-front-4; BOUND=24 ;;
    rq16) NAME=random-q16-0; BOUND=16 ;;
  esac
  log "launch $TASK-seed$SEED"
  AB_RUNS_ROOT=$RUNS_ROOT-$TASK-$SEED env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    ${PY:-.venv/bin/python} scripts/modular_family_probe.py run --runs $NAME:$SEED $CELL \
    --state-bound $BOUND --out $OUT/$TASK-seed$SEED $EXTRA >> $OUT/$TASK-seed$SEED.out 2>&1
  log "ended $TASK-seed$SEED: $(tail -1 $OUT/$TASK-seed$SEED.out)"
}
export -f run log
export OUT RUNS_ROOT EXTRA
echo $JOBS | tr ' ' '\n' | xargs -P $SLOTS -I{} bash -c 'run {}'
log "probe ended"
