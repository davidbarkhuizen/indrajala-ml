#!/bin/bash
# The state formation workplan's D58, the continuous crystallisation on the random control (expectations
# written before their data, 2026-10-22, 02:50). D54's cell unchanged (D48's method, --continuous), on
# random-q12-0 seeds 0-3: D37's control backdrop (the general algebras' consolidation certified 2 of 3
# at about 12 states); beside D56's random-q16-0, it says whether the method holds on unstructured
# machines the earlier arms could solve; on klip's slots once D55's slice ended there.
# Feasibility (docs/information-flow.md): q 12, 11 codes a factor, m 121 (10.1 q), cap 132, width 32;
# row 15: depth 3 + k 1, within the crucible's 8. Up to 1,600 epochs or 690 minutes a run; the state
# bound 12 (D35's stop).
# Expectations (read as D54's: population, the per-epoch readings, the stops' networks):
#  - existence: at least 2 of 4 succeed (D41), the earlier arms' rate; refuted at 0 of 4 (the method
#    loses what the earlier arms had on an unstructured machine); each failure classified by kind,
#    strict-only certifications apart;
#  - any false discovery refutes D35's stop on this backdrop;
#  - the mechanics read as D54's (the coupling's rises, the trained length).
#
# Usage: probe-d58.sh CODE OUT RUNS_ROOT SLOTS "TASK:SEED ..." [--resume]   (TASK rq12)
set -u
CODE=$1; OUT=$2; RUNS_ROOT=$3; SLOTS=$4; JOBS=$5; EXTRA=${6:-}
mkdir -p $OUT
cd $CODE
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "D58 probe ($JOBS; $SLOTS at once) starts at $(git rev-parse --short HEAD) $EXTRA"
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
    s4) NAME=word-problem-s4; BOUND=24 ;;
    rq12) NAME=random-q12-0; BOUND=12 ;;
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
