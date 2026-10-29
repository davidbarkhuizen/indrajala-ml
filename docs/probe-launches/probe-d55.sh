#!/bin/bash
# The state formation workplan's D55, the continuous crystallisation's fresh-seed matrix (expectations
# written before its data, 2026-10-21, 20:50). D54's cell unchanged (D48's method, --continuous), on
# seeds no run has touched: word-problem-s3 seeds 24-39 and move-to-front-4 seeds 24-31 (D54 used 0-7
# and 0-3, the stopped core matrix (D46) 8-23).
# Feasibility as D54's (docs/information-flow.md): S3 q 6, 8 codes a factor, cap 69, width 32; mtf-4
# q 24, 15 codes a factor, cap 248. Up to 1,600 epochs or 690 minutes a run.
# Expectations (read as D54's: population, the per-epoch readings, the stops' networks):
#  - S3: at least 12 of 16 succeed (D41), D54's 6 of 8; refuted at 8 or fewer (no better than the
#    adaptive cool's (D33) bar of half);
#  - move-to-front-4: at least 2 of 8 succeed; the strict-only certifications (D35's stop on near
#    ties) counted apart; 0 of 8 is read, not a refutation;
#  - any false discovery (a stop the certificate rejects at both verdicts in a run that never
#    certifies) refutes;
#  - each rate reported with its Wilson 95% interval; the mechanics read as D54's (the coupling's rises
#    against the pre-continuous 0.05, the trained lengths), each failure classified by kind.
#
# Usage: probe-d55.sh CODE OUT RUNS_ROOT SLOTS "TASK:SEED ..." [--resume]
#   TASK s3 or mtf4; SLOTS runs at once, the rest queued; on stone, stopped at 06:55 by the caller
#   and resumed the next evening with --resume.
set -u
CODE=$1; OUT=$2; RUNS_ROOT=$3; SLOTS=$4; JOBS=$5; EXTRA=${6:-}
mkdir -p $OUT
cd $CODE
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "D55 matrix ($JOBS; $SLOTS at once) starts at $(git rev-parse --short HEAD) $EXTRA"
run() {
  TASK=${1%:*}; SEED=${1#*:}
  CELL="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 1600 \
    --budget-minutes 690 --anneal --anneal-tail 40 --anneal-never-commit --warm-up --melt-hot 2 \
    --melt-kappa 1e-2 --melt-hold 20 --melt-halving 30 --melt-shape exponential --crucible \
    --warm-up-screen 1.01 --continuous --workers 1"
  case $TASK in
    s3) NAME=word-problem-s3; BOUND=6 ;;
    mtf4) NAME=move-to-front-4; BOUND=24 ;;
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
