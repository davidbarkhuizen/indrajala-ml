#!/bin/bash
# The state formation workplan's D59, the margin stop's probe (expectations written before its data,
# 2026-10-22, 04:20; the owner: "yes, register the waiting stop probe"). D54's cell with D35's stop
# waiting for its margins (--margin-wait 200): a step D35's stop passes ends the run only when no near
# tie splits the snapped machine (the certificate's split_ties, read without the target) and no label
# is a near tie; at most 200 steps after the first such pass ("waited"), whose network is kept.
# Why: of D54's and D55's runs, 12 stopped at D35's first pass exact at the strict verdict but not the
# default (D41's success needs both): every one a near tie's branch into another state (1-13 wrong
# states, 2-188 near ties); on D55's 19 stopped networks split_ties > 0 on each of the 9 early stops
# and 0 on each of the 10 certified (up to 125 near ties), so the rule reads D41's default verdict
# there without the target.
# Population: the 12, each re-run from its seed (bit-identical to its first run up to the first pass,
# the replay a check): word-problem-s3 seeds 27, 29, 33, 35, 36 and move-to-front-4 seeds 24, 26, 29,
# 30 (D55's), move-to-front-4 seeds 1, 2, 3 (D54's). Feasibility as D54's; up to 1,600 epochs or 690
# minutes a run.
# Expectations (read with population, the per-epoch readings, the first-pass and stop networks):
#  - at least 8 of 12 succeed (D41) on the margin stop's network; refuted at 4 or fewer;
#  - the wait's harm: a run whose first pass was exact at 1.0 and whose stop's network isn't is a
#    lost machine; refuted if the lost outnumber the gained;
#  - any false discovery (a stop that certifies at neither verdict, the run never certifying) refutes;
#  - read: the steps waited, split_ties and the near ties over the wait, the trained length; each
#    run's rows up to its first pass against its first run's (bit-identical, but for the margin's).
#
# Usage: probe-d59.sh CODE OUT RUNS_ROOT SLOTS "TASK:SEED ..." [--resume]   (TASK s3 or mtf4)
set -u
CODE=$1; OUT=$2; RUNS_ROOT=$3; SLOTS=$4; JOBS=$5; EXTRA=${6:-}
mkdir -p $OUT
cd $CODE
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "D59 probe ($JOBS; $SLOTS at once) starts at $(git rev-parse --short HEAD) $EXTRA"
run() {
  TASK=${1%:*}; SEED=${1#*:}
  CELL="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 1600 \
    --budget-minutes 690 --anneal --anneal-tail 40 --anneal-never-commit --warm-up --melt-hot 2 \
    --melt-kappa 1e-2 --melt-hold 20 --melt-halving 30 --melt-shape exponential --crucible \
    --warm-up-screen 1.01 --continuous --workers 1 --margin-wait 200"
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
