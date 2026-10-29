#!/bin/bash
# The state formation workplan's D67, the deadline's completion (expectations written before its data,
# 2026-10-23, 06:45; the owner: "make epochs the binding budget from now on - that approach is
# thermodynamically consistent"; "systematically work through and re-process any data that might be
# affected").
# Why: a run's budget is its epochs (1,600); the 690-minute wall-clock budget was meant only as a guard
# but bound ten runs of the two probes still being read, cutting them short of their epoch budget on a
# loaded machine: D64's word-problem-s3 45 (1,584 epochs), move-to-front-4 34 (918) and 35 (1,135);
# D63's word-problem-s4 2-5 (892-1,091) and random-q16-0 4 (1,336), 5 (1,330) and 6. A deadline's stop
# wrote its checkpoint "ended", so a run can't be resumed; its seed run again from the start, the same
# cell at its probe's commit with the wall clock lifted (--budget-minutes 100000, a guard nothing
# normal reaches; the old code's default is 60), is the same run (every row but the seconds, bit
# parity across machines) up to the deadline, then runs on to its epoch budget.
# Population: those ten. D64's at 41f92e7 (its cell: D54's continuous crystallisation, --margin-wait
# 200), D63's at 0fdf237 (D64's cell plus --reheat). Not re-run: D33's and D38's cut runs (their
# verdicts stand on the runs that ended, their designs superseded) and D59's move-to-front-4 3 (its
# success by its rules holds with it censored); each re-read as censored in the workplan.
# Feasibility as D63's and D64's (docs/information-flow.md): S3 q 6 (8 codes, cap 69), mtf-4 and S4 q 24
# (15 codes, cap 248), random-q16-0 q 16 (12 codes, cap 159), width 32; 1,600 epochs a run.
# Expectations (read with cool, same-prefix, restart-cost, formation):
#  - integrity: each completion's rows identical to its cut run's (same-prefix, seconds aside) through
#    the cut run's last row before its deadline; a difference before it voids that completion;
#  - each completion's outcome at its epoch budget (D41: a certificate at both verdicts) stands in for
#    its cut run in its probe's reading: D64's S3 and move-to-front-4 rate, at least 8 of 12 met if any
#    of its three certifies, 7 of 12 (between its bars) if none; D63's paired verdict at equal epochs;
#  - any false discovery (a stop certifying at neither verdict, the run never certifying) refutes the
#    stop (D35 with D59-D60), as in both probes;
#  - read apart: whether a run frozen at its deadline (the clock's rate held at 0, cool) thaws in the
#    epochs after it, and when (the freeze's permanence: restarts' and the reheat's case).
#
# Usage: probe-d67.sh CODE OUT RUNS_ROOT MAX "PROBE:TASK:SEED ..."
#   PROBE d64 or d63 (its cell; CODE that probe's worktree, one PROBE per invocation); TASK s3, mtf4,
#   s4 or rq16; MAX the machine's runs at once (any modular_family_probe.py run), a queued run launched
#   only below it.
set -u
CODE=$1; OUT=$2; RUNS_ROOT=$3; MAX=$4; JOBS=$5
mkdir -p $OUT
cd $CODE
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "D67 probe ($JOBS; at most $MAX runs on the machine) starts at $(git rev-parse --short HEAD)"
run() {
  IFS=: read -r PROBE TASK SEED <<< "$1"
  CELL="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 1600 \
    --budget-minutes 100000 --anneal --anneal-tail 40 --anneal-never-commit --warm-up --melt-hot 2 \
    --melt-kappa 1e-2 --melt-hold 20 --melt-halving 30 --melt-shape exponential --crucible \
    --warm-up-screen 1.01 --continuous --workers 1 --margin-wait 200"
  [ "$PROBE" = d63 ] && CELL="$CELL --reheat"
  case $TASK in
    s3) NAME=word-problem-s3; BOUND=6 ;;
    mtf4) NAME=move-to-front-4; BOUND=24 ;;
    s4) NAME=word-problem-s4; BOUND=24 ;;
    rq16) NAME=random-q16-0; BOUND=16 ;;
  esac
  log "launch $PROBE-$TASK-seed$SEED"
  AB_RUNS_ROOT=$RUNS_ROOT-$PROBE-$TASK-$SEED env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    ${PY:-.venv/bin/python} scripts/modular_family_probe.py run --runs $NAME:$SEED $CELL \
    --state-bound $BOUND --out $OUT/$PROBE-$TASK-seed$SEED >> $OUT/$PROBE-$TASK-seed$SEED.out 2>&1
  log "ended $PROBE-$TASK-seed$SEED: $(tail -1 $OUT/$PROBE-$TASK-seed$SEED.out)"
}
# a run is its python process alone: its forkserver children carry its arguments too
running() { pgrep -f "^[^ ]*/python[0-9.]* scripts/[m]odular_family_probe.py run" | wc -l; }
for J in $JOBS; do
  while [ $(running) -ge $MAX ]; do sleep 30; done
  run $J &
  sleep 20  # the run's process appears before the next count
done
wait
log "probe ended"
