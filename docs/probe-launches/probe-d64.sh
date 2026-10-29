#!/bin/bash
# The state formation workplan's D64, the method's fresh-seed rate (expectations written before its
# data, 2026-10-22, 09:00; the owner: "you can run stone at a half-load today"). The current method,
# unchanged: D54's cell (D48's continuous crystallisation) with the margin stop under its deadline
# (--margin-wait 200; D59, D60), no D61 arm, on seeds no run has touched.
# Why: the method is statistical (the owner, 2026-10-22: "not expect every seed to always work, just
# for the method to work in general"): a remedy (D61's arms, D63) is adopted only on a fresh-seed rate
# against the current method's, and the current rate is known only from D59's re-runs of D55's seeds
# (S3 11 of 16, move-to-front-4 7 of 8). And "in general" means every backdrop class: the unstructured
# and larger machines are the gap (random-q16-0 0 of 4, D56; random-q12-0 1 of 4, D58; S4 1 of 3, D57),
# so the batch weights them.
# Population (22 runs): word-problem-s3 40-47, move-to-front-4 32-35, word-problem-s4 3-5,
# random-q16-0 4-6 on stone (6 at once by day, 10 at night, launched longest first); random-q12-0 4-7
# on klip (3 at once). Feasibility as D54's, D56's, D57's and D58's (docs/information-flow.md): S3 q 6
# (8 codes, cap 69), mtf-4 and S4 q 24 (15 codes, cap 248), random-q12-0 q 12 (11 codes, cap 132),
# random-q16-0 q 16 (12 codes, cap 159), width 32; up to 1,600 epochs or 690 minutes a run.
# Expectations (read with population, the per-epoch readings, the passes' and stop's networks):
#  - each backdrop's rate (D41) with its Wilson 95% interval; S3 and move-to-front-4 together, a
#    clear majority (at least 8 of 12) as the method's working rate, refuted as "working in general"
#    on these two at 6 or fewer;
#  - the larger and unstructured backdrops read, not barred (S4 3, random-q12-0 4, random-q16-0 3):
#    their rates against S3's, the scaling question; 0 on all ten refutes nothing but names the gap;
#  - any false discovery (a stop certifying at neither verdict, the run never certifying) refutes the
#    stop (D35 with D59-D60);
#  - read: each failure's kind and its order (the first coupling move of 0.1 or more against the
#    screen's first fall below 0.9; the frozen clock's epochs after), D63's motive sized; the margin's
#    waits (D59), the deadline's stops (D60); the seconds to each certificate (cost per machine).
# The same seeds are the pairs for any later arm (D61's, D63's) batch.
#
# Usage: probe-d64.sh CODE OUT RUNS_ROOT DAY/NIGHT "TASK:SEED ..." [--resume]
#   TASK s3, mtf4, s4, rq12 or rq16; DAY/NIGHT the runs at once 07:00-18:00 and after, a queued run
#   launched only below the hour's.
set -u
CODE=$1; OUT=$2; RUNS_ROOT=$3; SLOTS=$4; JOBS=$5; EXTRA=${6:-}
mkdir -p $OUT
cd $CODE
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "D64 probe ($JOBS; $SLOTS at once) starts at $(git rev-parse --short HEAD) $EXTRA"
run() {
  TASK=${1%:*}; SEED=${1#*:}
  CELL="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 1600 \
    --budget-minutes 690 --anneal --anneal-tail 40 --anneal-never-commit --warm-up --melt-hot 2 \
    --melt-kappa 1e-2 --melt-hold 20 --melt-halving 30 --melt-shape exponential --crucible \
    --warm-up-screen 1.01 --continuous --workers 1 --margin-wait 200"
  case $TASK in
    s3) NAME=word-problem-s3; BOUND=6 ;;
    mtf4) NAME=move-to-front-4; BOUND=24 ;;
    s4) NAME=word-problem-s4; BOUND=24 ;;
    rq12) NAME=random-q12-0; BOUND=12 ;;
    rq16) NAME=random-q16-0; BOUND=16 ;;
  esac
  log "launch $TASK-seed$SEED"
  AB_RUNS_ROOT=$RUNS_ROOT-$TASK-$SEED env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    ${PY:-.venv/bin/python} scripts/modular_family_probe.py run --runs $NAME:$SEED $CELL \
    --state-bound $BOUND --out $OUT/$TASK-seed$SEED $EXTRA >> $OUT/$TASK-seed$SEED.out 2>&1
  log "ended $TASK-seed$SEED: $(tail -1 $OUT/$TASK-seed$SEED.out)"
}
limit() { H=$((10#$(date +%H))); if [ $H -ge 7 ] && [ $H -lt 18 ]; then echo ${SLOTS%/*}; else echo ${SLOTS#*/}; fi; }
for J in $JOBS; do
  while [ $(jobs -rp | wc -l) -ge $(limit) ]; do sleep 30; done
  run $J &
done
wait
log "probe ended"
