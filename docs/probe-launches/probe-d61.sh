#!/bin/bash
# The state formation workplan's D61, the paced coupling's probe (expectations written before its data,
# 2026-10-22, 08:00; the owner: "i would like to test both b and c", each with no fixed scale: "we need
# something dynamically calculated"). D54's cell with the margin stop (--margin-wait 200) under D60's
# deadline (the wait counting every step past the first pass, ending on the best pass's network), in two
# arms:
#  - pace (b, --coupling-pace): each coupling moves toward its sharpness's at most its allowance an
#    epoch, the allowance its first rise, then times 2 ** (2 rate - 1) (doubled while the screen
#    holds, halved when the clock stops): a trust region the network's response sets;
#  - smooth (c, --sharpness-smoothing; the owner: "swap c to savitzky-golay"): the coupling read from
#    the sharpness smoothed by a one-sided Savitzky-Golay filter, its least-squares line over the
#    readings of the cool's last e-folding time of clock (30 / ln 2), read at the clock's end. Offline
#    on D54-D59's logs it removed every move of 0.1 or more with near-zero lag (the moving average
#    first proposed held the coupling 0.16-0.53 below).
# Why: the coupling, read straight from the sharpness (D50), jumps when a split moves it. Every run
# that spent its budget (D54 S3 1, 4; D55 S3 24, 25; D56 random-q16-0 0-3) moved it by 0.1 or more in
# an epoch 58-201 times (largest 0.34-0.77); the certified ones read at most 11 (largest 0.20); S3 35
# (D59) lost D35's pass two steps after a rise of 0.23 and never passed again (143 such moves).
# Population, each arm (16 runs): the failures word-problem-s3 24, 25 (D55), 35 (D59), random-q16-0 0,
# 1 (D56), move-to-front-4 3 (D54; lost its pass, 6 such moves: the jumps' counter-case); the clean
# word-problem-s3 26, move-to-front-4 24 (certified, few moves). Each control is its seed's earlier run.
# Feasibility as D54's and D56's; up to 1,600 epochs or 690 minutes a run.
# Expectations (read with population, the per-epoch readings, the passes' and stop's networks):
#  - mechanism, each arm: the coupling's largest move an epoch below 0.1 in every run's cool, and the
#    moves of 0.1 or more fewer than its control's in each failure;
#  - outcome, each arm (a diagnosis, not adoption: the owner, "only a clear majority ... the final
#    method [is] statistical"; adoption only on a fresh-seed rate against the current method's, S3
#    11 of 16 and move-to-front-4 7 of 8 on D55's seeds with the margin stop, registered after this):
#    its mechanism promising if it rescues (D41, on the stop's network) at least 2 of
#    the 6 failures and loses neither clean seed; refuted if it rescues none or loses both; between,
#    read; the two arms compared on the same seeds;
#  - any false discovery (a stop certifying at neither verdict, the run never certifying) refutes the
#    stop (D35 with D59-D60);
#  - D60: every run ends at its first pass with no split or label tie, or 200 steps after its first
#    pass on its best pass's network;
#  - read: each run's rows up to the cool's start against its control's (bit-identical, but for the
#    arm's new fields), the trained length, the steps to the stop, the screen's drops.
#  - prediction (the failures' order: the first coupling move of 0.1 or more against the screen's first
#    fall below 0.9): jump first in S3 35, the same epoch in S3 25 and random-q16-0 0, 1, drop first in
#    S3 24 (screen 0.74 at coupling 0): an arm rescues the first two kinds, not S3 24 (the counter-case);
#    the frozen clock after a drop (the rate at 0), shared by every failure, neither arm changes.
#
# Usage: probe-d61.sh CODE OUT RUNS_ROOT DAY/NIGHT "ARM:TASK:SEED ..." [--resume]
#   ARM pace or smooth; TASK s3, mtf4 or rq16; DAY/NIGHT the runs at once 06:00-18:00 and after (jebel:
#   8/4), a queued run launched only below the hour's.
set -u
CODE=$1; OUT=$2; RUNS_ROOT=$3; SLOTS=$4; JOBS=$5; EXTRA=${6:-}
mkdir -p $OUT
cd $CODE
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "D61 probe ($JOBS; $SLOTS at once) starts at $(git rev-parse --short HEAD) $EXTRA"
run() {
  ARM=${1%%:*}; REST=${1#*:}; TASK=${REST%:*}; SEED=${REST#*:}
  CELL="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 1600 \
    --budget-minutes 690 --anneal --anneal-tail 40 --anneal-never-commit --warm-up --melt-hot 2 \
    --melt-kappa 1e-2 --melt-hold 20 --melt-halving 30 --melt-shape exponential --crucible \
    --warm-up-screen 1.01 --continuous --workers 1 --margin-wait 200"
  case $ARM in
    pace) FLAG=--coupling-pace ;;
    smooth) FLAG=--sharpness-smoothing ;;
  esac
  case $TASK in
    s3) NAME=word-problem-s3; BOUND=6 ;;
    mtf4) NAME=move-to-front-4; BOUND=24 ;;
    rq16) NAME=random-q16-0; BOUND=16 ;;
  esac
  ID=$ARM-$TASK-seed$SEED
  log "launch $ID"
  AB_RUNS_ROOT=$RUNS_ROOT-$ARM-$TASK-$SEED env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    ${PY:-.venv/bin/python} scripts/modular_family_probe.py run --runs $NAME:$SEED $CELL $FLAG \
    --state-bound $BOUND --out $OUT/$ID $EXTRA >> $OUT/$ID.out 2>&1
  log "ended $ID: $(tail -1 $OUT/$ID.out)"
}
limit() { H=$((10#$(date +%H))); if [ $H -ge 6 ] && [ $H -lt 18 ]; then echo ${SLOTS%/*}; else echo ${SLOTS#*/}; fi; }
for J in $JOBS; do
  while [ $(jobs -rp | wc -l) -ge $(limit) ]; do sleep 30; done
  run $J &
done
wait
log "probe ended"
