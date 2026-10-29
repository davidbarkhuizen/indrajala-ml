#!/bin/bash
# The D33 existence probe's last two runs (probe-d33.sh's settings unchanged; only their scheduling): its
# launcher was stopped at about 08:30 with seeds 6 and 7 not started (5 at once, the running five at their
# 690-minute budgets till about 19:00); seed 6 now (6 jobs, half of stone by day), seed 7 at 18:00 (all
# of stone). Same arguments as probe-d33.sh.
set -u
CODE=$1; OUT=$2; RUNS_ROOT=$3
cd $CODE
PROBE="env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ${PY:-.venv/bin/python} scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
BASE="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 1600 --budget-minutes 690 \
  --crucible --anneal --anneal-tail 40 --anneal-never-commit --warm-up --handover --melt-hot 2 --melt-kappa 1e-2 \
  --melt-hold 20 --workers 1 --warm-up-screen 1.01 --melt-gate 0.99 --handover-octaves 3"
ADAPTIVE="--melt-shape adaptive --cool-pace 0.03 --cool-settle 10 --cool-within 0.75"
launch() {
  log "launch $1 (probe-d33-rest.sh at $(git rev-parse --short HEAD))"
  AB_RUNS_ROOT=$RUNS_ROOT-$1 $PROBE --runs $2 $BASE $ADAPTIVE --out $OUT/$1 >> $OUT/$1.out 2>&1 &
}
launch s3-seed6 word-problem-s3:6
now=$(date +%s); at=$(date -d 18:00 +%s); [ $at -gt $now ] && sleep $((at - now))
launch s3-seed7 word-problem-s3:7
wait
log "probe-d33-rest.sh ended"
