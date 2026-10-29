#!/bin/bash
# The panel's (4b) cost, before its registration: seconds an epoch and peak memory under --whole (D36's
# second arm, and the crucible arm's growth), ONE CELL AT A TIME on pyramidon (the first try ran four at
# once and swapped: the owner, "be careful about memory"). Not a study: no claims, timing only (the
# warm-up's continuous epochs, every length 1 to the cap from the first). S4 seed 0 first, the reference
# (D24: 61 s an epoch on stone, two at once). A watchdog kills a run whose machine's available memory
# falls below 800 MB, before it can swap.
set -u
OUT=${1:-$HOME/code/indrajala-ml/data/window-recurrent/state-formation/option1/panel-cost-serial}
mkdir -p $OUT
cd $HOME/code/indrajala-ml
PROBE="env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 .venv/bin/python scripts/modular_family_probe.py run"
CELL="--reads output --window-width 32 --budget-epochs 3 --budget-minutes 60 --workers 1 --whole \
  --anneal --anneal-tail 40 --anneal-never-commit --warm-up --handover --melt-hot 2 --melt-kappa 1e-2 \
  --melt-hold 20 --melt-cool 600 --melt-octaves 20 --melt-halving 30 --melt-shape convex \
  --warm-up-screen 1.01 --melt-gate 0.99 --handover-octaves 3"
log() { echo "$* $(date)" >> $OUT/launch.log; }
available() { awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo; }
timed() {  # name, then the run's arguments; one at a time, peak RSS by /usr/bin/time, the watchdog beside it
  local name=$1; shift
  log "start $name (available $(available) MB, swap used $(free -m | awk '/Swap/ {print $3}') MB)"
  setsid /usr/bin/time -f "peak_rss_kb %M wall %e" -o $OUT/$name.time \
    env AB_RUNS_ROOT=$HOME/code/ab-runs-panel-cost-$name $PROBE "$@" $CELL --out $OUT/$name > $OUT/$name.out 2>&1 &
  local pid=$!
  local low=100000
  while kill -0 $pid 2>/dev/null; do
    local now; now=$(available)
    [ "$now" -lt "$low" ] && low=$now
    if [ "$now" -lt 800 ]; then
      log "watchdog: $name killed at available $now MB"
      kill -- -$pid 2>/dev/null
      break
    fi
    sleep 5
  done
  wait $pid
  log "end $name: $(cat $OUT/$name.time 2>/dev/null | tail -1), lowest available $low MB"
}
log "panel cost (serial) at $(git rev-parse --short HEAD)"
timed s4 --runs word-problem-s4:0 --factors 2 --codes-ratio 32/11
timed s5 --runs word-problem-s5:1 --factors 2 --codes-ratio 32/11
timed division-13 --p 13 --arm family --factor 32 38 --factor 32 38 --seed 0
log "ended"
