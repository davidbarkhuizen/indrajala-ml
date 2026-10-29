#!/bin/bash
# Annealing's probe, arm 2 (docs/commitment-control-workplan.md, stage 3; a probe, it claims nothing; arm 1 and the
# expectations are in ~/code/anneal-probe.sh on pyramidon, written before the data): move-to-front-5 seeds 0 and 1,
# round 2's cell (two factors, --codes-ratio 32/11, reading the output, the fork, 600 epochs) at --window-width 32
# with --anneal, against C0's same seeds (failure A at 32: best state share 0.25 and 0.21 at 18:20, still running).
# Expectation, written before its data: either forms a state (share 0.5 at its best) -> annealing moves failure A at
# 32 too. On stone beside R1's probe (its 2 free slots), its own lock directory.
# Control check first: without --anneal, this commit repeats C0's seed 0 by bits (every row but the budget stop,
# seconds and the machine rows aside); a failed check stops the probe.
set -u
source ~/labour/env.sh
export AB_RUNS_ROOT=$LABOUR/ab-runs-anneal
CODE=$LABOUR/code/wt-anneal
OUT=$LABOUR/data/commitment-control/anneal-probe
C=$LABOUR/data/commitment-control/controls
mkdir -p $OUT $AB_RUNS_ROOT
cd $LABOUR/code/indrajala-ml
PROBE="env PYTHONPATH=$LABOUR/site-soft-codes:$CODE OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 $CODE/../wt-branches/.venv/bin/python $CODE/scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "anneal probe arm 2: control check starts at $(git -C $CODE rev-parse --short HEAD)"
K=$OUT/bits-check
$PROBE --runs move-to-front-5:0 --factors 2 --codes-ratio 32/11 --reads output --fork --window-width 32 \
  --budget-epochs 8 --budget-minutes 30 --workers 1 --out $K > $OUT/bits-check.out 2>&1
L=$K/move-to-front-5/seed0/log.jsonl
[ -f $L ] || L=$K/seed0/log.jsonl
n=$(( $(wc -l < $L) - 1 ))
strip() { sed -E 's/, "seconds": [0-9.]+//; s/, "machine": \{[^}]*\}//'; }
if [ "$n" -gt 2 ] && diff <(head -n $n $L | strip) <(head -n $n $C/c0-move-to-front-5-seed0.log.jsonl | strip) > $K/diff; then
  log "control check: move-to-front-5 seed 0 repeated C0 by bits ($n rows)"
else
  log "control check FAILED: see $K/diff; probe stopped"; exit 1
fi
log "arm 2 starts"
$PROBE --runs move-to-front-5:0 move-to-front-5:1 --factors 2 --codes-ratio 32/11 --reads output --fork \
  --window-width 32 --anneal --budget-epochs 600 --budget-minutes 900 --workers 2 --out $OUT/arm2 >> $OUT/arm2.out 2>&1
log "arm 2 ended exit $?"
