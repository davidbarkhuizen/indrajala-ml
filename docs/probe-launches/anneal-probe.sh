#!/bin/bash
# Annealing's probe (docs/commitment-control-workplan.md, stage 3, D6-D11; a probe, it claims nothing).
# Arm 1, the diagnostic's six failure-A runs (word-problem-s3 seeds 0, 1, 3, 4, 5; move-to-front-4 seed 3; round 1's
# cell: two factors, --codes-ratio 32/11, reading the output, the fork, 600 epochs, width 16) with --anneal (tau 1.0
# halved every 10 epochs, commitment 0 while soft, hard at 99% of windows above 0.99, commitment ramped back over 20).
# One change against round 1 (all six failure A), the diagnostic (commitment off for steps 1-5: 3 of 6 formed a state,
# S3 seed 5 certified) and the thermostat's two probes.
# Arm 2 (on stone: ~/labour/anneal-probe-stone.sh), move-to-front-5 seeds 0 and 1 at width 32 (round 2's cell, --window-width 32) with --anneal, against C0's
# same seeds (failure A at 32, provisional 17:30).
# Feasibility (docs/information-flow.md): S3 q 6, 8 codes a factor, m 64, cap 69; move-to-front-4 q 24, 15, 225, 248;
# move-to-front-5 q 120, 32, 1,024, 1,143; widths 16 and 32 each certified these cells' tasks before (round 1, C0).
# Expectations, written before its data (a run forms a state if its state share reaches 0.5 at its best):
#  - arm 1: 3 or more of 6 form a state and at least one certifies -> annealing does what the diagnostic's schedule did,
#    adaptively: it moves bottleneck 1, and is read on the scorecard's backdrops next. Fewer than 3 form -> the soft
#    phase isn't enough on its own; read when each factor hardened against when (if) its state formed, and whether a
#    state formed soft was lost on hardening or on the commitment's return.
#  - arm 2: either forms a state (C0's same seeds didn't) -> annealing moves failure A at 32 too.
# Runs from ~/code/wt-anneal (8e88c9d or its successor, the crate's soft-codes branch built into $SITE).
set -u
# after the thermostat's second probe (6 workers) ends
until grep -q "ended" ~/code/indrajala-ml/data/window-recurrent/commitment-control/thermostat-probe-2/launch.log; do sleep 60; done
cd ~/code/wt-anneal
SITE=/tmp/claude-1000/-home-david-code/e4ffafc9-db12-4b40-951c-48f07d056816/scratchpad/site
OUT=data/window-recurrent/commitment-control/anneal-probe
mkdir -p $OUT
PROBE="env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH=$SITE:. .venv/bin/python scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "anneal probe: control check starts at $(git rev-parse --short HEAD), crate $(git -C ~/code/wt-rust rev-parse --short HEAD)"
# without --anneal, this commit repeats round 1's S3 seed 0 by bits (every row but the budget stop, seconds and the
# machine rows aside); a failed check stops the probe
K=$OUT/bits-check
$PROBE --runs word-problem-s3:0 --factors 2 --codes-ratio 32/11 --reads output --fork --budget-epochs 8 \
  --budget-minutes 20 --workers 1 --out $K > $OUT/bits-check.out 2>&1
L=$K/word-problem-s3/seed0/log.jsonl
[ -f $L ] || L=$K/seed0/log.jsonl
n=$(( $(wc -l < $L) - 1 ))
strip() { sed -E 's/, "seconds": [0-9.]+//; s/, "machine": \{[^}]*\}//'; }
R=data/window-recurrent/general-algebras/round1/word-problem-s3/seed0/log.jsonl
if [ "$n" -gt 2 ] && diff <(head -n $n $L | strip) <(head -n $n $R | strip) > $K/diff; then
  log "control check: word-problem-s3 seed 0 repeated round 1 by bits ($n rows)"
else
  log "control check FAILED: see $K/diff; probe stopped"; exit 1
fi
log "arm 1 starts"
$PROBE --runs word-problem-s3:0 word-problem-s3:1 word-problem-s3:3 word-problem-s3:4 word-problem-s3:5 move-to-front-4:3 \
  --factors 2 --codes-ratio 32/11 --reads output --fork --anneal --budget-epochs 600 --budget-minutes 900 --workers 6 \
  --out $OUT/arm1 >> $OUT/arm1.out 2>&1
log "arm 1 ended exit $?"
log "anneal probe ended"
