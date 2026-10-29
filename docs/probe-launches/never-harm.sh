#!/bin/bash
# The commitment off for good, a harm check (docs/commitment-control-workplan.md; a probe, it claims nothing; the
# toolbox only, `--commitment-from 1000`). The never probe's arms 1 and 2 certified 4 of the 6 failure-A runs
# (word-problem-s3 seeds 0, 3, 5; move-to-front-4 seed 3) against at most 1 of 6 for every earlier remedy. Here, where
# the commitment was on and worked: C0's certified seeds at the base width 32, two a cell: word-problem-s4 seeds 0
# and 1 (C0: certified at steps 11 and 12), move-to-front-4 seeds 0 and 5 (C0: steps 12 and 26). Round 1's cell (two
# factors, --codes-ratio 32/11, reading the output, the fork, 600 epochs) at --window-width 32.
# Feasibility (docs/information-flow.md): S4 q 24, 15 codes a factor, m 225, cap 248; move-to-front-4 the same;
# width 32 certified both (C0).
# Expectations, written before its data:
#  - all 4 certify -> no harm seen where the commitment worked: the commitment off goes to the scorecard (S5,
#    mtf-5 with R3/R4, division 13 at 32), 2 seeds each, then 6 seeds for adoption (the owner's).
#  - 1-3 certify -> read each failure (its bottleneck, its snap margins: near ties are what the commitment
#    prevented); harm if a failure is a near-tie or margin failure the commitment would have avoided.
#  - 0 certify -> harm: the commitment is needed once a state exists; a schedule (off until formed, then on
#    gently) is the next probe, not off for good.
# On pyramidon from main beside annealing arm 1 (2 workers), its own lock directory. Control check: without the flag
# this commit repeats C0's word-problem-s4 seed 0 by bits (8 epochs); a failure stops the probe.
set -u
cd ~/code/indrajala-ml
export AB_RUNS_ROOT=$HOME/code/ab-runs-never-harm
C=data/window-recurrent/commitment-control
OUT=$C/never-harm
mkdir -p $OUT $AB_RUNS_ROOT
PROBE="env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 .venv/bin/python scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "never harm check: control check starts at $(git rev-parse --short HEAD)"
K=$OUT/bits-check
$PROBE --runs word-problem-s4:0 --factors 2 --codes-ratio 32/11 --reads output --fork --window-width 32 \
  --budget-epochs 8 --budget-minutes 20 --workers 1 --out $K > $OUT/bits-check.out 2>&1
L=$K/word-problem-s4/seed0/log.jsonl
[ -f $L ] || L=$K/seed0/log.jsonl
n=$(( $(wc -l < $L) - 1 ))
strip() { sed -E 's/, "seconds": [0-9.]+//; s/, "machine": \{[^}]*\}//'; }
R=data/window-recurrent/one-code-one-state/c0/word-problem-s4/seed0/log.jsonl
if [ "$n" -gt 2 ] && diff <(head -n $n $L | strip) <(head -n $n $R | strip) > $K/diff; then
  log "control check: word-problem-s4 seed 0 repeated C0 by bits ($n rows)"
else
  log "control check FAILED: see $K/diff; probe stopped"; exit 1
fi
log "never harm check starts"
$PROBE --runs word-problem-s4:0 word-problem-s4:1 move-to-front-4:0 move-to-front-4:5 --factors 2 --codes-ratio 32/11 \
  --reads output --fork --window-width 32 --commitment-from 1000 --budget-epochs 600 --budget-minutes 900 \
  --workers 2 --out $OUT/runs >> $OUT/runs.out 2>&1
log "never harm check ended exit $?"
