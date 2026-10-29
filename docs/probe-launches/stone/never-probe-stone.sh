#!/bin/bash
# The commitment never returns (docs/commitment-control-workplan.md, after the thermostat's two probes; a probe, it
# claims nothing; the toolbox only: `--commitment-from 1000`, past any run's last step). Word-problem-s3 seeds 0 and 3,
# round 1's cell (two factors, --codes-ratio 32/11, reading the output, the fork, 600 epochs, width 16), with one
# change against the diagnostic (`--commitment-from 6`): the commitment stays 0 after step 5. Steps 1-5 are the
# diagnostic's, so the runs continue it. In the diagnostic seed 0 formed a state (0.97 at step 5) that eroded when
# the commitment returned (0.56 at step 6, 0.43 by 10); seed 3 formed one (0.95 at step 4) and lost it at step 5 with
# the commitment still off (0.15). In the thermostat's second probe every state formed (4 of 6) eroded within 1-2
# steps of the commitment's return.
# Feasibility (docs/information-flow.md): S3 q 6, 8 codes a factor, m 64, cap 69 (round 1, the diagnostic).
# Expectations, written before its data (a run forms a state if its share reaches 0.5; keeps it if its share stays
# at 0.5 or more for 5 steps after it formed):
#  - seed 0 keeps its state past step 6, or either certifies -> the return erodes states, not training without the
#    commitment: the commitment off for good is a candidate, read next on the scorecard's backdrops (it gives up the
#    commitment's margins: the certificate's verdicts judge them).
#  - neither keeps a state -> states are lost without the commitment too: forming is the commitment's, keeping is
#    retention's (the fork, stop at certified), and the commitment isn't the remedy for that half.
# On stone beside R1's probes and annealing arm 2 (2 workers), its own lock directory, from wt-anneal (without
# --anneal it repeats round 1 by bits: arm 1's control check).
set -u
source ~/labour/env.sh
export AB_RUNS_ROOT=$LABOUR/ab-runs-never
CODE=$LABOUR/code/wt-anneal
OUT=$LABOUR/data/commitment-control/never-probe
C=$LABOUR/data/commitment-control/controls
mkdir -p $OUT $AB_RUNS_ROOT
cd $LABOUR/code/indrajala-ml
PROBE="env PYTHONPATH=$LABOUR/site-soft-codes:$CODE OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 $CODE/../wt-branches/.venv/bin/python $CODE/scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "never probe: control check starts at $(git -C $CODE rev-parse --short HEAD)"
# with --commitment-from 1000, this commit repeats the diagnostic's S3 seed 0 by bits through its first epochs
K=$OUT/bits-check
$PROBE --runs word-problem-s3:0 --factors 2 --codes-ratio 32/11 --reads output --fork --commitment-from 1000 \
  --budget-epochs 8 --budget-minutes 20 --workers 1 --out $K > $OUT/bits-check.out 2>&1
L=$K/word-problem-s3/seed0/log.jsonl
[ -f $L ] || L=$K/seed0/log.jsonl
n=$(( $(wc -l < $L) - 1 ))
strip() { sed -E 's/, "seconds": [0-9.]+//; s/, "machine": \{[^}]*\}//'; }
if [ "$n" -gt 2 ] && diff <(head -n $n $L | strip) <(head -n $n $C/diagnostic-s3-seed0.log.jsonl | strip) > $K/diff; then
  log "control check: word-problem-s3 seed 0 repeated the diagnostic by bits ($n rows)"
else
  log "control check FAILED: see $K/diff; probe stopped"; exit 1
fi
log "never probe starts"
$PROBE --runs word-problem-s3:0 word-problem-s3:3 --factors 2 --codes-ratio 32/11 --reads output --fork \
  --commitment-from 1000 --budget-epochs 600 --budget-minutes 900 --workers 2 --out $OUT/runs >> $OUT/runs.out 2>&1
log "never probe ended exit $?"
