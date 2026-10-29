#!/bin/bash
# The state formation workplan's option 1 panel (stage 4, D11; a probe set, its expectations written before
# its data): `--anneal --warm-up` (a continuous warm-up, then codes by k-means of the states seen, annealed
# from tau_0, D7'), round 1's and 2's cell (two factors, --codes-ratio 32/11, reading the output, the fork,
# 600 epochs) at --window-width 32; division 13 W1's cell (--factor 32 38 twice). Code: indrajala-ml #891
# (c875b1a), crate 60dc376. Split across stone (harm, mtf-5, S5) and pyramidon (S3, mtf-4 seed 3, controls).
# Feasibility (docs/information-flow.md): S3 q 6, 8 codes a factor, m 64, cap 69; S4 and mtf-4 q 24, 15,
# 225, 248; S5 and mtf-5 q 120, 32, 1,024, 1,143; division q 169, 38, 1,444, 1,612; width 32 certified each.
# Expectations (the plan's stage 2 findings):
#  - failure A: S3 seed 0 and mtf-4 seed 3 certify (as with the commitment never on); S3 seeds 1 and 4 not
#    expected (not dimensional collapse); mtf-5 seeds 0, 1 form a state (share >= 0.5), certifying a bonus.
#  - harm: S4 seeds 0, 1, mtf-4 seed 0, division seeds 0, 1 all certify (as with the commitment on: C0, W1).
#  - bottleneck 2: S5 seed 1 descriptive (C0 failed it at 32; C2's R3/R4 certified it).
#  - controls (pyramidon): S3 seeds 0, 1, 4 at 32 without the option, descriptive (round 1 at 16: failure A).
# D12: accepted if more failure-A runs certify than their controls (C0's 0 of 3 on mtf-4 3, mtf-5 0, 1; the
# S3 controls here), no harm run loses its certificate, and every certificate passes both verdicts; rejected
# on a lost harm certificate or no failure-A run moved; otherwise one follow-up from the readings, then decide.
set -u
CODE=$HOME/code/wt-warm
OUT=$HOME/code/indrajala-ml/data/window-recurrent/state-formation/option1/pyramidon
mkdir -p $OUT
cd $CODE
PROBE="env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 .venv/bin/python scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "option 1 panel (pyramidon): control check starts at $(git rev-parse --short HEAD)"
# without the flags this commit repeats C0's word-problem-s4 seed 0 by bits
K=$OUT/bits-check
AB_RUNS_ROOT=$HOME/code/ab-runs-option1 $PROBE --runs word-problem-s4:0 --factors 2 --codes-ratio 32/11 --reads output --fork \
  --window-width 32 --budget-epochs 8 --budget-minutes 20 --workers 1 --out $K > $OUT/bits-check.out 2>&1
L=$K/word-problem-s4/seed0/log.jsonl
[ -f $L ] || L=$K/seed0/log.jsonl
n=$(( $(wc -l < $L) - 1 ))
strip() { sed -E 's/, "seconds": [0-9.]+//; s/, "machine": \{[^}]*\}//'; }
R=$HOME/code/indrajala-ml/data/window-recurrent/one-code-one-state/c0/word-problem-s4/seed0/log.jsonl
if [ "$n" -gt 2 ] && diff <(head -n $n $L | strip) <(head -n $n $R | strip) > $K/diff; then
  log "control check: word-problem-s4 seed 0 repeated C0 by bits ($n rows)"
else
  log "control check FAILED: see $K/diff; panel stopped"; exit 1
fi
log "panel starts"
BASE="--factors 2 --codes-ratio 32/11 --reads output --fork --window-width 32 --budget-epochs 600 --budget-minutes 900"
AB_RUNS_ROOT=$HOME/code/ab-runs-option1 $PROBE --runs word-problem-s3:0 word-problem-s3:1 word-problem-s3:4 move-to-front-4:3 \
  $BASE --anneal --warm-up --workers 2 --out $OUT/option >> $OUT/option.out 2>&1 &
AB_RUNS_ROOT=$HOME/code/ab-runs-option1-controls $PROBE --runs word-problem-s3:0 word-problem-s3:1 word-problem-s3:4 \
  $BASE --workers 2 --out $OUT/controls >> $OUT/controls.out 2>&1 &
wait
log "panel ended"
