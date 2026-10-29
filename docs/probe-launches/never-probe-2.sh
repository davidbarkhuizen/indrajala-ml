#!/bin/bash
# The never probe's second arm (docs/commitment-control-workplan.md; a probe, it claims nothing; the toolbox only).
# Its first arm (stone, 19:34-19:43): word-problem-s3 seeds 0 and 3 with `--commitment-from 1000` (the commitment
# never returns) both certified (steps 6 and 8, both verdicts recertified on pyramidon); seed 3 lost its state at
# steps 5-6 and re-formed it at 7, which no run did with the commitment back. Here the diagnostic's other four
# failure-A runs, the same cell (round 1's: two factors, --codes-ratio 32/11, reading the output, the fork, 600
# epochs, width 16): word-problem-s3 seeds 1, 4 and 5, move-to-front-4 seed 3. Controls: round 1 (all failure A),
# the diagnostic (off for steps 1-5: seeds 1 and 4 formed nothing, seed 5 certified at step 6, move-to-front-4
# seed 3 formed 0.87 and spent at 0.94), the thermostat's two probes (0 of 6 certified each).
# Feasibility (docs/information-flow.md): S3 q 6, 8 codes a factor, m 64, cap 69; move-to-front-4 q 24, 15, 225, 248.
# Expectations, written before its data:
#  - 2 or more of 4 certify -> with the first arm, 4 or more of the six failure-A runs certify with the commitment
#    off for good: a cure for bottleneck 1 on these backdrops, read next for harm where the commitment was on and
#    worked (S4, move-to-front-4's certified seeds) and on the larger cells.
#  - seeds 1 or 4 certify -> more than the diagnostic's schedule did (they formed nothing there).
#  - 0 or 1 of 4 -> the first arm's seeds were the diagnostic's formers: the commitment's return explains them, not
#    failure A at large.
# On pyramidon from main beside annealing arm 1 (2 workers), its own lock directory. Control check: with the flag,
# S3 seed 1's first rows repeat the diagnostic's by bits (steps 1-5 are its).
set -u
cd ~/code/indrajala-ml
export AB_RUNS_ROOT=$HOME/code/ab-runs-never
C=data/window-recurrent/commitment-control
OUT=$C/never-probe/arm2
mkdir -p $OUT $AB_RUNS_ROOT
PROBE="env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 .venv/bin/python scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "never probe arm 2: control check starts at $(git rev-parse --short HEAD)"
K=$OUT/bits-check
$PROBE --runs word-problem-s3:1 --factors 2 --codes-ratio 32/11 --reads output --fork --commitment-from 1000 \
  --budget-epochs 8 --budget-minutes 20 --workers 1 --out $K > $OUT/bits-check.out 2>&1
L=$K/word-problem-s3/seed1/log.jsonl
[ -f $L ] || L=$K/seed1/log.jsonl
n=$(( $(wc -l < $L) - 1 ))
strip() { sed -E 's/, "seconds": [0-9.]+//; s/, "machine": \{[^}]*\}//'; }
R=$C/diagnostic/word-problem-s3/seed1/log.jsonl
if [ "$n" -gt 2 ] && diff <(head -n $n $L | strip) <(head -n $n $R | strip) > $K/diff; then
  log "control check: word-problem-s3 seed 1 repeated the diagnostic by bits ($n rows)"
else
  log "control check FAILED: see $K/diff; probe stopped"; exit 1
fi
log "arm 2 starts"
$PROBE --runs word-problem-s3:1 word-problem-s3:4 word-problem-s3:5 move-to-front-4:3 --factors 2 --codes-ratio 32/11 \
  --reads output --fork --commitment-from 1000 --budget-epochs 600 --budget-minutes 900 --workers 2 --out $OUT/runs \
  >> $OUT/runs.out 2>&1
log "never probe arm 2 ended exit $?"
