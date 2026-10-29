#!/bin/bash
# C2 (docs/one-code-one-state-workplan.md, registered before its data; on the base window per D4', width 32): R3 and
# R4 together (--keep-machine --counterexamples 0.25) on round 2's word-problem-s5 and move-to-front-5 (two
# factors, --codes-ratio 32/11, reading the output, the fork, 600 epochs, --window-width 32), seeds 0-5, 12 runs, 6
# at once on jebel. Its control: C0's same cells (0 of 6 each; S5 all bottleneck 2 at ~120 pairs per code, mtf-5
# five failure A and one near miss at 1.6). Helps: 4 or more of 6 on a cell (Fisher one-sided p 0.030); works: 1-3.
# Read every failure's bottleneck by pairs per code and state share. Feasibility: q 120, 32 codes a factor, m 1,024,
# cap 1,143 (the plan's table). From ~/code/wt-r34 at c452ed9 (C0's commit and venv, crate afa7204): without the
# flags the same commit repeats C0 by bits (checked first: seed 0 of each cell, 8 epochs; a failure stops C2).
cd ~/code/indrajala-ml
CODE=~/code/wt-r34
OUT=data/window-recurrent/one-code-one-state/c2
C0=$CODE/data/window-recurrent/one-code-one-state/c0
mkdir -p $OUT
PROBE="env PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 $CODE/.venv/bin/python $CODE/scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
K=$OUT/bits-check; rm -rf $K
$PROBE --runs word-problem-s5:0 move-to-front-5:0 --factors 2 --codes-ratio 32/11 --reads output --fork \
  --window-width 32 --budget-epochs 8 --budget-minutes 30 --workers 2 --out $K > $OUT/bits-check.out 2>&1
for task in word-problem-s5 move-to-front-5; do
  L=$K/$task/seed0/log.jsonl; R=$C0/$task/seed0/log.jsonl
  n=$(( $(wc -l < $L) - 1 ))
  strip() { sed -E 's/, "seconds": [0-9.]+//; s/, "machine": \{[^}]*\}//'; }
  if [ "$n" -gt 2 ] && diff <(head -n $n $L | strip) <(head -n $n $R | strip) > $K/$task.diff; then
    log "control check: $task seed 0 repeated C0 by bits ($n rows)"
  else log "control check FAILED on $task: see $K/$task.diff; C2 stopped"; exit 1; fi
done
log "C2 starts at $(git -C $CODE rev-parse --short HEAD)"
$PROBE --task word-problem-s5 move-to-front-5 --factors 2 --codes-ratio 32/11 --reads output --fork \
  --window-width 32 --keep-machine --counterexamples 0.25 --budget-epochs 600 --budget-minutes 900 \
  --first-seed 0 --seeds 6 --workers 6 --out $OUT >> $OUT/c2.out 2>&1
log "C2 ended exit $?"
