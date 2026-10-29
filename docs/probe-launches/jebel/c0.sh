#!/bin/bash
# C0 (docs/one-code-one-state-workplan.md, the amendment of 2026-10-19 afternoon, registered in #879 before its
# data): the base window at width 32 (--window-width 32) on round 1's word-problem-s4 and move-to-front-4 and round
# 2's word-problem-s5 and move-to-front-5 (two factors, --codes-ratio 32/11, reading the output, the fork, 600
# epochs), seeds 0-5, 24 runs, 6 at once on jebel once B3 ends. Its controls: the same cells at width 16 (round 1:
# S4 4 of 6, move-to-front-4 4 of 6; round 2: 0 of 6 each), the default width repeating each by bits on this commit
# (checked first: seed 0 of each, 8 epochs; a failure stops C0). From ~/code/wt-r34 at c452ed9 (R2's pairs per code
# logged; wt-b2's venv, crate afa7204). Resume: c0.sh --resume.
cd ~/code/indrajala-ml
CODE=~/code/wt-r34
OUT=data/window-recurrent/one-code-one-state/c0
mkdir -p $OUT
until grep -q "B3 algebras ended" data/window-recurrent/window-branches/b3/launch.log 2>/dev/null; do sleep 60; done
PROBE="env PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 $CODE/.venv/bin/python $CODE/scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
if [ "$1" != "--resume" ]; then
  K=$OUT/bits-check; rm -rf $K
  $PROBE --runs word-problem-s4:0 move-to-front-4:0 word-problem-s5:0 move-to-front-5:0 --factors 2 --codes-ratio 32/11 \
    --reads output --fork --budget-epochs 8 --budget-minutes 20 --workers 4 --out $K > $OUT/bits-check.out 2>&1
  for pair in "word-problem-s4 round1" "move-to-front-4 round1" "word-problem-s5 round2" "move-to-front-5 round2"; do
    read -r task round <<< "$pair"  # not set --, which would replace the script's own arguments
    L=$K/$task/seed0/log.jsonl; R=data/window-recurrent/general-algebras/$round/$task/seed0/log.jsonl
    n=$(( $(wc -l < $L) - 1 ))
    if [ "$n" -gt 2 ] && diff <(head -n $n $L | sed -E 's/, "seconds": [0-9.]+//; s/, "machine": \{[^}]*\}//') \
        <(head -n $n $R | sed -E 's/, "seconds": [0-9.]+//') > $K/$task.diff; then log "control check: $task seed 0 repeated by bits ($n rows)"
    else log "control check FAILED on $task: see $K/$task.diff; C0 stopped"; exit 1; fi
  done
fi
log "C0 starts at $(git -C $CODE rev-parse --short HEAD)"
$PROBE --task word-problem-s4 move-to-front-4 word-problem-s5 move-to-front-5 --factors 2 --codes-ratio 32/11 \
  --reads output --fork --window-width 32 --budget-epochs 600 --budget-minutes 900 --first-seed 0 --seeds 6 \
  --workers 6 --out $OUT "$@" >> $OUT/c0.out 2>&1
log "C0 ended exit $?"
