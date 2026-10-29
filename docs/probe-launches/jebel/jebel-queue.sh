#!/bin/bash
# jebel's stage 6 queue (docs/window-branches-workplan.md), after B2: W2, then B3's general-algebras cells.
# Each claim's by-bits control rule is checked first, on ~/code/wt-b2 at 393b3b1 (crate afa7204): a few epochs
# of seed 0 without the claim's change must repeat its control's log (every row but the budget stop, seconds
# aside). A failed check stops the queue before that claim, its diff kept; nothing is launched past it.
cd ~/code/indrajala-ml
CODE=~/code/wt-b2
WB=data/window-recurrent/window-branches
Q=$WB/queue.log
PROBE="env PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 $CODE/.venv/bin/python $CODE/scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $Q; }
same() {  # same CHECK_LOG CONTROL_LOG: every row of the check but its last equals the control's, seconds aside
  local n=$(( $(wc -l < "$1") - 1 ))
  [ "$n" -gt 2 ] && diff <(head -n $n "$1" | sed -E 's/, "seconds": [0-9.]+//') \
    <(head -n $n "$2" | sed -E 's/, "seconds": [0-9.]+//') > "$1.diff"
}
until grep -q "B2 ended" $WB/b2/launch.log; do sleep 60; done
log "queue starts at $(git -C $CODE rev-parse --short HEAD)"

# W2: L3's cell, the window 22 wide. Control L3 (stone, 440a680): its default width must repeat L3 here.
C=$WB/w2-bits-check; rm -rf $C; mkdir -p $C
$PROBE --p 13 --arm family --factor 32 38 --factor 32 38 --reads output --fork --budget-epochs 4 \
  --budget-minutes 20 --first-seed 0 --seeds 1 --workers 1 --out $C > $C/check.out 2>&1
if same $C/seed0/log.jsonl $WB/controls/l3-seed0.log.jsonl; then
  log "W2's control check: L3 seed 0 repeated by bits ($(( $(wc -l < $C/seed0/log.jsonl) - 1 )) rows)"
else log "W2's control check FAILED: see $C/seed0/log.jsonl.diff; queue stopped"; exit 1; fi
OUT=$WB/w2; mkdir -p $OUT
echo "W2 starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch.log
$PROBE --p 13 --arm family --factor 32 38 --factor 32 38 --reads output --fork --window-width 22 \
  --budget-epochs 600 --budget-minutes 900 --first-seed 0 --seeds 6 --workers 6 --out $OUT >> $OUT/w2.out 2>&1
echo "W2 ended $(date) exit $?" >> $OUT/launch.log

# B3's general-algebras cells: round 1's cell with --branches. Control round 1 (5a89bfe).
C=$WB/b3-bits-check; rm -rf $C; mkdir -p $C
$PROBE --runs word-problem-s4:0 move-to-front-4:0 --factors 2 --codes-ratio 32/11 --reads output --fork \
  --budget-epochs 8 --budget-minutes 20 --workers 2 --out $C > $C/check.out 2>&1
R=data/window-recurrent/general-algebras/round1
for t in word-problem-s4 move-to-front-4; do
  if same $C/$t/seed0/log.jsonl $R/$t/seed0/log.jsonl; then log "B3's control check: $t seed 0 repeated by bits"
  else log "B3's control check FAILED on $t: see $C/$t/seed0/log.jsonl.diff; queue stopped"; exit 1; fi
done
OUT=$WB/b3; mkdir -p $OUT
echo "B3 algebras starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch.log
$PROBE --task word-problem-s4 move-to-front-4 --branches --factors 2 --codes-ratio 32/11 --reads output --fork \
  --budget-epochs 600 --budget-minutes 900 --first-seed 0 --seeds 6 --workers 6 --out $OUT >> $OUT/b3.out 2>&1
echo "B3 algebras ended $(date) exit $?" >> $OUT/launch.log
log "queue ends"
