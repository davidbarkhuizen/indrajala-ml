#!/bin/bash
# R3 and R4's probe (docs/one-code-one-state-workplan.md, stage 5; a probe, it claims nothing): round 2's S5 cell
# (two factors, --codes-ratio 32/11, reading the output, the fork, 600 epochs; W11 16 wide, the base window until
# one is adopted, D4) with --keep-machine --counterexamples (0.25), seeds 0 and 1. Its control: round 2's same seeds
# (0 of 2; pairs per code at their best steps 120 and 120), the default repeating round 2 by bits on this commit.
# Expectations, written before its data: either certifies -> R3 and R4 move bottleneck 2: C2 on the adopted window;
# neither, but pairs per code at the best steps (the diagnosis rows' machine) fall well below 120 (under 30) ->
# they move it, not enough: read the counterexample counts and the kept restores for which part helps; neither and
# no fall -> they don't move it: read whether the counterexamples are drawn (counts above 0) and the kept machines.
# From ~/code/wt-r34 at c452ed9 (wt-b2's venv: crate afa7204, main's pin), 2 workers beside B3, under its own lock
# directory (AB_RUNS_ROOT=~/code/ab-runs-r34); a control check first (seed 0, 8 epochs, no flags).
cd ~/code/indrajala-ml
CODE=~/code/wt-r34
OUT=data/window-recurrent/one-code-one-state/r34-probe
mkdir -p $OUT ~/code/ab-runs-r34
PROBE="env AB_RUNS_ROOT=$HOME/code/ab-runs-r34 PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 $CODE/.venv/bin/python $CODE/scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "R3/R4 probe: control check at $(git -C $CODE rev-parse --short HEAD)"
K=$OUT/bits-check; rm -rf $K
$PROBE --runs word-problem-s5:0 --factors 2 --codes-ratio 32/11 --reads output --fork --budget-epochs 8 \
  --budget-minutes 20 --workers 1 --out $K > $OUT/bits-check.out 2>&1
R=data/window-recurrent/general-algebras/round2/word-problem-s5/seed0/log.jsonl
n=$(( $(wc -l < $K/seed0/log.jsonl) - 1 ))
if [ "$n" -gt 2 ] && diff <(head -n $n $K/seed0/log.jsonl | sed -E 's/, "seconds": [0-9.]+//; s/, "machine": \{[^}]*\}//') \
    <(head -n $n $R | sed -E 's/, "seconds": [0-9.]+//') > $K/diff.txt; then log "control check: round 2 seed 0 repeated by bits ($n rows)"
else log "control check FAILED: see $K/diff.txt; probe stopped"; exit 1; fi
log "R3/R4 probe starts"
$PROBE --task word-problem-s5 --factors 2 --codes-ratio 32/11 --reads output --fork --keep-machine --counterexamples \
  --budget-epochs 600 --budget-minutes 900 --first-seed 0 --seeds 2 --workers 2 --out $OUT >> $OUT/probe.out 2>&1
log "R3/R4 probe ended exit $?"
