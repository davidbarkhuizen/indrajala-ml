#!/bin/bash
# The commitment off for good on two more backdrops (docs/commitment-control-workplan.md; a probe, it claims nothing;
# the toolbox only). The never probe certified 4 of the 6 failure-A runs (S3 seeds 0, 3, 5; move-to-front-4 seed 3);
# its harm check (S4, move-to-front-4 at 32) runs beside this. Two cells, seeds 0 and 1, the base width 32:
#  (a) division p 13, W1's cell (two factors 32 wide, 38 codes each, reading the output, the fork, 600 epochs,
#      --window-width 32) with --commitment-from 1000: a harm check where the commitment was on and worked (W1:
#      seeds 0 and 1 certified at steps 11 and 8; 5 of 6 in all).
#  (b) word-problem-s5, round 2's cell (two factors, --codes-ratio 32/11) at 32 with --commitment-from 1000
#      --keep-machine --counterexamples 0.25: bottleneck 2 (C0: every S5 seed at ~120 pairs per code; R3/R4's probe
#      at 16 certified seed 1). Read beside C2 (R3/R4 at 32 with the commitment, same seeds, running on jebel).
# Feasibility (docs/information-flow.md): division q 169, 38 codes a factor, m 1,444, cap 1,612, width 32 (W1);
# S5 q 120, 32, 1,024, 1,143 (round 2).
# Expectations, written before its data:
#  - (a) both certify -> no harm on division; neither -> harm: read their bottleneck and margins.
#  - (b) either certifies where C2's same seed doesn't -> the commitment off helps bottleneck 2 too; read pairs per
#    code against C2's.
# Control: W1 ran at 85a9e44 (crate 40a2761) and C0 at c452ed9; this commit's repeat of them by bits is checked and
# logged (division seed 0 against W1, 4 epochs), but a mismatch doesn't stop the probe: its comparisons are counts.
# On pyramidon from main beside the harm check (4 workers), its own lock directory.
set -u
cd ~/code/indrajala-ml
export AB_RUNS_ROOT=$HOME/code/ab-runs-never-scorecard
OUT=data/window-recurrent/commitment-control/never-scorecard
mkdir -p $OUT $AB_RUNS_ROOT
PROBE="env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 .venv/bin/python scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "never scorecard: control check starts at $(git rev-parse --short HEAD)"
K=$OUT/bits-check
$PROBE --p 13 --arm family --factor 32 38 --factor 32 38 --reads output --fork --window-width 32 --budget-epochs 4 \
  --budget-minutes 20 --first-seed 0 --seeds 1 --workers 1 --out $K > $OUT/bits-check.out 2>&1
L=$K/seed0/log.jsonl
n=$(( $(wc -l < $L) - 1 ))
strip() { sed -E 's/, "seconds": [0-9.]+//; s/, "machine": \{[^}]*\}//'; }
R=data/window-recurrent/window-branches/w1/seed0/log.jsonl
if [ "$n" -gt 2 ] && diff <(head -n $n $L | strip) <(head -n $n $R | strip) > $K/diff; then
  log "control check: division seed 0 repeated W1 by bits ($n rows)"
else
  log "control check: division seed 0 did NOT repeat W1 by bits ($n rows; see $K/diff); going ahead (counts only)"
fi
log "never scorecard starts"
$PROBE --p 13 --arm family --factor 32 38 --factor 32 38 --reads output --fork --window-width 32 --commitment-from 1000 \
  --budget-epochs 600 --budget-minutes 900 --first-seed 0 --seeds 2 --workers 2 --out $OUT/division \
  >> $OUT/division.out 2>&1 &
$PROBE --runs word-problem-s5:0 word-problem-s5:1 --factors 2 --codes-ratio 32/11 --reads output --fork \
  --window-width 32 --commitment-from 1000 --keep-machine --counterexamples 0.25 --budget-epochs 600 \
  --budget-minutes 900 --workers 2 --out $OUT/s5 >> $OUT/s5.out 2>&1 &
wait
log "never scorecard ended"
