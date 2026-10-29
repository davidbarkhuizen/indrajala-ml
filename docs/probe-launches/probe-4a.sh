#!/bin/bash
# The state formation workplan's stage 4a, the mechanics probe (D14-D21; expectations written before its
# data): option 1 rebuilt as heat, hold, cool, crystallise (`--anneal --warm-up`, no fork: D18), round 1's
# and 2's cell (two factors, --codes-ratio 32/11, reading the output, 600 epochs) at --window-width 32,
# the tail 40 epochs and the commitment's ramp 60 (D19). Three arms (D16, D17): full = T_hot 2 T_c, kappa
# 1e-2; quiet = T_hot 2 T_c, kappa 1e-4; partial = T_hot 0.5 T_c, kappa 1e-2. TASK seed 0 per arm.
# Feasibility (docs/information-flow.md): S3 q 6, 8 codes a factor, m 64, cap 69; S4 q 24, 15, 225, 248;
# width 32 certified each (C0).
# Expectations (mechanics, the probe's questions):
#  - every warm-up ends at step 2 or later, its distinct states at least 4 times its codes (D14);
#  - the melt: at the hold's end the full arms' codes in 1-2 groups (D20), the partial arm's in more;
#  - the splits: the groups rise during the cool, the first within an octave of the warm-up end's
#    melting point (D21), and no factor ends with codes identical in pairs (D17's noise splits them);
#  - hard during the cool, then the tail and the ramp; the commitment's ramp erodes less than the void
#    panel's (codes in use at the ramp's end at least 3/4 of those at hard);
#  - states (descriptive): S4 seed 0 share >= 0.5 in at least one arm; S3 seed 0 certifies in at least one.
# Settles D16 and D17: the arm whose factors reach the most codes in use at hard with the larger state
# share; a tie, the full melt with kappa 1e-2 (the owner's profile). None splitting back: the profile is
# redesigned before any panel (the melting curves and the logged groups say where it failed).
set -u
TASK=$1; CODE=$2; OUT=$3; RUNS_ROOT=$4
mkdir -p $OUT
cd $CODE
# PY: the machine's Python with the crate (stone: PYTHONPATH to its wheel and a shared venv)
PROBE="env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ${PY:-.venv/bin/python} scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "stage 4a ($TASK) starts at $(git rev-parse --short HEAD)"
BASE="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 600 --budget-minutes 150 \
  --anneal --anneal-tail 40 --anneal-ramp 60 --warm-up --workers 1"
AB_RUNS_ROOT=$RUNS_ROOT-full $PROBE --runs $TASK:0 $BASE --melt-hot 2 --melt-kappa 1e-2 --out $OUT/full > $OUT/full.out 2>&1 &
AB_RUNS_ROOT=$RUNS_ROOT-quiet $PROBE --runs $TASK:0 $BASE --melt-hot 2 --melt-kappa 1e-4 --out $OUT/quiet > $OUT/quiet.out 2>&1 &
AB_RUNS_ROOT=$RUNS_ROOT-partial $PROBE --runs $TASK:0 $BASE --melt-hot 0.5 --melt-kappa 1e-2 --out $OUT/partial > $OUT/partial.out 2>&1 &
wait
log "probe ended"
