#!/bin/bash
# The state formation workplan's stage 4a'', the handover probe (D25 on D22-D24; expectations written
# before its data). Option 1 with no curriculum (--whole), the gradual handover (--handover: each soft
# factor's next state (1 - l) h + l q, l 0 through the heat and the hold, then log-linear in T from its
# melting point to its last transition), the full melt (T_hot 2 T_c), held 20 epochs, thermal noise kappa
# 1e-2, the commitment never back, the cool falling 20 octaves in 600 epochs as each shape (concave,
# convex, s), then halved every 30, hardness read once the shaped cool has run; round 1's and 2's cell
# (two factors, --codes-ratio 32/11, reading the output) at --window-width 32; no fork (D18).
# Backdrops: word-problem-s3 seed 0 and move-to-front-4 seed 3 (failure-A seeds that certified with the
# commitment never on). Up to 1,600 epochs a run (about 9 h and 27 h; multi-day allowed where justified).
# Feasibility (docs/information-flow.md): S3 q 6, 8 codes a factor, m 64, cap 69; mtf-4 q 24, 15, 225, 248;
# width 32 certified each (C0); every epoch trains lengths 1..cap, so the exclusion bound holds throughout.
# Why (the separation check, the plan's findings): the no-curriculum warm-up's continuous states carry the
# target's at every length (mtf-4 seed 3, joint nearest neighbour 1.00 at lengths 4-128); 4a lost them at
# the abrupt snap itself (0.99 to 0.82 and 0.98 to 0.25), so the handover is the test.
# Expectations (mechanics, read offline with state-separation on each step's network, and the log):
#  - while l is 0 (heat, hold) the screen holds near the warm-up's (no melt of the forward path), and the
#    codes' groups fall (the codes melt on the liquid);
#  - as l rises the joint separation holds (>= 0.9 at length 16, read snapped from the step networks)
#    and each factor's alone rises toward the certified networks' 1.00 (the factors become clusterable);
#  - states: at least one run reaches a state share >= 0.5; at least one certifies.
# Existence only (one seed a shape). Decision: a shape that certifies or forms a state goes forward to the
# panel (4b), its commitment's return then tested; separation lost as l rises: the handover's window or
# speed is the next design (where it was lost says which); held but no state: the readings decide.
# Flaw found mid-run (2026-10-20, 14:05, recorded in the plan): the block loop ended at the cap (the
# curriculum's last length), so a run passing a block an epoch stopped at block 248 (mtf-4) or 69 (S3),
# before its cool ended. Fixed in code (blocks go on until the budget) and the runs resumed from their
# checkpoints (bit-identical to the uninterrupted runs up to the stop): the 5th argument, --resume.
set -u
RUN=$1; CODE=$2; OUT=$3; RUNS_ROOT=$4  # RUN: task:seed
EXTRA=${5:-}  # --resume: each run goes on from its checkpoint
mkdir -p $OUT
cd $CODE
# PY: the machine's Python with the crate (stone: PYTHONPATH to its wheel and a shared venv)
PROBE="env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ${PY:-.venv/bin/python} scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "stage 4a'' ($RUN) starts at $(git rev-parse --short HEAD) $EXTRA"
BASE="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 1600 --budget-minutes 2880 \
  --whole --anneal --anneal-tail 40 --anneal-never-commit --warm-up --handover --melt-hot 2 --melt-kappa 1e-2 \
  --melt-hold 20 --melt-cool 600 --melt-octaves 20 --melt-halving 30 --workers 1"
for SHAPE in concave convex s; do
  AB_RUNS_ROOT=$RUNS_ROOT-$SHAPE $PROBE --runs $RUN $BASE --melt-shape $SHAPE --out $OUT/$SHAPE $EXTRA >> $OUT/$SHAPE.out 2>&1 &
done
wait
log "probe ended"
