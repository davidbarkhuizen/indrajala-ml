#!/bin/bash
# The state formation workplan's stage 4a'''', the quasi-static crucible (D28-D30; expectations written
# before its data). The seeded crystallisation of 4a''' with three fixes, each for a systematic failure
# of its population (read in flight with `population`): D28 the crucible at the warm-up's cap
# (--warm-up-screen 1.01: three crucibles ended at length 3 and never formed the state); D29 a
# quasi-static cool (--melt-gate 0.99: the profile's clock holds through an epoch whose screen before it
# is below 0.99 of the hold's best; three formed states were lost as the open-loop cool ran on through a
# screen's dip to coupling 1); D30 each coupling window at least 3 octaves (--handover-octaves 3: one
# factor melted in a single transition and handed over within an octave). Everything else as 4a'''.
# Population: word-problem-s3 seeds 1, 2, 3, 4, 6, 7; seeds 0 and 5 carried over from 4a''' (under
# D28-D30 the same runs: warm-ups ended at the cap, windows 3.25-4.25 octaves, cool screens never below
# 0.99), so the population is 0-7. Existence and the rate p (Wilson 95%), against 2 of 8 in 4a'''.
# Feasibility (docs/information-flow.md): the cell of 4a''' (q 6, 8 codes a factor, m 64, cap 69, width
# 32). Up to 1,600 epochs or 690 minutes a run, on stone from about 19:30 (13 workers on 12 cores until
# 4a''' ends about 05:45; about half the cores by day after 07:00).
# Expectations (read offline with `population` on each log; state-separation snapped at length 16 on
# the crucible's last block's network, as the rule of 4a'''; approximations on any certified network):
#  - D28: every crucible length 8; seed formation (the cool's best state share >= 0.9, and the snapped
#    joint separation at length 16 >= 0.9) in at least 4 of the 6;
#  - D29: the clock holds (held > 0) in every run whose screen dips in the cool, and no run that formed
#    its state (share >= 0.9) loses it (its codes in use halved) before it is hard;
#  - certificates: at least 2 of the 6 (p at least 4 of 8 with seeds 0 and 5).
# Refuted: a formed state lost while the clock holds (the dip isn't the schedule's: the coupling's own
# level, at fixed T, loses it; next, the coupling's end below 1 until a certificate, or D26 (b)); runs
# stalled at a held clock to the budget, never recovering (the gate too strict, or a dip the network
# can't climb out of at fixed T: the logs' screens through the holds say which); no certificate (the
# fixes don't reach the failure).
set -u
CODE=$1; OUT=$2; RUNS_ROOT=$3; SEEDS=${4:-1 2 3 4 6 7}
EXTRA=${5:-}  # --resume: each run goes on from its checkpoint
mkdir -p $OUT
cd $CODE
# PY: the machine's Python with the crate (stone: PYTHONPATH to its wheel and a shared venv)
PROBE="env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ${PY:-.venv/bin/python} scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "stage 4a'''' (word-problem-s3 seeds $SEEDS) starts at $(git rev-parse --short HEAD) $EXTRA"
BASE="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 1600 --budget-minutes 690 \
  --crucible --anneal --anneal-tail 40 --anneal-never-commit --warm-up --handover --melt-hot 2 --melt-kappa 1e-2 \
  --melt-hold 20 --melt-cool 600 --melt-octaves 20 --melt-halving 30 --melt-shape convex --workers 1 \
  --warm-up-screen 1.01 --melt-gate 0.99 --handover-octaves 3"
for SEED in $SEEDS; do
  AB_RUNS_ROOT=$RUNS_ROOT-$SEED $PROBE --runs word-problem-s3:$SEED $BASE --out $OUT/seed$SEED $EXTRA \
    >> $OUT/seed$SEED.out 2>&1 &
done
wait
log "probe ended"
