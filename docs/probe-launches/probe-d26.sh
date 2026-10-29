#!/bin/bash
# The state formation workplan's stage 4a''', seeded crystallisation (D26, D27; expectations written
# before its data). Each run's seed grown in its own crucible: the curriculum's warm-up (D14, no --whole),
# then the material held at the warm-up's last step's lengths through the whole profile (the full melt,
# T_hot 2 T_c, held 20 epochs, thermal noise kappa 1e-2, the convex cool falling 20 octaves in 600 epochs,
# the gradual handover, tail 40, the commitment never back) until every factor is hard; then growth: the
# whole distribution, the codes hard, the forward path coupled to them. No fork (D18).
# Population (D27): word-problem-s3 seeds 0-7, independent (each its own warm-up, codes and noise), one
# cell: two factors, --codes-ratio 32/11, reading the output, --window-width 32. Existence; the rate a
# screen (Wilson 95%). Population annealing deferred (the owner) until this is read.
# Feasibility (docs/information-flow.md): q 6, 8 codes a factor (m 64, row 1), cap 69 = m + q - 1 (row 2,
# trained in growth; the crucible below it, its seed corrected by growth, a certificate a proof whatever
# the lengths), width 32 certified on S3 before (row 3). Up to 1,600 epochs or 690 minutes a run (about
# 10 h: the crucible's 666 epochs at about 10 s, growth's at about 28 s), in stone's window 18:00-07:00.
# Why (the plan's understanding step): S3's no-curriculum warm-up never formed the state (centroid at
# chance from length 16); the curriculum's did (4a: 1.00 to 32), but snapped onto its own k-means codes it
# fell to chance (0.41 at 16, 0.20 at 64): a group has no synchronising word, so its seed must be
# crystallised on its own material before the long words.
# Expectations (read offline: state-separation, snapped, on the crucible's last block's network, the
# step before the first "grown" block; approximations on any certified network):
#  - seed formation: joint nearest-neighbour separation at length 16 >= 0.9 in at least 4 of the 8 runs;
#  - growth: at least one run certifies.
# Refuted: no crucible crystal separates (the seed doesn't form: the crucible's design is wrong); seeds
# form but none certifies (growth loses them: D26 (b), reheating below the seed's melting point, next,
# where the logs say).
set -u
CODE=$1; OUT=$2; RUNS_ROOT=$3; SEEDS=${4:-0 1 2 3 4 5 6 7}
EXTRA=${5:-}  # --resume: each run goes on from its checkpoint
mkdir -p $OUT
cd $CODE
# PY: the machine's Python with the crate (stone: PYTHONPATH to its wheel and a shared venv)
PROBE="env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ${PY:-.venv/bin/python} scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "stage 4a''' (word-problem-s3 seeds $SEEDS) starts at $(git rev-parse --short HEAD) $EXTRA"
BASE="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 1600 --budget-minutes 690 \
  --crucible --anneal --anneal-tail 40 --anneal-never-commit --warm-up --handover --melt-hot 2 --melt-kappa 1e-2 \
  --melt-hold 20 --melt-cool 600 --melt-octaves 20 --melt-halving 30 --melt-shape convex --workers 1"
for SEED in $SEEDS; do
  AB_RUNS_ROOT=$RUNS_ROOT-$SEED $PROBE --runs word-problem-s3:$SEED $BASE --out $OUT/seed$SEED $EXTRA \
    >> $OUT/seed$SEED.out 2>&1 &
done
wait
log "probe ended"
