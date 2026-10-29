#!/bin/bash
# The state formation workplan's core matrix (4b-core; expectations written before its data, 2026-10-21
# about 16:30, before the coupling probe's (D38) data). The owner: "is there an affordable matrix we can
# run to rather just provide a better statistical basis to what we already know?" The panel's two arms
# (D36: the quasi-static crucible, D26-D30, against the same cool with --whole) on the two backdrops
# option 1 has results on: word-problem-s3 (a group) and move-to-front-4 (a reset machine), on fresh
# seeds (8-23; none run before under option 1). The large tier (S5, mtf-5, division) is set aside:
# under --whole its runs take 5-27 days on stone (the panel's timing, 2026-10-21).
#
# Cells (each arm: the rest as 4a''''s cell, the commitment never back, no fork):
#   S3 x crucible   seeds 8-23 (16)    S3 x whole    seeds 8-15 (8)
#   mtf-4 x whole   seeds 8-23 (16)    mtf-4 x crucible seeds 8-15 (8)
# Feasibility (docs/information-flow.md): S3 q 6, 8 x 8 codes, m 64, cap 69 (the exclusion bound); mtf-4
# q 24, 15 x 15, m 225, cap 248; width 32 (certified at 32 before, each); row 15: S3 depth 2 + k 1, mtf-4
# 3 + 2, within the crucible's 8. The budget: 600 epochs, the same on every machine (minutes 2,400, a
# safety only; the machines run at different speeds, so a minutes budget would give a slower machine's
# runs fewer epochs): every certified run of option 1 so far took at most 294 epochs (S3 crucible
# 27-175, 4a''' and 4a''''; mtf-4 whole 110-294, 4a'').
# The stop (D35, D43): --state-bound at the target's size (S3 6, mtf-4 24), the certificate judging it;
# a run succeeds when its stop's network certifies at both verdicts (D41); a stop exact only at 1.0
# reported, not counted. Each certified run's trained length reported (D42).
# The coupling (D45's conditional rule, registered here before D38's data): CAP 0 (--coupling-cap 0) if
# the coupling probe holds (S3 at least 6 of 8 certified), else CAP 1 (the handover as built, D25, D30).
#
# Expectations (read offline: population on each log, answers-stop --snapped where needed, the stop's
# network certified at both verdicts, approximations on every certified network):
#  - S3 x crucible: p at least 0.5 (at least 8 of 16; 4a'''': 8 of 8 on the design seeds);
#  - S3 x whole: at most 1 of 8 (formation fails without a seed: 4a'''s S3 arm);
#  - mtf-4 x whole: at least 4 of 16 (seed 3 certified under every shape, 4a'');
#  - mtf-4 x crucible: fewer than mtf-4 x whole (the D33 probe's crucible control never certified);
#  - the group-against-reset claim: S3 crucible above S3 whole (Fisher's exact, one-sided, p < 0.05)
#    and mtf-4 whole at or above mtf-4 crucible;
#  - the stop: no false discovery (a stop on a network the certificate rejects at both verdicts, in a
#    run that never certifies).
# Refuted: S3 crucible at 4 or fewer of 16 (the 8 of 8 was the design seeds'); S3 whole at 3 or more of
# 8 (a group forms without a seed); mtf-4 crucible above mtf-4 whole (the crucible doesn't harm a reset
# machine); any false discovery (D35's stop unsafe at the exact bound).
# Analysis: Wilson 95% per cell, Fisher's exact (one-sided) for the two contrasts; each failure
# classified by kind (failure-kinds.md: state-separation snapped and --live; breaks).
#
# Machines: stone 6 workers (its daytime limit), jebel 4 (the owner: at night at half capacity; one
# pool, so it holds jebel's machine lock against an A/B), pyramidon 3 (the cheap cells: it runs about
# 1.85x stone's seconds an epoch). The split rebalanced at 16:45, before any data (no setting changed). Code: main at 7d2b4eb (training
# as 3754155, the coupling probe's; only reading tools changed), crate 5689b2a.
#
# Usage: probe-4b-core.sh MACHINE CAP OUT CODE   (MACHINE stone|jebel|pyramidon; CAP 0|1)
set -u
MACHINE=$1; CAP=$2; OUT=$3; CODE=$4
case $CAP in 0) COUPLING="--coupling-cap 0" ;; 1) COUPLING="" ;; *) echo "CAP must be 0 or 1"; exit 2 ;; esac
mkdir -p $OUT
cd $CODE
# PY: the machine's Python with the crate (stone: PYTHONPATH to its wheel and a shared venv)
PROBE="env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ${PY:-.venv/bin/python} scripts/modular_family_probe.py run"
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "4b-core ($MACHINE, CAP $CAP) starts at $(git rev-parse --short HEAD)"
CELL="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 600 --budget-minutes 2400 \
  --anneal --anneal-tail 40 --anneal-never-commit --warm-up --handover --melt-hot 2 --melt-kappa 1e-2 \
  --melt-hold 20 --melt-cool 600 --melt-octaves 20 --melt-halving 30 --melt-shape convex \
  --warm-up-screen 1.01 --melt-gate 0.99 --handover-octaves 3 $COUPLING"
seeds() { local task=$1 from=$2 to=$3; for s in $(seq $from $to); do echo -n "$task:$s "; done; }
pool() {  # name, workers, arm flags, state bound, runs; its own lock root unless LOCK_DEFAULT (jebel)
  local name=$1 workers=$2 arm=$3 bound=$4; shift 4
  log "launch $name ($workers workers): $*"
  local root=${LOCK_DEFAULT:+$HOME/code/ab-runs}
  AB_RUNS_ROOT=${root:-$HOME/code/ab-runs-4b-$name} $PROBE --runs "$@" $CELL $arm --state-bound $bound \
    --workers $workers --out $OUT/$name >> $OUT/$name.out 2>&1 &
}
CRUCIBLE="--crucible"
WHOLE="--whole"
case $MACHINE in
  stone)
    pool mtf4-whole-b 3 "$WHOLE" 24 $(seeds move-to-front-4 16 23)
    pool s3-whole 2 "$WHOLE" 6 $(seeds word-problem-s3 8 15)
    pool s3-crucible-a 1 "$CRUCIBLE" 6 $(seeds word-problem-s3 8 15) ;;
  jebel)
    LOCK_DEFAULT=1 pool mtf4-whole-a 4 "$WHOLE" 24 $(seeds move-to-front-4 8 15) ;;
  pyramidon)
    pool mtf4-crucible 2 "$CRUCIBLE" 24 $(seeds move-to-front-4 8 15)
    pool s3-crucible-b 1 "$CRUCIBLE" 6 $(seeds word-problem-s3 16 23) ;;
  *) echo "MACHINE must be stone, jebel or pyramidon"; exit 2 ;;
esac
wait
log "4b-core ($MACHINE) ended"
