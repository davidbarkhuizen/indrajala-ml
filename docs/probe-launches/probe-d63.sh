#!/bin/bash
# The state formation workplan's D63, the collapse: the reheat with restarts against the plain method
# with restarts (expectations written before its data, 2026-10-22, after D61's reading; the owner,
# 2026-10-22: "i would imagine that we need both, if they both work").
# The cell: D64's (D54's continuous crystallisation, the margin stop under its deadline, --margin-wait
# 200; D59, D60) plus the reheat (--reheat, D63 (a): the cool's rate below its low mark runs the clock
# back, never past the cool's start, the temperatures it passes back over released; growth only
# forward). Code at main (the reheat, #921, and the run's state object, #919, are after D64's 41f92e7).
# Why: every failure of D54-D59 ends frozen (the clock held 738-1,472 epochs); the literature (the
# 2026-10-22 review, indrajala-papers/freeze-and-restarts.tex) says no schedule passes a freeze, the
# remedies being escape and restart. The reheat acts within a run (it changes q(t), a run's chance of
# success by t); restarts at a fixed cutoff act across runs (Luby, Sinclair and Zuckerman 1993: a
# certified run is a Las Vegas run). They combine: each arm is read at its own optimal cutoff.
# Population: D64's 22 seeds, paired with D64's runs: word-problem-s3 40-47, move-to-front-4 32-35,
# word-problem-s4 3-5, random-q16-0 4-6, random-q12-0 4-7; and, as a diagnosis not counted in any
# rate, word-problem-s4 0 and 2 (D57's collapses: does a reheat revive their dead codes?). Placement
# (the owner, 2026-10-22: stone's whole machine until 07:00, the laptops at full load, jebel off
# overnight): stone (12 runs at once on the machine, D64's included) the five S4, random-q16-0 4-6,
# random-q12-0 6-7 and move-to-front-4 32-35, stopped at 06:55 and resumed that evening (--resume);
# pyramidon (3 at once) word-problem-s3 40-47; klip (3 on the machine, D64's included) random-q12-0
# 4-5. Every machine's runs are the same cell (cross-machine bit parity), each at commit 0fdf237.
# Feasibility as D64's (docs/information-flow.md): S3 q 6 (8 codes, cap
# 69), mtf-4 and S4 q 24 (15 codes, cap 248), random-q12-0 q 12 (11 codes, cap 132), random-q16-0 q 16
# (12 codes, cap 159), width 32; up to 1,600 epochs or 690 minutes a run. S3 24 of the earlier cell
# list is left out: its drop came at the cool's start, leaving no room to run back (D63 (a)'s cons).
# Expectations (read with D64's twins once both have ended; restart-cost, weight-norms, state-spread):
#  - integrity: each run's rows identical to its D64 twin's up to its first reheat (a rate below 0),
#    but for the reheat's own fields; any false discovery (a stop certifying at neither verdict, the
#    run never certifying) refutes the stop;
#  - the paired verdict on the 22 seeds (McNemar: b, the seeds only the reheat certifies; c, those
#    only the plain method does): the reheat helps if b >= c + 2, is refuted (not adopted) if c >= b;
#  - the cost: Luby's l(t) per arm at its own cutoff (t* plain, t*' reheat, from each arm's runs,
#    spent runs censored), pooled over S3 and move-to-front-4 and per gap backdrop: the reheat with
#    restarts is adopted, together with restarts, if it helps on the paired verdict and l(t*') <= l(t*)
#    on the pooled ordered backdrops; restarts at t* are adopted for the plain method in any case
#    (D63 (e), offline on its runs);
#  - existence on random-q16-0 (0 of 11 runs on 7 seeds before it): any certificate reported; none
#    refutes nothing but says the reheat does not make forming possible there;
#  - read apart: certificates reached only after a reheat (outside Luby's model); each reheat's depth
#    (the clock run back) and whether the re-cool froze again and where (D66's trigger: re-freezing at
#    the same point runs D66); on S4, codes in use and dimension before and after each reheat.
#
# Usage: probe-d63.sh CODE OUT RUNS_ROOT MAX "TASK:SEED ..." [--resume]
#   TASK s3, mtf4, s4, rq12 or rq16; MAX the machine's runs at once (this probe's and any other
#   modular_family_probe.py run), a queued run launched only below it; --resume to go on from each
#   run's checkpoint (a run that ended isn't trained again).
set -u
CODE=$1; OUT=$2; RUNS_ROOT=$3; MAX=$4; JOBS=$5; EXTRA=${6:-}
mkdir -p $OUT
cd $CODE
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "D63 probe ($JOBS; at most $MAX runs on the machine) starts at $(git rev-parse --short HEAD) $EXTRA"
run() {
  TASK=${1%:*}; SEED=${1#*:}
  CELL="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 1600 \
    --budget-minutes 690 --anneal --anneal-tail 40 --anneal-never-commit --warm-up --melt-hot 2 \
    --melt-kappa 1e-2 --melt-hold 20 --melt-halving 30 --melt-shape exponential --crucible \
    --warm-up-screen 1.01 --continuous --workers 1 --margin-wait 200 --reheat"
  case $TASK in
    s3) NAME=word-problem-s3; BOUND=6 ;;
    mtf4) NAME=move-to-front-4; BOUND=24 ;;
    s4) NAME=word-problem-s4; BOUND=24 ;;
    rq12) NAME=random-q12-0; BOUND=12 ;;
    rq16) NAME=random-q16-0; BOUND=16 ;;
  esac
  log "launch $TASK-seed$SEED"
  AB_RUNS_ROOT=$RUNS_ROOT-$TASK-$SEED env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    ${PY:-.venv/bin/python} scripts/modular_family_probe.py run --runs $NAME:$SEED $CELL \
    --state-bound $BOUND --out $OUT/$TASK-seed$SEED $EXTRA >> $OUT/$TASK-seed$SEED.out 2>&1
  log "ended $TASK-seed$SEED: $(tail -1 $OUT/$TASK-seed$SEED.out)"
}
# a run is its python process alone: its forkserver children carry its arguments too
running() { pgrep -f "^[^ ]*/python[0-9.]* scripts/[m]odular_family_probe.py run" | wc -l; }
for J in $JOBS; do
  while [ $(running) -ge $MAX ]; do sleep 30; done
  run $J &
  sleep 20  # the run's process appears before the next count
done
wait
log "probe ended"
