#!/bin/bash
# never-scorecard.sh's division half (a), relaunched with its own lock directory: the first launch shared the S5
# half's AB_RUNS_ROOT and was refused before it started (no data). Unchanged otherwise; its header holds the
# expectations, written before the data; its control check (division seed 0 repeated W1 by bits, 8 rows) stands.
set -u
cd ~/code/indrajala-ml
export AB_RUNS_ROOT=$HOME/code/ab-runs-never-scorecard-division
OUT=data/window-recurrent/commitment-control/never-scorecard
mkdir -p $AB_RUNS_ROOT
log() { echo "$* $(date)" >> $OUT/launch.log; }
log "division half refused by the shared lock (no data); relaunched with its own"
rm -f $OUT/division.out
env OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 .venv/bin/python scripts/modular_family_probe.py run --p 13 --arm family \
  --factor 32 38 --factor 32 38 --reads output --fork --window-width 32 --commitment-from 1000 --budget-epochs 600 \
  --budget-minutes 900 --first-seed 0 --seeds 2 --workers 2 --out $OUT/division >> $OUT/division.out 2>&1
log "division half ended exit $?"
