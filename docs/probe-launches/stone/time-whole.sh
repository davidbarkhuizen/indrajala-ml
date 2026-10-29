#!/bin/bash
# Time a few whole-distribution epochs (D22) per task before setting the follow-up probe's budget.
set -u
source ~/labour/env.sh
CODE=$LABOUR/code/wt-warm
OUT=$LABOUR/data/state-formation/option1/time-whole
mkdir -p $OUT
cd $CODE
PY="env PYTHONPATH=$LABOUR/site-continuous:$CODE OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 $LABOUR/code/wt-branches/.venv/bin/python"
for TASK in word-problem-s3 word-problem-s4; do
  AB_RUNS_ROOT=$LABOUR/ab-runs-time-$TASK $PY scripts/modular_family_probe.py run --runs $TASK:0 --factors 2 --codes-ratio 32/11 \
    --reads output --window-width 32 --budget-epochs 3 --budget-minutes 30 --anneal --warm-up --whole --workers 1 \
    --out $OUT/$TASK > $OUT/$TASK.out 2>&1 &
done
wait
echo "timing ended $(date)" >> $OUT/launch.log
