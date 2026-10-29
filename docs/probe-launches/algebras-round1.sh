#!/bin/bash
# The general algebras, round 1 (docs/general-algebras-workplan.md, E1 registered in #819): n 3 and 4,
# S_n's word problem, move-to-front and random-n-0, seeds 0-5, 6 at once on pyramidon. Code
# ~/code/wt-algebras (pinned 5a89bfe, #820); the probe holds the machine lock.
cd ~/code/indrajala-ml
CODE=~/code/wt-algebras
OUT=data/window-recurrent/general-algebras/round1
mkdir -p $OUT
echo "round 1 starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch.log
PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 .venv/bin/python $CODE/scripts/modular_family_probe.py run \
  --task word-problem-s3 move-to-front-3 random-3-0 word-problem-s4 move-to-front-4 random-4-0 \
  --factors 2 --codes-ratio 32/11 --reads output --fork --budget-epochs 600 --budget-minutes 900 \
  --first-seed 0 --seeds 6 --workers 6 --out $OUT >> $OUT/round1.out 2>&1
echo "round 1 ended $(date) exit $?" >> $OUT/launch.log
