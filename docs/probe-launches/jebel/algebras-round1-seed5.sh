#!/bin/bash
# General algebras round 1, moved by whole runs (experiment-design.md section 5): seed 5 of every task, queued on
# pyramidon, run on jebel at the same commit (5a89bfe); pyramidon's duplicates of seed 5 are discarded.
cd ~/code/indrajala-ml
CODE=~/code/wt-algebras
OUT=data/window-recurrent/general-algebras/round1-seed5-jebel
mkdir -p $OUT
echo "seed 5 starts $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/launch.log
PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 .venv/bin/python $CODE/scripts/modular_family_probe.py run \
  --task word-problem-s3 move-to-front-3 random-3-0 word-problem-s4 move-to-front-4 random-4-0 --factors 2 \
  --codes-ratio 32/11 --reads output --fork --budget-epochs 600 --budget-minutes 900 --first-seed 5 --seeds 1 \
  --workers 6 --out $OUT >> $OUT/round1.out 2>&1
rc=$?
echo "seed 5 ended $(date) exit $rc" >> $OUT/launch.log
