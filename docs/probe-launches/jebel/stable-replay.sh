#!/bin/bash
# Stable-seed replay probe (exploratory, 2026-10-16): P1's replay with the wall off on the step-7
# networks of 4 E2b' seeds that never lost a proof (1, 3, 8, 9), 5 replays each (generators
# snap-wall-p1-0..4, P1's first 5, so the data order matches seed 6's off-0..4), 2 epochs of
# step 8's data. Registered before the data:
#   STABLE NETWORKS: 3 or more of the 4 keep the proof (certified after both epochs) in >= 4 of 5.
#   GENERIC COLLAPSE: 10 or more of the 20 replays lose the proof (seed 6's: 9 of 10).
#   Anything else: inconclusive, recorded.
# Code ~/code/wt-replay (pinned 7618dfe), data in the main checkout, 2 workers.
cd ~/code/indrajala-ml
OUT=data/window-recurrent/stable-replay
CODE=~/code/wt-replay
MODELS=data/window-recurrent/e2b-prime/models
mkdir -p $OUT/start
for s in 1 3 8 9; do cp $MODELS/curriculum-linear-k1-s16-m16-seed$s/step-7.network.json $OUT/start/seed$s-step-7.network.json; done
echo "started $(date) at $(git -C $CODE rev-parse --short HEAD)" >> $OUT/replay.log
(cd $OUT/start && sha256sum *.json) >> $OUT/replay.log
jobs_list() { for i in $(seq 0 4); do for s in 1 3 8 9; do echo "$s $i"; done; done; }
run_one() {
  s=$1; i=$2
  PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 .venv/bin/python $CODE/scripts/window_recurrent_study.py replay \
    $OUT/start/seed$s-step-7.network.json \
    --config $CODE/scripts/window_recurrent_e2b_prime.json --run curriculum-linear-k1-s16-m16-seed$s --step 8 \
    --epochs 2 --generator snap-wall-p1-$i --numpy --wall 0 --json $OUT/seed$s-$i.json 2>&1 | sed "s/^/seed$s /"
}
export -f run_one; export OUT CODE
jobs_list | xargs -P 2 -L 1 bash -c 'run_one $0 $1' >> $OUT/replay.log
echo "ended $(date) exit $?" >> $OUT/replay.log
