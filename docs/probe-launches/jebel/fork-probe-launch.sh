#!/bin/bash
# The collapse-fork probe (docs/collapse-fork-workplan.md, D6): starts when E2b's continuous arm (75846) ends.
cd ~/code/indrajala-ml
while kill -0 75846 2>/dev/null; do sleep 60; done
D=data/window-recurrent/fork-probe; mkdir -p $D
echo "fork probe launched $(date) from ~/code/wt-fork at $(git -C ~/code/wt-fork rev-parse --short HEAD)" > $D/launch.txt
P="--arm family --reads output --p 7 --factor 16 32 --factor 16 32 --first-seed 0 --seeds 4 --budget-epochs 400 --budget-minutes 240"
export PYTHONPATH=$HOME/code/wt-fork OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
.venv/bin/python ~/code/wt-fork/scripts/modular_family_probe.py run $P --fork --out $D/fork > $D/fork.out 2>&1 &
.venv/bin/python ~/code/wt-fork/scripts/modular_family_probe.py run $P --out $D/nofork > $D/nofork.out 2>&1 &
wait
