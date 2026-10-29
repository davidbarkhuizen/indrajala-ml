#!/bin/bash
# Long tests (division D4: U1 and S1 at 128, 256, 1,024) of the fork probe's 5 certified p 7
# networks, one at a time, after E2b's long tests (3812641) end. Code: main checkout's main.
cd ~/code/indrajala-ml
while kill -0 3812641 2>/dev/null; do sleep 60; done
D=data/window-recurrent/fork-probe
L=$D/long-tests.log
echo "started $(date) at $(git rev-parse --short HEAD)" > $L
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
for n in $D/{fork,nofork}/seed*/certified-step-*.network.json; do
  arm=$(basename "$(dirname "$(dirname "$n")")"); seed=$(basename "$(dirname "$n")")
  echo "start $arm-$seed $(date)" >> $L
  .venv/bin/python scripts/modular_family_probe.py long-test "$n" --task modular-division-7 --arm family \
    --out $D/long-tests/$arm-$seed >> $L 2>&1
  echo "done $arm-$seed $(date) exit $?" >> $L
done
echo "ended $(date)" >> $L
