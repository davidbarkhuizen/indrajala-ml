#!/bin/bash
# Division stage 5's long tests (division-workplan.md, D4), shared between pyramidon and jebel:
# each worker claims a certified network by an atomic mkdir on pyramidon
# (division-claim/long-tests-claims/<arm>-<seed>), long-tests it with the main checkout's code, and
# on jebel copies the result to pyramidon. Each machine's log: division-claim/long-tests-<host>.txt.
# Usage: stage5-long-tests.sh WAIT_PID [reverse]   (waits for WAIT_PID to end first; 0: no wait)
cd ~/code/indrajala-ml || exit 1
D=data/window-recurrent/division-claim
CLAIMS=code/indrajala-ml/$D/long-tests-claims  # relative to the home directory on pyramidon
HOST=$(hostname)
LOG=$D/long-tests-$HOST.txt
while [ "$1" != 0 ] && kill -0 "$1" 2>/dev/null; do sleep 60; done
NETS=$(ls $D/{family,quotient}/seed*/certified-step-*.network.json)
[ "$2" = reverse ] && NETS=$(echo "$NETS" | tac)
claim() {  # 0: claimed; 1: claimed already; retries while ssh fails (255)
  local rc
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if [ "$HOST" = pyramidon ]; then mkdir ~/"$CLAIMS/$1" 2>/dev/null; rc=$?
    else ssh -o BatchMode=yes pyramidon mkdir "$CLAIMS/$1" 2>/dev/null; rc=$?; fi
    [ $rc -ne 255 ] && return $rc
    sleep 60
  done
  echo "claim $1: ssh failed, skipped $(date)" >> $LOG
  return 1
}
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "worker $$ started $(date) at $(git rev-parse --short HEAD)" >> $LOG
for n in $NETS; do
  arm=$(basename "$(dirname "$(dirname "$n")")"); seed=$(basename "$(dirname "$n")"); name=$arm-$seed
  claim $name || continue
  echo "start $name $(date) worker $$" >> $LOG
  .venv/bin/python scripts/modular_family_probe.py long-test "$n" --task modular-division-5 --arm "$arm" \
    --out $D/long-tests/$name >> $LOG 2>&1
  if [ "$HOST" != pyramidon ]; then
    rsync -a $D/long-tests/$name/ pyramidon:code/indrajala-ml/$D/long-tests/$name/ >> $LOG 2>&1 \
      || echo "rsync $name to pyramidon failed" >> $LOG
  fi
  echo "done $name $(date) worker $$" >> $LOG
done
echo "worker $$ finished $(date)" >> $LOG
