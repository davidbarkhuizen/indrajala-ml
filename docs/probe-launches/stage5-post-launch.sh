#!/bin/bash
# Division stage 5 follow-up (division-workplan.md, stage 5; D8): when pyramidon's arms (3778260,
# 3778261) end, pull jebel's seeds into division-claim/, recertify every certified network in both
# verdicts, then long-test each one at a time (beside E2b snapped's 3 workers). Runs the main
# checkout's code as it stands (the code stage 5 trained with); no git operations.
cd ~/code/indrajala-ml || exit 1
D=data/window-recurrent/division-claim
while kill -0 3778260 2>/dev/null || kill -0 3778261 2>/dev/null; do sleep 60; done
mkdir -p $D/certify $D/long-tests
LOG=$D/post-pyramidon.txt
echo "stage 5 follow-up started $(date) at $(git rev-parse --short HEAD)" > $LOG
rsync -a --ignore-existing jebel:code/indrajala-ml/$D/ $D/ >> $LOG 2>&1 || { echo "rsync failed $(date)" >> $LOG; exit 1; }
echo "merged from jebel $(date)" >> $LOG
export OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
P=scripts/modular_family_probe.py
NETS=$(ls $D/{family,quotient}/seed*/certified-step-*.network.json 2>/dev/null)
for n in $NETS; do
  arm=$(basename "$(dirname "$(dirname "$n")")"); seed=$(basename "$(dirname "$n")")
  stem=$arm-$seed-$(basename "$n" .network.json)-modular-division-5
  .venv/bin/python $P certify "$n" --task modular-division-5 --arm "$arm" --out $D/certify/$stem.certificate.json >> $LOG 2>&1
  .venv/bin/python $P certify "$n" --task modular-division-5 --arm "$arm" --snap-ratio 1.0 --out $D/certify/$stem.strict.certificate.json >> $LOG 2>&1
  echo "certified $n $(date)" >> $LOG
done
for n in $NETS; do
  arm=$(basename "$(dirname "$(dirname "$n")")"); seed=$(basename "$(dirname "$n")")
  .venv/bin/python $P long-test "$n" --task modular-division-5 --arm "$arm" --out $D/long-tests/$arm-$seed >> $LOG 2>&1
  echo "long-tested $n $(date)" >> $LOG
done
echo "stage 5 follow-up done $(date)" >> $LOG
