#!/bin/bash
# The approximation claim's follow-up (docs/approximation-claim-workplan.md, stage 2): recertify every
# certified network in both verdicts (default snap ratio and strict), then long-test each (D4 of the
# division workplan: U1, S1 at 128, 256, 1,024), 6 at a time. On jebel: pyramidon has no memory to
# spare beside its long tests; runs are deterministic. Code: ~/code/wt-p11 (7618dfe, the code the runs
# trained with); don't develop there.
cd ~/code/indrajala-ml || exit 1
D=data/window-recurrent/approximation-claim
CODE=~/code/wt-p11
LOG=$D/post-jebel.txt
mkdir -p $D/certify $D/long-tests
echo "follow-up started $(date) at $(git -C $CODE rev-parse --short HEAD)" > $LOG
export PYTHONPATH=$CODE OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
P=$CODE/scripts/modular_family_probe.py
V="--vocabulary sum difference product"
NETS=$(ls $D/{quotient,sum,family,product}/seed*/certified-step-*.network.json 2>/dev/null)
for n in $NETS; do
  arm=$(basename "$(dirname "$(dirname "$n")")"); seed=$(basename "$(dirname "$n")")
  stem=$arm-$seed-$(basename "$n" .network.json)-modular-division-5
  .venv/bin/python $P certify "$n" --task modular-division-5 --arm "$arm" $V --out $D/certify/$stem.certificate.json >> $LOG 2>&1
  .venv/bin/python $P certify "$n" --task modular-division-5 --arm "$arm" $V --snap-ratio 1.0 --out $D/certify/$stem.strict.certificate.json >> $LOG 2>&1
  echo "certified $n $(date)" >> $LOG
done
echo "recertification done $(date)" >> $LOG
long_test() {
  n=$1; arm=$(basename "$(dirname "$(dirname "$n")")"); seed=$(basename "$(dirname "$n")")
  .venv/bin/python $P long-test "$n" --task modular-division-5 --arm "$arm" $V --out $D/long-tests/$arm-$seed > $D/long-tests/$arm-$seed.log 2>&1
  echo "long-tested $n exit $? $(date)" >> $LOG
}
export -f long_test; export D P V LOG
printf '%s\n' $NETS | xargs -P 6 -I{} bash -c 'long_test {}'
echo "follow-up done $(date)" >> $LOG
