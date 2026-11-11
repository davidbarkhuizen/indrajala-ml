#!/bin/bash
# Kept from the crate performance workplan: it names these machines, and writes logs beside itself. Read it before running it.
# Its sequence benchmark (SEQ) is the one that resets jebel: docs/working-notes.md. Bundle or fetch a merged crate commit for stone and klip first.
# ab_all.sh TAG OLD NEW: one kernel change's A/Bs: jebel claims, stone and klip guards, every one with the d 512 backdrop.
# Output: a local log per machine, in this script's directory, each ending "<machine> <tag> ended".
T=$1; O=$2; N=$3; SP=$(cd "$(dirname "$0")" && pwd); ML=/home/david/code/indrajala-ml
FOCUSED='focused_benchmark -- --shape 5408 --shape 784 --shape attention --op forward_batch --op downstream_batch --op accumulate_gradient_batch'
ONE='--rust-threads 1 --openblas-threads 1'
D512='cmd -- scripts/measurement/sequence_training_timing.py --windows 128 --repeats 1 --features 512 --heads 8'
# the window program: indrajala-studies' window_study_timing.py, whose default is ml's learn and forward cases and the
# long test (the window-timing-guard workplan), by its absolute path in each host's studies checkout
WINDOW_SCRIPT='scripts/measurement/window_study_timing.py --repeats 3'
WINDOW="cmd -- /home/david/code/indrajala-studies/$WINDOW_SCRIPT"
WINDOW_STONE="cmd -- /home/david/labour/code/studies-main/$WINDOW_SCRIPT"
SEQ='cmd -- scripts/measurement/sequence_training_timing.py --windows 1024 --repeats 2'
report() { for r in "$@"; do .venv/bin/python scripts/measurement/ab.py report $(date +%F)-$r --brief 2>&1 | grep -v "^profile\|^controls" | cut -c1-165; done; }

# SKIP_GUARDS: only jebel, the guards having run
if [ -z "$SKIP_GUARDS" ]; then
# stone gets code as bundles
B=$SP/b; rm -rf "$B"; mkdir -p "$B"; cd $ML
git bundle create -q "$B/indrajala-ml.bundle" main
git -C rust branch -f bundle-main origin/main >/dev/null; git -C rust branch -f bundle-new "$N" >/dev/null
git -C rust bundle create -q "$B/indrajala-math-rust.bundle" bundle-main bundle-new
git -C rust branch -D bundle-main bundle-new -q
git -C ../indrajala-studies bundle create -q "$B/indrajala-studies.bundle" main
scp -q "$B"/*.bundle stone:labour/bundles/ || exit 3
cat > "$B/stone.sh" <<S
source ~/labour/env.sh; export AB_CRATE_REPO=\$HOME/labour/code/indrajala-math-rust
cd \$AB_CRATE_REPO && git fetch -q origin '+refs/heads/*:refs/remotes/origin/*'
cd ~/labour/code/indrajala-ml && git fetch -q origin main:refs/remotes/origin/main && git merge -q --ff-only origin/main
if [ -d ~/labour/code/studies-main ]; then git -C ~/labour/code/studies-main fetch -q ~/labour/bundles/indrajala-studies.bundle main && git -C ~/labour/code/studies-main merge -q --ff-only FETCH_HEAD
else git clone -q -b main ~/labour/bundles/indrajala-studies.bundle ~/labour/code/studies-main; fi
AB=".venv/bin/python scripts/measurement/ab.py run --old main --new main --old-crate $O --new-crate $N --order ONNONO"
{
\$AB --name $T-guard-d512 --bench $D512 2>&1 | tail -1
\$AB --name $T-guard-focused --bench $FOCUSED $ONE 2>&1 | tail -1
\$AB --name $T-guard-epochs --bench prepared_dataset_timing -- --configs "conv B=32" "dense B=32" --repeats 2 2>&1 | tail -1
\$AB --name $T-guard-window --bench $WINDOW_STONE 2>&1 | tail -1
for r in $T-guard-d512 $T-guard-focused $T-guard-epochs $T-guard-window; do .venv/bin/python scripts/measurement/ab.py report $(date +%F)-\$r --brief 2>&1 | grep -v "^profile\|^controls" | cut -c1-165; done
echo "stone $T ended"
} > ~/labour/stone-$T.log 2>&1
S
scp -q "$B/stone.sh" stone:labour/stone-$T.sh; ssh stone "setsid nohup bash ~/labour/stone-$T.sh >/dev/null 2>&1 < /dev/null & disown"

cat > "$B/klip.sh" <<K
export PATH=\$HOME/.cargo/bin:\$PATH
git -C ~/code/indrajala-math-rust fetch -q origin
cd ~/code/indrajala-ml && git pull -q --ff-only && git submodule update -q --init
git -C ~/code/indrajala-studies pull -q --ff-only
ulimit -v 4500000
AB=".venv/bin/python scripts/measurement/ab.py run --old main --new main --old-crate $O --new-crate $N --order ONNONO"
{
\$AB --name $T-guard-d512 --bench $D512 2>&1 | tail -1
\$AB --name $T-guard-focused --bench $FOCUSED $ONE 2>&1 | tail -1
\$AB --name $T-guard-window --bench $WINDOW 2>&1 | tail -1
\$AB --name $T-guard-sequence --bench $SEQ 2>&1 | tail -1
for r in $T-guard-d512 $T-guard-focused $T-guard-window $T-guard-sequence; do .venv/bin/python scripts/measurement/ab.py report $(date +%F)-\$r --brief 2>&1 | grep -v "^profile\|^controls" | cut -c1-165; done
echo "klip $T ended"
} > ~/klip-$T.log 2>&1
K
scp -q "$B/klip.sh" klip:klip-$T.sh; ssh klip "setsid nohup ~/code/thermal-guard.sh ~/klip-$T-temp.log 85 -- bash ~/klip-$T.sh >/dev/null 2>&1 < /dev/null & disown"

fi
# jebel, driven from here
# JEBEL_REBOOT_WAIT seconds: leave jebel alone until it has rebooted, or that long has passed
if [ -n "$JEBEL_REBOOT_WAIT" ]; then
  id0=$(ssh -o ConnectTimeout=10 jebel cat /proc/sys/kernel/random/boot_id); end=$((SECONDS + JEBEL_REBOOT_WAIT))
  while [ $SECONDS -lt $end ]; do
    id=$(ssh -o ConnectTimeout=10 jebel cat /proc/sys/kernel/random/boot_id 2>/dev/null)
    if [ -n "$id" ] && [ "$id" != "$id0" ]; then echo "jebel rebooted"; sleep 60; break; fi
    sleep 20
  done
fi
cd $ML; AB=".venv/bin/python scripts/measurement/ab.py remote --host jebel run --old main --new main --old-crate $O --new-crate $N --order ONNONO"
$AB --name $T-sequence-d512 --bench cmd -- scripts/measurement/sequence_training_timing.py --windows 256 --repeats 1 --features 512 --heads 8 2>&1 | tail -1
$AB --name $T-attention-d512 --bench attention_timing -- --no-memory --repeats 2 --tokens 256 --features 512 --heads 8 --masks causal 2>&1 | tail -1
$AB --name $T-focused-1 --bench $FOCUSED $ONE 2>&1 | tail -1
$AB --name $T-focused --bench $FOCUSED 2>&1 | tail -1
$AB --name $T-epochs --bench prepared_dataset_timing -- --configs "conv B=32" "conv B=512" "dense B=32" "dense B=512" --repeats 2 2>&1 | tail -1
$AB --name $T-sequence --bench $SEQ 2>&1 | tail -1
ssh jebel 'git -C ~/code/indrajala-studies pull -q --ff-only'
$AB --name $T-window --bench $WINDOW 2>&1 | tail -1
for r in sequence-d512 attention-d512 focused-1 focused epochs sequence window; do ssh jebel "cd ~/code/indrajala-ml; .venv/bin/python scripts/measurement/ab.py report $(date +%F)-$T-$r --brief 2>&1" | grep -v "^profile\|^controls" | cut -c1-165; done
until ssh -o ConnectTimeout=10 stone "grep -q 'stone $T ended' ~/labour/stone-$T.log" 2>/dev/null; do sleep 60; done
echo === STONE; ssh stone "grep -v '^done' ~/labour/stone-$T.log | cut -c1-165"
until ssh -o ConnectTimeout=10 klip "grep -q 'klip $T ended' ~/klip-$T.log || grep -q ' ended, status' ~/klip-$T-temp.log" 2>/dev/null; do sleep 60; done
echo === KLIP; ssh klip "grep -v '^done' ~/klip-$T.log | cut -c1-165; tail -1 ~/klip-$T-temp.log"
echo "all $T ended"
