#!/bin/bash
# Kept from the crate performance workplan, stage 8: its crate commit and output directory are that day's. Read it before running it.
# thread_readings.sh: stage 8's readings at crate main, each an A/A through ab.py, jebel, stone and klip at once.
# Output: pooled tables in $OUT, one per machine and reading, and a log per machine, each ending "<machine> ended".
SP=$(cd "$(dirname "$0")" && pwd); ML=/home/david/code/indrajala-ml; OUT=$HOME/data/indrajala-ml/runs/thread-readings-2026-11-04
CRATE=362e34c169eaf61f6ba36edcd7d99f59ddd4861e; mkdir -p "$OUT/stone" "$OUT/klip" "$OUT/jebel"
P=scripts/measurement/thread_cost_probe.py
PREPARED='epochs --backdrop prepared --threads 0 1 2 4 8 --repeats 1 -- --configs "conv B=32" "conv B=512" "dense B=32" "dense B=512"'
D512='epochs --backdrop sequence --threads 1 2 4 8 --repeats 1 -- --windows 256 --features 512 --heads 8'
ATTENTION='epochs --backdrop attention --threads 1 2 4 8 --repeats 1 -- --tokens 256 --features 512 --heads 8 --masks causal --repeats 2'
over() { echo "epochs --backdrop sequence --processes $1 --threads 1 2 4 8 --repeats 1 -- --windows 64 --features 512 --heads 8"; }

# the commands one guard runs, its ab.py command in $AB, each reading's table in $T
guard() { cat <<G
run() { name=\$1; shift; eval "\$AB --name thr-\$name --bench cmd -- $P \$*" 2>&1 | tail -1
  .venv/bin/python scripts/measurement/ab.py report \$(date +%F)-thr-\$name --pooled \$T/\$name.md >/dev/null 2>&1; }
$(for r in "$@"; do echo "$r"; done)
G
}

# stone gets code as a bundle
B=$SP/b; rm -rf "$B"; mkdir -p "$B"; cd $ML
git bundle create -q "$B/indrajala-ml.bundle" main; scp -q "$B/indrajala-ml.bundle" stone:labour/bundles/ || exit 3
{ cat <<S
source ~/labour/env.sh; export AB_CRATE_REPO=\$HOME/labour/code/indrajala-math-rust
cd ~/labour/code/indrajala-ml && git fetch -q origin main:refs/remotes/origin/main && git merge -q --ff-only origin/main
AB=".venv/bin/python scripts/measurement/ab.py run --old main --new main --old-crate $CRATE --new-crate $CRATE --order ON"
T=~/labour/thr; mkdir -p \$T
{
S
guard "run calls calls --threads 1 2 4 8 12" "run scaling scaling --threads 1 2 4 8 12 --repeats 1" "run prepared '$PREPARED'" "run d512 '$D512'" "run over '$(over '1 2 4')'"
echo 'echo "stone ended"; } > ~/labour/stone-thr.log 2>&1'; } > "$B/stone.sh"
scp -q "$B/stone.sh" stone:labour/stone-thr.sh; ssh stone "setsid nohup bash ~/labour/stone-thr.sh >/dev/null 2>&1 < /dev/null & disown"

{ cat <<K
export PATH=\$HOME/.cargo/bin:\$PATH; cd ~/code/indrajala-ml; ulimit -v 4500000
AB=".venv/bin/python scripts/measurement/ab.py run --old main --new main --old-crate $CRATE --new-crate $CRATE --order ON"
T=~/thr; mkdir -p \$T
{
K
guard "run calls calls" "run scaling scaling --repeats 1" "run prepared '$PREPARED'" "run d512 '$D512'" "run over '$(over '1 2 3')'"
echo 'echo "klip ended"; } > ~/klip-thr.log 2>&1'; } > "$B/klip.sh"
scp -q "$B/klip.sh" klip:klip-thr.sh; ssh klip "setsid nohup ~/code/thermal-guard.sh ~/klip-thr-temp.log 85 -- bash ~/klip-thr.sh >/dev/null 2>&1 < /dev/null & disown"

# jebel, driven from here
AB=".venv/bin/python scripts/measurement/ab.py remote --host jebel run --old main --new main --order ON"
jrun() { name=$1; shift; eval "$AB --name thr-$name --bench cmd -- $P $*" 2>&1 | tail -1
  ssh jebel "cd ~/code/indrajala-ml; export PATH=\$HOME/.cargo/bin:\$PATH; .venv/bin/python scripts/measurement/ab.py report \$(date +%F)-thr-$name --pooled /tmp/thr-$name.md >/dev/null 2>&1; uptime -s"
  scp -q jebel:/tmp/thr-$name.md "$OUT/jebel/$name.md"; echo "jebel $name done $(date +%T)"; }
ssh jebel "cd ~/code/indrajala-ml; export PATH=\$HOME/.cargo/bin:\$PATH; .venv/bin/python scripts/measurement/ab.py report \$(date +%F)-thr-calls --pooled /tmp/thr-calls.md >/dev/null 2>&1"; scp -q jebel:/tmp/thr-calls.md "$OUT/jebel/calls.md"
jrun scaling scaling --repeats 1
jrun prepared "$PREPARED"
jrun d512 "$D512"
jrun attention "$ATTENTION"
jrun over "$(over '1 2 4 8')"
echo "jebel ended"

until ssh -o ConnectTimeout=10 stone "grep -q 'stone ended' ~/labour/stone-thr.log" 2>/dev/null; do sleep 60; done
scp -q 'stone:labour/thr/*.md' "$OUT/stone/"; ssh stone "cut -c1-165 ~/labour/stone-thr.log" | sed 's/^/stone: /'
until ssh -o ConnectTimeout=10 klip "grep -q 'klip ended' ~/klip-thr.log || grep -q ' ended, status' ~/klip-thr-temp.log" 2>/dev/null; do sleep 60; done
scp -q 'klip:thr/*.md' "$OUT/klip/"; ssh klip "cut -c1-165 ~/klip-thr.log; tail -1 ~/klip-thr-temp.log" | sed 's/^/klip: /'
ls -la "$OUT"/*/ | cut -c30-; echo "all ended"
