#!/bin/bash
# The deadline's completion (D67): pyramidon's move-to-front-4 34 and 35 moved to jebel at their
# checkpoints (the owner, 2026-10-23: "move them to jebel ... when the capacity becomes available").
# Placement is where, not what (D67's registration: bit parity across machines); the same command at
# the probe's commit (41f92e7) with --resume, as D63's moved runs. Run on pyramidon.
set -eu
SRC=~/code/data-d67/probe-d67
DST=code/data-d67/probe-d67-d64  # on jebel, beside D64's S3 45 (the same probe, worktree wt-p61 at 41f92e7)
PGID=$(ps -o pgid= -p "$(pgrep -f '^[^ ]*/python[0-9.]* scripts/[m]odular_family_probe.py run --runs move-to-front-4:34' | head -1)" | tr -d ' ')
[ "$(ps -o cmd= -p "$PGID")" = "bash $HOME/code/probe-d67.sh $HOME/code/wt-d64 $SRC $HOME/code/ab-runs-d67 4 d64:mtf4:34 d64:mtf4:35" ] \
  || { echo "process group $PGID isn't D67's move-to-front-4 probe"; exit 1; }
echo "stop $PGID $(date)" >> $SRC/launch.log
kill -TERM -- "-$PGID"
while pgrep -f "^[^ ]*/python[0-9.]* scripts/[m]odular_family_probe.py run --runs move-to-front-4:3[45]" > /dev/null; do sleep 2; done
for s in 34 35; do
  place=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['place'] if isinstance(json.load(open(sys.argv[1]))['place'], str) else 'mid-run')" $SRC/d64-mtf4-seed$s/seed$s/checkpoint.json)
  [ "$place" = mid-run ] || { echo "seed $s checkpoint is $place, not resumable"; exit 1; }
done
ssh jebel "mkdir -p ~/$DST"
rsync -a $SRC/d64-mtf4-seed34 $SRC/d64-mtf4-seed34.out $SRC/d64-mtf4-seed35 $SRC/d64-mtf4-seed35.out jebel:$DST/
mkdir -p ~/code/data-d67/moved-to-jebel
mv $SRC/d64-mtf4-seed34 $SRC/d64-mtf4-seed34.out $SRC/d64-mtf4-seed35 $SRC/d64-mtf4-seed35.out ~/code/data-d67/moved-to-jebel/
echo "moved d64-mtf4-seed34 d64-mtf4-seed35 to jebel:$DST $(date)" >> $SRC/launch.log
ssh jebel bash -s <<'REMOTE'
set -eu
OUT=~/code/data-d67/probe-d67-d64
CELL="--factors 2 --codes-ratio 32/11 --reads output --window-width 32 --budget-epochs 1600 \
  --budget-minutes 100000 --anneal --anneal-tail 40 --anneal-never-commit --warm-up --melt-hot 2 \
  --melt-kappa 1e-2 --melt-hold 20 --melt-halving 30 --melt-shape exponential --crucible \
  --warm-up-screen 1.01 --continuous --workers 1 --margin-wait 200"
cd ~/code/wt-p61
[ "$(git rev-parse --short HEAD)" = 41f92e7 ]
for s in 34 35; do
  name=d64-mtf4-seed$s
  echo "launch $name --resume (moved from pyramidon) at $(git rev-parse --short HEAD) $(date)" >> $OUT/launch.log
  setsid nohup bash -c "AB_RUNS_ROOT=$HOME/code/ab-runs-d67-d64-mtf4-$s OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=$HOME/code/site-straight:$HOME/code/wt-p61 $HOME/code/indrajala-ml/.venv/bin/python \
    scripts/modular_family_probe.py run --runs move-to-front-4:$s $CELL --state-bound 24 --resume \
    --out $OUT/$name >> $OUT/$name.out 2>&1; echo \"ended $name: \$(tail -1 $OUT/$name.out) \$(date)\" >> $OUT/launch.log" \
    > /dev/null 2>&1 < /dev/null &
  sleep 20
done
REMOTE
sleep 60
ssh jebel 'tail -3 ~/code/data-d67/probe-d67-d64/launch.log; for s in 34 35; do tail -1 ~/code/data-d67/probe-d67-d64/d64-mtf4-seed$s/seed$s/log.jsonl | cut -c1-120; done'
