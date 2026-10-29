#!/bin/bash
# Stop D63's four remaining runs on stone (moved to pyramidon and klip, the owner 2026-10-23 05:30),
# their launcher subshells first so no false "ended" line is logged.
L=~/labour/data/state-formation/option1/probe-d63/launch.log
for R in random-q12-0:6 random-q12-0:7 move-to-front-4:34 move-to-front-4:35; do
  P=$(pgrep -f "^[^ ]*/python[0-9.]* scripts/[m]odular_family_probe.py run --runs $R ")
  [ -z "$P" ] && { echo "no run $R"; continue; }
  PP=$(ps -o ppid= -p $P | tr -d ' ')
  grep -q "probe-d63.sh" /proc/$PP/cmdline && kill -KILL $PP
  kill -TERM $P
  for i in $(seq 60); do kill -0 $P 2>/dev/null || break; sleep 1; done
  pkill -TERM -f "\-\-runs $R .*option1/probe-[d]63/" ; sleep 2
  echo "moved $R (pid $P, subshell $PP) $(date)" | tee -a $L
done
