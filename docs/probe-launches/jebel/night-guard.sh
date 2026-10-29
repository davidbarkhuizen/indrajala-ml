#!/bin/bash
# jebel at night (18:00-06:00): at most 4 study runs going (the owner: half capacity overnight). The newest
# runs past 4 are paused (SIGSTOP to the run's whole process tree: the training is in a forkserver
# worker) and resumed (SIGCONT), oldest first, as slots free or at 06:00. Its paused runs in LOG.paused.
# Usage: night-guard.sh LOG   (runs until killed)
LOG=$1; PAUSED=$LOG.paused; touch $PAUSED
runs() { ps -eo pid=,args= --sort=start_time | grep "scripts/modular_family_probe.py run" | grep -v forkserver | grep -v grep | awk '{print $1}'; }
tree() { echo $1; for C in $(pgrep -P $1); do tree $C; done; }
while true; do
  H=$((10#$(date +%H)))
  if [ $H -ge 18 ] || [ $H -lt 6 ]; then LIMIT=4; else LIMIT=1000; fi
  for P in $(cat $PAUSED); do kill -0 $P 2>/dev/null || sed -i "/^$P\$/d" $PAUSED; done  # ended while paused
  GOING=$(runs | grep -vxF -f $PAUSED | wc -l)
  if [ $GOING -gt $LIMIT ]; then
    for P in $(runs | grep -vxF -f $PAUSED | tail -n $((GOING - LIMIT))); do
      kill -STOP $(tree $P) 2>/dev/null && echo $P >> $PAUSED && echo "paused $P $(date)" >> $LOG
    done
  elif [ $GOING -lt $LIMIT ] && [ -s $PAUSED ]; then
    for P in $(head -n $((LIMIT - GOING)) $PAUSED); do
      kill -CONT $(tree $P) 2>/dev/null; sed -i "/^$P\$/d" $PAUSED; echo "resumed $P $(date)" >> $LOG
    done
  fi
  sleep 60
done
