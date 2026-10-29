#!/bin/bash
# D64 on stone: stopped at 06:55 on Friday 2026-10-23 (the owner allowed half-load on Thursday only), resumed that evening with --resume.
sleep $(( $(date -d "2026-10-23 06:55" +%s) - $(date +%s) ))
L=~/labour/data/state-formation/option1/probe-d64/launch.log
pkill -TERM -f "[p]robe-d64.sh"
pkill -TERM -f "option1/probe-[d]64/"
echo "stopped $(date)" >> $L
