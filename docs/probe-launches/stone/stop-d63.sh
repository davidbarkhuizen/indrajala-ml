#!/bin/bash
# D63 on stone: stopped at 06:55 on Friday 2026-10-23 (the owner: stone's whole machine until 07:00),
# resumed that evening with --resume. A script file, so the command line that runs it names no pattern.
sleep $(( $(date -d "2026-10-23 06:55" +%s) - $(date +%s) ))
L=~/labour/data/state-formation/option1/probe-d63/launch.log
pkill -TERM -f "[p]robe-d63.sh"
pkill -TERM -f "option1/probe-[d]63/"
echo "stopped $(date)" >> $L
