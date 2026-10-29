#!/bin/bash
# D38's coupling probe (probe-d38.sh, its settings unchanged) started at 19:00, once the D33 probe's
# five runs started at 07:24 have reached their 690-minute budgets: 9 runs and D33's seed 7 on
# stone's 12 cores. Arguments as probe-d38.sh's.
set -u
now=$(date +%s); at=$(date -d 19:00 +%s); [ $at -gt $now ] && sleep $((at - now))
exec ~/labour/probe-d38.sh "$@"
