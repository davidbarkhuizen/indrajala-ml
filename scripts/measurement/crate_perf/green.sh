#!/bin/bash
# green.sh DIR BRANCH: wait for every check on BRANCH's PR to end; exit 0 only if all passed, and say which failed.
cd "$1" || exit 2; sleep 15
while :; do
  rows=$(gh pr view "$2" --json statusCheckRollup -q '.statusCheckRollup[]|[.status,.conclusion,.name]|@tsv')
  [ -n "$rows" ] && ! grep -qvP '^COMPLETED\t' <<<"$rows" && break
  sleep 30
done
bad=$(grep -vP '^COMPLETED\t(SUCCESS|SKIPPED|NEUTRAL)\t' <<<"$rows" | cut -f3)
if [ -n "$bad" ]; then echo "FAILED $2:"; echo "$bad"; exit 1; fi
echo "GREEN $2 ($(wc -l <<<"$rows") checks)"
