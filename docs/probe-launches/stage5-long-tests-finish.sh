#!/bin/bash
# Division stage 5's long tests, the end (pyramidon): when every certified network's long test is
# in division-claim/long-tests/ (jebel's copied by its workers), write the summary
# (long-tests-summary.txt: each network's cases, passed, failures) and the manifest
# data/window-recurrent/division-claim-long-tests.sha256 (run from data/window-recurrent).
cd ~/code/indrajala-ml/data/window-recurrent || exit 1
D=division-claim
want=$(ls $D/{family,quotient}/seed*/certified-step-*.network.json | wc -l)
while [ "$(ls $D/long-tests/*/*.long-test.json 2>/dev/null | wc -l)" -lt "$want" ]; do sleep 300; done
sleep 120  # let a last rsync finish
python3 - "$D" > $D/long-tests-summary.txt <<'EOF'
import json, sys
from pathlib import Path
total = passed = 0
bad = []
for path in sorted(Path(sys.argv[1], "long-tests").glob("*/*.long-test.json")):
    d = json.loads(path.read_text())
    cases = sum(p["cases"] for n in d["lengths"].values() for p in n["properties"].values())
    ok = sum(p["passed"] for n in d["lengths"].values() for p in n["properties"].values())
    fails = sum(len(p["failures"]) for n in d["lengths"].values() for p in n["properties"].values())
    total, passed = total + cases, passed + ok
    if ok != cases or fails:
        bad.append(path.parent.name)
    print(f"{path.parent.name}: {ok:,} of {cases:,} passed, {fails} failures listed, sha256 {d['sha256'][:12]}")
print(f"all: {passed:,} of {total:,} passed; networks with a failure: {bad or 'none'}")
EOF
find $D/long-tests -type f | sort | xargs sha256sum > division-claim-long-tests.sha256
echo "manifest $(sha256sum division-claim-long-tests.sha256 | cut -c1-64) $(date)" >> $D/long-tests-summary.txt
