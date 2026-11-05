#!/bin/bash
# golden_at.sh SHA: the golden check for the crate at SHA, on jebel (from its A/B wheel, or built) and on pyramidon (built here, under a memory limit).
sha=$1; ML=/home/david/code/indrajala-ml; export PATH=$HOME/.cargo/bin:$PATH
ulimit -v 8000000
W=$(mktemp -d "$(dirname "$0")/golden.XXXX"); git -C $ML/rust worktree add -q --detach "$W/src" "$sha" || exit 2
( cd "$W/src" && $ML/.venv/bin/maturin build --release --out "$W/whl" -i $ML/.venv/bin/python 2>&1 | tail -1 ) 
$ML/.venv/bin/python -m pip install -q --no-deps --no-index --target "$W/site" "$W"/whl/*.whl
ulimit -v unlimited 2>/dev/null
( ulimit -v 5000000; cd $ML && PYTHONPATH="$W/site" .venv/bin/python scripts/measurement/golden_training_run.py check data/refactoring/golden_run.json 2>&1 | tail -3 | sed 's/^/pyramidon: /' )
git -C $ML/rust worktree remove --force "$W/src"; rm -rf "$W"
echo "pyramidon golden ended"
