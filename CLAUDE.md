# indrajala-ml

## Workflow

- **One focused PR per chunk of work, from a feature branch**, never on `main`: branch, commit,
  push, open the PR, wait for CI to pass, squash-merge, `git pull` on `main`, then delete the local
  and remote branch.
- **Multi-stage changes start with a workplan doc** (`docs/<name>-workplan.md`). Settle its owner
  decisions (D1, D2, ...) and write the answers into the doc before opening its PR. Each stage is
  its own PR, merged before the next. A stage that changes the crate lands in
  `indrajala-math-rust` first, then a "Bump rust/" PR here.
- **Retire a workplan only when it is complete** (every stage and decision resolved); its leftover
  work moves into [docs/next-steps.md](docs/next-steps.md).
- **"Bit-identical" is claimed only when checked** (golden run, parity tests). The golden run may
  be re-recorded only for an owner-approved correctness improvement (docs/measurement.md, §8).

## Timing

Timing a change follows [docs/measurement.md](docs/measurement.md); read it before any benchmark.

- **Every old-against-new timing goes through `scripts/ab.py`** (`run`, then `report --brief`).
  No hand-built driver or pooling scripts: a measurement it can't express gets a probe or an
  adapter in `ab.py`, in the same PR.
- **Launch `ab.py run` in the background and don't poll.** Its exit is the signal; use `status`
  only when the owner asks, and schedule no wake-ups for runs under an hour.
- **Read `report --brief` only.** Open a raw pass file only when the brief report flags something
  it can't explain.
- **In a crate A/B, the header's two `.so` hashes must differ** when the crate's Rust changed; the
  same hash twice means one crate was timed on both sides (docs/measurement.md, Gotchas).
- **Put `report --md FILE` into a PR body by concatenating files**, not by reading and retyping it.
- **Check the machine, then go; don't ask.** Before a run, check `uptime` and `ps`; `pkill` Brave
  and Zed if they are running, and wait out a high load. Keep the machine quiet during the run:
  no tests, lint or builds; reading and writing are fine.
- A PR that changes no `learn*` or `classify_rows` path needs no A/B, and neither does one confined
  to pure-Python code (never timed). Every change passes the golden run:
  `.venv/bin/python scripts/golden_training_run.py check data/refactoring/golden_run.json`.

## Testing

- `./cli test` runs both test suites, `./cli lint` runs ruff and strict pyright (ruff also formats
  Python blocks inside Markdown, so lint doc-only changes too).
