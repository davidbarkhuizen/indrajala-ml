# indrajala-ml

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
- **Keep the machine quiet during a run:** no tests, lint or builds; reading and writing are fine.
  Ask the owner to close the browser and editor before starting one.
- A PR that changes no `learn*` or `classify_rows` path needs no A/B. Every change passes the
  golden run: `.venv/bin/python scripts/golden_training_run.py check data/refactoring/golden_run.json`.
- `./cli test` runs both test suites, `./cli lint` runs ruff and strict pyright (ruff also formats
  Python blocks inside Markdown, so lint doc-only changes too).
