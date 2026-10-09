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

- **Development runs on `pyramidon`; timing runs on `jebel`,** started from `pyramidon` with
  `ab.py remote --host jebel run ...` (docs/measurement.md, §4). Push both sides first.
- **Every old-against-new timing goes through `scripts/ab.py`** (`run`, then `report --brief`).
  No hand-built driver or pooling scripts: a measurement it can't express gets a probe or an
  adapter in `ab.py`, in the same PR.
- **Launch `ab.py remote --host jebel run` in the background and don't poll.** Its exit is the
  signal, with the brief report; use `status` only when the owner asks, and schedule no wake-ups
  for runs under an hour. A dropped ssh session ends only the wait: run the `remote ... wait` it
  names.
- **Read `report --brief` only.** Open a raw pass file only when the brief report flags something
  it can't explain.
- **In a crate A/B, the header's two `.so` hashes must differ** when the crate's Rust changed; the
  same hash twice means one crate was timed on both sides (docs/measurement.md, Gotchas).
- **Put `remote ... report RUN --md FILE` into a PR body by concatenating files**, not by reading
  and retyping it.
- **After a reboot, `ab.py remote run` re-applies `jebel`'s setup itself** (frequency policy, PL1
  65 W, `thermald` stopped, snap refreshes held), through the root-owned copy the owner installs
  with `sudo scripts/install_benchmark_setup.sh`. When it says the copy is missing or stale, ask
  the owner to run that in their own terminal on `jebel`.
- **Check `jebel`, then go; don't ask.** Before a run, check its `uptime` and `ps` over ssh;
  `pkill -x` Brave, Firefox and Zed there if they are running, and wait out a high load, an apt
  job or a snap refresh (measurement.md §2). Nothing else runs on `jebel` during the run;
  development on `pyramidon` carries on.
- **A PR's tier decides its timing** (docs/measurement.md, §1): **0**, no timed path changed
  (new functionality that leaves existing timed paths alone, docs, pure-Python code, never timed):
  no A/B; **1**, a timed path (`learn*`, `classify_rows`, the crate) changed with no speedup
  claimed: one A/B of the most relevant benchmark (§1's table) with `--order ONNO`, and a
  consistent slower row is extended (`extend --order NO`) before it counts; **2**, a speedup
  claimed: the full protocol, 6 passes; **3**, a release, toolchain, numpy/BLAS, kernel or BIOS
  change, or new machine: the full baseline. Tier 2 and 3 runs go to the archive once the PR
  merges (`ab.py archive`, §10). Every change that could reach training results (the
  package, the crate, the golden run's script or its data) passes the golden run:
  `.venv/bin/python scripts/golden_training_run.py check data/refactoring/golden_run.json`.
  A change that can't (docs, machine profiles, CI config) needs no golden run.

## Testing

- `./cli test` runs both test suites, `./cli lint` runs ruff and strict pyright (ruff also formats
  Python blocks inside Markdown, so lint doc-only changes too).
