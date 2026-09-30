# Workplan: the A/B harness and a stand-alone measurement guide

**Status: done (2026-09-30), stages 1-4 (#488-#491). Decisions D1-D7 settled by the owner (2026-09-30), D4's verdict revised in stage 1.**

This plan adds `scripts/ab.py`, one tool that runs a timing A/B between two commits by this repo's
protocol and reports it. Every A/B so far has been hand-built instead. It also rewrites the
measurement doc as a guide that stands alone: it tells anyone timing a change how to prepare the
machine, which tool answers which question, and how to run and read an A/B. It also states that
A/Bs go through `ab.py` rather than a new driver script each time. No library code changes, so the
golden run is unaffected.

## Why

- **Every A/B was built by hand.** Each recent timing PR (#468, #477, #479, #480)
  wrote a shell driver in a temporary directory, then a pooling
  script, then a table for the PR body. The protocol is the same each time; only the benchmark
  and the commits change. Examples from this week's sessions:
  - #479's driver failed on its first run, because the neutral working directory had no `data/`.
  - #480's pooling script was written three times. Its balancing pair (passes 5 and 6) was added
    by editing the driver with `sed`.
  - #477's old side ran with the old tree on `PYTHONPATH` from the repo's own directory, the
    setup the namespace-package pitfall warns about. Only the `trainers:` line in each pass's log
    showed which tree was imported.
- **Waiting and reading cost the most.** Most of this repo's A/Bs are run by an agent (Claude
  Code), and there the cost is agent turns. The transcripts of eight sessions from 2026-09-29/30
  show the pattern. A session that ran A/Bs made 20-30 `tail`/`cat` checks on running or
  finished passes, plus `sleep` loops and fallback wake-ups. It also read 40-130K characters of raw
  logs and JSON, where the decision needed about ten table rows. Each check is a full turn that
  re-reads the whole conversation.
- **The rules are applied by hand.** [measurement.md](measurement.md) sets them:
  - alternate the builds, commit both sides first, and run one process per measurement;
  - read the other backend, or `prepare`, as the control;
  - treat changes under about 20% as noise unless the passes agree;
  - add a balancing pair when one pass runs fast.

  Some mechanics aren't written down in this repo at all: the old side's worktree, the neutral
  working directory, `PYTHONPATH` order, the `data` symlink, and the crate's `.so` hash check. A
  tool applies all of these the same way every time, and records what it did.

## Where things are now

Counts are from `main` at de59a75.

- **Timing scripts** (`scripts/`) each time one tree. `process_runs.py` gives them
  `interleaved_runs` (one process per cell and repeat, rotating order) and `run_json_worker`.
  Old against new is done outside them:
  - The Python side is compared by putting another checkout first on `PYTHONPATH`, as the
    docstrings of `prepared_dataset_timing.py` and `op_call_timing.py` describe.
  - The crate side is compared by rebuilding or reinstalling the extension between passes
    (`./cli build-rust`, or the pyo3 upgrade's `alternate.sh`, which reinstalled cached wheels
    into the venv and checked the `.so` hash each pass).
- **Raw outputs differ by script:**
  - `prepared_dataset_timing.py --out` writes `{"<config> / <backend>": [{metric: seconds}, ...
    one per repeat]}`;
  - `focused_benchmark.py --json` writes `{settings, results: [{shape, op, batch, median_us,
    ...}]}`;
  - `epoch_op_profile.py --out` writes `{label, runs: {"<architecture> / <trainer>": [{total,
    ops: {op: [seconds, calls]}}]}}`;
  - `op_call_timing.py --json` writes `{settings, runs: {cell: ...}}`;
  - `accuracy_pass_timing.py` and `batch_size_timing.py` write `--out` too; stage 2 maps both.
- **The extension** lives in the venv: one build at a time, from `maturin develop --release`.
- **The measurement doc** is `docs/optimizations/measurement.md`, 134 lines: the machine, a tools
  table, protocols, gotchas, judging correctness, and the rules for an optimization PR. Three
  workplans and every timing PR rely on it, not only optimization work. It is linked 14
  times from 8 docs (README included), and from the docstrings of four scripts.
- **`machine_profile.py compare`** checks the machine's identity. Run in a shell without
  `~/.cargo/bin` on `PATH`, it reads `rustc` as null (seen while writing this plan).
- **The raw data of past A/Bs** is kept outside the repo in `~/code/ab-runs/archive/`. It covers
  #477 (both A/Bs), #479, #480, and the pyo3 upgrade's crate A/B (`epoch_op_profile.py`,
  `focused_benchmark.py` and a boundary probe, with its `alternate*.sh` drivers). The copy comes
  from temporary directories that are cleared at reboot. Each PR body has the table published from
  its data.

## Decisions (settled)

The owner settled D1-D7 on 2026-09-30, each as recommended.

- **D1: the tool lives in this repo.** `scripts/ab.py`, with tests in `tests/test_ab.py`. It is
  versioned with the scripts whose output it parses and tested in CI. Crate A/Bs are run from
  here too, since all timing is done through the Python API. A separate tools repo, like
  `upgrade-tools`, would be untested and drift from the scripts.
- **D2: both sides are commits, each in a detached worktree.** `--old` (default `main`) and
  `--new` (default `HEAD`) resolve to commits. Each gets a detached worktree under
  `~/code/ab-worktrees/<sha7>`, created once and reused. The working checkout is never timed, so
  "commit both builds" is enforced and the checkout stays free for other work. `run` refuses if the
  checkout has uncommitted changes under `indrajala_ml/`, `scripts/` or `rust/` and `--new` is
  `HEAD`, since that usually means a change that wasn't committed yet.
- **D3: the crate is switched by cached wheels in target directories.** One wheel per crate
  commit, built once and cached, installed with `pip install --no-deps --target` into a site
  directory for that commit. Each pass puts that directory on `PYTHONPATH` after its tree. The
  venv is never touched, so a failed run leaves nothing to restore. Each pass checks the `.so`
  hash it imported. Reinstalling into the venv each pass, as `alternate.sh` did, was rejected: a
  run that dies partway can leave the venv on the wrong build.
- **D4: the report gives a verdict per row, and a pass is shifted at 5%.** Its rules, read from
  the protocol:
  - **The pooled table** has, per (case, metric), old and new medians, the min-max over all runs,
    Δ median, and each pass's own median. This is #480's table.
  - **A verdict per row.** *Consistent* means every per-pass median of one side lies beyond every
    per-pass median of the other, with at least 2 passes a side, **and the gap between the sides
    is wider than each side's own spread of per-pass medians**; the report gives Δ. Otherwise the
    row is *within noise*, and the report gives |Δ| and counts the rows that are separated but
    inside their spread. This is "two passes agree", made exact.
  - **Revised in stage 1 (owner, 2026-09-30).** Separation alone happens by chance 1 time in 3 at 2
    passes a side (2 of the 6 orderings), 1 in 10 at 3. On the archived A/Bs it flagged 13 of 24
    rows of #479 (no change) and 9 of 24 of #480's first four passes; the rule above flags 3 and 0,
    and none of #480's six passes. The default order became `ONNONO`, 3 passes a side.
  - **Controls.** Each benchmark names its control rows (`prepare`, or the other backend when
    `--control-backend` is given). If a control row is not within noise, the report says so
    first: the A/B can't resolve a change at this level, and `epoch_op_profile.py` is the next
    step.
  - **Shifted passes.** A pass is shifted when its rows, the control included, sit 5% or more from
    the pooled medians in the same direction (the ratio of pass median to pooled median, taken as
    the median over rows). #480's passes 1 and 5 were 5-15% fast, while the other passes of most
    rows agreed within about 2%. When shifted passes are unbalanced between the sides, the report
    names the `extend` order that balances them.
  - The report advises; the claim in a PR remains its author's.
- **D5: runs are kept in `~/code/ab-runs/<YYYY-MM-DD>-<name>/`,** outside every repo, which
  persists across sessions and reboots. The name defaults to `<branch>-<bench>`. A run holds
  `manifest.json` and each pass's raw output and log, so `report` can be re-run later, and a past
  run can become a test fixture. The wheel cache (D3) is `~/code/ab-runs/wheels/`. A git-ignored
  directory in the repo was rejected: each worktree would have its own, and `git clean -fdx` would
  delete the history.
- **D6: the guide moves to `docs/measurement.md`.** `docs/optimizations/measurement.md` is
  rewritten there as the stand-alone guide, and `optimizations.md` keeps a link. Every timing PR
  and workplan uses it, so it isn't an optimizations sub-page. The move updates 14 doc links and
  4 script docstrings.
- **D7: a `CLAUDE.md` at the repo root** (there is none today), about 15 lines. It points to the
  guide and states the A/B rule and the output rules below ([The agent
  side](#the-agent-side)). Claude Code loads it in every session in this repo, so the rules don't
  depend on an agent's memory notes or on reading the guide first.

## The design

### Commands

```
python scripts/ab.py run --bench prepared_dataset_timing [--old main] [--new HEAD]
                         [--order ONNONO] [--name stage-3b] [--control-backend numpy]
                         [--allow-profile-change] [-- <benchmark arguments>]
python scripts/ab.py status [RUN]
python scripts/ab.py extend [RUN] [--order NO]
python scripts/ab.py report [RUN] [--brief] [--md FILE]
python scripts/ab.py clean [--worktrees] [--wheels]
```

`RUN` defaults to the most recent run. `--bench cmd -- <probe.py ...>` runs any probe that follows
the [probe contract](#benchmarks-adapters-and-the-probe-contract).

**`run`**, in order:
1. **Resolve.** Turn `--old` and `--new` into commits and create or reuse their worktrees. Read
   each tree's `rust/` submodule commit:
   - If they are equal, both sides use the venv's extension, and its `.so` hash is recorded.
   - If they differ, the crate is switched per pass (D3). Before stage 3, `run` refuses.
2. **Pre-flight.**
   - Put `~/.cargo/bin` on `PATH`.
   - Run `machine_profile.py compare` against the reference, and refuse on an identity change
     unless `--allow-profile-change` is given.
   - Record the thread env vars (`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`), the load average,
     and any process above 10% CPU.
   - Symlink `data` into the run directory, which is the neutral working directory.
3. **Smoke.** Run each side once with the benchmark's smallest settings (its adapter's
   `smoke_args`, e.g. one config and one repeat). This catches an import, path or data mistake in
   about a minute, not after the first 40-minute pass.
4. **Passes.** Run the passes in `--order` (default `ONNONO`); each one:
   - runs in its own process tree with the run directory as the working directory;
   - sets `PYTHONPATH=<tree>[:<crate site>]` in the environment, so the benchmark's worker
     processes inherit it;
   - runs the benchmark script from the new tree (`--script-from old` overrides this; a
     benchmark script uses only interfaces both sides have).
5. **Provenance.** Before each pass, a probe process in the pass's environment prints the files
   `indrajala_ml.train` and `indrajala_math_rust` resolve to, and the `.so` hash. The pass aborts
   unless they are under the expected tree and crate.
6. **Records.**
   - `progress.jsonl` gets a line per pass: start, end, exit status, load average, and
     busy processes.
   - `manifest.json` gets the commits, the crate hashes, the order, the command lines, the profile
     result and the environment.

**Output is bounded.** Raw data only goes to files.
- `run` prints one line when it starts, with an ETA when a previous run of the same benchmark
  and arguments exists. It prints one line when it finishes: `done <RUN>: 4 passes, 38 min; next:
  ab.py report --brief`.
- On a failure, `run` prints the failing step's last 20 lines of stderr and exits 1.
- `status` prints one line: which pass is running, how many are done, and the ETA.
- `report --brief` prints at most about 15 lines:
  - a header (commits, order, profile check, load warnings);
  - control and shifted-pass findings, if any;
  - one line per *consistent* row;
  - one summary line for the rest ("12 rows within noise, max |Δ| 1.5%").
- `report --md FILE` writes the full table and a protocol paragraph for the PR body. The
  paragraph gives the commits, the order, the command, the neutral working directory, the
  provenance check, and why any balancing pair was added.

**`extend`** adds passes to a finished run in the given order, numbered after the existing ones,
under the same pre-flight and provenance checks. **`clean`** removes worktrees and cached wheels
that no run under `~/code/ab-runs` from the last 14 days refers to.

### Benchmarks: adapters and the probe contract

An adapter is a small class per timing script:
- `argv(out_path, extra)` builds the command line;
- `smoke_args` gives the smallest settings;
- `default_args` gives the default settings;
- `rows(out_path)` yields `Row(case, metric, value, unit)`;
- `controls` names the control rows.

Stage 1 has `prepared_dataset_timing`. Its control is `prepare`, and with `--control-backend` also
the other backend's rows. Stage 2 adds `focused_benchmark`, `epoch_op_profile`,
`accuracy_pass_timing`, `op_call_timing` and `batch_size_timing`.

**The probe contract** covers the in-process probes written while investigating, such as #477's
optimizer step probe and the pyo3 upgrade's boundary probe. A probe prints one JSON object per line
to stdout: `{"case": ..., "metric": ..., "value": ..., "unit": ...}`. `ab.py` runs it once per pass
like any benchmark. A probe that proves useful becomes a script with an adapter in the same PR, not
a scratch file.

### The agent side

These rules go in the guide and in `CLAUDE.md` (D7):

- **Launch in the background and don't poll.** Start `ab.py run` as a background command, and let
  its exit be the signal. Use `status` only when the owner asks. Don't schedule wake-ups for runs
  under an hour.
- **Read the brief report only.** Read `report --brief` and nothing else from the run. Open a raw
  pass file only when the brief report flags something it can't explain.
- **Pass the table on without reading it.** Put `report --md` into the PR body by concatenating
  files, not by reading the table and retyping it.
- **Keep the machine quiet.** Nothing CPU-heavy runs while an A/B does: no tests, lint or builds.
  Reading code and writing docs are fine.
- **Extend `ab.py` instead of working around it.** A measurement `ab.py` can't express gets a
  probe (the contract above) or an adapter, not a new driver script.

## Pitfalls to design around

- **The namespace package.** `indrajala_ml` resolves by `sys.path` order, so running from inside a
  tree can import that tree whatever `PYTHONPATH` says. The fix is to always run from the run
  directory and to check provenance each pass. #477's A/B ran from the repo's directory and came
  out right only because of the order `PYTHONPATH` gave. Nothing checked it.
- **`data/` is relative to the working directory** (`prepared_dataset_timing.py` reads
  `data/mnist/`), so the run directory gets a `data` symlink. #479's first run failed on this.
- **Worker processes** (`run_json_worker`) inherit the environment, not `sys.path` edits. The tree
  goes into the environment's `PYTHONPATH`, never into `sys.path` inside `ab.py`.
- **Worktrees don't initialize submodules.** The Python side doesn't need them: the crate comes
  from the venv or the wheel cache. Wheels are built in a worktree of `~/code/indrajala-math-rust`
  at the submodule's commit, fetched first if missing, with the toolchain its
  `rust-toolchain.toml` pins.
- **`rustc` on `PATH`.** Without `~/.cargo/bin`, builds fail and the profile reads `rustc` as null.
  `ab.py` sets it itself.
- **Whole passes shift** by 5-15% in every config, the controls included (#480, and #477's first
  A/B). The report detects this and balances it with `extend`. It never drops a pass.
- **Times can be bimodal between processes** (the max-pool downstream gotcha). The report shows the
  min-max next to the medians, and per-pass medians, so a second mode is visible. It doesn't try to
  classify modes.
- **BLAS threads.** `ab.py` doesn't change any benchmark's threading. It records the thread env
  vars in the manifest, so a comparison can't silently mix settings.
- **The script's own version.** By default the benchmark script comes from the new tree. If the
  PR changes the script itself, both sides still run the new script, which is the point. Old
  results from an old script are not comparable, and the manifest records which script ran.
- **Interrupted runs** leave worktrees and partial passes. `extend` and `report` use the complete
  passes only, and `clean` removes what's unreferenced.
- **CI must not time anything.** `tests/test_ab.py` covers the report on archived fixtures, and
  `run`/`extend` on a toy probe (`--bench cmd`) in a temporary git repo with two commits. It needs
  no MNIST, crate build or real benchmark.

## Stages

Each stage is one PR, merged before the next. A stage that touches the tool also updates the tools
table and the A/B protocol in the measurement doc. Stage 4 rewrites the doc as the guide.

### Stage 1: `ab.py` for worktree A/Bs of `prepared_dataset_timing`

- `run`, `status`, `extend`, `report` (`--brief`, `--md`) and `clean` (worktrees only), with the
  pre-flight, smoke, provenance and output rules above, and the `cmd` probe benchmark.
- Fixtures: the archived raw data of #477, #479 and #480, trimmed to the JSON the report reads,
  under `tests/fixtures/ab/`, with each PR's published table as the expected output.
- Tests:
  - the report on #480's fixture reproduces its PR table: all 8 rows, the medians, ranges, Δ
    and per-pass medians;
  - it marks passes 1 and 5 as shifted and balanced;
  - it matches #479's and #477's tables;
  - the order parsing and the numbering after `extend`;
  - provenance: a pass whose tree lacks the module aborts;
  - `run` and `extend` end to end on a toy probe in a temporary repo;
  - the output bounds: `--brief` stays within its line limit on every fixture.
- **Gate:** an A/A on the benchmark machine (`--old main --new main`, `ONNONO`, `--repeats 5`), with
  the browser and editor closed. The brief report must call every row within noise, with no
  control finding. The PR quotes that brief report verbatim, with its line count.

- **As built** (differences from the design above):
  - the machine check runs from the checkout, not the new worktree: worktrees have no `rust/`
    submodule, so `machine_profile.py` there reads the crate's release profile as null, and the
    crate both sides import is the venv's, built from the checkout's `rust/`;
  - `--skip-profile` (tests and toy probes; the report says "skipped"), `--repo` (the tests' toy
    repository), and `AB_RUNS_ROOT` / `AB_WORKTREES_ROOT` to move the two directories;
  - `extend --note` records why passes were added; by default, the shifted passes they balance;
  - the run check refuses uncommitted changes to tracked files only, so scratch probes in
    `scripts/` don't block a run;
  - a probe row may carry `"control": true`.

### Stage 2: the other benchmarks' adapters

- Adapters for `focused_benchmark`, `epoch_op_profile`, `accuracy_pass_timing`, `op_call_timing`
  and `batch_size_timing`:
  - each with its `smoke_args`;
  - each with its controls: the other backend where the script runs both, none where it is
    Rust-only (the report then says "no control").
- Tests: from the pyo3 upgrade's archived data, the report reproduces the tables its
  `summarize.py` produced (the epoch_op_profile and focused_benchmark rows), and the boundary probe
  goes through the probe contract.
- **Gate:** one short real run per adapter on the machine, as an A/A, with every row within noise.

- **As built:**
  - rows per adapter: `focused_benchmark` gives µs per call per (shape op batch, malloc, backend),
    each of its `--passes` a run; `epoch_op_profile` gives seconds per `<config> <op>` and the
    profiled total; `accuracy_pass_timing` leaves out the mismatch counts (not timings);
    `op_call_timing` gives the whole run's seconds and µs per call per argument shapes;
    `batch_size_timing` gives each measure per `B=<size> / <backend>`;
  - controls: the two-backend scripts take `--control-backend`; the brief report says which
    benchmarks are Rust only;
  - the pyo3 fixtures are the archive trimmed to the fields read, and the boundary probe's lines
    were rewritten to the probe contract (`per call`, ns). `summarize.py` dropped a run's op under
    5 ms, so the epoch test compares the ops at or above it in every run.

### Stage 3: crate A/Bs

- `--old-crate` and `--new-crate` default to each tree's `rust/` submodule commit, and a crate
  commit can be given alone for a crate-only A/B.
- Wheels are cached in `~/code/ab-runs/wheels/<crate sha>/`, installed with `--target` (D3), and
  built as in [Pitfalls](#pitfalls-to-design-around).
- Every pass checks the `.so` hash, and `clean --wheels` removes unreferenced wheels.
- Tests: the site-directory and hash-check logic on a fake extension module. No build runs in CI.
- **Gate:** a crate A/B of two commits whose diff doesn't touch `src/` (crate `750d83a` against its
  parent, #41: a test reference only). The report calls every row within noise and shows the
  hashes.

- **As built:**
  - wheels are built from a `git archive` of the crate commit, not a worktree, so there is no crate
    worktree to clean; the crate repository is fetched once when the commit is missing;
  - builds share one cargo target directory, `~/code/ab-runs/wheels/target/`, so a second crate's
    build is incremental; `clean --wheels` keeps it;
  - maturin is the venv's (`./cli`'s pin), building for the venv's interpreter;
  - a crate A/B is any run whose sides' crate commits differ, so `--old main --new main
    --old-crate X --new-crate Y` is a crate-only A/B;
  - the brief report's header and the protocol paragraph give each side's crate commit and
    extension hash.

### Stage 4: the stand-alone measurement guide

The guide is written to the tool as built, at `docs/measurement.md` (D6). Its contents:

1. **What to measure, and when not to.** Pure Python is never timed. A PR that changes no
   `learn*` or `classify_rows` path needs no A/B, as #481-#483 said.
2. **Preparing the machine.** The profile check, closing the browser and editor, `PATH`,
   `perf_event_paranoid`, and BLAS threads. This moves in from the owner's notes and the current
   doc.
3. **Choosing the measurement.** The tools table, from the quick survey to the decisive number.
4. **Running an A/B with `ab.py`.**
   - Worked examples: a worktree A/B, a crate A/B, a probe, `extend` after a shifted pass, and
     putting `--md` into a PR.
   - The rule: A/Bs go through `ab.py`. Hand-built drivers and pooling scripts are not used. What
     it can't express becomes a probe or an adapter in the same PR.
5. **Reading the report.** The verdicts, the controls, shifted passes, and what to do next in each
   case.
6. **Protocols.** The rules and why each exists, including the ones `ab.py` now applies for you.
7. **Gotchas.** The current list, plus the namespace package, `data/`, and pass shifts.
8. **Judging correctness.** Unchanged.
9. **Rules for a timing claim in a PR, and for an optimization PR.** Unchanged in substance.
10. **For agents.** The rules in [The agent side](#the-agent-side), which `CLAUDE.md` repeats.

The stage also:
- updates every link to the old path;
- repoints the four scripts' docstrings that describe running old/new by hand to `ab.py`;
- updates the README's `scripts/` row, and adds `CLAUDE.md` (D7).

- **Gate:**
  - `./cli lint`, since ruff also formats Python blocks in Markdown;
  - a grep that finds no link to the old path;
  - the owner reads the guide as a newcomer would.

- **As built:** the guide is [measurement.md](measurement.md), numbered 1-10 as above so its
  anchors are stable; `optimizations.md` links to it. `CLAUDE.md` is 18 lines. The scripts whose
  docstrings described old/new by hand (`prepared_dataset_timing.py`, `epoch_op_profile.py`,
  `op_call_timing.py`) point to `ab.py`; `machine_profile.py` and `focused_benchmark.py` point to
  the new path. The workplan's own history still names the old path, as text, not a link.

## After this plan

- Update the agent's memory notes to point at the guide, and retire `upgrade-tools/alternate*.sh`
  and the machine-notes entries the guide now covers.
- After the first two real A/Bs through `ab.py`, count the agent turns and characters read per A/B
  from the session transcripts, against this plan's numbers (20-30 checks, 40-130K characters).
  Record the result in the stage 4 PR or a follow-up.
- Worth a similar treatment: waiting on CI (`gh pr checks --watch` loops). This plan doesn't
  cover it.

## Out of scope

- **Non-timing gates.** The golden run, tests and lint are each already one command.
- **`perf_region.py` counters** need `sudo` and a driver marking regions. An adapter can come
  later.
- **Significance tests** (Mann-Whitney and the like). The protocol judges by ranges and agreement
  between passes. The report makes that exact and adds no p-values.
- **Other machines.** The profile check refuses them. Comparing machines is a different question.
