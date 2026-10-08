# Measurement

How to time a change in this repository: when to measure, how to prepare the machine, which tool
answers which question, how to run an A/B with `scripts/ab.py` and read its report, the protocols
behind it, what goes wrong, and the rules a timing claim in a PR must meet. It stands alone: every
timing PR and workplan relies on it, not only optimization work.

## 1. What to measure, and when not to

- **Pure-Python networks are never timed.** They are for correctness and parity only.
- **A PR that changes no `learn*` or `classify_rows` path needs no A/B.** Say so in the PR, as
  #481-#483 did. A refactor through those paths needs one, even if it should cost nothing: #477's
  first A/B found a 5-8% regression from a runtime `cast` on a generic Protocol.
- **Correctness first.** A change is judged by parity and the golden run
  ([§8](#8-judging-correctness)), never by timing or accuracy.
- **A timing claim is measured, not argued.** A number in a PR comes from a run by this guide,
  with its commits and protocol.

## 2. Preparing the machine

The benchmark machine is a Core i7-9700K desktop (`jebel`): 8 cores / 8 threads (Coffee Lake, no
SMT), 32 KB L1d and 256 KB L2 per core, 12 MB L3, AVX2 and FMA without AVX-512, `intel_pstate`
(active, hardware-managed clocks) with the `powersave` governor, EPP `balance_performance` and
turbo on, and the package power limit PL1 lowered to 65 W (at the stock 95 W a long all-core load
reaches 100 C and throttles). Its full record is `docs/machine_profiles/i7-9700k.json`, and its
noise and baseline, from six A/As of every benchmark, are in
[`docs/machine_profiles/i7-9700k.md`](machine_profiles/i7-9700k.md). The noise figures below are
this machine's. The gotchas in §7 were first measured on the Ryzen 7 3700U laptop before it
(`docs/machine_profiles/ryzen7-3700u.json`) and re-checked here (the benchmark machine
workplan's stage 6, `git show 480164d:docs/benchmark-machine-workplan.md`): §7 has the i7's numbers, and the
findings that didn't hold here are in its "Measured on the Ryzen laptop" note.

- **Run the setup script once per boot.** `sudo scripts/benchmark_machine_setup.sh` sets the
  frequency policy the profile records (governor, EPP, turbo, PL1), `perf_event_paranoid` 2, and
  holds snap refreshes for 24 hours, then prints what it set. None of it survives a reboot (the
  snap hold expires), and the machine check below refuses a run until it is applied.
- **Check the machine yourself, then go.** Before a timing run or a long sweep, read the 1-minute
  load (`uptime`) and the running processes (`ps`). If a browser (Brave, Firefox) or the editor
  (Zed) is running, close it (`pkill brave`, `pkill firefox`, `pkill zed`); don't stop to ask the
  owner. If the load is high, wait and check again. Keep other work light while it runs: reading
  and writing are fine; tests, lint and builds are not. A background IDE once spoiled a whole
  measurement.
- **Package jobs.** `unattended-upgrades` is enabled and snaps refresh on their own schedule; either
  can start a large job mid-run. Before a run, `systemctl is-active apt-daily.service
  apt-daily-upgrade.service` prints `inactive` twice when no apt job runs (`unattended-upgrades`
  itself is a shutdown hook, always active), `snap changes` lists none that isn't `Done`, and
  `snap refresh --time` shows the hold ("next: ... (but held)"). Wait out a running job; re-apply
  the setup script if the hold has expired.
- **A run a browser or package job overlapped is re-run in full.** The pre-flight check only covers
  the moment before the run. If the browser was opened during one, stop it by PID (`ab.py`, its benchmark and
  its worker processes), rename its run directory to `<run>-browser-open` without reading it, and
  start again once the 1-minute load is below 1.0.
- **Check the machine's identity.** `ab.py` does this itself. By hand:
  `python scripts/machine_profile.py compare docs/machine_profiles/i7-9700k.json`, in the
  same shell and environment as the benchmark. It exits 1 and names each changed identity field:
  CPU, caches, cpufreq policy (governor, EPP, turbo, the `intel_pstate` mode), the package power
  limits, kernel,
  `perf_event_paranoid`, Python, numpy and its BLAS, rustc, the crate's release
  profile, the thread env vars. `profile --out FILE` records a new machine.
- **Each machine has its own profile and noise rules.** `ab.py` picks the profile in
  `docs/machine_profiles/` recorded on the host it runs on (`jebel`: `i7-9700k.json`,
  `pyramidon`: `ryzen7-3700u.json`), and refuses a host with none unless
  `--allow-profile-change`. A profile's `noise_rules`, set by hand from that machine's A/As, are
  the thresholds its reports use ([§5](#5-reading-the-report)): `shifted_pass`, `high_load` and
  `small_consistent`. `profile --out` onto an existing profile keeps them.
- **The policy can change mid-run.** `thermald` puts PL1 back to 95 W within 15-40 minutes of
  the setup script, restarted or not, so the script stops it for the session (it starts again at
  boot). `ab.py` re-reads the frequency policy and the power limits after every pass. When they
  changed, it fails that pass and stops the run; re-run the setup script, then `ab.py extend`.
- **`PATH`.** Non-login shells lack `~/.cargo/bin`: builds fail and the profile reads `rustc` as
  null. `export PATH=$HOME/.cargo/bin:$PATH` (`ab.py` sets it itself). The crate's toolchain is
  pinned in `rust/rust-toolchain.toml`, which only rustup honours.
- **BLAS threads.** numpy's OpenBLAS worker threads compete with Rust for the cores (see
  [Gotchas](#7-gotchas)). Keep the default unless the measurement is about kernels, and never mix
  settings between the sides: `ab.py` records `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and
  `MKL_NUM_THREADS` in each run's manifest.
- **`perf`** needs `kernel.perf_event_paranoid` <= 2; it is 4 by default here, and the setup
  script sets 2 until reboot. Without it, internals are timed in a local probe
  build of the crate (timers and counters behind a Python-callable switch).

## 3. Choosing the measurement

From the quick survey to the decisive number:

| question | tool | notes |
| --- | --- | --- |
| Which ops look slow? | `python -m indrajala_ml.demos.benchmarks.demo_layer_op_timing` | every layer op, numpy and Rust interleaved; batch rows can be far off (interleaving). Finds candidates, never judges them. |
| How fast is one op? | `python scripts/focused_benchmark.py` | loops of about 20 ms, median of 9, each (case, backend) in its own process; faults per call; `--matmul MxKxN`, `--rust-threads`, `--openblas-threads`, `--malloc both`, `--kernel-overrides`. **The number to quote.** |
| Faults or compute? | `focused_benchmark.py --malloc both` | glibc defaults against both allocator thresholds at 1e9; a time that drops with the faults was paying for them. |
| Why is it slow (or slow in some processes)? | `python scripts/perf_region.py -- driver.py` | hardware counters for only the region a driver marks (`with counted():`), per unit of work, one row per process; `OPENBLAS_NUM_THREADS=1` unless `--openblas-threads`. Needs `perf_event_paranoid` <= 2. For cycles per instruction, `perf record` the driver and `perf annotate` the op. |
| One op per call inside real training | `python scripts/op_call_timing.py` | every call of chosen crate functions timed in a conv-demo training run, grouped by argument shapes (so layers come apart), plus the run's seconds; `--rust-threads`, `--kernel-overrides`. Sees training's thread count and cache state. |
| One op's share of real epochs | `python scripts/epoch_op_profile.py` | cProfile of Rust training by crate op. Resolves changes of a few % that epoch timing can't. |
| A training-path change | `python scripts/prepared_dataset_timing.py time` | one trainer epoch per process, dense full MNIST and the conv subset, both backends; `--epochs N`. The default A/B benchmark. |
| An accuracy pass, per row against batched | `python scripts/accuracy_pass_timing.py time` | all demo architectures, both backends; counts differing predictions. |
| Dense full-MNIST epochs, broken down | `python scripts/batch_size_timing.py time` / `profile` | epoch, step loop, one accuracy pass and conversions apart, per batch size. Its accuracy-pass column times the old tuple path. |
| The Rust/numpy ratios end to end | `python -m indrajala_ml.demos.conv.demo_conv_rust_vs_vectorized_digit_recognition` | about 3 minutes; median of 5 from identical weights, UCI digits and a 2000-row MNIST subset, plus a Rust op profile. |
| Old against new, any of the above | `python scripts/ab.py run --bench <benchmark>` | [§4](#4-running-an-ab-with-abpy). Every script above with an adapter, and any probe. |
| A threading setting | `set_matmul_threading(t, threshold)` | in one process, no rebuild; accept only on end-to-end numbers. |
| A kernel setting | `set_kernel_overrides(rows_per_block, k_block)` | `matmul_narrow`'s rows per block and `matmul_long_k`'s slab rows, no rebuild; `--kernel-overrides R:K` in the scripts above. |

## 4. Running an A/B with `ab.py`

**The rule: every old-against-new timing goes through `scripts/ab.py`.** Hand-built driver
scripts, pooling scripts and tables retyped into PR bodies are not used. A measurement `ab.py`
can't express gets a probe or an adapter, in the same PR (below).

`ab.py` resolves both sides to commits and checks each out once as a detached worktree under
`~/code/ab-worktrees/<sha7>`, so both builds are committed and the working checkout stays free. A
run lives in `~/code/ab-runs/<date>-<name>/`: its manifest (commits, crate hashes, commands,
machine check, environment), `progress.jsonl`, and each pass's raw output and logs. That
directory is also the neutral working directory every pass runs from.

Benchmarks: `prepared_dataset_timing`, `focused_benchmark`, `epoch_op_profile`,
`accuracy_pass_timing`, `op_call_timing`, `batch_size_timing`, and `cmd` (a probe).

**A training-path change** (commit it first; `--new` defaults to `HEAD`, `--old` to `main`):

```
python scripts/ab.py run --bench prepared_dataset_timing        # ONNONO, --repeats 5, ~30 min
python scripts/ab.py report --brief                              # when it exits
python scripts/ab.py report --md ab-table.md                     # the table for the PR body
```

`run` checks the machine, runs each side once at the benchmark's smallest settings (a smoke run,
about a minute), then the passes. Before each pass a probe checks that the trainer
(`indrajala_ml.training.train`, or `indrajala_ml.train` before the source layout) imports from
that side's tree and that the crate extension's hash is the expected one. Arguments after `--` go
to the benchmark: `-- --configs "conv B=32" --repeats 5 --epochs 2`.

**A crate change** (committed in `indrajala-math-rust`):

```
python scripts/ab.py run --bench prepared_dataset_timing --old main --new main \
    --old-crate 638ff13 --new-crate 750d83a --control-backend numpy
```

The sides' crate commits default to each tree's `rust/` submodule, so a "Bump rust/" branch
against `main` is a crate A/B with no flags. Each crate commit is built once into a release wheel
under `~/code/ab-runs/wheels/<sha>/` and installed with `pip install --target` into a site
directory on `PYTHONPATH` after the tree. The venv is never touched, and every pass checks the
extension's hash. The numpy rows are the control: a crate change can't move them.

The builds share one cargo target directory, so each is incremental. `ab.py` extracts each crate
commit with fresh modification times, so cargo recompiles whatever changed. It also refuses a run
whose two crate commits differ in what the extension is built from (`src/`, `Cargo.toml`,
`Cargo.lock` and the build configuration) but built the same extension. A cached wheel is reused
for as long as it exists, so a wrong build stays wrong until its `wheels/<sha>/` directory is
removed ([Gotchas](#7-gotchas)).

**A probe.** A question no script answers yet gets a probe: a script that prints one JSON object
per line, `{"case": ..., "metric": ..., "value": ..., "unit": ...}`, optionally `"control":
true`. Other lines are ignored.

```
python scripts/ab.py run --bench cmd -- scripts/my_probe.py --its-args
```

A relative probe path is taken from the new tree when it's there, else from the current
directory. A probe that proves useful becomes a script with an adapter (a small class in
`ab.py`: its command, smoke settings, rows and controls) in the same PR, not a scratch file.

**A shifted pass.** When the report names one, add the passes it asks for:

```
python scripts/ab.py extend --order NO      # numbered after the existing passes
```

**Into a PR.** Put the `--md` file into the PR body by concatenating files, not by retyping it.
The file has the protocol paragraph (commits, order, command, working directory, provenance,
machine check, why any passes were added) and one table per metric.

**Other commands.** `status` prints one line (the pass running, how many are done, an ETA).
`clean --worktrees` and `clean --wheels` remove the worktrees and crate builds no run of the last
14 days refers to. `RUN` defaults to the most recent run everywhere.

## 5. Reading the report

`report --brief` prints at most 15 lines:

1. **The header:** commits, pass order, benchmark and arguments, and for a crate A/B each side's
   crate commit and extension hash. When the crate's Rust changed, the two hashes must differ;
   `ab.py` refuses the run otherwise.
2. **The machine:** the profile check, the pre-flight 1-minute load (flagged above the machine's
   `high_load`, 1.5 on both machines), and
   processes that were above 10% of a CPU around the passes.
3. **Controls.** The control rows are `prepare` for `prepared_dataset_timing`, and the other
   backend's rows with `--control-backend`. Rust-only benchmarks have none, and the line says so.
   If a control row is *consistent*, the report says so first: this A/B can't resolve a change of
   about that size, and `epoch_op_profile.py` is the next step.
4. **Shifted passes.** A pass whose rows, the controls included, sit the machine's
   `shifted_pass` or more from their side's pooled medians in the same direction (the median over
   rows of pass median / pooled median): 2% on the i7, where no pass of 36 in its A/As moved more
   than 0.9%, and 5% on the laptop, where whole passes ran 5-15% fast or slow, the untouched
   control included. The
   report says whether shifted passes are balanced between the sides, or names the `extend
   --order` that balances them. A shifted pass is never dropped.
5. **Consistent rows**, largest |Δ| first, with old and new medians. A row under the machine's
   `small_consistent` (2% on the i7) is marked `consistent, small`: an A/A there flags rows that
   size by chance, so one A/B doesn't establish it.
6. **A summary:** how many rows are within noise, how many of those are separated but inside their
   spread, and the largest |Δ|.

**The verdict.** A row is *consistent* when every per-pass median of one side lies beyond every
per-pass median of the other, with 2 or more passes a side, **and** the gap between the sides is
wider than each side's own spread of per-pass medians. Otherwise it is *within noise*.
Separation alone happens by chance 1 time in 3 at 2 passes a side, and 1 in 10 at 3: on #479 (no
change) it flagged 13 of 24 rows, and an A/A of `main` against itself flags some rows that way
too. The report advises; the claim in a PR remains its author's.

**The table** (`--md`), per metric and case: old and new pooled medians with the min-max over all
runs, Δ median, each pass's own median (numbered), and the verdict. The per-pass medians show a
second mode or a drifting pass that pooled numbers hide.

What to do next:

| the report says | next |
|---|---|
| everything within noise | the change is below what this benchmark resolves: say so, with max \|Δ\|. |
| consistent rows, controls quiet, passes balanced | a measured change: quote Δ and the table. |
| a control moved | the A/B can't resolve changes of about that size: `epoch_op_profile.py` or `focused_benchmark.py`. |
| unbalanced shifted passes | `ab.py extend --order <named>`, then report again. |
| a large per-pass spread in one row | look at that row's raw runs for two modes ([Gotchas](#7-gotchas)). |
| the machine line has a high load or a busy process | re-run with the machine quiet. |

## 6. Protocols

`ab.py` applies the ones marked ✓; they are why its design is what it is.

- ✓ **Commit both sides first.** Each side is a commit, run from its own worktree. An uncommitted
  change under `indrajala_ml/`, `scripts/` or `rust/` with `--new HEAD` is refused.
- ✓ **One process per measurement**, and **numpy and Rust in separate processes**, always.
- ✓ **Alternate the sides** (`ONNONO`), so drift over the run falls on both. Rotate the order
  within a pass too (the scripts' `interleaved_runs`).
- ✓ **Numbers vary between passes.** On the i7 a row's per-pass medians spread about 1-4% (the
  median over rows, per benchmark), and 6-13% for the noisiest tenth of rows. A single process
  can still land far off (one numpy op ran 50% slow in one pass of six), and a few ops settle in
  distinct per-process modes. On the laptop it was 20-30%. Treat a change as real only when the
  passes agree: the verdict above makes "agree" exact.
- ✓ **Read the control.** A change to one backend can't move the other; `prepare` converts the
  dataset and runs no training code. When the control moves as much as the change, the numbers
  can't resolve it.
- ✓ **Balance shifted passes; never drop one.**
- ✓ **Know which tree ran.** `indrajala_ml` is a namespace package, so running from inside a tree
  can import that tree whatever `PYTHONPATH` says. Passes run from a neutral directory with only
  their tree (and crate site directory) on `PYTHONPATH`, and a probe checks each pass.
- **Don't judge a training-path change on trainer epoch time alone.** The accuracy passes (n + 1
  for n epochs) are a large share of a short run: time them apart (`accuracy_pass_timing`).
- **Kernel changes end to end:** one epoch of MNIST, `ConvSpec(3, 8)`, dense 32, mini-batch 32,
  lr 0.5 on the demo's 2000-row subset, from a snapshot of `randomized(...)` after
  `np.random.seed(0)` (the network's generator seeded 0 since the RNG generators workplan's stage
  3) with the shuffle seeded 0 before each epoch; the same at mini-batch 512; one
  dense MNIST epoch at batch 32 and 512. Then the conv demo for the ratio table.
- **A quick look at a crate change by hand** (not for a PR's numbers): commit it, then alternate
  `./cli build-rust` (about 6 s) between the builds, switching with `git checkout main --
  <changed files>` and back, never a stash.

## 7. Gotchas

- **The namespace package** (above): an A/B run from the repository's own directory once came
  out right only because of the order `PYTHONPATH` gave, and nothing checked it.
- **`data/` is relative to the working directory** (`prepared_dataset_timing.py` reads
  `data/mnist/`). `ab.py` symlinks the checkout's `data` into the run directory.
- **Worker processes inherit the environment, not `sys.path` edits.** A tree goes into
  `PYTHONPATH`, never into `sys.path` in a driver.
- **Whole passes shift** on a noisy machine: by 5-15% in every config on the laptop, the controls
  included (#480, and #477's first A/B). On the i7, with the setup script applied, no pass of 36
  moved more than 0.9%. Balance a shifted pass ([§5](#5-reading-the-report)) on either.
- **numpy's OpenBLAS threads slow a Rust call run soon after.** After a BLAS call its workers
  spin for a while (`OPENBLAS_THREAD_TIMEOUT`) before they sleep. On the i7, the conv tail's
  `downstream_batch` right after a threaded numpy product took 1.56x its time after a 1 s sleep at
  batch 512 (threaded in Rust: 4.0 against 2.56 ms) and 1.09x at batch 32 (one thread)
  (`scripts/openblas_spin_probe.py`, stage 6). The numpy product also evicts the op's data: even
  after the sleep, batch 32 read 8.7% slower than with no numpy call. Any per-op ratio measured
  interleaved with threaded numpy is suspect; the conv demo interleaves (unmeasured effect, at
  most the start of each run).
- **numpy's default threading makes numpy look fast in hot loops.** Compare on one thread each
  (`--openblas-threads 1 --rust-threads 1`) to see the kernels. On the i7 it pays in training too:
  at `OPENBLAS_NUM_THREADS=1` the MNIST conv mini-batch 32 epoch was 6.8% slower (0.433 against
  0.405 s), conv 512 4.8% and dense 32 6.3% (stage 6).
- **Clocks.** An idle core clocks up in about 3 ms on the i7 under D1's policy (the first small
  products about 6x slow), so time loops, not single calls. In rotated runs settings barely carry
  over (pass shifts within about 1%), but all-core work does: run in a fixed order, the same
  8-thread products read 5-20% apart from one setting to the next, the package power budget
  (PL1 65 W, PL2 120 W) spent by what ran before. Rotate the order.
- **`--rust-threads 1` hides what training's threading does.** Past 8M flops training threads
  the op. On the i7, 1-row blocks in `matmul_narrow` (against the default 4) cost 5-8% on the
  batch-512 conv `downstream_batch` ops on one thread and within 2% threaded over rows; at batch
  32, under the threshold, 10-20% either way (stage 6). Time a kernel change at the thread count
  training uses too.
- **Isolated loops flatter threading.** Back-to-back calls keep every core clocked up; in
  training the cores idle between calls and each threaded call pays the cold clock. On the i7, 8
  threads ran 2.3-3.2x faster than 1 on 1.6-5.5M-flop products in isolation, yet lowering the
  8M threshold to 2M or 4M changed no epoch by more than 1.6% (stage 6).
- **A probe's allocation pattern is not the real call path's.** A probe loop allocating fresh
  outputs faulted on every page (5410 faults a call, 40% of its time) where the real op had 0.1;
  glibc's mmap threshold adapts to what the process freed before. Quote fault costs from the
  real op, or compare probe and op with both allocator thresholds raised.
- **A bare product's benchmark can miss its cost inside the op.** Conv forward's product alone
  scaled linearly at N = 32, but inside the op, after im2col had streamed 1.56 MB through the
  cache, the same call cost 1.5-2x as much. Time an op's parts in place before ruling one out.
- **Zero-filling a buffer can be what brings it into cache.** Removing a fill made a later strided
  write into the buffer slower (it now paid the first touch). Remove a fill only when whatever
  writes the buffer first writes it sequentially.
- **glibc heap trimming can fault a batch op's buffers back in on every call** when calls are
  chained (freed top-of-heap returned to the OS). `focused_benchmark.py --malloc both` separates
  it.
- **A time can be modal between processes, not only noisy.** On the i7, the single-example Rust
  `downstream` on the 32 x 5408 conv tail settles per process at about 20.8, 22.3 or 23.8 µs,
  each process steady (20 processes, stage 6). `perf_region.py` shows the same instructions and
  L1 misses in every mode; the slow ones send about 50% more demand loads past L2 to L3 (714
  against about 1085 L3 hits a call), and the extra cycles are stalls on those misses. Physical
  page placement in the L2's sets is the likely cause (not tested). A median of loops in one
  process can't see it: run several processes and quote the modes. `ab.py`'s per-pass medians
  and ranges show a second mode; it doesn't classify modes.
- **The conv demo's mini-batch runs barely train** (about 10% accuracy in 1-2 epochs at lr 0.5):
  their timings are valid, their accuracy columns are not. conv-conv's single-example MNIST run is
  unstable at lr 0.5 too (numpy collapsed to 0.11 where Rust reached 0.52; it trains at 0.2).
- **Back-to-back runs inherit load.** The pre-flight load of a run started right after another
  counts the previous run's benchmark; `ab.py` flags it.
- **A crate A/B once timed the old crate on both sides** (stage 4c, 2026-09-30). `git archive`
  stamps every file with the commit time, and cargo decides freshness in the shared target
  directory by modification time. The new commit (made 15:41) was built right after the old one
  (built 15:53), so cargo found its sources older than the last build and compiled nothing
  (`Finished ... in 0.02s` in its `build.log`, with no `Compiling indrajala_math_rust` line). Both
  sides got the same extension (`.so dbe73bea6421` twice in the header), the provenance check
  passed because it compares each pass with its own side's build, and the report said "within
  noise". Any crate commit older than the target directory's last build was skipped this way;
  stage 3's A/B compiled because its new commit was newer than that build. `ab.py` now
  extracts with fresh modification times and refuses same-extension runs
  ([§4](#4-running-an-ab-with-abpy)).
  After a bad build, remove its `~/code/ab-runs/wheels/<sha>/` directory before running again.
- **`pkill -f <pattern>` also matches the invoking shell's own command line.** Stop a stray
  benchmark by PID. Waiting is the same: `while pgrep -f "<pattern>"` matches its own loop's
  command line and never exits, and a `pgrep` right after `(nohup cmd &)` can catch the wrapper
  shell. Read the Python process's PID from `pgrep -af` and wait with `kill -0 <PID>`.

**Measured on the Ryzen laptop** (Zen 2, 4 cores; the benchmark machine workplan's D9). These
didn't hold on the i7, or held with other numbers; the i7's are above, and
[machine_profiles/i7-9700k.md](machine_profiles/i7-9700k.md) has each side by side.

- A Rust batch op right after a threaded numpy call ran 2-5x slow, whatever Rust's thread count
  (the i7: 1.56x threaded, 1.09x on one thread).
- numpy's threading didn't pay in training: the MNIST conv mini-batch 32 epoch was faster at
  `OPENBLAS_NUM_THREADS=1` (0.94 against 1.18 s; the i7: 6.8% slower).
- Idle cores dropped to 1.1-1.5 GHz and took hundreds of ms of load to clock up to 3.8 GHz.
  Clocks carried over between settings (an 8-thread setting run after another read up to 2x
  faster) and between paired processes (the second of a pair once read 26-27 against 41-43 µs
  whichever setting ran second).
- 4-row blocks in `matmul_narrow` took 40% off conv-conv's second-conv accumulate (8 rows) on one
  thread and nothing when threaded over rows, where each thread already had 2 rows.
- The one-pass max-pool downstream at batch 32 (probe and op) was bimodal between processes:
  about 17 cycles a window (190-200 µs) in some and 42-49 (470-570 µs) in others, tight within
  each. The counters showed the same instructions, L1 and L2 accesses in both modes; the slow one
  was integer-scheduler stalls (ALU-token stalls 24 against 1 a window). Ruled out, each measured:
  page faults and allocator thresholds, clock frequency, the core and its SMT sibling, virtual
  placement (buffers pinned to a 2^28-aligned arena, ASLR off), physical pages (re-paged between
  trials), the AVX upper state, SSBD, the `+=` read (a store-only pass) and the division (a
  slot-offset table: still 4.7-8.5 ns a window). The cause is unknown. On the i7 the same op is
  unimodal (157.6-161.6 µs over 20 processes).

## 8. Judging correctness

Single-example training is chaotically sensitive to rounding: networks trained from the same
weights by numpy and Rust, or by numpy against itself with one weight nudged by 1 ULP, agree on
only 71-83% of test predictions after training, though they stay within 1e-15 through the first
UCI epoch. So a change is judged by step-by-step parity (per-step agreement to about 1e-15), never
by end-of-run accuracy. The golden run (`scripts/golden_training_run.py check
data/refactoring/golden_run.json`, about 1 s) pins 106 networks bit for bit.

The golden run is the default gate, not a sacred one. A change that is genuinely more correct may
move its bits: propose it to the owner first, and if they accept it, re-record the golden file and
say so in the PR, with the measured difference and why the new result is more correct. Bits are
never moved for style or for parity alone.

## 9. Rules for a timing claim in a PR, and for an optimization PR

**Any timing claim:**
- It comes from an `ab.py` run on the benchmark machine, with the machine check passing (or the
  change it names explained).
- The PR gives the `--md` output: the protocol paragraph and the table, with its verdicts. A
  claim of a change quotes consistent rows; a claim of no change quotes max |Δ| within noise.
- A numeric difference in results is explained, never accepted within a tolerance.

**An optimization PR, in addition:**
- **Two repos.** Crate changes land in `indrajala-math-rust` (`rust/`) first, with its own
  numpy-only tests; then a PR here bumps the submodule and runs the full suite (`./cli build-rust
  && ./cli test`). Neither merges until both pass.
- **Bit-identical claims are tested, not assumed:** a crate test pins the new op with `==` on
  `tolist()` at shapes that reach every kernel path, threading and blocking threshold; it passes
  on the old build, and a mutation (a second rounding, a changed start value or order) fails it.
- **Bit-changing changes:** the `rtol` parity tests (`tests/model/layers/test_*fused_layer_ops.py`, the
  step-by-step network parity tests) pass unchanged; a moved end-to-end pin is compared with a
  1-ULP control on the old code (nudge one initial weight) and updated only if it moves similarly,
  with the control recorded in the PR; record the max abs and ULP difference from the old op.
- **Measure first; close what doesn't pay.** A stage 0 measures the stake against a bar fixed
  before measuring (about 5% of an epoch in a trained configuration). Each stage is its own PR,
  merged before the next. A stage with no gain is closed with its reason in the PR. Every PR
  quotes before/after per-op rows for the ops it touches and the end-to-end effect; the
  measurements behind a change live in its PR.

## 10. For agents

Most A/Bs here are run by an agent, where the cost is turns: each check on a running job re-reads
the whole conversation. The repository's `CLAUDE.md` repeats these rules.

- **Launch in the background and don't poll.** Start `ab.py run` as a background command and let
  its exit be the signal. Use `status` only when the owner asks. Don't schedule wake-ups for runs
  under an hour.
- **Read the brief report only.** Read `report --brief` and nothing else from the run. Open a raw
  pass file only when the brief report flags something it can't explain.
- **In a crate A/B, check the header before the verdict.** The two `.so` hashes must differ when
  the crate's Rust changed. The same hash on both sides means one crate was timed twice, whatever
  the rows say ([Gotchas](#7-gotchas)). `ab.py` refuses such a run now; the check still costs
  nothing.
- **Pass the table on without reading it.** Put `report --md` into the PR body by concatenating
  files.
- **Keep the machine quiet.** Nothing CPU-heavy runs while an A/B does: no tests, lint or builds.
  Reading code and writing docs are fine. Check the load and close the browser and editor yourself
  first ([§2](#2-preparing-the-machine)).
- **Extend `ab.py` instead of working around it.** A measurement it can't express gets a probe or
  an adapter, not a new driver script.
