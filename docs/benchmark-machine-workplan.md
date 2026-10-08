# Workplan: a baseline on the new benchmark machine

**Status: D1-D11 settled; stage 0 done (2026-10-07: both test suites, lint and a fresh golden
run pass here); stages 1-4 done; stages 5-7 are planned.**

Benchmarking moves from the Ryzen 7 3700U laptop (`pyramidon`) to a desktop Core i7-9700K
(`jebel`). Every timing rule in [measurement.md](measurement.md) was written and calibrated on the
laptop: the machine check names its profile, the noise figures ("20-30% between passes", "whole
passes shift 5-15%") are its numbers, and several gotchas are its CPU's behaviour. The crate's
threading constants were tuned on it too (`rust/src/linalg.rs`, `matmul_thread_count`: "measured
on this laptop (4 cores / 8 threads), not portable").

This plan sets the new machine up, makes the tooling accept it, measures how noisy it is, and
records a baseline of every benchmark at one fixed commit. After it, a timing number from this
machine can be checked against something, and measurement.md describes the machine it runs on.

## Why

- **`ab.py` refuses the machine.** Its machine check compares against
  `docs/machine_profiles/ryzen7-3700u.json` (`PROFILE_REFERENCE`), so no A/B can run here until a
  new profile is recorded and becomes the reference.
- **The noise numbers decide verdicts.** The protocols (2 or more passes a side, `ONNONO`, the 5%
  shifted-pass rule) were chosen for the laptop's 20-30% pass-to-pass variation. A desktop with
  fixed cooling and 8 real cores may be quieter or noisy in other ways (turbo bins, power limits,
  background services). The rules have to be checked against measured numbers, not assumed.
- **A baseline is the reference for later work.** It lets a later number be checked for drift
  (kernel, BLAS or toolchain updates, a changed BIOS setting), and gives the Rust/numpy ratios
  every optimization quotes. The last one was the laptop's, retired with the optimization docs
  (`git show 0a04977:docs/optimizations/current-baseline.md`).
- **Machine-specific claims become visible.** Some laptop findings may not hold here: the
  threading threshold, `matmul_narrow`'s 4-row blocks, OpenBLAS's spin-wait penalty, and the
  bimodal max-pool downstream (integer-scheduler stalls on Zen 2). The plan records which ones
  still hold. It changes none of them.

## Where things are now

**The new machine,** read on 2026-10-07:

| | i7-9700K (`jebel`) | Ryzen 7 3700U (`pyramidon`) |
| --- | --- | --- |
| cores / threads | 8 / 8 (no SMT) | 4 / 8 |
| L1d, L2 per core | 32 KB, 256 KB | 32 KB, 512 KB |
| L3 (shared) | 12 MB | 4 MB |
| ISA | AVX2, FMA; no AVX-512 | AVX2, FMA; no AVX-512 |
| clocks | 800-4900 MHz (base 3600) | 1400-2300 MHz, boost to ~3800 |
| frequency driver | `intel_pstate` (active, HWP), governor `powersave`, EPP `balance_performance`, turbo on (`no_turbo` 0) | `acpi-cpufreq`, `schedutil`, boost on |
| memory | 31 GB, 8 GB swap | 5.3 GB, 4 GB swap |
| power | desktop (no battery) | laptop |
| kernel, distro | 7.0.0-38-generic | 7.0.0-34-generic, Ubuntu 26.04.1 |
| `perf_event_paranoid` | 4 | 2 when profiled |
| sensors | `coretemp`, `asus` hwmon; `turbostat` and `cpupower` installed | — |

Things that differ in ways that matter:

- **No SMT.** On the laptop, `available_parallelism()` returned 8 on 4 cores, so the crate's
  8 threads shared cores. Here, 8 threads get 8 cores. The threading threshold (8M flops) and the
  finding that 2 and 4 threads never pay may not hold.
- **Half the L2, three times the L3.** Every working set that was "fits L2" on the laptop (a 32 x
  5408 conv tail, about 1.4 MB) is now in L3, and every one that spilled the laptop's L3 now fits.
  The per-op ratios may move in both directions.
- **OpenBLAS's kernel.** The laptop's numpy reported OpenBLAS `DYNAMIC_ARCH ... Haswell`. A Coffee
  Lake without AVX-512 should select the same Haswell kernels. If it does, and the crate takes its
  AVX2 path on both, the golden run may be bit-identical across the two machines. The laptop's
  golden file is gone, but the laptop isn't, so stage 5 tests it (D5); stage 1's profile records
  the BLAS build either way.
- **Intel frequency controls.** `machine_profile_capture.py` reads `cpufreq/boost` or `cpb`
  (AMD) for `boost_enabled`. Neither exists under `intel_pstate`, so the field would read `null`.
  It records no EPP (`energy_performance_preference`), and EPP decides how fast the cores clock
  up here, as the governor does on the laptop.
- **Background services.** Brave and Firefox are snaps, `snapd` refreshes on its own schedule,
  and `unattended-upgrades` is enabled. Either can start a large job mid-run. The laptop's
  checklist (close Brave and Zed) doesn't cover them.

**The machine is not set up yet:** no `.venv`, no rustup (`~/.cargo/bin` is missing), `rust/`
not checked out (the submodule is uninitialized), no `data/mnist/`, no golden run file
(`data/refactoring/` is ignored by git and was never on this machine), no `~/code/ab-runs/`.
Python 3.14.4 is at `/usr/bin/python3.14`, the same patch release as the laptop's profile.

**Where the old machine is named:** `scripts/ab.py:69` (`PROFILE_REFERENCE`),
`tests/measurement/test_machine_profile.py:13` (the schema tests' fixture),
`scripts/machine_profile.py`'s usage line, measurement.md §2 (the description and the `compare`
command) and its noise figures in §5-§7, and the crate's `linalg.rs` comments. The PyPI
workplan's stages 1-2 say "on the Ryzen machine".

## Decisions

Settled with the owner on 2026-10-07.

- **D1. The frequency policy timed under: decided in stage 2, from measurement.** Stage 2 runs
  an A/A under (a) the defaults (`powersave` with EPP `balance_performance`, turbo on) and under
  (b) EPP `performance`, turbo on, and adopts the one with the smaller pass-to-pass spread. (c)
  Turbo off (a flat 3.6 GHz) is the fallback only if both are noisy: it is the quietest and the
  least realistic, and it hides the cold-clock effects in training (threaded calls on idle cores)
  that the laptop's threading findings rest on. The chosen policy is recorded in the profile, so
  the machine check catches a reboot that reset it.

  **Answer (stage 2, 2026-10-08): (a), the defaults.** Both A/As were clean and close; (a) had the
  smaller pass-to-pass spread, so it stays (`prepared_dataset_timing`, 16 timed rows each, with
  D10's 65 W PL1 applied):

  | | (a) `balance_performance` | (b) `performance` |
  | --- | --- | --- |
  | pass-to-pass spread, median / max over rows | 3.2% / 5.6% | 3.5% / 7.2% |
  | largest A/A median difference | 3.3% | 2.9% |
  | controls, busy processes | all within noise, none | all within noise, none |
  | clock-up from idle (one 96x96 product, back to back) | about 6x slower for the first 3 ms | none |

  Epoch times under the two agree within 2%. (a) also keeps the cold-clock start that training
  sees; on this machine it lasts about 3 ms, not the laptop's hundreds. (c) wasn't needed.
- **D10. PL1 (RAPL's long-term package power limit): 65 W, set by the D2 script** (owner,
  2026-10-08, from stage 2's step 1). At the stock 95 W, ten minutes of all eight cores reached
  100 C: 460 package thermal-throttle events, clocks falling from 4.2 to 3.8 GHz while power fell
  under the limit. The cooling can't be improved. At 65 W the same load held 3.69 GHz and about
  78 C with no throttling, about 8% slower per epoch. One busy core draws about 37 W and the
  threaded conv case about 40 W, so neither is affected. PL2 stays at 120 W. The profile records
  both (`power_limits`, `schema_version` 3), so the machine check catches a reboot that reset it.
  **`thermald` is stopped for the session** (owner, 2026-10-08, stage 4): running, it put PL1 back
  to 95 W within 15-40 minutes, twice, the second time though restarted after PL1 was set. Stopped,
  PL1 held for all six A/As (109 minutes). The setup script stops it; it starts again at boot.
- **D2. Reapplying the settings after a reboot: a checked-in script.**
  `scripts/benchmark_machine_setup.sh` (it needs sudo) sets the D1 policy, D3's
  `perf_event_paranoid` and holds snap refreshes, and the owner runs it after boot. The machine
  check refuses a run if it wasn't. No systemd unit: each change stays visible.
- **D3. `perf_event_paranoid`: set to 2 by the D2 script**, per boot, not permanently in
  `/etc/sysctl.d`, so `perf_region.py` works and the change is visible.
- **D4. The reference profile: one benchmark machine at a time.**
  `docs/machine_profiles/i7-9700k.json` becomes `PROFILE_REFERENCE`. The Ryzen file stays as
  history and as the schema tests' second fixture. No lookup by hostname: two reference machines
  would mean two sets of noise rules.

  **Replaced (owner, 2026-10-08, stage 4): a profile per host, with its own noise rules.** The
  noise thresholds are a property of the machine, so each profile carries `noise_rules`
  (`shifted_pass`, `high_load`, `small_consistent`; schema version 4), and `ab.py` picks the
  profile recorded on the host it runs on. `PROFILE_REFERENCE` remains the benchmark machine:
  the one an unknown host is compared to under `--allow-profile-change`, and the one whose numbers
  the docs quote. The i7's rules come from stage 4's A/As; the laptop keeps the rules it had until
  its own runs (D11) revise them.
- **D5. The golden run: recorded fresh here, then checked across the machines.** It is
  machine-specific by design (numpy's BLAS) and lives in an ignored file. Stage 0 records a new
  one on `main`. The laptop's file is gone, but the laptop is still available (owner, 2026-10-08;
  D11). Stage 5 records a golden run on it at the baseline tag, and each machine checks the
  other's file, which settles whether the two compute the same bits.
- **D6. The baseline commit: tagged in both repos.** The `main` commit stage 3 merges on is
  tagged `baseline-i7-9700k` here, and the submodule's commit with the same name in
  `indrajala-math-rust`. Every number in the baseline comes from that commit pair, so it can be
  rerun exactly later.
- **D11. The laptop: occasional runs at the baseline tag** (owner, 2026-10-08). The Ryzen laptop
  (`pyramidon`) still exists, so its columns are measured, not quoted from retired docs: stage 5
  runs its golden run (D5) and the non-A/B measurements there, at the same commit pair. Since
  D4's replacement, `ab.py` on the laptop checks against its own profile and noise rules. Before its
  first run, the laptop is set up for ssh from `jebel` (owner), checked out at the tags, built, and
  its profile re-recorded at schema 4 (keeping its noise rules). Its baseline numbers stay context
  for the i7's.
- **D7. What the baseline covers: everything.** An A/A of every `ab.py` adapter at its default
  arguments (stage 4), plus the per-op table, the conv demo's ratio table and the kernel protocol
  configurations (stage 5), as in the laptop's retired baseline, with a laptop column where one
  exists. Rust and numpy at default threading, and both at one thread for the per-op table.
- **D8. Where the baseline lives: beside the machine's profile,**
  `docs/machine_profiles/i7-9700k.md`. One file per machine: its profile's companion, holding the
  D1 policy, the tagged commit pair, the noise figures and every baseline table, replaced in place
  when re-recorded, with the history in git. measurement.md links to it and keeps no numbers of
  its own beyond the noise rules.
- **D9. Laptop findings that don't reproduce: moved to a Ryzen note.** Stage 6 re-measures them.
  measurement.md keeps each one that still holds, rewrites each that changed with the new
  numbers, and moves each that no longer holds to a "Measured on the Ryzen laptop" note with its
  PR. Code that cites one (the `linalg.rs` comments) changes only in the crate workplan that
  changes the code (Out of scope).

## Pitfalls to design around

- **Turbo bins and power limits.** The 9700K runs one core at up to 4.9 GHz and all eight lower
  (about 4.6 GHz), and a long all-core load can hit the package power limit (PL1, 95 W at stock)
  or, with weak cooling, a thermal limit. So a single-threaded op and a threaded one run at
  different clocks, and a long sweep can slow down partway. Stage 2 measures this with
  `turbostat` before any timing is trusted.
- **The machine check's own fields.** Recording the profile before the Intel fields exist
  (stage 1) gives `boost_enabled: null` and no EPP, and the check would then pass whatever the
  turbo and EPP setting were.
- **Profile timing.** `capture()` reads clocks first because later imports boost the cores. With
  HWP, clocks are per core and change in microseconds, so `cpu_mhz_now` is state, not identity.
  It must stay out of `compare`.
- **The A/A is not free of a verdict.** Under the verdict rule, an A/A should flag no row
  *consistent*. One that does is a finding about the machine (two modes, drift within a run), not
  a result to average away. It is investigated before the baseline is recorded.
- **Snap and apt jobs.** A `snapd` refresh or an `unattended-upgrades` run mid-pass shows up as a
  shifted pass or a busy process in the report. The D2 script holds snap refreshes for the
  session (`snap refresh --hold=<duration>`); `unattended-upgrades` is checked as a running
  process by the pre-flight, and a run it overlapped is re-run, as the browser rule says.
- **The desktop is in use.** GNOME, the terminal and Claude Code itself run during every pass.
  Their CPU share is recorded by `ab.py` (processes above 10% of a CPU); stage 2 checks it stays
  small. A run with a browser open is re-run in full, as now.
- **Comparing with the laptop.** Its columns are measured at the baseline tag (D11), so the code
  is the same, but the machines aren't quieted the same way and their noise rules differ. A
  cross-machine column is context, never a timing claim: the ratios (Rust/numpy) are the
  comparable part, the absolute times are rougher. Numbers from the retired docs, at older
  commits, are quoted only where no laptop run exists.

## Stages

Each stage is one PR, merged before the next, with `./cli test`, `./cli lint` and the golden run
green. Stages 0-1 change tooling and docs. Stages 2-6 are measurement PRs: their numbers come from
`ab.py` runs or the named scripts, their raw run directories stay in `~/code/ab-runs/`, and each
PR body carries its `report --md` tables.

### Stage 0: set the machine up (owner and agent, no PR)

1. Install rustup and let `rust/rust-toolchain.toml` pick 1.98.1
   (`rustup default 1.98.1` afterwards, as the crate's CLAUDE.md says).
2. `./cli setup`: OS packages, `.venv` from Python 3.14, submodules, requirements, maturin 1.15.0,
   the lint tools, the MNIST fetch (`scripts/fetch_datasets.py`, checksum-verified).
3. `./cli build-rust`, `./cli test`, `./cli lint`; the crate's own tests
   (`rust/`: the README's "Build and test").
4. `python scripts/golden_training_run.py record data/refactoring/golden_run.json` on `main`
   (D5), then `check` it once to see it pass on a second run.
5. Record the software versions that came out of the lock (numpy and its OpenBLAS build, the
   `maturin`, `rustc`) for stage 1's profile.

Done when both test suites, lint and the golden run pass on this machine.

### Stage 1: the profile knows Intel, and the reference moves

1. `machine_profile_capture.py`: under `intel_pstate`, read `no_turbo` into `boost_enabled`
   (inverted) and `intel_pstate/status`; for every driver, read the EPP when the file exists. The
   EPP and the `intel_pstate` mode are new identity fields: a schema change
   (`machine_profile.schema.json`, `schema_version` 2), with the Ryzen profile migrated (its new
   fields `null`) so it still validates. Tests for both parsers, from captured sysfs text.
2. `scripts/benchmark_machine_setup.sh` (D2, D3): sets the chosen frequency policy (the
   defaults until stage 2 decides), `perf_event_paranoid` 2, and holds snap refreshes, then
   prints the state it set. It changes nothing that isn't on its list.
3. Record `docs/machine_profiles/i7-9700k.json` with the script applied. Point
   `PROFILE_REFERENCE` and `machine_profile.py`'s usage line at it (D4). The schema tests keep the
   Ryzen file as a fixture and add the new one.
4. measurement.md §2: the machine's description and its record, the setup script in place of
   "close Brave and Zed" (which stays, with Firefox added), the snap and apt checks.

This profile is provisional: stage 2 may change the frequency policy, and then re-records it.

Done when `ab.py run --bench cmd -- <a trivial probe>` passes its machine check here.

### Stage 2: characterize the machine and settle D1

Measurement only; the PR records findings in this workplan and changes the setup script.

1. **Clocks under load,** with `turbostat --interval 1` beside each: idle; one busy core (a
   single-threaded `focused_benchmark.py` loop); all eight (the conv mini-batch 512 epoch, the
   threaded case, repeated for 10 minutes). Record each case's per-core MHz, package watts and
   temperature, and whether a power or thermal limit was hit (turbostat's throttle counters). If
   the 10-minute all-core run slows down, the cooling or the power limit is the owner's to fix
   before any number is recorded.
2. **Clock-up time from idle:** a probe timing one small product every few ms after an idle
   second, under (a) and (b). This measures the laptop gotcha "taking hundreds of ms of load to
   clock up" on this machine.
3. **A/A spread under (a) and (b):** `ab.py run --bench prepared_dataset_timing --old main --new
   main` once under each policy, the machine otherwise quiet. Compare the per-pass spread, the
   shifted passes and their size, and the controls.
4. **Background load:** the processes `ab.py` reports above 10% during those runs.
5. Settle D1 from 1-3, write the answer and its numbers into this plan's Decisions, set it in the
   setup script, and re-record the profile if the policy changed.

Done when D1 is settled and recorded with its numbers.

**Done 2026-10-08.** Step 1, with `turbostat --interval 1`, 95 W PL1 unless marked:

| case | busy (of 8 CPUs) | clock | package power | package temp, max | throttling |
| --- | --- | --- | --- | --- | --- |
| idle, 60 s | 0.6% | 800 MHz | 2 W | 32 C | none |
| one core: conv B=32, Rust matmul capped at 1 thread, 120 s | 12.5% | 4.8 GHz, flat | 35-38 W | 72 C | none |
| the threaded case: conv B=512, 120 s | 16% | 4.6 GHz, flat | 37-41 W | 79 C | none |
| all eight: 8 copies of the one-core load, 10 min | 99.6% | 4.2 GHz falling to 3.8 | 95 W, then 76-85 | 100 C | 460 package events |
| the same at 65 W PL1 (D10) | 99.6% | 3.69 GHz, flat | 65 W | 78 C (93 for one second) | none |

- **The conv mini-batch 512 epoch is not an all-core load:** about 1.3 CPUs busy, so step 1's
  all-core case ran eight single-threaded copies instead.
- **Eight copies each ran 2.7x slower than one alone** (0.9 s an epoch against 0.33), of which the
  clock explains about 1.2x. The rest is most likely shared L3 and memory bandwidth: a question for
  stage 6's threading re-check.
- **Clock-up (step 2)** is about 3 ms from idle under (a) and nothing under (b) (D1).
- **Background load (step 4):** `ab.py` reported no process above 10% of a CPU in either A/A; the
  1-minute load stayed under 0.7.

The probes (the load loop, the `turbostat` driver and the clock-up timer) were throwaway scripts;
the A/A runs are `~/code/ab-runs/2026-10-08-stage2-aa-a-balance-performance` and
`-aa-b-performance`.

### Stage 3: the baseline commit

1. Merge stages 1-2 and tag the merge commit `baseline-i7-9700k` here, and the submodule's
   commit with the same name in `indrajala-math-rust` (D6).
2. Create `docs/machine_profiles/i7-9700k.md` (D8): the machine (a link to its profile and the
   D1 policy), the commit pair, how to rerun it (stages 4-5's commands), and empty sections for
   stages 4-6.

Done when the tags exist and the skeleton is merged.

**Done 2026-10-08.** Both repositories have an annotated `baseline-i7-9700k` tag: indrajala-ml at
`6f0e9de` (stage 2's merge) and indrajala-math-rust at `66ecc0d`, its `rust/`. The skeleton is
[machine_profiles/i7-9700k.md](machine_profiles/i7-9700k.md). Stage 5 has one gap to close: the
conv mini-batch 512 epoch (step 3) has no script config yet, so stage 5 adds one to
`prepared_dataset_timing.py`.

### Stage 4: noise and baseline from A/As of every benchmark

At the tagged commits, one A/A run per `ab.py` adapter at its default arguments
(`--old baseline-i7-9700k --new baseline-i7-9700k`, `ONNONO`):
`prepared_dataset_timing`, `focused_benchmark`, `epoch_op_profile`, `accuracy_pass_timing`,
`op_call_timing`, `batch_size_timing`. A run is launched in the background, one at a time, the
machine checked first (CLAUDE.md, Timing).

Each run gives two things, written into `docs/machine_profiles/i7-9700k.md`:

- **The baseline numbers:** the pooled medians and min-max over all six passes, per row.
- **The noise:** per row, the spread of per-pass medians as a % of the median; the number of
  shifted passes and their size; the controls' spread; any row flagged *consistent* (each one is
  investigated before the stage merges: Pitfalls).

Then, from those numbers, update measurement.md's figures for this machine: "numbers vary 20-30%
between passes", "whole passes shift 5-15%", the HIGH_LOAD flag (1.5) and the 5% shifted-pass
threshold. Keep a rule as it is if the numbers support it. If one doesn't fit this machine,
propose the new value with its numbers in the PR; the owner decides.

Done when all six A/As are recorded and measurement.md's noise figures are this machine's.

**Done 2026-10-08.** The six A/As (`~/code/ab-runs/2026-10-08-baseline-aa-*`) are in
[machine_profiles/i7-9700k.md](machine_profiles/i7-9700k.md), each as `ab.py report --pooled`
(added for this stage), with the noise table. Per benchmark, the median spread of a row's
per-pass medians was 0.7-4.3%; no pass of 36 shifted more than 0.9%; 7 of 591 rows were flagged
*consistent*, all under 2% and by chance. measurement.md's figures are this machine's now. The
owner made the noise rules per machine (#582; D4 replaced): the i7's are `shifted_pass` 2%,
`high_load` 1.5 and `small_consistent` 2%. Along the way:

- **`thermald` reset PL1 to 95 W mid-session**, twice. `ab.py` now re-reads the frequency policy
  and power limits after every pass (#579), and the setup script stops `thermald` (#581, D10).
  The first two attempts at these A/As were discarded; the failed one is
  `-prepared_dataset_timing-pl1-reset`.
- **Per-process modes** in the single-example Rust `downstream` on the 32 x 5408 conv tail
  (about 20.8, 22.1-22.4 and 23.8 µs), in both of the day's `focused_benchmark` A/As: a lead
  for stage 6, beside the laptop's bimodal max-pool.
- **numpy batch ops ran 15-20% slower** in the afternoon A/A (PL1 65 W throughout) than in the
  morning one (PL1 at 95 W for part of it): whether 65 W costs all-core numpy bursts is a lead
  for stage 6.

### Stage 5: the end-to-end and per-op baseline

At the tagged commits, the measurements that aren't A/As, each run by the protocol in
measurement.md §6 (one process per measurement, numpy and Rust apart, rotated order):

1. **The per-op table:** `focused_benchmark.py` over the shapes in the laptop's retired table
   (32 x 5408 and 30 x 784, `forward_batch`, `downstream_batch`, `accumulate_gradient_batch`, at
   batch 32 and 512), numpy and Rust, at default threading and at `--openblas-threads 1
   --rust-threads 1`. A Rust/numpy column, and the laptop's numbers beside them as context.
2. **The conv demo's ratio table:**
   `python -m indrajala_ml.demos.conv.demo_conv_rust_vs_vectorized_digit_recognition`, all five
   architectures, UCI digits and the MNIST subset, single and mini-batch; with the laptop's
   ratios beside them.
3. **The kernel protocol configurations** (§6, "Kernel changes end to end"): the conv subset
   epoch at mini-batch 32 and 512, and the dense MNIST epoch at batch 32 and 512, both backends.
4. **Where a Rust epoch spends its time:** `epoch_op_profile.py` for the conv configuration,
   the top ops by share.
5. **On the laptop, at the same tags (D11):** its profile re-recorded; a golden run recorded
   there and checked against `jebel`'s file, and `jebel` checking the laptop's (D5); and steps
   1-4 there, for the laptop columns. The laptop keeps its old checklist: Brave and Zed closed,
   on AC power, its own machine profile compared before each measurement.

Done when `docs/machine_profiles/i7-9700k.md` has every table, each with its command, commit
pair and date, the laptop columns measured, and D5's answer recorded.

### Stage 6: re-check the laptop's machine-specific findings

Measurement only, under D9. Each finding gets its measurement and one of three outcomes: still
holds, holds with other numbers, or doesn't hold here.

1. **The threading threshold and thread count** (`matmul_thread_count`): `focused_benchmark.py
   --matmul MxKxN --rust-threads {1,2,4,8}` over the products in the `linalg.rs` comment (the
   conv dense tail, 32 x 5408, at batch 32 and 512; conv's accumulate; dense MNIST's 12M), and
   `set_matmul_threading` thresholds from 2M to 32M on the conv and dense epochs at batch 32 and
   512. The question: where does threading start to pay with 8 real cores, and do 2 or 4 threads
   now beat both 1 and 8 anywhere.
2. **The kernel blocking constants:** `--kernel-overrides R:K` for `matmul_narrow`'s rows per
   block (4) and `matmul_long_k`'s slab (64), on the shapes they were tuned on, at one thread
   and at training's thread count (§7: one-thread numbers hide threading).
3. **OpenBLAS's spin-wait penalty:** a Rust batch op timed right after a threaded numpy call,
   against the same op after a sleep longer than `OPENBLAS_THREAD_TIMEOUT`.
4. **numpy's threading in training:** the conv mini-batch 32 epoch at default and at
   `OPENBLAS_NUM_THREADS=1`, numpy backend.
5. **The bimodal max-pool downstream** (batch 32): its time over 20 or more processes; whether
   two modes appear. If they do, `perf_region.py` (with D3's `perf_event_paranoid`) for the same
   counters as the laptop's investigation.
6. **Clocks carry over between settings and paired processes** (§7, Clocks): from stage 2's
   clock-up probe and the rotated runs above.

Each finding's outcome goes into measurement.md under D9, and each measured difference large
enough to act on (over the 5% bar, measurement.md §9) becomes an entry in
[next-steps.md](next-steps.md) for its own crate workplan.

Done when every finding has an outcome recorded with its numbers.

### Stage 7: docs, and retire the plan

1. measurement.md: §2 describes only this machine. §5-§7 carry its noise numbers and the stage 6
   outcomes. The laptop's findings that don't hold here are in the "Measured on the Ryzen laptop"
   note (D9).
2. CLAUDE.md's Timing section: the machine's setup script before a run, and Firefox and snap
   refreshes alongside Brave and Zed.
3. The PyPI workplan: its "on the Ryzen machine" (stages 1-2 and Pitfalls) becomes "on the
   benchmark machine", and its timing baselines are this plan's.
4. The README links `docs/machine_profiles/i7-9700k.md` where it describes measurement.
5. Retire this workplan into next-steps.md, with stage 6's leads.

Done when this file is deleted and next-steps.md lists it.

## After this plan

- **Crate tuning for this machine:** any threading or blocking constant that stage 6 finds
  wrong here, each in its own crate workplan with an A/B (measurement.md §9).
- **Rerunning the baseline** after a change that could move every number (kernel, BIOS,
  numpy/OpenBLAS, rustc, the frequency policy): rerun stages 4-5 at the same tags, then at
  `main`, and replace `docs/machine_profiles/i7-9700k.md`'s tables, with the reason, in one PR.

## Out of scope

- Any change to the crate's kernels, threading or blocking constants, or to a network's
  numerics. This plan measures; it changes nothing that runs in training.
- Re-recording any parity test or the golden run for bits (the golden run is recorded fresh here
  because it is machine-specific, not to move bits).
- Laptop runs beyond D11's (stage 5's golden run and columns), and claiming a cross-machine
  speedup from the laptop columns.
- GPU work (the UHD 630 is unused), BIOS settings beyond what the owner chooses for cooling and
  power limits, and kernel boot parameters (`isolcpus`, `nohz_full`): only if stage 2 shows the
  noise needs them, as a new decision.
