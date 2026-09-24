# Optimizations: method

Part of the optimization record; the index is [../optimizations.md](../optimizations.md).

## How to measure

- **First, check the machine:** `python scripts/machine_profile.py compare
  docs/machine_profiles/ryzen7-3700u.json` before quoting new numbers against the ones here. It
  exits 1 and names each changed identity field (CPU, caches, ISA, cpufreq driver and governor,
  boost, memory, GPUs, kernel, Python, numpy and its BLAS build, rustc, the crate's release
  profile, the thread env vars), and prints the state (clocks, load, free memory, power, commits)
  side by side for context. Run it in the same shell and env as the benchmark: a thread env var
  set for the run shows up as a difference. `profile --out FILE` records a new machine.
- **Per op, quick survey:** `python -m indrajala_ml.demos.demo_layer_op_timing`. It times every
  dense and conv layer method, single-example and batch 1/32/512, numpy and Rust interleaved,
  300 calls per loop (300 // batch for batch ops, at least 10), median of 5 loops. **Its batch
  rows are noisy and can be far off**, mostly because of the interleaving (next item): it
  reported 30 x 784 `forward_batch` at batch 512 as 3469 µs where a focused benchmark measured
  1350, and `downstream_batch` at 32 x 5408, batch 32 as 9.5x numpy where the focused
  benchmark in separate processes measures 2.4-2.8x (an interleaved focused benchmark had read
  13.9x). Use it to find candidates, not to judge them.
- **Per op, focused:** time the op in loops of about 20 ms, median of 9. Two passes per build
  at least. This is the number to quote. `python scripts/focused_benchmark.py` does this: it
  runs every (case, backend) in its own process and reports minor page faults per call. It
  covers the demo's layer ops, the parts of the backward batch ops and bare `--matmul MxKxN`
  products. It can also force threads (`--rust-threads`, `--openblas-threads`). See its
  `--help`. **Time numpy and Rust in separate processes**, never
  interleaved in one. After a numpy BLAS call, OpenBLAS's threads keep spinning for between 100
  and 500 ms, and a Rust batch op run in that time measured 2-5x slow (see "Other findings").
- **End to end:** `python -m indrajala_ml.demos.demo_conv_rust_vs_vectorized_digit_recognition`
  (about 3 minutes; median of 5 interleaved runs from identical initial weights, UCI digits and
  a 2000-row MNIST subset, single-example and mini-batch 32, plus a cProfile of Rust time by
  op; `rust_op_breakdown` in the demo gives it for any architecture and trainer). For a dense op
  change, also one epoch of dense MNIST 784 -> 30 -> 10 from identical weights, single-example
  and mini-batch 32, median of 3.
- **One op's share of real epochs:** `python scripts/epoch_op_profile.py` runs the conv demo's
  `rust_op_breakdown` (cProfile of one Rust training run on the MNIST subset), one process per
  (architecture, trainer, repeat), and reports each crate op's min-max seconds and calls
  (`--op` filters, `--label` names the build). Run it per build, builds alternated, for a change
  worth a few % of an epoch: candidate 4's gain was lost in whole-epoch timing (numpy's
  control moved as much) but separated cleanly here.
- **A training-path change, before and after:** `python scripts/prepared_dataset_timing.py time`
  times one trainer epoch (dense full MNIST and the conv demo's subset, single-example and B =
  32, both backends), one process per measurement. Run it once as is and once with the old
  checkout first on `PYTHONPATH` (see its docstring for the namespace-package caveat); a
  `git worktree add` of `main` in a scratch directory makes that checkout. `--epochs N` trains
  each run for N epochs (one accuracy pass per epoch, plus one before), for a long run's share.
  Read the other backend's cells as a control: a change to one backend can't move the other,
  so when the control moves as much between builds (numpy -5.1% against Rust -5.6% in
  candidate 4's A/B), the epoch numbers can't resolve the change; use `epoch_op_profile.py`.
- **One accuracy pass, per row against batched:** `python scripts/accuracy_pass_timing.py time`
  (candidate 2's stage 0; `report runs.json` reprints a saved run). Dense full MNIST and the conv
  subset, both backends, one process per measurement, with the saving as a share of a one-epoch
  and a long run and a count of rows whose batched prediction differs.
- **Dense full-MNIST epochs, broken down:** `python scripts/batch_size_timing.py time` and
  `... profile` (see its docstring). It times the trainer epoch, the step loop, one accuracy pass
  and the batch and row conversions separately, one process per (backend, batch size, repeat),
  with the order rotated each repeat. **Don't judge a training-path change on trainer epoch time
  alone.** The two accuracy passes were 52-73% of a full-MNIST epoch before candidates 1 and 2
  (see "Other findings"); after candidate 2 a batched pass is 0.21 s in Rust and 0.18 s in numpy,
  still enough to blur a step-loop change. Its "accuracy pass" column still times the tuple path
  (`_training_accuracy` without a prepared dataset: `classify_state` and a row conversion each),
  which the trainers no longer use for array networks, so it overstates their pass.
- **End to end, old against new build** (for a kernel change, before the demo): one epoch of
  MNIST, one `ConvSpec(3, 8)`, dense 32, mini-batch 32, lr 0.5, the demo's 2000-row subset, from
  a snapshot of `randomized(...)` after `np.random.seed(0)`, with `random.seed(0)` before each
  epoch. One process per epoch, builds alternated with the order swapped every run, medians of
  5 or more. The same at mini-batch 512, and one dense MNIST epoch at batch 32 and 512. Run
  these, and the demo, in the background with an ETA.
- **Old vs new builds:** alternate the builds (old, new, old, new) and run each benchmark on
  both. Builds take about 6 s (`./cli build-rust`). Commit the crate change before switching,
  and switch with `git checkout main -- <changed files>` (for example `src/linalg.rs`, or
  `src/fused.rs` too for stage B) and back, not a stash.
- **This machine** (Ryzen 7 3700U laptop, 4 cores / 8 threads, 512 KB L2 per core, 4 MB L3;
  the full record is `docs/machine_profiles/ryzen7-3700u.json`, and `python
  scripts/machine_profile.py compare docs/machine_profiles/ryzen7-3700u.json` checks that the
  hardware, cpufreq policy, OS and software stack still match it) varies 20-30% between passes, sometimes more. A background IDE made a first measurement
  unusable once. Treat changes under about 20% as noise unless both passes agree, and re-check a
  surprising result with the build order reversed. Idle cores drop to 1.1-1.5 GHz and a busy one
  boosts to 3.8 GHz (`schedutil`), so a single call after a pause measures slow. Time loops,
  not single calls. Clocks also carry over between settings: in a sweep, an 8-thread
  setting run right after another one measured up to 2x faster at `(32, 512) @ (512, 5408)`
  (5.3-5.8 against 9.9-10.1 ms), since cores take hundreds of ms of load to clock up. Rotate
  the order of settings. `perf` can't be used without root (`perf_event_paranoid` is 4), so probes
  go in a local crate build instead (timers and counters behind a Python-callable switch).
- **Faults or compute:** `focused_benchmark.py --malloc both` times each case with glibc's
  defaults and with `MALLOC_TRIM_THRESHOLD_` and `MALLOC_MMAP_THRESHOLD_` at 1e9 (nothing is
  returned to the OS, so nothing faults in again), each in its own process. A time that drops
  with the faults was paying for them.
- **A probe's allocation pattern is not the real call path's.** Check faults on the real op.
  In optimization 7's probe, a tight Rust loop allocating a fresh 22 MB output every call
  (with a reused buffer of the same size also live) faulted on every page: 5410 faults per
  call, about 10 ms (40%) at `(512, 32) @ (32, 5408)`. The same product through the Python
  op had 0.1 faults per call. So quote allocation and fault costs from the real op
  (`focused_benchmark.py` reports faults per call), not from a probe loop. Candidate 4's
  stage 0 probe showed it twice: freeing its buffers inside each repetition, it faulted 1024
  times a call at N = 32 and its parts summed to 3.5 ms against the real op's 1.5; freeing them
  in the real call path's order (the previous call's `A` and `cols` after the new ones exist)
  it still faulted 195-490 times a call against the real op's 0-20, since glibc's mmap threshold
  adapts to what the process freed before. Raising both glibc thresholds (`--malloc raised`
  in `focused_benchmark.py`, or the env vars in a probe's process) removes the faults on both
  sides, so compare a probe's parts with the real op there.
- **Threading:** `set_matmul_threading(t, threshold)` forces a thread count and threshold in
  one process, so a sweep needs no rebuild. Accept a threading change on the end-to-end number
  only (see "Threading").
- **Never time pure Python.** It is for correctness and parity only.

## Rules for an optimization PR

- **Two repos.** Crate changes land in `indrajala-math-rust` (`rust/`) first, with its own
  numpy-only tests. Then a PR here bumps the submodule and runs the full suite
  (`./cli build-rust && ./cli test`). Neither merges until both pass.
- **Bit-identical claims are tested, not assumed.** A change claimed as bit-identical gets a
  crate test that pins the new op with exact equality (`==` on `tolist()`, not `approx`), at
  shapes that reach every kernel path, threading and blocking threshold. The test must also pass
  on the old build, and a mutation (a second rounding, a changed start value or order) must
  fail it. If exact equality fails, the change is bit-changing.
- **Bit-changing protocol.** For a change to summation order:
  1. The existing `rtol` parity tests (`tests/test_*fused_layer_ops.py`, the step-by-step
     network parity tests) must pass unchanged.
  2. If an end-to-end pinned result moves (for example the Rust conv network's 0.9875 /
     epoch 10 / 0.925), don't loosen it silently. Compare it with a 1-ULP control: nudge one
     initial weight by 1 ULP on the *old* code. If the pin moves by a similar amount, it is
     rounding sensitivity (see "Other findings"); update the pin and record the control in
     the PR. If not, treat it as a bug.
  3. Record the max abs and max ULP difference from the old op at the benchmark shapes.
- **Measure first, and close what doesn't pay.** Each stage is its own PR, merged before the
  next starts. A stage that measures no gain is closed with its numbers recorded here, not
  merged. Every PR quotes the before/after per-op rows for the ops it touches and the end-to-end
  ratios.
