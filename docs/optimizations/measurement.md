# Optimizations: measurement

Part of the optimization docs; the index is [../optimizations.md](../optimizations.md).

How to measure a performance change here, what goes wrong, and the rules a change must meet.
Pure-Python networks are never timed: they are for correctness and parity only.

## The machine

Ryzen 7 3700U laptop: 4 cores / 8 threads (Zen 2), 32 KB L1d and 512 KB L2 per core, 4 MB L3,
`schedutil` governor. The full record is `docs/machine_profiles/ryzen7-3700u.json`. Before quoting
numbers against the docs, run `python scripts/machine_profile.py compare
docs/machine_profiles/ryzen7-3700u.json` in the same shell and env as the benchmark: it exits 1 and
names each changed identity field (CPU, caches, cpufreq policy, kernel, Python, numpy and its
BLAS, rustc, the crate's release profile, thread env vars). `profile --out FILE` records a new
machine. `perf` needs `kernel.perf_event_paranoid` <= 2, and it is 4 by default here (`sudo
sysctl` lowers it until reboot); without it, internals are timed in a local probe build of the
crate (timers and counters behind a Python-callable switch).

## Tools

From the quick survey to the decisive measurement:

| question | tool | notes |
| --- | --- | --- |
| Which ops look slow? | `python -m indrajala_ml.demos.demo_layer_op_timing` | every layer op, numpy and Rust interleaved; batch rows can be far off (interleaving). Finds candidates, never judges them. |
| How fast is one op? | `python scripts/focused_benchmark.py` | loops of about 20 ms, median of 9, each (case, backend) in its own process; faults per call; `--matmul MxKxN`, `--rust-threads`, `--openblas-threads`, `--malloc both`. **The number to quote.** |
| Faults or compute? | `focused_benchmark.py --malloc both` | glibc defaults against both allocator thresholds at 1e9; a time that drops with the faults was paying for them. |
| Why is it slow (or slow in some processes)? | `python scripts/perf_region.py -- driver.py` | hardware counters for only the region a driver marks (`with counted():`), per unit of work, one row per process; `OPENBLAS_NUM_THREADS=1` unless `--openblas-threads`. Needs `kernel.perf_event_paranoid` <= 2 (`sudo sysctl`, until reboot). For cycles per instruction, `perf record` the driver and `perf annotate` the op. |
| One op's share of real epochs | `python scripts/epoch_op_profile.py` | cProfile of Rust training by crate op, one process per (architecture, trainer, repeat); `--op`, `--label`. Resolves changes of a few % that epoch timing can't. |
| A training-path change, old against new | `python scripts/prepared_dataset_timing.py time` | one trainer epoch per process, dense full MNIST and the conv subset, both backends; old checkout first on `PYTHONPATH` (a `git worktree` of `main`); `--epochs N`. |
| An accuracy pass, per row against batched | `python scripts/accuracy_pass_timing.py time` | all demo architectures, both backends; counts differing predictions. |
| Dense full-MNIST epochs, broken down | `python scripts/batch_size_timing.py time` / `profile` | epoch, step loop, one accuracy pass and conversions apart, per batch size. Its accuracy-pass column times the old tuple path. |
| The Rust/numpy ratios end to end | `python -m indrajala_ml.demos.demo_conv_rust_vs_vectorized_digit_recognition` | about 3 minutes; median of 5 from identical weights, UCI digits and a 2000-row MNIST subset, plus a Rust op profile (`rust_op_breakdown`). |
| A threading setting | `set_matmul_threading(t, threshold)` | in one process, no rebuild; accept only on end-to-end numbers. |

## Protocols

- **Numbers vary 20-30% between passes on this machine**, sometimes more (a background IDE once
  spoiled a whole measurement). Treat changes under about 20% as noise unless two passes agree,
  and re-check a surprising result with the order reversed.
- **One process per measurement**, and **numpy and Rust in separate processes**, always.
- **Old against new build:** commit the crate change first, then alternate builds (old, new, new,
  old, ...), switching with `git checkout main -- <changed files>` and back, not a stash, and
  `./cli build-rust` (about 6 s) each time. Medians of 5 or more; run multi-minute runs in the
  background with an ETA.
- **Kernel changes end to end:** one epoch of MNIST, `ConvSpec(3, 8)`, dense 32, mini-batch 32, lr
  0.5 on the demo's 2000-row subset, from a snapshot of `randomized(...)` after `np.random.seed(0)`
  with `random.seed(0)` before each epoch; the same at mini-batch 512; one dense MNIST epoch at
  batch 32 and 512. Then the conv demo for the ratio table.
- **Read the other backend as a control.** A change to one backend can't move the other; when
  the control moves as much between builds, the epoch numbers can't resolve the change, so use
  `epoch_op_profile.py`.
- **Don't judge a training-path change on trainer epoch time alone.** The accuracy passes (n + 1
  for n epochs) are a large share of a short run: time them apart.

## Gotchas

- **numpy's OpenBLAS threads slow a Rust call run soon after.** After a BLAS call its workers
  spin for 100-500 ms (`OPENBLAS_THREAD_TIMEOUT`). A Rust batch op in that window measured 2-5x
  slow, whatever Rust's own thread count. Any per-op ratio measured interleaved with threaded
  numpy is suspect; the conv demo interleaves (unmeasured effect, at most the start of each run).
- **numpy's default threading makes numpy look fast in hot loops** but doesn't pay in training
  (its MNIST conv mini-batch 32 epoch was faster at `OPENBLAS_NUM_THREADS=1`). Compare on one
  thread each (`--openblas-threads 1 --rust-threads 1`) to see the kernels.
- **Clocks.** Idle cores drop to 1.1-1.5 GHz and a busy one boosts to 3.8 GHz, taking hundreds of
  ms of load to clock up. Time loops, not single calls. Clocks carry over between settings (an
  8-thread setting run after another read up to 2x faster), and between paired processes (the
  second of a pair once read 26-27 against 41-43 µs whichever setting ran second): rotate the
  order.
- **`--rust-threads 1` hides what training's threading does.** Past 8M flops training threads
  the op over output rows. conv-conv's second-conv accumulate (8 rows) lost 40% to 2-row blocks
  on one thread and nothing at default threads, where each thread already has 2 rows. Time a
  kernel change at the thread count training uses too.
- **Isolated loops flatter threading.** Back-to-back calls keep every core clocked up; in
  training the cores idle between calls and each threaded call pays the cold clock.
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
- **A time can be bimodal between processes, not only noisy.** The one-pass max-pool downstream
  at batch 32 (probe and op) runs at about 17 cycles a window (190-200 µs) in some processes
  and 42-49 (470-570 µs) in others, tight within each. The counters show the same instructions, L1 and L2
  accesses in both modes; the slow one is integer-scheduler stalls (ALU-token stalls 24 against 1
  a window). Ruled out, each measured: page faults and allocator thresholds, clock frequency,
  the core and its SMT sibling, virtual placement (buffers pinned to a 2^28-aligned arena, ASLR
  off), physical pages (re-paged between trials), the AVX upper state, SSBD, the `+=` read (a
  store-only pass) and the division (a slot-offset table: still 4.7-8.5 ns a window). The cause is
  unknown. A median of loops in one process can't see it: run several processes and quote both
  modes.
- **The conv demo's mini-batch runs barely train** (about 10% accuracy in 1-2 epochs at lr 0.5):
  their timings are valid, their accuracy columns are not.
- **Rust dense-MNIST epoch times from before #365 aren't comparable** with later ones: the shared
  pixel floats made row conversion 12% cheaper.

## Judging correctness

Single-example training is chaotically sensitive to rounding: networks trained from the same
weights by numpy and Rust, or by numpy against itself with one weight nudged by 1 ULP, agree on
only 71-83% of test predictions after training, though they stay within 1e-15 through the first
UCI epoch. So a change is judged by step-by-step parity (per-step agreement to about 1e-15), never
by end-of-run accuracy.

## Rules for an optimization PR

- **Two repos.** Crate changes land in `indrajala-math-rust` (`rust/`) first, with its own
  numpy-only tests; then a PR here bumps the submodule and runs the full suite (`./cli build-rust
  && ./cli test`). Neither merges until both pass.
- **Bit-identical claims are tested, not assumed:** a crate test pins the new op with `==` on
  `tolist()` at shapes that reach every kernel path, threading and blocking threshold; it passes
  on the old build, and a mutation (a second rounding, a changed start value or order) fails it.
- **Bit-changing changes:** the `rtol` parity tests (`tests/test_*fused_layer_ops.py`, the
  step-by-step network parity tests) pass unchanged; a moved end-to-end pin is compared with a
  1-ULP control on the old code (nudge one initial weight) and updated only if it moves similarly,
  with the control recorded in the PR; record the max abs and ULP difference from the old op.
- **Measure first; close what doesn't pay.** A stage 0 measures the stake against a bar fixed
  before measuring (about 5% of an epoch in a trained configuration). Each stage is its own PR,
  merged before the next. A stage with no gain is closed and its reason added to
  [Rejected](rejected.md). Every PR quotes before/after per-op rows for the ops it touches and the
  end-to-end effect.
- **Update the docs to the new state, don't append to them:** replace changed numbers in
  [Current baseline](current-baseline.md), move the item between [Candidates](candidates.md),
  [Implemented](implemented.md) and [Rejected](rejected.md). The measurements behind a change live
  in its PR.
