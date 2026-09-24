# Workplan: optimization 6, threading in `for_each_row_range`

Candidate 1 in [`../optimizations.md`](../optimizations.md), the highest-ranked open item. All
crate code is in `rust/src/linalg.rs`. The rules in `../optimizations.md` apply: each stage is
its own crate PR plus a submodule bump PR here, measured before and after. A stage that measures
no gain is closed and its numbers recorded.

## Context

`for_each_row_range(out, rows, cols, total_flops, compute)` is the only threading in the crate.
Below `THREADING_THRESHOLD_FLOPS` (4M) it calls `compute` once. Otherwise it takes
`min(available_parallelism, 8, rows)` threads, gives each a contiguous block of
`rows.div_ceil(threads)` output rows, and starts them with `std::thread::scope`. Three kernels
use it:

| kernel | products | threaded work split |
| --- | --- | --- |
| `matmul_2d` (via `matmul`) | dense `downstream_batch` (`delta_batch @ W`), `accumulate_gradient_batch` (`delta_batch.T @ X`) | rows of `a`; every thread reads all of `b` |
| `matmul_nt` | every dense `forward_batch` (`X @ W.T`) | rows of `X`; every thread reads all of `W` |
| `matmul_narrow` | conv forward, downstream, accumulate | rows of `cols` / `D` |

What `../optimizations.md` measured, with threading forced on for the tiled kernel:

- Starting threads costs 100-200 µs per call. A 64x64x64 matmul takes 28 µs on 1 thread and
  120-200 µs on 2-8.
- At 32 x 5408, batch 32, `downstream_batch` takes about 550 µs on 1 thread and 550-630 µs on 8.
  The per-op table has it at about 3000 µs (numpy: 238-258).
- Large products still scale: `(32, 512) @ (512, 5408)` goes from 32 ms on 1 thread to 6-7 ms
  on 8. 30 x 784 at batch 512 goes from 1.3-1.7 ms to 0.8 ms on 4 threads, and no better on 8.

**Open question: the 550 vs 3000 µs gap. Answered in stage 0b step 1:** the 3000 µs came from
the harness (numpy's OpenBLAS threads still spinning), and the causes below were each measured
there. Kept as first written:

- The two numbers came from different harnesses: a Rust-level microbenchmark and the Python
  per-op harness.
- The output `vec![0.0; m * n]` is zeroed on the calling thread, then written from other cores,
  so its cache lines move between cores. First-touch page faults may also land on 8 threads at
  once.
- 8 threads on 4 cores: two SMT siblings share one core's FMA units and L2.
- The CPU clocks down when all cores wake. This laptop varies 20-30% between passes already.

**Invariant to keep:** every output is computed by exactly one thread with its kernel's fixed
summation order. Any split (by rows, by columns, by a pool) keeps that, so every stage below is
expected to be **bit-identical**, and each stage tests that claim.

**Where it matters end to end.** Dense MNIST 784 -> 30 -> 10 at batch 32 is below the threshold
everywhere (about 0.75M flops per matmul). So the end-to-end target is the MNIST conv
mini-batch runs, where the dense layer after the conv stack is 5408 inputs wide (ratios
0.44-0.54, the worst in the table). Stage 0 confirms this by counting which calls cross the
threshold.

## Stage 0: instrument and measure (a crate PR for the override, then measurement only)

### 0a. A threading override for tests and benchmarks

**Done** (crate PR #15). One change from the plan below: the threshold override is read *before*
the default threshold comparison, since threshold 1 has to lower the threshold. On x86 it is one
relaxed load. Mutations "later threads start one row late" and "last block drops a row" each
fail all 20 new tests.

Add a crate-level override, exposed to Python:

```rust
/// Test/benchmark hook: max threads (0 = default) and threshold (0 = default) for every
/// threaded matmul. Can't change any output's value; that is what the tests check.
#[pyfunction]
fn set_matmul_threading(max_threads: usize, threshold_flops: usize)
```

The two settings are held in `AtomicUsize` statics read once per call, after the default
threshold check (so the unthreaded path doesn't slow down). An env var would be read once per
process, so it couldn't sweep settings within one run, which is why this is a function. Resetting
with `(0, 0)` restores today's behaviour exactly.

Tests (crate `tests/test_linalg.py`):

- **Thread count can't change bits.** For each of `matmul` (matrix @ matrix), `matmul_nt` (via
  `layer_relu_forward_batch` as today) and `matmul_narrow` (via a conv op), at the `BIG_SHAPES`
  plus a conv shape: compute once with `(1, huge threshold)` (unthreaded), then with thread
  counts 2, 3, 5, 8 and threshold 1. Every result must equal the unthreaded one with `==` on
  `tolist()`. Counts 3 and 5 give uneven row blocks, and `rows < threads` is covered by
  `(5, 3000, 21)`.
- A fixture resets the override after each test so it can't leak into other tests.
- Mutation check (with `PYTHONDONTWRITEBYTECODE=1` and `-B`): make one thread's range start one
  row late, or give the last block a different `rows_per_block`. The test must fail. Then restore.

### 0b. Measurements (recorded in the stage 0 PR description and `../optimizations.md`)

**Step 1 done (2026-09-24).** The gap came from the harness. Interleaved numpy loops leave
OpenBLAS threads spinning, and they slow the Rust call 2-5x. Timed on its own, Rust takes
528-570 µs at every thread count (1: 528-539, 2: 854-902, 4: 729-737, 8: 565-685, default
559-570); numpy takes 239-244. The listed causes, each tested on a local experiment build
(crate branch `exp/threading-0b`, not pushed):

- **Spawn only**, with an empty `compute`: 25-30 µs per call unthreaded, then 89-96 at 2
  threads, 141-165 at 4 and 211-232 at 8. So spawning costs about 60-70, 110-140 and 180-200 µs.
- **First touch:** pre-touching the output on the caller, or leaving it uninitialized for the
  workers to write first, made no consistent difference at any thread count.
- **Clock and SMT, the main cost.** Timed inside each worker, a worker's share of the work takes
  430-800 µs at 2 threads (16 rows) and 250-315 at 8 (4 rows), against 535 µs for all 32 rows
  on the caller. With register-only FMA work in place of the matmul (no memory traffic), a
  spawned worker runs 1.6-3.1x slower per row than the caller, at every thread count. Per-core
  clocks read during the runs: one core at 3.8 GHz and the idle ones at 1.1-1.5 GHz with 1
  thread, and every core at 2.2-3.1 GHz with 8. The reading: `schedutil` doesn't raise the
  clock for a thread spawned on every call, since it has no load history. That reading is
  inferred, not measured directly.
- **Caller computes chunk 0** (spawn one thread fewer): 443-584 µs at 4 threads and 564-621 at
  8, against 729-779 and 723-728 for spawn-all in the same runs.

What this means for stage A-C: stage C (`b` traffic) can't be the main cost here, since the
register-only probe shows the same slowdown with no memory traffic. Stage B (a pool) could remove
both the spawn cost and the idle clock, and the caller should do one chunk itself. Steps 2 and 3
still decide it, measured in separate processes from numpy.

All on the current kernel, focused per-op benchmark (loops of about 20 ms, median of 9, two
passes). Use a scratch script, not a demo:

1. **Explain the gap.** `downstream_batch` at 32 x 5408, batch 32, in the Python harness at
   threads = 1, 2, 4, 8, with the output buffer pre-touched and not. If 1 thread is about 550 µs
   and 8 about 3000, the cost is in threading. Then test the causes above one at a time:
   - spawn only, with an empty `compute`;
   - 4 threads against 8 (SMT);
   - `vec![0.0]` against an uninitialized buffer written by the worker (first touch).
2. **Thread-count sweep** at every shape in the `../optimizations.md` op table, plus `(32, 512)
   @ (512, 5408)`, conv forward/downstream/accumulate at MNIST 28x28 N = 32 and 512, and a flop
   ladder (1M, 2M, 4M, 8M, 16M, 32M, 64M) for each kernel. Record µs at 1/2/4/8 threads.
3. **Which production calls cross the threshold.** Count per call site in one epoch of each conv
   demo configuration and dense MNIST at batch 32 (a temporary counter in a local build, not
   committed).

**Decision recorded at the end of stage 0.** For each kernel, the flop count where 2, 4 and 8
threads start to beat 1, with the pool-free spawn cost subtracted out. This decides:

- whether stage A alone (threshold and thread cap) closes most of the gap;
- whether spawn cost, which stage B removes, or `b` traffic, which stage C removes, is the
  larger remaining cost at each shape.

Stages A-C below run in the order stage 0 justifies. Skip any stage stage 0 shows can't pay.

## Stage A: threshold and thread-count policy (smallest change)

Replace the single 4M-flop constant with a policy taken from the stage 0 sweep. Candidates:

- a **minimum work per thread** (`threads = min(cap, total_flops / min_flops_per_thread)`),
  so a 5M-flop product gets 1 thread and a 90M one gets 8. It replaces the all-or-nothing jump
  from 1 to 8 threads at 4M;
- a **cap at physical cores** (4 here) if stage 0 shows 8 no better than 4. `available_parallelism`
  reports logical CPUs, so a physical-core cap needs a constant or a measured default. Keep it
  simple: a constant cap, documented as machine-dependent;
- a **separate threshold per kernel** only if the ladder shows them differ by more than 2x.

The comment in `for_each_row_range` about the rejected rows-per-thread floor (a mini-batch
step got worse with it) must be re-checked, not deferred to. Rerun the end-to-end benchmark
that it cites alongside the new policy.

Tests: 0a's bit-identity tests already cover every thread count. Add a crate test that the
policy picks 1 thread at the smallest production shape that is threaded today, so a policy
regression shows up as a test failure, not as a benchmark surprise.

## Stage B: persistent thread pool (removes spawn cost)

Only if stage 0 shows spawn cost is still a large share once stage A is in.

**The dependency decision (make it at the start of stage B, recorded in the PR):**

- **rayon** (recommended). A global pool that already exists and has been tested for years.
  `out.par_chunks_mut(rows_per_thread * cols).enumerate()` keeps the contiguous row blocks as
  they are, so each output is still computed by one thread with the same order. It adds the
  crate's first dependency after pyo3, plus build time; measure both.
- **A small hand-rolled pool** (`OnceLock` of workers, a barrier per call). No dependency, but
  it is new unsafe-adjacent code to get right (scoped borrows across a persistent pool need
  either `unsafe` lifetime erasure or copying). Prefer it only if rayon's own per-call overhead
  measures worse.

Whichever is chosen, the split must stay **static** (fixed row blocks decided before any
thread runs). Work stealing moves blocks between threads, but each output is still computed by
one thread in one order, so it doesn't change bits. The static split just makes that argument
obvious. The 0a tests prove it either way.

Watch for: the pool's threads competing with Python's own threads (none in the trainers
today), and pool start-up at import time. Initialize lazily, as `available_parallelism_cached`
already does.

## Stage C: split by columns when `b` is the larger operand

Only if stage 0 shows `b` traffic is the cost at shapes like 32 x 5408, batch 32 (where 8 threads
barely beat 1).

- **`matmul_2d` / `matmul_narrow`:** give each thread a range of output **columns** (a multiple of
  16, the tile width) for all rows. Each thread then reads only its `K x cols_per_thread` panel of
  `b`, not all of it. `tiled_row_range` gets `col_start`/`col_end` and an output row stride
  (today it assumes the row stride is `n`). Each output's FMA chain is unchanged.
- **`matmul_nt`:** the column split is a split over `W`'s rows. Each thread runs
  `dot_products_into` against its slice of `W` for every row of `X`, so each thread reads only
  its part of `W`. `dot_products_into`'s 8/4/2-row blocks then start at the thread's first
  `W` row. Each output's grouping is the same, since blocking only chooses which rows run
  together.
- **Output ownership.** Column ranges aren't contiguous slices, so `split_at_mut` doesn't work.
  Options: raw pointer plus disjointness argument in a `// SAFETY:` comment, or each thread
  writes a private `rows x cols_per_thread` buffer that is copied back. Measure the copy. The
  raw-pointer version is expected, and it needs the disjointness stated and tested.
- **Choosing the split:** columns when `b`'s bytes exceed `a`'s (or a cutoff from the stage 0
  sweep), rows otherwise.

Tests: extend 0a with shapes that take the column split at uneven column counts (`n` not a
multiple of `16 * threads`, and `n < 16 * threads`) and assert `==` against the unthreaded
result. Mutation: shift one thread's column start by 4; the test must fail.

## Measurement and acceptance for every stage

- Per op: the focused benchmark at every row of the `../optimizations.md` op table plus the
  stage 0 ladder, old and new builds alternated (commit the crate change first, switch with
  `git checkout main -- src/linalg.rs`, not a stash). Two passes. Nothing below the threshold
  may get slower by more than noise.
- End to end: `demo_conv_rust_vs_vectorized_digit_recognition` (the MNIST mini-batch ratios are
  the target), plus one dense MNIST epoch at batch 32 and 512. Run them in the background and
  give an ETA; the conv demo takes about 3 minutes.
- Bit identity: all 0a tests pass on the new build and the old build; the full suite
  (`./cli build-rust && ./cli test`) passes with no pin changes. **A pin that moves means the
  change isn't bit-identical and is a bug**, since every stage here is claimed bit-identical.
- Update `../optimizations.md`: the op table, the end-to-end table, candidate 1's entry (or its
  removal), and a row in "Completed" or "Closed with no measured gain" per stage. Update
  candidate 2 (dense `forward_batch` at large batches), which was waiting on this.

## Out of scope

- Threading anything other than the three matmul kernels (pooling, elementwise ops).
- Splitting over `k` (needs a cross-thread reduction, which changes summation order).
- Releasing the GIL around crate calls (`py.allow_threads`). The trainers are single-threaded
  Python, so there is nothing to overlap with.
- Tuning for machines other than this one. Defaults stay constants documented as measured on
  this laptop (Ryzen 7 3700U, 4 cores / 8 threads).
