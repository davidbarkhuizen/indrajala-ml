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

**Step 2 done (2026-09-24).** Thread-count sweep over 40 cases, numpy and Rust in separate
processes, two passes, with the focused benchmark. Rust ran at `(t, 1)` for t = 1, 2, 4, 8, then
at the default `(0, 0)`. The full per-case table is in the stage 0 PR. Best of 2/4/8 threads
against 1, from the flop ladder:

| kernel (ladder shape) | 1M | 2M | 4M | 8M | 16M | 32M | 64M |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `matmul_2d`, `(m, 64) @ (64, 512)` | 1.27 | 1.02 | 0.87 | 0.80 | 0.50 | 0.51 | 0.61 |
| `matmul_nt`, 30 x 784 `forward_batch` | 1.50 | 1.18 | 0.83 | 0.65 | 0.56 | 0.55 | 0.57 |
| `matmul_narrow`, conv 28x28 `forward_batch` | 1.25 | 1.11 | 1.15 | 1.07 | 1.01 | 0.90 | 0.77 |

- **2 threads never beat 1** up to 64M, except `matmul_2d` at 64M (0.68). 4 threads beat 1 at
  `matmul_2d` 4M and above, marginally, and at `matmul_nt` only from 32M. 8 threads start to pay
  at about 8M for `matmul_2d` and `matmul_nt`. Subtracting 8 threads' spawn cost (180-200 µs) moves
  that to about 4-8M.
- **Conv:** threading barely moves conv forward and downstream at any N (0.9-1.2 at N = 32 and
  512), because the matmul is a small part of those ops. Conv accumulate, with 8 output rows and
  a long `k`, halves on 8 threads at N = 512 (58-60 → 30 ms) and doesn't gain at N = 32.
- **Large products scale:** `(32, 512) @ (512, 5408)` goes from 31-32 ms on 1 thread to 10-11 on
  8; 32 x 5408 `accumulate_gradient_batch` at batch 512 from 30 to 6.5-10 ms.
- **Clock warm-up.** The default setting is the same as `(8, 1)` on this machine, but in the
  sweep it often measured faster, because it always ran right after t=8. With the order swapped,
  `(32, 512) @ (512, 5408)` takes 9.9-10.1 ms for whichever 8-thread setting runs first after
  t=1, and 5.3-5.8 ms for the one after it. Cores take hundreds of ms of load to clock up. The
  32 x 5408, batch 32 call shows no such effect (600-626 µs either way). So the sweep's t=8
  column for large products is partly cold, and its default column partly warm.

**Step 3 done (2026-09-24).** A per-shape call counter in the local experiment build, one epoch
of each conv demo configuration (UCI digits and the 2000-row MNIST subset, mini-batch 32 and
single-example) and one 60000-row epoch of dense MNIST 784 -> 30 -> 10 at batch 32. **Only one
configuration ever crosses the threshold:** MNIST, a single `ConvSpec(3, 8)`, mini-batch 32. 186
of its 4504 threaded-kernel calls per epoch cross it, all in the 32 x 5408 dense tail: 124
`matmul_2d` (`(32, 32) @ (32, 5408)`, downstream and accumulate) and 62 `matmul_nt` (forward),
each 5.5M flops. conv-pool-conv and conv-conv-stride2 have dense tails 968 and 1152 wide, which
stay below the threshold. Nothing crosses in any UCI run, any single-example run, or dense MNIST.

**What threading costs end to end.** That configuration's epoch, default threading against
threading off (`(1, 2**60)`), alternated, 5 each, same initial weights: **0.830-0.862 s against
0.746-0.788 s, medians 0.846 and 0.752. Threading makes it 11% slower**, about 500 µs per
threaded call. In isolation, repeated back to back, the same calls come out about even. In
training the other cores idle between these calls, so each call pays the full cold-clock cost.

### Decision (end of stage 0)

- **Stage A first, and it is the only stage the demos justify.** Raising the threshold so that
  5.5M-flop products run on one thread removes the only measured production cost (11% of the
  MNIST conv mini-batch epoch). 8 threads pay from about 8M in the isolated ladder, but the
  end-to-end result says the isolated ladder flatters threading, so the new threshold needs an
  end-to-end check at batch 512 too, not just the ladder. Drop 2 and 4 threads: 2 never paid, and
  4 only marginally.
- **Stage B (pool): deferred, not justified by any demo.** A pool removes spawn cost (180-200 µs
  at 8 threads), and threads that persist might keep their cores warm. But in training the
  workers would still idle between calls, so the cold clock may remain unless the workers spin.
  That is untested. It only pays at batch 512 and up, which no demo uses.
- **Stage C (split by columns): skip.** Nothing points at `b` traffic. The register-only probe
  in step 1 showed the same worker slowdown with no memory traffic at all.
- **The remaining batch-32 gaps are kernel gaps, not threading.** Unthreaded at 32 x 5408, batch
  32: `accumulate_gradient_batch` 1214-1374 µs against numpy's 411-453, and `downstream_batch`
  587-1453 against 232-246. That is outside this workplan.

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

**Done (2026-09-24, crate PR #16).** The threshold goes from 4M to 8M flops. Above it the thread
count is still `min(available_parallelism, 8, rows)`, all or nothing. The policy is now its own
function, `matmul_thread_count`, and `matmul_threads_for(m, k, n)` exposes it to the tests.
Bit-identical: the full suite passes with no pin changes.

Choosing the threshold, before any crate change. `set_matmul_threading(8, T)` on the old build
at T = 6M, 13M and 26M, and off. Medians of 5, same weights, settings rotated every run:

| epoch | 4M | 6M | 13M | 26M | off |
| --- | --- | --- | --- | --- | --- |
| MNIST conv, mini-batch 32 | 0.820 | 0.777 | | | |
| MNIST conv, mini-batch 512 | | 0.812 | 0.823 | 0.916 | 1.056 |
| dense MNIST, batch 512 | | 5.736 | 5.718 | | 5.937 |

At batch 512, threading pays for the 24.9M conv ops (13M → 26M, +11%) and the 88.6M dense
tail (26M → off, +15%). Dense MNIST's 12M calls come out even (6M against 13M). 6M and 13M
thread the same conv 512 calls, so their difference is noise. Every threshold from 5.5M up to
12M gives the same demo calls. 8M is where the isolated ladder shows 8 threads clearly paying.

Acceptance, old build (main) against new, one process per epoch, builds alternated with the
order swapped every run, medians of 5 (seconds):

| epoch | old | new |
| --- | --- | --- |
| MNIST conv, mini-batch 32 | 0.879 (0.875-0.886) | **0.770** (0.765-0.789), -12% |
| MNIST conv, mini-batch 512 | 0.852 (0.838-0.863) | 0.867 (0.861-0.876) |
| the same, rerun, 10 each | 0.882 (0.863-0.955) | 0.885 (0.855-0.930) |
| dense MNIST, batch 512 | 5.980 (5.663-6.105) | 5.845 (5.482-6.034) |
| dense MNIST, batch 32 | 4.604 (4.433-4.921) | 4.352 (4.341-4.447) |

The new policy matches stage 0's threading-off time at batch 32 (0.752 then, 0.770 here).
Conv 512 and both dense runs make the same threading decisions on both builds. So their
differences are noise between builds: up to 5%, and 2% for conv 512 before its rerun came out
equal.

**The rows-per-thread floor, re-checked.** A local build with a 32-rows-per-thread floor on top
of the new policy (branch `exp/threading-a-floor`, not pushed): conv 512 took 1.048 s
(1.026-1.117) against 0.867, 21% slower. Dense 512 was 5.775 (5.619-5.857), within noise. The
old comment's conclusion holds, and the comment now cites these numbers.

**Per op, isolated (the focused benchmark, old against new, two processes each):** above 8M
and below 4M every shape is within noise (the same calls). Between 4M and 8M, the products that
are now unthreaded are slower in isolation: 32 x 5408 `forward_batch` at batch 32 739-913 µs
against 650-743 (1.23x), and the ladder's `(244, 64) @ (64, 512)` (7.995M) 1531-1573 against
1001-1225 (1.40x). The same product's `downstream_batch` and `accumulate_gradient_batch` got
faster (0.84-0.85x). This is the cold-clock effect from stage 0: back-to-back isolated calls
keep the cores warm, training doesn't. The end-to-end number is the acceptance number, so this
is accepted and recorded.

Tests: `test_policy_*` in the crate's `tests/test_linalg.py`. The 5.5M tail gets 1 thread.
Dense 512, conv accumulate 512 and the 88.6M tail all get the same count, more than 1. The
override and the row cap work. Thresholds of 5M and 30M each fail one of them.

The original brief follows.

The goal is that the 32 x 5408, batch-32 calls (5.5M
flops) run on one thread, with no 2- or 4-thread counts. The acceptance number is end to end:
one epoch of MNIST, one `ConvSpec(3, 8)`, dense 32, mini-batch 32, lr 0.5, the demo's
2000-row subset. Build the network from a snapshot of `randomized(...)` taken after
`np.random.seed(0)`, and call `random.seed(0)` before each epoch. Time 5 runs of each build,
alternating builds and swapping the order every run. Stage 0 measured 0.846 s with the
current policy against 0.752 s with `set_matmul_threading(1, 2**60)`; the new policy should
match the second. Repeat at batch 512, where threading does pay (stage 0 step 2).

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
  removal), and a row in "Completed" or "Closed with no measured gain" per stage. Candidate 2
  (dense `forward_batch` at large batches), which was waiting on stage A, is done (#17): its
  entry records where threading still pays on the new kernel.

## Out of scope

- Threading anything other than the three matmul kernels (pooling, elementwise ops).
- Splitting over `k` (needs a cross-thread reduction, which changes summation order).
- Releasing the GIL around crate calls (`py.allow_threads`). The trainers are single-threaded
  Python, so there is nothing to overlap with.
- Tuning for machines other than this one. Defaults stay constants documented as measured on
  this laptop (Ryzen 7 3700U, 4 cores / 8 threads).
