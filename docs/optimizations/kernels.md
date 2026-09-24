# Optimizations: the kernels

Part of the optimization record; the index is [../optimizations.md](../optimizations.md).

## Threading

The only threading in the crate is `for_each_row_range` in `rust/src/linalg.rs`, used by the
three matmul kernels: `matmul_2d` (dense `downstream_batch`, `accumulate_gradient_batch`),
`matmul_nt` (every dense `forward_batch`) and `matmul_narrow` (conv forward, downstream,
accumulate). `matmul_thread_count` is the policy: below 8M flops (counted as `m * k * n`
multiply-adds, as everywhere in this document) one thread, otherwise
`min(available_parallelism, 8, rows)`, all or nothing. Each thread gets a contiguous block of
output rows under `std::thread::scope`, spawned on every call. `set_matmul_threading(max_threads,
threshold_flops)` overrides both (`(0, 0)` restores the default) and `matmul_threads_for(m, k,
n)` reports the policy's choice; both are for tests and benchmarks. The crate's
`tests/test_linalg.py` checks every kernel at thread counts 1, 2, 3, 5 and 8 for `==` against
the unthreaded result, and `test_policy_*` pins the policy at the production shapes.

What the measurements found (crate #15, #16; a local probe build for the internals):

- **Spawning costs** about 60-70 µs per call at 2 threads, 110-140 at 4 and 180-200 at 8
  (an empty `compute`).
- **A spawned worker runs 1.6-3.1x slower per row than the caller**, at every thread count, even
  on register-only FMA work with no memory traffic. Per-core clocks during the runs: one core
  at 3.8 GHz and the idle ones at 1.1-1.5 GHz with 1 thread, every core at 2.2-3.1 GHz with 8.
  The reading, inferred rather than measured directly: `schedutil` doesn't raise the clock for
  a thread spawned on every call, since it has no load history. This, not spawn cost, is the
  main cost of threading a mid-sized product.
- **First touch doesn't matter:** pre-touching the output or leaving it for the workers to write
  first made no consistent difference.
- **Break-even**, best of 2/4/8 threads against 1, isolated and repeated back to back:

  | kernel (ladder shape) | 1M | 2M | 4M | 8M | 16M | 32M | 64M |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | `matmul_2d`, `(m, 64) @ (64, 512)` | 1.27 | 1.02 | 0.87 | 0.80 | 0.50 | 0.51 | 0.61 |
  | `matmul_nt`, 30 x 784 `forward_batch` | 1.50 | 1.18 | 0.83 | 0.65 | 0.56 | 0.55 | 0.57 |
  | `matmul_narrow`, conv 28x28 `forward_batch` | 1.25 | 1.11 | 1.15 | 1.07 | 1.01 | 0.90 | 0.77 |

  2 threads never beat 1, except `matmul_2d` at 64M (0.68); 4 only marginally; 8 pays from about
  8M for `matmul_2d` and `matmul_nt`. Threading barely moves conv forward and downstream (the
  matmul is a small part of those ops). Conv accumulate halves on 8 threads at N = 512 (58-60 →
  30 ms). Large products scale: `(32, 512) @ (512, 5408)` goes from 31-32 ms on 1 thread to 10-11
  on 8.
- **Isolated loops flatter threading.** Back-to-back calls keep every core clocked up; in
  training the cores idle between calls, so each threaded call pays the cold clock. The 32 x
  5408, batch 32 calls came out about even in isolation but made the MNIST conv mini-batch 32
  epoch 11% slower (0.846 against 0.752 s, threading off), about 500 µs per call. **Threading
  decisions are judged end to end.** Since #16, the products between 4M and 8M flops are
  slower in isolation than before (up to 1.4x at `(244, 64) @ (64, 512)`) and faster in
  training.
- **Which demo calls are threaded.** Counted per call site over one epoch of each demo
  configuration: at batch 32 or single-example, only MNIST `ConvSpec(3, 8)` mini-batch 32 ever
  crossed the old 4M threshold (its 32 x 5408 tail, 186 calls of 5.5M flops per epoch);
  conv-pool-conv and conv-conv-stride2 have tails 968 and 1152 wide. Nothing is threaded in any
  demo at batch 32 since #16. The only demo that trains larger batches is
  `demo_batch_size_scaling` (dense, B = 128 to 1024), whose 30 x 784 products are threaded from
  B = 512 (12M flops). In the batch-512 epochs of "How to measure", the conv ops (24.9M) and the
  32 x 5408 tail (88.6M) are threaded and pay (+11% and +15% when forced unthreaded); dense
  MNIST's 12M calls come out even.
- **Choosing 8M.** A product is threaded at or above the threshold, so every threshold above
  5.54M (the tail at batch 32, 5,537,792) and up to 12.04M (dense MNIST at batch 512, 12,042,240)
  threads the same calls in these epochs; 8M is where the ladder shows 8 threads clearly paying.
  A 32-rows-per-thread floor on top of it made the conv mini-batch 512 epoch 21% slower (1.048
  against 0.867 s), so there is none.

## Kernel invariants

Every Rust result is bit-identical between the scalar and AVX2+FMA paths, so a training run
doesn't depend on the machine. Each kernel has one fixed summation order:

- **Matrix @ vector and `matmul_nt` (`X @ W.T`, every forward):** `dot_product`'s grouping.
  Lane `j` of a 4-lane accumulator sums indices `j, j+4, ...` by FMA, then
  `(l0 + l1) + (l2 + l3)`, then the `k % 4` tail sequentially. `dot_products_into` runs 8, then
  4, then 2 rows at once in that grouping; `matmul_nt` runs 4 rows of `X` against 2 of `W` in
  it (#17), and the rows and columns left over through `dot_products_into`. A batched
  forward's rows equal the single-example forward exactly.
- **Matrix @ matrix (`matmul_2d`, `matmul_narrow`) and vector @ matrix:** one FMA chain per
  output, `k` increasing from 0.0. `tiled_row_range` holds 16-column output tiles in registers
  across all of `k`: `matmul_2d` in row blocks of about 16 KB of `a`, `matmul_narrow` one row at
  a time. Vector @ matrix is the one-row product (crate #20; it had its own `axpy_row` loop
  with the same chain).
- **Threading** splits output rows across threads, so each output is computed by one thread and
  the thread count can't change any value (see "Threading").
- These orders differ from numpy's in the last few ULPs, so parity with numpy is checked with
  `rtol`. Crate tests pin each order exactly against a `Fraction`-emulated FMA reference.
