# Optimizations: implemented

Part of the optimization docs; the index is [../optimizations.md](../optimizations.md).

What the code does for speed, and why each choice is right for this workload. Each item names
the mechanism it exploits, the evidence, and where it lives. Crate PR numbers (`crate #N`) are
`indrajala-math-rust`'s (`rust/`); plain `#N` are this repo's. Numbers are µs per call on the
benchmark machine (see [Measurement](measurement.md#the-machine)) unless marked otherwise.

## The constraint every kernel keeps: one fixed summation order

Every Rust result is bit-identical between the scalar and AVX2+FMA paths and at every thread
count, so a training run doesn't depend on the machine. That matters because training is
chaotically sensitive to rounding (a 1-ULP change to one weight moves end-of-run accuracy as
much as a real change; see [Measurement](measurement.md#judging-correctness)), so bit-identity is
the only way an optimization can be proven not to change results. Each kernel has one order:

- **Matrix @ vector and `matmul_nt` (`X @ W.T`, every forward):** `dot_product`'s grouping. Lane
  `j` of a 4-lane accumulator sums indices `j, j+4, ...` by FMA, then `(l0 + l1) + (l2 + l3)`,
  then the `k % 4` tail in order. So a batched forward's rows equal the single-example forward
  exactly, which the batched accuracy pass relies on.
- **Matrix @ matrix (`matmul_2d`, `matmul_narrow`, `matmul_long_k`) and vector @ matrix:** one FMA
  chain per output, `k` increasing from 0.0 (`matmul_long_k` stores and resumes it between slabs
  of `k`; a stored double is exact).
- **Threading** splits output rows (column chunks in `matmul_long_k`), so each output is computed
  by one thread.

These differ from numpy's orders in the last few ULPs, so parity with numpy is checked with
`rtol`; crate tests pin each order exactly against a `Fraction`-emulated FMA reference. Every
optimization below is bit-identical unless it says otherwise.

## Register tiling: keep outputs in registers across all of `k`

A product whose output lives in memory pays a load and a store per FMA; one whose output tile
stays in registers pays neither, and each loaded input is reused across the tile. On Zen 2 (two
FMA pipes, about 5 cycles' latency) a tile also needs about 10 independent chains in flight to
keep the pipes busy. All three matmul kernels are built on that:

- **`tiled_row_range`** (`rust/src/linalg.rs`) holds 16-column output tiles (4 AVX2
  accumulators) in registers across all of `k` (or a slab of it), for `matmul_2d`,
  `matmul_narrow`, `matmul_long_k` and vector @ matrix. `matmul_2d` runs it in row blocks of about 16 KB of `a` (so a block of `a` and `b`'s
  `k x 16` panel share L1); `matmul_narrow` in blocks of 4 rows, for conv's narrow products
  (`cols @ W.T` with only `C*k*k` or `O` columns). Replacing the old load/FMA/store loops (crate
  #11, #12, #14) cut conv forward 13-64%, downstream 4-56%, accumulate 16-43%, and unthreaded
  dense batch downstream/accumulate to 0.1-0.6x their time.
- **Its 16-wide tiles cover 2 rows at once** (crate #26): 8 independent FMA chains instead of 4,
  and each `b` row of the tile loaded once for both rows; a block's leftover row runs the 1-row
  tile. Dense batch 32 `downstream_batch` went 551-677 → 445-476 µs at 32 x 5408 and accumulate
  81-89 → 67-70 µs at 30 x 784; one-thread 30 x 784 accumulate at batch 512 0.63-0.70x. In the
  conv mini-batch 32 epoch profile, dense downstream went 45-60 → 41-44 ms and accumulate 64-83 →
  62-63 ms. A 3-row tile won only at `k` = 128 in the probe. Vector @ matrix passes one-row blocks.
- **`matmul_narrow` passes 4-row blocks** (crate #35), so the conv accumulate (8 rows of
  `N*P`, where `matmul_2d`'s 16 KB rule gives one) reaches the 2-row tiles. On one thread at
  N = 32 the second-conv accumulate went about 40% faster (13x13x8 568-671 → 364-370 µs), 4 rows
  ahead of 2 and 8; the downstream 10-15% faster in places. In the epoch profile (builds
  alternated) the accumulate saved 0.6-2.0% of each second-conv network's epoch and the
  downstream up to 1.3%; nothing where training threads the accumulate (conv-conv mini-batch 32,
  10.6M flops: each thread already has 2 rows).
- **The conv accumulate runs `k` in slabs, threaded over columns** (`matmul_long_k`, crate #36).
  `D @ cols` (`(O, N*P) @ (N*P, C*k*k)`) makes one pass over all of `k` per row pair and column
  tile, re-reading `cols` 4 times over; past the 4 MB L3 each pass came from memory, and row
  threading (above 8M flops) made every thread stream all of `cols`. Now each chain runs 64-row
  slabs of `cols`, stored and resumed, so a slab serves every row and tile from cache, and threads
  take 16-wide column chunks, reading `cols` about once. In training (MNIST subset, builds
  alternated, 4 processes each) the accumulate went, per call: conv-conv's second conv (`cols`
  10.6 MB at N = 32) 4777 → 2656 µs, conv-pool-conv's 774 → 627, conv-conv-stride2's 953 → 755,
  the first conv 1220-1374 → 1049-1165, single-example first conv 33-40 → 29-37. That is 0.135 s
  of a 2.4 s conv-conv mini-batch 32 run (5.5%), 3.4-3.5% of the other second-conv mini-batch
  runs, 1.6% of conv's, 0.2-1.5% single-example. Slabs of 64-1024 rows were within 5%, 4096
  slower; column chunks alone (one slab) gained nothing. Other ops within noise.
  `set_kernel_overrides(narrow_rows_per_block, long_k_block)` keeps both kernel choices
  switchable at runtime for re-measurement (`--kernel-overrides` in `focused_benchmark.py` and
  `op_call_timing.py`); the crate tests that no setting changes a bit.
- **One kernel, compile-time structure for whole products.** `tiled_row_range` takes a `Panel`
  (rows, columns, a `k` range, an output stride) and a mode (`OVERWRITE`, `ADD`, `RESUME`). With
  those offsets at runtime the single-example conv forward and downstream ran 5-7% slower (26x26x8
  forward 150-155 → 160-166 µs); outside `RESUME` the kernel takes the whole panel as known
  structure, which measured level again.
- **Vector @ matrix goes through the same kernel as a one-row product** (crate #20), not its own
  `axpy_row` loop, which made 32 load/FMA/store passes over a 43 KB output row: single-example
  dense `downstream` at 32 x 5408 went 44-48 → 26 µs (numpy 29).
- **`matmul_nt` runs 4 rows of `X` against 2 rows of `W`** (crate #17; 2 x 4, 3 x 3 and 2 x 2
  measured slower or tied), and `dot_products_into` 8, then 4, then 2 rows at once (crate #13),
  all in `dot_product`'s grouping. Each row of `X` used to re-read all of `W` (1.4 MB at 32 x
  5408, past L2); now `W` streams once per 4 rows. Unthreaded dense `forward_batch` went to
  0.6-0.9x its old time; on one thread Rust is now faster than numpy at every measured shape.

## Avoiding copies and passes

A pass over an array the size of the output costs about as much as a small product, so every
temporary removed is a real saving at these shapes:

- **No transpose copies:** single-example `downstream` is `delta @ W`, not `W.T @ delta` (crate
  #7, 308 → 43 µs at 32 x 5408; this one changed bits, within 4 ULPs), and `forward_batch` uses
  `matmul_nt` for `X @ W.T` (crate #8, 330 → 58 µs at batch 1).
- **One-pass dense `accumulate_gradient`**, adding `delta ⊗ x` into the gradient without an
  `outer` temporary (crate #5, 596 → 62 µs at 32 x 5408).
- **Dense `accumulate_gradient_batch` adds `grad_W` in the product's store** (`matmul_add`,
  crate #27): `tiled_row_range::<true>` stores `out + chain` into a copy of `grad_W`, so there is
  no separate add pass or update array. Each chain still starts from 0.0, so it is the separate
  add's single rounding, bit-identical. At 32 x 5408, batch 32, one thread: 940-948 → 793-810 µs;
  30 x 784 unchanged within noise. In the conv mini-batch 32 epoch profile the op went 60-92 →
  55-79 ms, inside the block-to-block drift but in line with 100-150 µs on each of 63 calls.
- **A blocked transpose** (crate #28): `.T` (the `delta_batch.T` copy feeding the dense
  accumulate) copies in 8 x 8 blocks. The row-by-row loop wrote one output line per element at a
  stride of `rows`; at batch 512 that stride is 4 KB, so every write fell in the same L1 set.
  (512, 30): 89-93 → 14-15 µs (numpy's copy 8-11); batch 32 unchanged. The 30 x 784 accumulate
  at batch 512 gains about the transpose (one thread, medians 1680 → 1596 µs); in the B = 512
  step-loop profile the op went 0.170-0.182 → 0.148-0.170 s, about 0.5% of the step loop.
- **A fused single-example SGD step**, `layer_sgd_step` (crate #6): the Rust layers' `sgd_step`
  updates weights in one call, with a Python fallback for momentum, Adam, L2 and conv. Per-layer
  step 463 → 64 µs; dense MNIST epoch -15%.
- **Conv forward doesn't return `Z`** (crate #9), which the backward pass doesn't need: about 14%
  at N = 512.
- **Conv and pool ops take a vector as N = 1** (crate #10), so single-example calls don't reshape:
  per-call Rust overhead on the UCI conv layer 28.5% → 0.3%.

## Conv `forward_batch` one example at a time

`conv_forward_batch` (`rust/src/conv.rs`) runs im2col, the product and bias + ReLU per example:
each example's rows are appended to `cols` (kept whole for the backward pass), its product goes
into one `(P, O)` buffer reused for every example, and its rows of `A` are appended in order
(crate #21).

Why: the whole-batch version zero-filled three batch-sized buffers (1.4-1.6 MB each at N = 32,
past L2) that were then overwritten in full, and scattered `A` with a stride. At N = 1 those
buffers sit in L1/L2 and cost nothing; at N = 32 each fill and each strided pass went to L3, and
the batch op cost 1.5-2.1x its single calls. Per example, the 48 KB slab of `cols` and the 43 KB
product stay hot, and nothing is zero-filled. Just dropping the fills was not enough: the strided
scatter then paid for fetching `A`'s lines itself (see [the first-touch
gotcha](measurement.md#gotchas)); writing `A` in order is what made it pay. N = 32: 1351-1694 →
920-1010 µs; N = 512: 34-40 → 15-16 ms. The per-example product doesn't thread; at N = 512
threading had given nothing.

## Max-pool forward: no division per slot, and a fixed 2x2 body

`max_pool_forward_batch` (`rust/src/conv.rs`) pools one channel plane at a time. The general
kernel walks each window's rows as slices and counts the slot alongside; 2x2 windows at stride 2
(the demos' `PoolSpec(2)`) take the same scan unrolled (crate #24). Both keep the row-major scan
and strict `>`, so the first maximal slot wins with its sign of zero, bit for bit as before.

Why: the old scan computed `slot / k`, `slot % k` and a full input index for every slot, with `k`
known only at run time. Walking row slices removes that (16 → 9 µs a call in the stage 0 probe),
but what is left is the loops with a run-time trip count, not the branches (a select-based
general kernel was no faster), so only the fixed-size body gets near the 0.9 µs floor. At
26x26x8: 16.6-17.2 → 4.1-4.2 µs single, 504-525 → 112-114 at N = 32, 9.2 → 5.7-5.8 ms at N = 512
(its 22 MB input bounds it). In the conv-pool-conv epoch the op went from 95-101 ms to 23-32 ms,
8.5-9.5% of the profiled run.

## Max-pool downstream: one pass without overlap

`max_pool_downstream_batch` (`rust/src/conv.rs`) takes one pass per channel plane when windows
don't overlap (stride ≥ `k`, the demos' `PoolSpec(2)`) (crate #25). Each input then receives at
most one delta, so the pass stores `0.0 + d` at the winning input: the bits the add into the
zeroed `dX` gave, `-0.0` deltas coming out `+0.0`. Slot offsets come from a `k * k` table, and
the argmax check is folded in as an exact integer test. Overlapping windows keep the slot-outer
loop, which fixes numpy's summation order.

Why: the old op swept every window once per slot (`k * k` passes) after a separate validation
pass. At 26x26x8: 22.8-24.4 → 5.7-6.3 µs single, 1211-1276 → 166-177 or 373-394 µs at N = 32 (two
modes between processes; see [Measurement](measurement.md#gotchas)), 26-28 → 9.7-12.9 ms at
N = 512. In the conv-pool-conv epoch the op went from 41-45 to 11.5-12.2 ms single-example and
48-53 to 17.2-18.4 ms at mini-batch 32, about 4% of each run. A plain store measured the same as
the add, and the offset table the same as a division or faster.

## Threading policy

The threading is `for_each_row_range` in `rust/src/linalg.rs`, used by `matmul_2d`, `matmul_nt`
and `matmul_narrow`, and `matmul_long_k`'s split over column chunks (the same thread count,
capped by the chunk count). `matmul_thread_count`: below 8M flops (`m * k * n`) one thread, otherwise
`min(available_parallelism, 8, rows)`, all or nothing, contiguous output-row blocks under
`std::thread::scope`, spawned per call (crate #15, #16). `set_matmul_threading(max_threads,
threshold)` overrides it for tests and benchmarks; the crate tests every kernel for `==` at 1, 2,
3, 5 and 8 threads and pin the policy at the production shapes.

Why the threshold is high:

- **A spawned worker runs 1.6-3.1x slower per row than the caller**, even on register-only work.
  Idle cores sit at 1.1-1.5 GHz and a busy one boosts to 3.8 GHz; `schedutil` doesn't raise the
  clock for a thread spawned on every call (inferred, not measured directly). This, more than the
  60-200 µs spawn cost (2 to 8 threads), is what threading a mid-sized product costs.
- **In training the cores idle between calls**, so each threaded call pays the cold clock that
  back-to-back benchmark loops hide. At the old 4M threshold the 32 x 5408, batch 32 products
  (5.5M flops) came out even in isolation but made the MNIST conv mini-batch 32 epoch 12% slower.
- **8 threads pay clearly from about 8M flops** for `matmul_2d` and `matmul_nt` (0.65-0.80 of the
  one-thread time at 8M, 0.50-0.56 at 16M; 2 threads almost never beat 1). Any threshold in
  5.54M-12.04M threads the same calls in the demos; 8M is where the ladder shows the gain. A
  32-rows-per-thread floor on top made the conv mini-batch 512 epoch 21% slower, so there is none.

So at batch 32 only conv-conv's second-conv accumulate (10.6M flops) is threaded, over column
chunks; the batch-512 conv ops and the dense products at B ≥ 512 are too, and pay (+11-15% when
forced unthreaded).

## The dataset as one backend array

Array networks train from a `PreparedDataset` (`indrajala_ml/prepared_dataset.py`): the training
set as one backend matrix plus labels, built once per run (or loaded straight into one by
`prepared_mnist`), read through `learn_row`, `learn_batch_rows` and `classify_row` (crate #19's
`Array.from_rows`, `row`, `take_rows`; #372-#374). The trainers shuffle row
indices, which gives the same permutation for a seed, so training is unchanged bit for bit.

Why: converting a Python tuple to an array costs 15 µs a 784-pixel row in Rust and 30 µs in
numpy, and the trainers did it for every batch and every accuracy-pass row. For Rust at dense
MNIST B = 32 that was about 65% of the epoch, a fixed cost neither backend's arithmetic can
remove. Rust epoch 3.72 → 1.69 s (1.22 from the loader), numpy 7.92 → 5.37; conv Rust B = 32
-12%. Preparation is 0.43 s in Rust (`from_rows` reads tuples by index) and 1.4 s in numpy.

## A batched training-set accuracy pass

`_training_accuracy` (`train.py`) calls `classify_rows(prepared)`, which runs `forward_batch`
over chunks of `CLASSIFY_CHUNK_ROWS` = 32 rows and takes each row's argmax (#378-#379).

Why: the trainers run n + 1 accuracy passes for n epochs, and a per-row pass is one small
forward per layer per row, dominated by per-call overhead; on dense full MNIST the two passes of
a one-epoch run were over half the epoch. Batching cut the dense B = 32 epoch 1.61 → 1.31 s in
Rust and 4.85 → 2.44 in numpy. Chunk 32 takes 97% of numpy's gain at 512, and Rust at 512 is
slower (it crosses the threading threshold). Rust dense rows are exact (above); numpy's batched
outputs differ from its per-row ones by 1 ULP in some outputs, so numpy is tested for equal
predictions, not bits.

**Rust conv keeps the per-row pass** (`ConvRustArrayMultiClassBackpropClassifierNetwork.classify_rows`):
batched it ties for one conv layer and is 14-20% slower with pooling or a second conv, because
the batched forward still writes a whole-batch `cols` that inference never reads (see
[Candidates](candidates.md#leads)).

## Loading MNIST shares one float per pixel value

`load_mnist_dataset` reuses 256 float objects instead of boxing 47 million (#365). Training-set
load 1.9 GB → 0.45 GB and 5.1 → 2.4 s, identical values; without it 4 sweep workers don't fit in
5 GB. It also makes Rust row conversion about 12% cheaper (the floats are already shared).
