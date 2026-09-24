# Optimizations

Performance work on the Rust backend (`rust/`, the `indrajala-math-rust` crate) against numpy,
the benchmark it is compared with. It started from the Rust CNN timing (#317-#320 and the Rust
CNN stages), which found where Rust was slower than numpy or slower than it needed to be. Each
item is measured before and after, and must keep every parity test passing.

This document is the only record: there are no separate workplans. Each open item, with its plan
where it has one, is a candidate below. Three workplans have been folded in: threading
(optimization 6, finished; its findings are in "Threading", and what it left open is candidate
9), dense batch ops at short `k` (optimization 7; stage 0 done, stages A and B planned in
candidate 1) and the dataset as one backend array (optimization 5, not started; candidate 8).

## Where things stand (2026-09-24)

Rust / numpy wall-clock ratio end to end (below 1 means Rust is faster), from
`demo_conv_rust_vs_vectorized_digit_recognition` after indrajala-math-rust#16. After #17 every
cell was within 0.05 of these (MNIST mini-batch 0.47, 0.43, 0.48), which is run-to-run noise:

| architecture | UCI digits, single | UCI digits, mini-batch | MNIST subset, single | MNIST subset, mini-batch |
| --- | --- | --- | --- | --- |
| conv | 0.12 | 0.17 | 0.26 | 0.49 |
| conv-pool-conv | 0.09 | 0.09 | 0.30 | 0.41 |
| conv-conv-stride2 | 0.11 | 0.12 | 0.35 | 0.46 |

At the start, MNIST conv single-example was 1.21 (Rust slower) and conv-pool-conv mini-batch was
0.66. Dense MNIST 784 -> 30 -> 10, one 60000-example epoch: Rust/numpy about 0.33
single-example and 0.49-0.53 mini-batch 32.

Per op, the single-example ops are all at or better than numpy. The gaps left are in batch
ops. Rust / numpy µs per call, focused benchmark (see "How to measure"), numpy and Rust in
separate processes, Rust with the default threading (threshold 8M flops since #16). The last
two columns put both backends on one thread (`--openblas-threads 1 --rust-threads 1`,
optimization 7 stage 0, 2026-09-24); "-" is not measured:

| shape | op | batch | numpy | Rust | Rust/numpy | numpy, 1 thread | Rust, 1 thread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 x 5408 | `downstream_batch` | 32 | 232-246 | 581-654 | 2.4-2.8x | 573-578 | 610-625 |
| 32 x 5408 | `accumulate_gradient_batch` | 32 | 411-453 | 976-1180 | 2.2-2.9x | 807-871 | 1040-1083 |
| 32 x 5408 | `downstream_batch` | 512 | 11753-11777 | 10648-11847 | 0.9-1.0x | 12236-12361 | 14746-15199 |
| 32 x 5408 | `accumulate_gradient_batch` | 512 | 12926-13493 | 6811-11720 | 0.5-0.9x | 9425-10298 | 31253-31958 |
| 32 x 5408 | `forward_batch` | 512 | 3695-4110 | 4747-4898 | 1.2-1.3x | - | - |
| 32 x 5408 | `forward_batch` | 32 | 294-415 | 470-659 | 1.1-2.2x | - | - |
| 30 x 784 | `downstream_batch` | 512 | 416-945 | 1193-1401 | 1.3-3.4x | 1153-1466 | 1383-1560 |
| 30 x 784 | `accumulate_gradient_batch` | 512 | 714-776 | 997-1336 | 1.3-1.9x | 1269-1436 | 2335-2715 |
| 30 x 784 | `forward_batch` | 512 | 750-1027 | 1038-1275 | 1.0-1.7x | - | - |
| 30 x 784 | `downstream_batch` | 32 | 46-50 | 73-85 | 1.5-1.8x | 71-84 | 75-86 |
| 30 x 784 | `accumulate_gradient_batch` | 32 | 70-73 | 82-97 | 1.1-1.4x | 90-102 | 84-93 |

The first two rows (5.5M flops) run on one thread since #16; their Rust numbers are from then.
The two 32 x 5408 batch-512 rows are from optimization 7 stage 0's default-threading run (88M
flops, threaded).
The `forward_batch` rows are after #17, except 32 x 5408 at batch 512, whose threaded time moved
between 4.6 and 9.8 ms from run to run in the same session, so it keeps its earlier numbers. The
batch-512 Rust numbers are partly warm-clock numbers (see "Threading"). An earlier version of
this table was measured with numpy and Rust interleaved in one process and read 12-13x, 7x,
4-8x and 4x on the first, second, fourth and fifth rows; that was numpy's OpenBLAS threads
taking cores from Rust (see "Other findings"). numpy's numbers are with OpenBLAS's default
threading, which it uses even at 5.5M flops. On one thread each, the batch-32 rows are level
or close, and the gap that remains is at batch 512 with a long `k`: accumulate (`k` = 512) is
3.1x numpy at 32 x 5408 and 1.8x at 30 x 784 (see candidate 3).

## Open candidates, in priority order

1. **Dense `downstream_batch` and `accumulate_gradient_batch` at short `k`** (`matmul_2d`,
   optimization 7). **Status: stage 0 done (#354); next stage A, then B; stage C skipped.** At
   32 x 5408, batch 32 these are the largest per-op ratios in the table (2.4-2.8x and 2.2-2.9x
   numpy), but stage 0 found that most of that is numpy's OpenBLAS threading: on one thread
   each they are level. What is left for Rust is its own kernel, and stages A and B together
   are worth about 4% of the worst end-to-end cell.

   Both ops are `(32, 32) @ (32, 5408)` products, 5.5M flops with a 1.4 MB output and `k` (the
   batch or the layer width) only 32. They run on one thread since #16 and are in the MNIST
   conv mini-batch 32 dense tail, the worst end-to-end cell: 62 calls of each per epoch. The
   crate paths are in `rust/src/fused.rs`: `layer_downstream_batch` calls `matmul` with nothing
   else; `layer_accumulate_gradient_batch` makes a `transpose()` copy of `delta_batch`, the
   product `delta_batch.T @ X`, then `grad_w.combine_with_array(update, g + u)` and `sum_axis0`
   for `grad_b`. `matmul_2d` zeroes `out`, picks `rows_per_block = 16 KB / (k * 8)` (all 32 rows
   in one block at `k` = 32) and calls `tiled_row_range` (`rust/src/linalg.rs`), whose AVX2 path
   runs 4 accumulators over `k` per 16-column tile and row. `matmul_narrow` (every conv matmul)
   shares that function with one row per block. The same kernel runs the 30 x 784 rows of the
   op table, so dense MNIST mini-batch gains too.

   **Stage 0 decision (2026-09-24).** Measured with `scripts/focused_benchmark.py` (two
   passes, one process per case) and a local probe build (crate branch
   `probe/opt7-stage0`, local only) that times kernel variants inside Rust. Ranges are the two
   passes' medians, µs per call:

   - **Most of the gap is numpy's OpenBLAS threading, not the kernel.** numpy's 228-389 µs for
     `downstream_batch` at 32 x 5408, batch 32 uses several threads. With both backends on one
     thread, Rust is level with numpy or faster: `downstream_batch` 610-625 against numpy's
     573-578, `accumulate_gradient_batch` 1040-1083 against 807-871. On the bare products alone,
     Rust took 549-637 against numpy's 850-928. Oddly, numpy's own layer op was faster on one
     thread than its bare `delta_batch @ W` of the same shapes and value ranges (573-578 against
     850-897), in the same runs, with no explanation found. So the single-threaded numpy target
     is 573-897 µs depending on which call is timed. At 30 x 784, batch 32, Rust single-threaded
     is level with numpy single-threaded as well. numpy's threading doesn't pay in training
     either: its own MNIST conv mini-batch 32 epoch took 1.22-1.32 s at `OPENBLAS_NUM_THREADS=1`
     against 1.39-1.56 s by default. So the 2.2-2.9x in the op table measures hot-loop
     threading. It doesn't show a slower kernel.
   - **Call boundary: 3-20 µs** (fused op against a bare `@`), 0-4%. Nothing to do.
   - **Allocation and page faults: not a cost.** The real call path has 0.0 minor
     faults per call at every shape (0.1 at 32 x 5408, batch 512); presumably glibc reuses the
     freed pages (inferred from the counts, not observed). A whole MNIST conv mini-batch 32
     epoch has 12.5-17k faults in total. If each dense-tail output faulted, the two ops alone
     would account for about 42k. Zeroing a 1.4 MB output costs 25-27 µs (4-5%). Stage C is
     skipped.
   - **Multi-row register tiles: 7-35% off the kernel.** The hypothesis was FMA latency: per row
     and tile the kernel has 4 independent chains of `k` dependent FMAs, where Zen 2's 2 FMA
     pipes at about 5 cycles' latency need about 10 in flight. The fix is the one #17 made for
     `matmul_nt`. 2-row tiles, bit-identical (2133 shapes checked with `==`):
     - 32 x 32 x 5408: 524-588 → 450-504 (-14%)
     - 32 x 30 x 784 and 30 x 32 x 784: -10 to -12%
     - 512 x 32 x 5408: -11%
     - 512 x 30 x 784: -18%
     - 30 x 512 x 784: 2184-2225 → 1474-1548 (-31%)
     - 32 x 512 x 5408: -7 to -10%

     3-row tiles were slower than 2-row at every shape except `k` = 128. Stage A runs first.
   - **But the kernel is not simply FMA-latency-bound.** At equal flops, `k` = 128 (`(32, 128) @
     (128, 1352)`) took 871-880 µs against 524-588 at `k` = 32, where latency-bound would be
     even and output-traffic-bound would be faster. numpy went the other way: on one thread,
     578-586 µs at `k` = 128 against 898-911 at `k` = 32 (Rust on one thread: 1003-1031 against
     646-650). At `k` = 128 in the probe, 3-row tiles gave -35% and 2-row -25%. The likely
     reading, without hardware counters to confirm it: the cost is the loads of `b`'s `k x 16` panel (4 KB at `k` = 32, 16
     KB at 128, 64 KB at 512, past the 32 KB L1). Each extra row in the tile reuses those loads.
   - **Accumulate's extra pass: 140-160 µs, plus about 240 µs unattributed.**
     At 32 x 5408, batch 32 the fused op took 952-1036 µs. Its parts summed to 715-780: the bare
     product 572-614, `add` 142-163, and the transpose and `sum_axis0` 1 µs each. The likely
     cause of the rest is that the fused op keeps four 1.4 MB arrays live (`X`, the update,
     `grad_W` and the sum) against a 4 MB L3. That is unmeasured. Stage B removes one of them
     and the pass, so it runs second.
   - **Revised stake:** stages A and B together might save about 450 µs per pair of calls,
     about 28 ms (4%) of the 0.72 s epoch, not the 8% estimated from the per-op gap.
   - **Found on the way:** a larger gap at long `k` (candidate 3), and Rust's `transpose` of
     `delta_batch` at batch 512 taking 87-107 µs against numpy's 8-12 µs copy, about 9% of the
     30 x 784 accumulate.

   **Stage A: multi-row register tiles in `tiled_row_range`.** Hold a tile of R rows x 16
   columns (4R accumulators) across all of `k`, R = 2 (and R chosen by `k` only if the op table
   shows it matters; 3 won only at `k` = 128). The probe's `probe_tiled_avx2_fma` on crate
   branch `probe/opt7-stage0` (local only, not pushed) is a starting point. Rows left over
   (`rows % R`) run the current 1-row tile; the 4-wide and scalar column tails keep their loops.
   `matmul_narrow` calls the same function with one row per block: measure the conv ops too
   (forward, downstream, accumulate at 28x28, N = 32 and 512), and if they don't gain, keep them
   on the 1-row tile by passing the tile height, the way `rows_per_block` is passed now. Each
   output is still one FMA chain, `k` increasing from 0.0, so it is bit-identical. Extend the
   crate's `test_matrix_at_matrix_is_the_fma_chain_exactly` (`tests/test_linalg.py`) so `m`
   reaches every row remainder (1 to 2R + 1) at the existing `TILE_WIDTHS`; `BIG_SHAPES` and the
   thread-count tests cover the large shapes. Mutations that must fail: give the tile's second
   row the first row's `a` value at one `k`; drop the last remainder row.

   **Stage B: fuse the gradient add into the kernel (accumulate only).** A `matmul_2d_add(a, b,
   c)` variant, or a `tiled_row_range` parameter, whose store writes `c + acc` in place of `acc`.
   `layer_accumulate_gradient_batch` then allocates one output, not two, and skips
   `combine_with_array`; `sum_axis0` for `grad_b` stays. Today's `grad_w + (delta.T @ X)` is `g +
   u` with `u` the finished chain, one rounding, and the fused store is the same operation on
   the same values, so it is bit-identical. There is only an `rtol` test of this today: add a
   crate test that `grad_w` equals `grad_w + (delta.T @ X)` from separate crate calls, `==` on
   `tolist()`, at `BIG_SHAPES` and the `TILE_WIDTHS` shapes, with `grad_w` containing `-0.0` and
   zeros. Mutation that must fail: start the chain from `g` (a different rounding).

   **Stage C: allocation (skipped).** Stage 0 measured 0 faults per call and zeroing at 4-5%.
   The options, should that change: don't zero outputs the kernel fully writes (`unsafe`
   uninitialized allocation with a `// SAFETY:` argument that every element is written; removes
   a pass, not faults), or raise glibc's mmap threshold at module init (process-wide, so it
   affects numpy in the same process too). Reusing output buffers through the Python API is
   ruled out: it breaks the immutable `RustArray` contract every caller relies on.

   **Acceptance for A and B**, beyond the rules below: per op, every op table row (the
   `matmul_nt` rows must not move), plus the conv ops if `tiled_row_range` changed; end to end,
   the old-against-new epochs in "How to measure", then the conv demo for the ratio table. Every
   stage is claimed bit-identical, so the full suite must pass with no pin changes. Out of scope:
   threading these products (see "Threading") and any change to summation order.

2. **Dense `forward_batch` at large batches** (`matmul_nt`): register tiles done (#17), the
   rest open. The recorded 5.6x at batch 64 was the interleaving: in separate processes batch
   64 was 1.8x numpy (1161-1164 against 625-653 µs), and unthreaded Rust was linear in the batch
   at about 2x numpy per row. So the cost was the kernel: every row of `X` re-read all of `W`
   (1.4 MB at 32 x 5408, past L2). #17 runs 4 rows of `X` against 2 rows of `W` at a time,
   bit-identical. Two focused passes, old and new in separate processes, the second with the
   build order reversed (µs, unthreaded):

   | shape | batch | old | new |
   | --- | --- | --- | --- |
   | 32 x 5408 | 32 | 635-916 | 490-903, then 545-555 |
   | 32 x 5408 | 128 | 2621-3708 | 2339-2769 |
   | 32 x 5408 | 512 | 10452-15660 | 8233-13628 (9331-10204 in the second pass) |
   | 30 x 784 | 32 | 95-110 | 69-98 (76-77) |
   | 30 x 784 | 512 | 1537-2109 | 1208-1478 |

   **Threading on the new kernel** (default against unthreaded, new build, both passes):
   threading still pays at 32 x 5408 from batch 128 (1455-2016 against 2339-2769 µs) and at
   batch 512 (6226-7672 against 8233-13628), and a little at 30 x 784, batch 512 (1038-1275
   against 1208-1478). At batch 64 (11M flops, threaded) the two overlap (887-1121 against
   998-1361). With default threading, 30 x 784 at batch 512 went from 1472-1655 to 1038-1275.

   **End to end**, one epoch, old against new build, one process per epoch, builds
   alternated with the order swapped every run, medians of 6 (seconds):

   | epoch | old | new |
   | --- | --- | --- |
   | MNIST conv, mini-batch 32 | 0.751 (0.710-0.856) | 0.743 (0.706-0.841) |
   | MNIST conv, mini-batch 512 | 0.870 (0.787-0.878) | 0.843 (0.774-0.870) |
   | dense MNIST, batch 32 | 4.170 (3.967-4.365) | 4.094 (4.007-4.337) |
   | dense MNIST, batch 512 | 5.814 (5.774-6.054) | 6.000 (5.658-6.114) |

   All within noise, as the arithmetic predicts: MNIST conv mini-batch 32's 62 forward calls
   save about 12 ms of 0.75 s. What is left: 32 x 5408 is still 1.1-2.2x numpy at batch 32
   (470-659 against 294-415 µs), because `W` still streams in from L3 once per 4 rows of `X`.
   Candidate: block over `k` so that a panel of `W` stays in L2 across all rows of `X`,
   storing and reloading each pair's 4-lane accumulator between panels. That keeps each
   output's grouping, so it is bit-identical. Low value while no demo runs large batches.

3. **Dense `accumulate_gradient_batch` at long `k`** (`matmul_2d`, found in optimization 7's
   stage 0). Single-threaded, at batch 512 (`k` = 512) it is 3.1x numpy's single-threaded time
   at 32 x 5408 (31.3-32.0 against 9.4-10.3 ms) and 1.8x at 30 x 784 (2335-2715 against
   1269-1436 µs). 2-row tiles recover only 7-31% there. The likely cause is `b`'s `k x 16` panel
   (64 KB at `k` = 512) no longer fitting the 32 KB L1. Default threading hides it (6.8-11.7 ms
   against numpy's 12.9-13.5), and it matters only at batch 512, which no demo's default runs.
   Candidate: block over `k`, or pack `b`'s panel, storing and reloading the tile's accumulators
   between `k` blocks so each output keeps its chain. The same idea is the remaining step for
   candidates 2 and 5, so one `k`-blocked `tiled_row_range` may serve all three.

4. **Conv `forward_batch` at N = 32 costs more than 32 single-example calls.** Measured at 28x28
   (2280 vs 1715 µs, before the `matmul_narrow` kernel). The likely cause, unmeasured: the 1.5 MB
   `cols` falls out of cache between im2col and the matmul. Candidate: im2col and multiply one
   block of output positions at a time, keeping `cols` for the backward pass. Re-measure first:
   the kernel has changed since.

5. **Conv accumulate with a large `cols`.** `D @ cols` goes through `matmul_narrow`, which gave
   no gain at 13x13x8, N = 32 (+2%, +4%, -1% at O = 4, 8, 32; `cols` 0.3-2.2 MB), where the old
   `matmul_2d` had `k`-blocking. That blocking is gone since #14. Moving the op to `matmul_2d`'s
   16 KB row blocks made it slower at 28x28, N = 512 (35-39 vs 30-32 ms), so row blocking isn't
   the fix. Candidate: block over `k` and store and reload the tile's accumulators between `k`
   blocks rather than resetting them. The per-output `k` order is unchanged, so it is
   bit-identical.

6. **`max_pool_forward_batch`** is the second- or third-largest Rust conv op, about 10% of
   profiled conv-pool-conv training (0.10 s of 0.9 s single-example, 6000 calls). It has never
   been examined. It does no arithmetic, so the likely costs are the per-window index arithmetic
   and the separate `argmax` output.

7. **Dense single-example `downstream` at 32 x 5408 is 1.5x numpy** (50 vs 33 µs, per-op
   harness). It is vector @ matrix, `delta @ W` through `axpy_row`: 32 load/FMA/store passes
   over a 43 KB output row, the pattern #14 removed from `matmul_2d`. The same product as a
   one-row `downstream_batch` goes through the tiled kernel and measured 29 vs numpy's 36 µs.
   Candidate: route vector @ matrix through `tiled_row_range` as a one-row matrix. It is the
   same FMA chain, so bit-identical, and the crate tests that compare the two would need
   another reference. Affects the conv-tail single-example step only. Low value.

8. **The dataset as one backend array** (optimization 5, low value, not started). Both backends
   convert Python tuples to an array on every call (`pa.Array(list(state))` in
   `RustArrayNetworkBase._forward`/`learn`, a list of rows in `learn_batch`, `np.array` in the
   numpy `ArrayNetworkBase`): 49 µs for one 784-pixel MNIST row in Rust, 55 µs in numpy. That
   doesn't change the ratio, but it is a fixed cost neither backend's maths can remove, and a
   larger share now the maths is faster. The trainers (`train.py`) take `list[tuple[state,
   label]]`, shuffle it in Python and pass tuples through, a contract shared with the
   pure-Python networks, the ensembles and the sweeps.

   **Stage 0, go/no-go, fixed before measuring:** measure the conversion's share of one epoch,
   single-example and mini-batch 32, dense 784 -> 30 -> 10 and the conv networks, both backends
   (cProfile own time of `Array.__new__`/`np.array`, plus an A/B against pre-converted inputs
   through a test-only path), and what the per-epoch `_training_accuracy` passes spend
   re-converting every row. Proceed only if the conversion is at least about 10% of epoch time
   for some production configuration; otherwise record the numbers here and close.

   **Design, if it goes ahead:** add a fast path, keep the existing interface. A
   `PreparedDataset` built once per run holds the states as one backend matrix, the labels as a
   list and its backend; the two array network bases get `prepare_dataset(rows)`, and
   `learn_row`/`learn_batch_rows` that share their bodies with `learn`/`learn_batch` (which
   become convert-then-call wrappers, so the paths can't drift). numpy row slices are free
   views; a Rust slice copies, so a `gather_rows(matrix, indices)` crate op only if that copy
   measures material. The trainers shuffle an index list in place of the tuple list, which
   consumes the RNG identically (`shuffle` on a list of the same length makes the same
   permutation), so every seeded end-to-end pin must stay exact, with a test that the same seed
   gives the same visiting order on both paths. Tests: `learn_row` gives exactly the weights of
   `learn`, step by step, for every numpy and Rust network class, enumerated from a registry so a
   new class can't be missed. Out of scope: changing the `load_*` functions, and anything the
   trainers compute.

9. **Threading past the threshold (deferred).** No demo runs a product above 8M flops except at
   batch 512, so none of these pays in a demo today. Revisit only for a large-batch use case:
   - **A persistent pool.** Removes the 180-200 µs spawn cost at 8 threads. Workers that persist
     might also keep their cores warm, but in training they would still idle between calls, so
     the cold clock may remain unless they spin; that is untested. Design notes: rayon's global
     pool with `out.par_chunks_mut(rows_per_thread * cols)` keeps the contiguous row blocks
     (the crate's first dependency after pyo3; measure its build time and per-call overhead),
     or a small hand-rolled pool (no dependency, but scoped borrows across persistent workers
     need `unsafe` lifetime erasure or copying). Keep the split static and the pool lazily
     initialized.
   - **The caller computes the first block** and spawns one thread fewer. In isolation at 32 x
     5408, batch 32 it took 443-584 µs at 4 threads against 729-779 for spawn-all, and 564-621
     against 723-728 at 8. It is the cheapest threading change left, bit-identical, and untried
     end to end.
   - **Split by columns** when `b` is the larger operand, so each thread reads only its panel of
     `b`: skipped. Nothing pointed at `b` traffic; a register-only probe (no memory traffic)
     showed the same worker slowdown.

   Out of scope for threading: splitting over `k` (a cross-thread reduction changes summation
   order), threading pooling or elementwise ops, and releasing the GIL (the trainers are
   single-threaded Python, so there is nothing to overlap with).

## Threading

The only threading in the crate is `for_each_row_range` in `rust/src/linalg.rs`, used by the
three matmul kernels: `matmul_2d` (dense `downstream_batch`, `accumulate_gradient_batch`),
`matmul_nt` (every dense `forward_batch`) and `matmul_narrow` (conv forward, downstream,
accumulate). `matmul_thread_count` is the policy: below 8M flops one thread, otherwise
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

  2 threads never beat 1, except `matmul_2d` at 64M (0.68); 4 only marginally; 8 pays from
  about 8M for `matmul_2d` and `matmul_nt`. Threading barely moves conv forward and downstream (the matmul is a small
  part of those ops). Conv accumulate halves on 8 threads at N = 512 (58-60 → 30 ms). Large
  products scale: `(32, 512) @ (512, 5408)` goes from 31-32 ms on 1 thread to 10-11 on 8.
- **Isolated loops flatter threading.** Back-to-back calls keep every core clocked up; in
  training the cores idle between calls, so each threaded call pays the cold clock. The 32 x
  5408, batch 32 calls came out about even in isolation but made the MNIST conv mini-batch 32
  epoch 11% slower (0.846 against 0.752 s, threading off), about 500 µs per call. **Threading
  decisions are judged end to end.** Since #16, the products between 4M and 8M flops are
  slower in isolation than before (up to 1.4x at `(244, 64) @ (64, 512)`) and faster in
  training.
- **Which demo calls are threaded.** Counted per call site over one epoch of each demo
  configuration: at batch 32 or single-example, only MNIST `ConvSpec(3, 8)` mini-batch 32
  ever crossed the old 4M threshold (its 32 x 5408 tail, 186 calls of 5.5M flops per epoch);
  conv-pool-conv and conv-conv-stride2 have tails 968 and 1152 wide. Nothing is threaded in
  any demo at batch 32 since #16. At batch 512 the conv ops (24.9M) and the 32 x 5408 tail
  (88.6M) are threaded and pay (+11% and +15% when forced unthreaded); dense MNIST's 12M
  calls come out even.
- **Choosing 8M.** Every threshold from 5.5M to 12M threads the same demo calls; 8M is where
  the ladder shows 8 threads clearly paying. A 32-rows-per-thread floor on top of it made the
  conv mini-batch 512 epoch 21% slower (1.048 against 0.867 s), so there is none.

## How to measure

- **Per op, quick survey:** `python -m indrajala_ml.demos.demo_layer_op_timing`. It times every
  dense and conv layer method, single-example and batch 1/32/512, numpy and Rust interleaved,
  300 calls per loop (300 // batch for batch ops, at least 10), median of 5 loops. **Its batch
  rows are noisy and can be off either way**, partly because of the interleaving (next item): it reported 30 x 784 `forward_batch` at batch 512
  as 3469 µs where a focused benchmark measured 1350, and `downstream_batch` at 32 x 5408, batch
  32 as 9.5x numpy where a focused benchmark measured 13.9x. Use it to find candidates, not to
  judge them.
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
  op). For a dense op change, also one epoch of dense MNIST 784 -> 30 -> 10 from identical
  weights, single-example and mini-batch 32, median of 3.
- **End to end, old against new build** (for a kernel change, before the demo): one epoch of
  MNIST, one `ConvSpec(3, 8)`, dense 32, mini-batch 32, lr 0.5, the demo's 2000-row subset, from
  a snapshot of `randomized(...)` after `np.random.seed(0)`, with `random.seed(0)` before each
  epoch. One process per epoch, builds alternated with the order swapped every run, medians of
  5 or more. The same at mini-batch 512, and one dense MNIST epoch at batch 32 and 512. Run
  these, and the demo, in the background with an ETA.
- **Old vs new builds:** alternate the builds (old, new, old, new) and run each benchmark on
  both. Builds take about 6 s (`./cli build-rust`). Commit the crate change before switching,
  and switch with `git checkout main -- src/linalg.rs` and back, not a stash.
- **This machine** (Ryzen 7 3700U laptop, 4 cores / 8 threads, 512 KB L2 per core, 4 MB L3)
  varies 20-30% between passes, sometimes more. A background IDE made a first measurement
  unusable once. Treat changes under about 20% as noise unless both passes agree, and re-check a
  surprising result with the build order reversed. Idle cores drop to 1.1-1.5 GHz and a busy one
  boosts to 3.8 GHz (`schedutil`), so a single call after a pause measures slow. Time loops,
  not single calls. Clocks also carry over between settings: in a sweep, an 8-thread
  setting run right after another one measured up to 2x faster at `(32, 512) @ (512, 5408)`
  (5.3-5.8 against 9.9-10.1 ms), since cores take hundreds of ms of load to clock up. Rotate
  the order of settings. `perf` can't be used without root (`perf_event_paranoid` is 4), so probes
  go in a local crate build instead (timers and counters behind a Python-callable switch).
- **A probe's allocation pattern is not the real call path's.** Check faults on the real op.
  In optimization 7's probe, a tight Rust loop allocating a fresh 22 MB output every call
  (with a reused buffer of the same size also live) faulted on every page: 5410 faults per
  call, about 10 ms (40%) at `(512, 32) @ (32, 5408)`. The same product through the Python
  op had 0.1 faults per call. So quote allocation and fault costs from the real op
  (`focused_benchmark.py` reports faults per call), not from a probe loop.
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

## Kernel invariants

Every Rust result is bit-identical between the scalar and AVX2+FMA paths, so a training run
doesn't depend on the machine. Each kernel has one fixed summation order:

- **Matrix @ vector and `matmul_nt` (`X @ W.T`, every forward):** `dot_product`'s grouping.
  Lane `j` of a 4-lane accumulator sums indices `j, j+4, ...` by FMA, then
  `(l0 + l1) + (l2 + l3)`, then the `k % 4` tail sequentially. `dot_products_into` runs 8, then
  4, then 2 rows at once in that grouping. A batched forward's rows equal the single-example
  forward exactly.
- **Matrix @ matrix (`matmul_2d`, `matmul_narrow`) and vector @ matrix:** one FMA chain per
  output, `k` increasing from 0.0. `tiled_row_range` holds 16-column output tiles in registers
  across all of `k`: `matmul_2d` in row blocks of about 16 KB of `a`, `matmul_narrow` one row at
  a time. Vector @ matrix goes through `axpy_row` with the same chain.
- **Threading** splits output rows across threads, so each output is computed by one thread and
  the thread count can't change any value (see "Threading").
- These orders differ from numpy's in the last few ULPs, so parity with numpy is checked with
  `rtol`. Crate tests pin each order exactly against a `Fraction`-emulated FMA reference.

## Completed

Crate PR numbers are `indrajala-math-rust`'s. "Bit-identical" means every output kept its bits.

| change | crate PR | result |
| --- | --- | --- |
| One-pass dense `accumulate_gradient` (no `outer` temporary) | #5 | 596 → 62 µs at 32 x 5408; bit-identical |
| Fused single-example SGD step, `layer_sgd_step` (a Python contract change: `sgd_step` on the Rust layers, with a fallback for momentum/Adam/L2/conv) | #6 | per-layer step 463 → 64 µs at 32 x 5408; dense MNIST epoch -15%; bit-identical |
| `layer_downstream` as `delta @ W`, no `W.T` copy | #7 | 308 → 43 µs at 32 x 5408; bits changed (within 4 ULPs) |
| `matmul_nt` for `X @ W.T` in `forward_batch`, no `W.T` copy | #8 | 330 → 58 µs at 32 x 5408, batch 1; batch rows now equal the single-example forward exactly |
| Conv forward stops returning `Z` | #9 | about 14% at N = 512, noise at N = 1 and 32; bit-identical |
| Conv and pool ops take a vector as N = 1, so no reshapes | #10 | Rust N = 1 overhead 28.5% → 0.3% (UCI conv layer); bit-identical |
| `matmul_narrow`: register-held output rows for conv `cols @ W.T` | #11 | conv forward 33-64% less at N = 1, 13-63% at N = 32; MNIST conv-pool-conv mini-batch 0.63 → 0.46; bit-identical |
| Conv downstream and accumulate via `matmul_narrow` | #12 | downstream 4-56% less, accumulate 16-43% less in 21 of 24 configurations; bit-identical |
| `dot_products_into`: 8/4/2 rows at once in `dot_product`'s grouping | #13 | 30 x 784 forward 9.0 → 3.7 µs, batch 32 260 → 93 µs; bit-identical |
| `matmul_2d` register-tiled, sharing `matmul_narrow`'s kernel | #14 | unthreaded dense batch downstream/accumulate 0.1-0.6x their old time; mini-batch dense MNIST epoch about -4%; bit-identical |
| `set_matmul_threading` override, and tests that the thread count can't change any kernel's bits | #15 | measurement and test infrastructure; no default behaviour changed |
| Threading threshold 4M → 8M flops, policy in `matmul_thread_count` | #16 | MNIST conv mini-batch 32 epoch 0.879 → 0.770 s (-12%); batch 512 and dense MNIST unchanged; bit-identical |
| `matmul_nt` in 4 x 2 register tiles (4 rows of `X` against 2 rows of `W`; 2 x 4, 3 x 3 and 2 x 2 measured slower or tied) | #17 | unthreaded dense `forward_batch` 0.6-0.9x its old time (32 x 5408, batch 32: 771-916 → 545-555 µs; 30 x 784, batch 32: 102-110 → 76-77); epochs within noise; bit-identical |

Closed with no measured gain, kept as findings:

- **Transposed-left matmul for `accumulate_gradient_batch`** (crate branch `matmul-tn`,
  `e59c511`). Bit-identical, and within ±4% at every shape. The copy it removed was
  `delta_batch.T`, batch x `M`, small next to the matmul.
- **Forward-only conv path for evaluation** (crate branch `conv-infer-batch`, `ee432d0`), which
  skipped storing `cols`. At N = 1 `cols` is only 48 KB, and a whole-network evaluation pass was
  1-8% slower with it.
- **Conv formulations** (crate branch `conv-forward-formulations-proto`, `ef85831`). `W @ colsT`
  had the fastest forward at small `O`, but its backward cost more, and it lost the training step
  to the `matmul_narrow` kernel in 17 of 24 configurations. A direct kernel with no `cols` only
  paid at large `O` and N, and can't serve training.
- **The threading guess.** The dense batch gaps were first blamed on `matmul_2d` threading
  just past its threshold. At batch 8 and 16 the same ops ran on one thread and were already
  3.5-9.4x numpy; the kernel was the main cause (#14). Threading is a real but separate cost
  (see "Threading").
- **Row blocks for `matmul_2d`.** 1-row blocks were 2-3x slower than 16 KB blocks where `K` is
  in the hundreds (`(32, 512) @ (512, 5408)`: 40-47 vs 15-16 ms). One block for all rows was
  close to 16 KB blocks but not better.

## Other findings from the same measurements

- **Single-example training is chaotically sensitive to rounding.** numpy and Rust networks
  trained from the same weights can end up classifying only 71-83% of test rows the same. numpy
  against itself, with one weight nudged by 1 ULP, diverges just as much (73-81%). Both pairs
  stay within about 1e-15 through the first UCI epoch, then jump to differences of about 0.7 in
  the second. So a bit-changing optimization shifts end-of-training numbers the same way. Judge
  it by the step-by-step parity tests (per-step agreement to about 1e-15), not by end-of-run
  accuracy.
- **numpy's OpenBLAS threads slow a Rust call that runs soon after.** After a BLAS call,
  OpenBLAS keeps its worker threads spinning for a while (`OPENBLAS_THREAD_TIMEOUT`). At 32 x
  5408, batch 32, Rust `downstream_batch` measured 547-564 µs on its own. In loops interleaved
  with numpy's it measured 996-1165 µs on 1 thread and 1752-2872 on 8. Tried one at a time, each
  of these brought it back to 509-585 µs: `OPENBLAS_NUM_THREADS=1`, `OPENBLAS_THREAD_TIMEOUT=4`
  (the shortest spin), or timing numpy in a separate process. With the short spin, numpy itself
  slowed from about 250 to 300-420 µs. A single Rust call was still slowed 100 ms after the
  numpy loop, and no longer at 500 ms. The Rust training path never calls numpy, so this only
  affects benchmarks. Every per-op ratio past the threading threshold that was measured
  interleaved is suspect.
- **The conv demo's mini-batch runs barely train.** They reach about 10% accuracy at lr 0.5 in
  1-2 epochs, and lr 2, 4 and 8 don't fix every configuration. Their timings are valid; their
  accuracy columns are not informative.
