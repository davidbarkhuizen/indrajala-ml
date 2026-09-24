# Optimizations: open candidates

Part of the optimization record; the index is [../optimizations.md](../optimizations.md).

## Open candidates, in priority order

Ranked by the Rust time each would save in configurations that are actually trained (the demos'
single-example and batch-32 runs; batch 512 and up only in the batch-size-scaling study), then
by how well the stake and the fix are established. Not by the Rust / numpy ratio: candidates 1
and 2 speed up both backends. Re-ranked 2026-09-24 after the batch-size-scaling study.


5. **`max_pool_forward_batch`** is the second- or third-largest Rust conv op: in profiled MNIST
   conv-pool-conv training, 0.095 s of 0.86 s single-example (11%, 6000 calls, 4000 of them the
   accuracy passes) and 10% at mini-batch 32. It has never been examined, and there is no
   comparison with numpy's. It does no arithmetic, so the likely costs are the per-window index
   arithmetic and the separate `argmax` output. Removing it entirely would save 10-11% of
   conv-pool-conv; how much is recoverable is unknown, so it ranks below the candidates with a
   measured stake but above candidate 6's smaller, better-established one.

6. **Dense `downstream_batch` and `accumulate_gradient_batch` at short `k`** (`matmul_2d`,
   optimization 7). **Status: stage 0 done (#354); next stage A, then B; stage C skipped.** At
   32 x 5408, batch 32 these are the largest per-op ratios in the table (2.4-2.8x and 2.2-2.9x
   numpy), but stage 0 found that most of that is numpy's OpenBLAS threading: on one thread
   each, `downstream_batch` is level and `accumulate_gradient_batch` is 1.2-1.3x, and Rust's
   bare product is faster than numpy's. What is left is Rust's own kernel (stage A) and
   accumulate's extra pass (stage B), together worth about 4% of the worst end-to-end cell.

   Both ops are `(32, 32) @ (32, 5408)` products, 5.5M flops with a 1.4 MB output and `k` (the
   batch or the layer width) only 32. They run on one thread since #16 and are in the MNIST
   conv mini-batch 32 dense tail, the worst end-to-end cell: 63 calls of each per epoch (62 at
   batch 32, the last at 16). The
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

   - **Most of the gap is numpy's OpenBLAS threading, not the kernel.** numpy's `downstream_batch`
     at 32 x 5408, batch 32 (228-389 µs in stage 0's passes, 232-246 in the op table) uses
     several threads. With both backends on one thread, `downstream_batch` is level (610-625
     against numpy's 573-578) and `accumulate_gradient_batch` is 1.2-1.3x (1040-1083 against
     807-871), most of which is its extra pass (see below). On the bare products alone,
     Rust took 549-637 against numpy's 850-928. Oddly, numpy's own layer op was faster on one
     thread than its bare `delta_batch @ W` of the same shapes and value ranges (573-578 against
     850-897), in the same runs, with no explanation found. So the single-threaded numpy target
     is 573-897 µs depending on which call is timed. At 30 x 784, batch 32, Rust single-threaded
     is level with numpy single-threaded as well. numpy's threading doesn't pay in training
     either: its own MNIST conv mini-batch 32 epoch took 1.22-1.32 s at `OPENBLAS_NUM_THREADS=1`
     against 1.39-1.56 s by default. So the 2.2-2.9x in the op table measures hot-loop
     threading. It doesn't show a slower kernel; what Rust adds is accumulate's extra pass.
   - **Call boundary: 3-20 µs** (fused op against a bare `@`), 0-4%. Nothing to do.
   - **Allocation and page faults: not a cost.** The real call path has 0.0 minor
     faults per call at every shape but 32 x 5408, batch 512 (0.1); presumably glibc reuses the
     freed pages (inferred from the counts, not observed). A whole MNIST conv mini-batch 32
     epoch has 12.5-17k faults in total. If each dense-tail output faulted, the two ops alone
     would account for about 64k (three 1.4 MB outputs per pair of calls, 62 pairs).
     Zeroing a 1.4 MB output costs 25-27 µs (4-5%). Stage C is skipped.
   - **Multi-row register tiles: 7-35% off the kernel.** The hypothesis was FMA latency: per row
     and tile the kernel has 4 independent chains of `k` dependent FMAs, where Zen 2's 2 FMA
     pipes at about 5 cycles' latency need about 10 in flight. The fix is the one #17 made for
     `matmul_nt`. 2-row tiles, bit-identical (2133 shapes checked with `==`), probe kernel
     time by product `m x k x n`:
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
     reading, without hardware counters to confirm it: the cost is the loads of `b`'s `k x 16`
     panel, which must stay in L1 alongside the current block of `a` (16 KB, or all of `a` when
     smaller). At `k` = 32 the two take 12 KB of the 32 KB L1; at 128 they fill it (16 + 16
     KB); at 512 the panel alone (64 KB) doesn't fit. Each extra row in the tile reuses those
     loads.
   - **Accumulate's extra pass: 140-160 µs, plus about 240 µs unattributed.**
     At 32 x 5408, batch 32 the fused op took 952-1036 µs. Its parts summed to 715-780: the bare
     product 572-614, `add` 142-163, and the transpose and `sum_axis0` 1 µs each. The likely
     cause of the rest is that the fused op keeps four 1.4 MB arrays live (`X`, the update,
     `grad_W` and the sum) against a 4 MB L3. That is unmeasured. Stage B removes one of them
     and the pass, so it runs second. The three arrays left still come to 4.2 MB, more than the
     L3, so stage B may recover little of the 240.
   - **Revised stake:** stages A and B together might save about 450 µs per pair of calls:
     about 150 from stage A (-14% on each of the two 524-588 µs products), 140-160 from stage
     B's removed `add`, and part of the 240 unattributed if it is the extra live array. Over
     62 pairs that is about 28 ms (4%) of the 0.72-0.75 s epoch, not the 8% the per-op ratios
     suggested before stage 0. Without the unattributed part it is about 19 ms (2.5%), the more
     likely figure (see above). Dense MNIST mini-batch gains a little too (-10 to -12% on its 30
     x 784 products; accumulate is 11.5% of its step loop at B = 32, candidate 7), about 1-2%.
   - **Found on the way:** a larger gap at long `k`, and Rust's `transpose` of `delta_batch` at
     batch 512 taking 87-107 µs against numpy's 8-12 µs copy, about 9% of the 30 x 784
     accumulate (both in candidate 7).

   **Stage A: multi-row register tiles in `tiled_row_range`.** Hold a tile of R rows x 16 columns
   (4R accumulators) across all of `k`, R = 2 (and R chosen by `k` only if the op table shows it
   matters; 3 won only at `k` = 128). The probe's `probe_tiled_avx2_fma` on crate branch
   `probe/opt7-stage0` (local only, not pushed) is a starting point. The rows left in each block
   (its rows `% R`) run the current 1-row tile; the 4-wide and scalar column tails keep their
   loops. The tile runs within a row block, so `matmul_narrow`, which passes one row per block,
   keeps the 1-row tile with no change. Whether conv gains is a separate step: give
   `matmul_narrow` blocks of R rows, measure the conv ops (forward, downstream, accumulate at
   28x28, N = 32 and 512), and keep one row per block if they don't gain. Conv accumulate is the
   likeliest to gain: each row of `D` streams all of `cols` from memory at large N, and an R-row
   tile reads it once for R rows (candidate 8). Each output
   is still one FMA chain, `k` increasing from 0.0, so it is bit-identical. Extend the crate's
   `test_matrix_at_matrix_is_the_fma_chain_exactly` (`tests/test_linalg.py`) so `m` takes every
   value from 1 to 2R + 1 (every remainder, with and without a full tile) at the existing
   `TILE_WIDTHS`, and add a `k` large enough that `rows_per_block` is odd (`k` from 513 to 682
   gives 3), so a remainder falls at every block's end, not only the last; `BIG_SHAPES` and the
   thread-count tests cover the large shapes and a remainder at each thread's range end. Mutations
   that must fail: give the tile's second row the first row's `a` value at one `k`; drop the last
   remainder row.

   **Stage B: fuse the gradient add into the kernel (accumulate only).** A `matmul_2d_add(a, b,
   c)` variant, or a `tiled_row_range` parameter, whose store writes `c + acc` in place of `acc`.
   `layer_accumulate_gradient_batch` then allocates one output, not two, and skips
   `combine_with_array`; `sum_axis0` for `grad_b` stays. Today's `grad_w + (delta.T @ X)` is `g +
   u` with `u` the finished chain, one rounding, and the fused store is the same operation on the
   same values, so it is bit-identical. There is only an `rtol` test of this today: add a crate
   test that the fused op's new `grad_w` equals `grad_w + (delta.T @ X)` computed by separate
   crate calls (`transpose`, `matmul`, `+`), `==` on `tolist()`, at `BIG_SHAPES` and the
   `TILE_WIDTHS` shapes, with `grad_w` containing `-0.0` and zeros. Mutation that must fail: start
   the chain from `g` (a different rounding).

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

7. **Dense `accumulate_gradient_batch` at long `k`** (`matmul_2d`, found in optimization 7's stage
   0). Single-threaded, at batch 512 (`k` = 512) it is 3.0-3.4x numpy's single-threaded time at 32
   x 5408 (31.3-32.0 against 9.4-10.3 ms) and 1.6-2.1x at 30 x 784 (2335-2715 against 1269-1436
   µs). 2-row tiles recover only 7-31% there. The likely cause is `b`'s `k x 16` panel (64 KB at
   `k` = 512) no longer fitting the 32 KB L1. At 32 x 5408 default threading hides it (6.8-11.7 ms
   against numpy's 12.9-13.5); at 30 x 784 it doesn't (997-1336 against 714-776 µs). It matters
   only at batch 512 and up, which only `demo_batch_size_scaling` and the batch-size-scaling
   study train at (dense, B up to 1024). Candidate: block over `k`, or pack `b`'s
   panel, storing and reloading the tile's accumulators between `k` blocks so each output keeps
   its chain. Candidate 8 needs the same change in the same function (`matmul_narrow` also runs
   `tiled_row_range`), so one `k`-blocked `tiled_row_range` may serve both.

   Separately, the `transpose()` of `delta_batch` that feeds this product takes 87-107 µs at
   batch 512 against numpy's 8-12 µs, about 9% of the 30 x 784 op (a naive element loop in
   `rust/src/array.rs` whose writes stride by `rows`). The closed transposed-left matmul
   ("Completed") removed this copy and measured within ±4%, calling it small next to the
   matmul; at 30 x 784, batch 512 that no longer holds. A blocked transpose is a copy, so
   trivially bit-identical, and cheaper to try than another kernel.

   **Measured end to end (batch-size-scaling study, #367, 2026-09-24).** Dense 784 -> 30 -> 10,
   one full-MNIST epoch, `scripts/batch_size_timing.py profile` (cProfile own time over one Rust
   step loop, plain SGD; seconds, share of the profiled loop):

   | B | step loop, unprofiled | Rust ops | `accumulate_gradient_batch` | `forward_batch` | the other three ops |
   | --- | --- | --- | --- | --- | --- |
   | 32 | 1.59 | 0.37 | 0.178 (3750 calls, 11.5%) | 0.161 (10.4%) | 0.028 (1.8%) |
   | 512 | 2.12 | 0.38 | 0.223 (236 calls, 0.94 ms each, 10.7%) | 0.147 (7.1%) | 0.011 (0.5%) |
   | 1024 | 2.06 | 0.40 | 0.251 (118 calls, 2.13 ms each, 12.9%) | 0.141 (7.3%) | 0.009 (0.5%) |

   The other three are `apply_accumulated_gradient`, `hidden_delta_batch` and `output_delta`.
   The Rust ops are about 0.4 s of every epoch whatever the batch; the rest of the step loop is
   Python and batch conversion (candidate 1). At B = 512 and 1024 accumulate passes the
   promotion bar fixed before measuring (about 5% of the step loop). But the same op is already
   0.178 s, 11.5%, at B = 32 (3750 calls). Long `k` adds only 0.05-0.07 s per epoch on top of
   the short-`k` cost. The 0.1 s saving estimated when this was decided, 5% of the step loop and
   2% of the trainer's 4.85 s epoch, applied the one-thread ratio (Rust 1.6-2.1x numpy at 30 x
   784) to these calls. But they are threaded in training (12M flops at B = 512), and at 0.94 ms
   each they already beat numpy's one-thread 1.27-1.44 ms. A faster kernel would still speed up
   each thread, but threaded calls pay the cold clock and spawn cost (see "Threading"), so the
   saving is likely under 0.1 s and is unmeasured. **Decision: do it after candidate 1, and only
   if a one-thread `k`-blocking probe gains enough to leave a threaded saving above the 5% bar.**
   Candidate 1 (done) took the Rust B = 32 epoch from 3.72 to 1.69 s, most of it batch and row
   conversion, so this op's share of the step loop is now larger; re-profile before deciding. Also note the
   profile table's percentages imply a profiled loop slightly shorter than the unprofiled one
   (for example 0.178 / 11.5% = 1.55 s at B = 32), which suggests the two columns come from
   different runs.

8. **Conv accumulate with a large `cols`.** `D @ cols` (`D` the deltas by channel, `(O, N*P)`;
   `cols` `(N*P, C*k*k)`) goes through `matmul_narrow`, so `k` is N*P and there are only O rows.
   At 28x28, `ConvSpec(3, 8)`, on one thread (2026-09-24, as in candidate 4) it took 56.7-58.0 ms
   at N = 512 against 14.4-15.0 ms for 512 single-example calls (3.8-4.0x); the bare `(8, 346112)
   @ (346112, 9)` product took 42.5-46.9 ms of that. The likely cause, from the kernel's loop
   order rather than counters: every output row makes one pass over `k` per column tile (two
   4-wide tiles and a scalar column at 9 columns), and each pass streams all of `cols` (25 MB,
   past the 4 MB L3), so 8 rows read it 24 times, about 600 MB per call, 13-14 GB/s. At N = 1
   `cols` is 48 KB and stays in L2; at N = 32 (1.56 MB) the op is 1.1-1.2x its single calls, and
   in the MNIST conv mini-batch 32 epoch it is 10.5% (78 ms, profiled). Candidate: block over
   `k` so a slab of `cols` serves every row and column tile before the next, storing and
   reloading the tile's accumulators between `k` blocks. The per-output `k` order is unchanged,
   so it is bit-identical. Stage A's R-row tiles (candidate 6), given to `matmul_narrow`, would
   divide the passes by R as a first step. The stake in trained configurations is small: no
   demo trains conv at N = 512, and at N = 32 the op is only 1.1-1.2x its single calls, so
   closing that gap would save about 8-12 ms of the 78 (1-1.5%). It is worth doing alongside
   candidate 7, which needs the same `k`-blocked `tiled_row_range`.

   Earlier findings: at 13x13x8, N = 32, `matmul_narrow` gave no gain over the old `matmul_2d`,
   which had `k`-blocking (+2%, +4%, -1% at O = 4, 8, 32; `cols` 0.3-2.2 MB); that blocking is
   gone since #14. Moving the op to `matmul_2d` at 28x28, N = 512 measured 35-39 against 30-32 ms
   (presumably threaded: one thread takes 57-58 ms), and was read as "row blocks are slower". It
   can't show that: at `k` = 346112 `matmul_2d`'s 16 KB rule gives 1 row per block, the same code
   path as `matmul_narrow`, so the difference was run-to-run variation.
