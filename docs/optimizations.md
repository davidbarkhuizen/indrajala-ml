# Optimizations

Performance work on the Rust backend (`rust/`, the `indrajala-math-rust` crate) against numpy,
the benchmark it is compared with. It started from the Rust CNN timing (#317-#320 and the Rust
CNN stages), which found where Rust was slower than numpy or slower than it needed to be. Each
item is measured before and after, and must keep every parity test passing.

This document is the only record of optimization work: there are no separate optimization workplans. Each open item, with its plan
where it has one, is a candidate below. Four workplans have been folded in: threading
(optimization 6, finished; its findings are in "Threading", and what it left open is candidate
9), dense batch ops at short `k` (optimization 7; stage 0 done, stages A and B planned in
candidate 6), the dataset as one backend array (optimization 5, candidate 1, now done) and
the batch-size-scaling study (#365-#368, finished; its timing stage measured candidates 1 and 7
and the trainer's accuracy passes on full MNIST).

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
0.66. Dense MNIST 784 -> 30 -> 10, one 60000-example epoch: Rust/numpy about 0.21
single-example and 0.54 mini-batch 32 since candidate 2 (#379; 0.20 and 0.33 before it, 0.31 and
0.47 before candidate 1), both backends' epochs much shorter (see candidates 1 and 2 under
"Completed"). Candidate 2 raised the B = 32 ratio because numpy's accuracy pass gained more.

Two caveats on these ratios. numpy runs with OpenBLAS's default threading, which slows its own
training: its MNIST conv mini-batch 32 epoch took 1.22-1.32 s at `OPENBLAS_NUM_THREADS=1`
against 1.39-1.56 s by default (optimization 7 stage 0), so against single-threaded numpy that
cell would be nearer 0.55-0.6 (an estimate from runs in different sessions, not measured side
by side). And the demo runs numpy then Rust in one process, so each Rust run starts while
OpenBLAS's threads may still be spinning (see "Other findings"); that can only slow Rust, by at
most the first 100-500 ms of each run, and is unmeasured.

Per op, the single-example ops are all at or better than numpy except dense `downstream` at
32 x 5408 (1.5-1.6x, candidate 3). The other gaps left are in batch ops. Rust / numpy µs per
call, focused benchmark (see "How to measure"), numpy and Rust in separate processes, Rust
with the default threading (threshold 8M flops since #16). The last two columns put both
backends on one thread (`--openblas-threads 1 --rust-threads 1`, 2026-09-24: optimization 7
stage 0, and the `forward_batch` rows in a later pass of two):

| shape | op | batch | numpy | Rust | Rust/numpy | numpy, 1 thread | Rust, 1 thread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 x 5408 | `downstream_batch` | 32 | 232-246 | 581-654 | 2.4-2.8x | 573-578 | 610-625 |
| 32 x 5408 | `accumulate_gradient_batch` | 32 | 411-453 | 976-1180 | 2.2-2.9x | 807-871 | 1040-1083 |
| 32 x 5408 | `downstream_batch` | 512 | 11753-11777 | 10648-11847 | 0.9-1.0x | 12236-12361 | 14746-15199 |
| 32 x 5408 | `accumulate_gradient_batch` | 512 | 12926-13493 | 6811-11720 | 0.5-0.9x | 9425-10298 | 31253-31958 |
| 32 x 5408 | `forward_batch` | 512 | 3695-4110 | 4747-4898 | 1.2-1.3x | 8937-9918 | 8421-8477 |
| 32 x 5408 | `forward_batch` | 32 | 294-415 | 470-659 | 1.1-2.2x | 563-575 | 479 |
| 30 x 784 | `downstream_batch` | 512 | 416-945 | 1193-1401 | 1.3-3.4x | 1153-1466 | 1383-1560 |
| 30 x 784 | `accumulate_gradient_batch` | 512 | 714-776 | 997-1336 | 1.3-1.9x | 1269-1436 | 2335-2715 |
| 30 x 784 | `forward_batch` | 512 | 750-1027 | 1038-1275 | 1.0-1.7x | 1271-1308 | 1217-1257 |
| 30 x 784 | `downstream_batch` | 32 | 46-50 | 73-85 | 1.5-1.8x | 71-84 | 75-86 |
| 30 x 784 | `accumulate_gradient_batch` | 32 | 70-73 | 82-97 | 1.1-1.4x | 90-102 | 84-93 |

The first two rows (5.5M flops) run on one thread since #16; their Rust numbers are from then.
The two 32 x 5408 batch-512 rows are from optimization 7 stage 0's default-threading run (88M
flops, threaded).
The `forward_batch` rows are after #17, except 32 x 5408 at batch 512, whose threaded time moved
between 4.6 and 9.8 ms from run to run in the same session, so it keeps its earlier numbers. The
batch-512 Rust numbers are partly warm-clock numbers (see "Threading"). An earlier version of
this table was measured with numpy and Rust interleaved in one process and read 12-13x, 7x,
4-8x and 4x on the 32 x 5408 batch-32 `downstream_batch`, `accumulate_gradient_batch` and
`forward_batch` rows and the 30 x 784 batch-512 `downstream_batch` row (another interleaved run
read 13.9x on the first); that was numpy's OpenBLAS threads
taking cores from Rust (see "Other findings"). numpy's numbers are with OpenBLAS's default
threading, which it uses even at 5.5M flops. On one thread each, the batch-32 rows are level
or within 1.3x (32 x 5408 accumulate, its extra pass; candidate 6), Rust's `forward_batch` is
level or faster at every row, and `downstream_batch` at batch 512 (`k` = 32 or 30) is 1.2x at
32 x 5408 and 0.9-1.4x, overlapping, at 30 x 784. The large gap that remains is at batch 512
with a long `k`: accumulate (`k` = 512) is 3.0-3.4x numpy at 32 x 5408 and 1.6-2.1x at 30 x 784
(see candidate 7).

The per-op tables compare against numpy; the profile of a whole epoch finds other costs. Rust
time by op, cProfile over one MNIST epoch (2000 rows), 2026-09-24:

- **About a quarter of each timed MNIST conv epoch is not training.** The trainers run
  `_training_accuracy` (`train.py`) over every training row before the first epoch and after each
  one, one example at a time. In the conv mini-batch 32 epoch that is 4000 of the 4063
  `conv_forward_batch` calls and all 8000 `layer_forward` calls, about 0.19 s of 0.74 s; the
  single-example epoch is similar (4000 of its 6000 conv forwards). numpy runs the same passes,
  so the ratios include them, and every stake quoted below as a share of an epoch is a share of
  this timed epoch. That overstates the passes for longer runs: the trainers run n + 1 passes
  for n epochs, so a one-epoch run has two and a long run about one per epoch.
- **Conv MNIST, mini-batch 32** (0.74 s): `conv_forward_batch` 27% (its 63 batch calls about 1.5
  ms each, against 0.86 ms for 32 single-example calls; candidate 4),
  `conv_accumulate_gradient_batch` 10.5%, dense `accumulate_gradient_batch` 9.5% and
  `downstream_batch` 6.7% (candidate 6), `forward_batch` 6.3%.
- **Conv MNIST, single-example** (0.80 s): `layer_sgd_step` 20%, `conv_forward_batch` 20%,
  `layer_forward` 19%, dense `downstream` 11% (candidate 3), conv accumulate 7%.
- **Conv-pool-conv MNIST**: `conv_forward_batch` 41-42% in both trainers, `max_pool_forward_batch`
  10-11% (candidate 5).

## Open candidates, in priority order

Ranked by the Rust time each would save in configurations that are actually trained (the demos'
single-example and batch-32 runs; batch 512 and up only in the batch-size-scaling study), then
by how well the stake and the fix are established. Not by the Rust / numpy ratio: candidates 1
and 2 speed up both backends. Re-ranked 2026-09-24 after the batch-size-scaling study.

1. **The dataset as one backend array** (optimization 5): **done** (crate #19, #372-#374,
   2026-09-24). The dense MNIST Rust epoch at B = 32 went from 3.72 to 1.69 s. See
   "The dataset as one backend array" under "Completed". The entry stays here so the
   numbers below keep their meaning.

2. **A batched accuracy pass**: **done** (#378-#379, 2026-09-24). The trainers' accuracy pass
   runs `forward_batch` over 32-row chunks; the dense MNIST epoch at B = 32 went from 1.61 to
   1.31 s in Rust and 4.85 to 2.44 s in numpy. See "A batched accuracy pass" under
   "Completed". Rust conv keeps the per-row pass until candidate 4 is fixed.

3. **Dense single-example `downstream` at 32 x 5408 is 1.5-1.6x numpy.** First seen in the
   interleaved per-op harness (50 vs 33 µs); in separate processes (focused benchmark, two
   passes, 2026-09-24) it holds: 42.7-52.0 µs against 28.1-33.4. It is vector @ matrix, `delta
   @ W` through `axpy_row`: 32 load/FMA/store passes over a 43 KB output row, the pattern #14
   removed from `matmul_2d`. The same product as a one-row `downstream_batch` goes through the
   tiled kernel and took 21.4-22.1 µs against numpy's 28.3-28.8 in the same runs. In the MNIST
   conv single-example epoch its 2000 calls took 88 ms of 0.80 s (11%, profiled), so the tiled
   kernel's rate would save about 44 ms (5%). Dense MNIST's only downstream is the 10 x 30
   layer's, so this is the conv tail's. Candidate: route vector @ matrix through
   `tiled_row_range` as a one-row matrix. It is the same FMA chain, so bit-identical, and the
   crate tests that compare the two would need another reference. It ranks above candidate 4,
   whose stake is about the same, because the cause and the fix are both known and the change
   is small.

4. **Conv `forward_batch` costs more per example than single-example calls.** Re-measured
   2026-09-24 at 28x28, `ConvSpec(3, 8)`, Rust on one thread (focused benchmark, two passes):
   26.5-28.7 µs for one example, 1365-1790 µs at N = 32 (1.5-2.1x the 848-918 of 32 single calls)
   and 36.2-38.4 ms at N = 512 (2.5-2.8x). Before the `matmul_narrow` kernel it was 2280 against
   1715 µs at N = 32 (1.3x), so the gap has widened as the single-example path got faster. The
   other conv batch ops cost less extra at N = 32: downstream 1020-1023 µs against 800-867 for 32
   calls (1.2-1.3x), accumulate 1037-1106 against 899-934 (1.1-1.2x; candidate 8 at N = 512). In
   the MNIST conv mini-batch 32 epoch the 63 batch calls took about 96 ms (profiled, the 4000
   accuracy-pass calls taken out at 27 µs each), so at the single-example rate they would save
   about 40 ms, 5% of the worst end-to-end cell. That is about candidate 3's stake, but this has
   neither a confirmed cause nor a design, so it ranks after it. It ranks above candidate 6,
   whose stages are planned and measured but worth only 2.5-4%, and fixing it is what would let
   conv networks gain from candidate 2. The likely cause, unmeasured: `cols` (48 KB at N = 1, 1.56 MB at N = 32,
   25 MB at N = 512) falls out of L2 between im2col and the matmul. A second, found in candidate
   2's stage 0: glibc heap trimming, which faulted 27000 pages a pass into chained conv
   `forward_batch` calls at N = 32 and cost them about a third of their time (see "Other
   findings"). Check the faults per call first (`focused_benchmark.py` reports them), then time
   the op's parts (im2col, the matmul, the ReLU scatter). Candidate: im2col and multiply one block of output
   positions at a time, keeping `cols` for the backward pass.

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

9. **Threading past the threshold (deferred).** Only the batch-size-scaling study and its demo
   run products above 8M flops in training (the dense 30 x 784 products at B = 512 and 1024, 12-24M flops). The
   demos otherwise train at batch 32 or single-example. Even there, the Rust ops are 0.4 s of a
   2 s step loop, so threading changes would not pay much. Revisit only for a large-batch use case:
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
     end to end. That product has run on one thread since #16 (581-654 µs in the op table,
     overlapping the 4-thread numbers), so there it would pay only with a lower threshold, which
     the cold-clock finding in "Threading" argues against.
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
- **A training-path change, before and after:** `python scripts/prepared_dataset_timing.py time`
  times one trainer epoch (dense full MNIST and the conv demo's subset, single-example and B =
  32, both backends), one process per measurement. Run it once as is and once with the old
  checkout first on `PYTHONPATH` (see its docstring for the namespace-package caveat); a
  `git worktree add` of `main` in a scratch directory makes that checkout. `--epochs N` trains
  each run for N epochs (one accuracy pass per epoch, plus one before), for a long run's share.
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
  4, then 2 rows at once in that grouping; `matmul_nt` runs 4 rows of `X` against 2 of `W` in
  it (#17), and the rows and columns left over through `dot_products_into`. A batched
  forward's rows equal the single-example forward exactly.
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
| `Array.from_rows`, `row` and `take_rows`, for the dataset as one backend array (candidate 1; the Python side is #373-#374) | #19 | dense MNIST Rust epoch at B = 32 3.72 → 1.69 s, single-example 4.60 → 2.57; training bit-identical |
| `matmul_nt` in 4 x 2 register tiles (4 rows of `X` against 2 rows of `W`; 2 x 4, 3 x 3 and 2 x 2 measured slower or tied) | #17 | unthreaded dense `forward_batch` 0.6-0.9x its old time (second pass: 32 x 5408, batch 32: 771-916 → 545-555 µs; 30 x 784, batch 32: 102-110 → 76-77); epochs within noise; bit-identical |
| A batched training-set accuracy pass: `classify_rows`, `forward_batch` over 32-row chunks (candidate 2) | - | dense MNIST epoch at B = 32 1.61 → 1.31 s in Rust, 4.85 → 2.44 in numpy; predictions equal, pinned results unchanged |

**A batched accuracy pass** (candidate 2; #378 stage 0, #379, 2026-09-24). The trainers'
`_training_accuracy` now calls `classify_rows(prepared)` on the array networks: both bases run
`forward_batch` over chunks of `CLASSIFY_CHUNK_ROWS` = 32 rows (`prepared_dataset.py`) and the
shape classes' `_classify_output_batch` takes each row's argmax (`np.argmax(axis=1)`; in Rust
`tolist` and a Python first-max, since the crate's `argmax` takes a vector) or 0.5 threshold.
The Rust conv network overrides it with the per-row loop, since its batched forward is slower
(below). `accuracy()` in `multiclass_evaluate.py` is unchanged. Tests
(`tests/test_prepared_dataset.py`): for all 22 classes, `classify_rows` equals `classify_row`
row for row over 70 rows (three chunks, the last partial) and the prepared accuracy pass equals
the tuple one; both dropout classes classify in inference mode (numpy's draws nothing from
`np.random`; Rust's at drop probability 0.5 predicts as `classify_row` does); the Rust row
argmax breaks ties, signed zeros and NaNs as `pa.argmax` does; the batched binary threshold is
strictly above 0.5 on both backends. Seven source mutations (last-max argmax, `>=` thresholds,
a dropped or truncated chunk, float labels, misaligned predictions) each failed them. The full
suite passed with every pinned training result unchanged.

**The A/B.** `scripts/prepared_dataset_timing.py` (which gained `--epochs`), median of 5 (ranges),
one process per measurement, against `main` at #378 on `PYTHONPATH`; one-epoch runs before then
after, three-epoch runs after then before. Seconds:

| config | backend | before | after | after / before | from the loader, before → after |
| --- | --- | --- | --- | --- | --- |
| dense MNIST, B = 32 | Rust | 1.61 (1.59-1.71) | 1.31 (1.31-1.32) | 0.81 | 1.15 → 0.89 |
| dense MNIST, B = 32 | numpy | 4.85 (4.81-5.04) | 2.44 (2.42-2.45) | 0.50 | 3.49 → 1.03 |
| dense MNIST, single | Rust | 2.58 (2.29-2.78) | 2.11 (2.06-2.30) | 0.82 | 2.00 → 1.89 |
| dense MNIST, single | numpy | 12.96 (12.01-13.98) | 9.97 (9.69-10.85) | 0.77 | 11.61 → 8.44 |
| conv MNIST 2000, B = 32 | Rust | 0.61 (0.61-0.64) | 0.64 (0.63-0.64) | 1.05 | 0.59 → 0.61 |
| conv MNIST 2000, B = 32 | numpy | 1.24 (1.24-1.35) | 1.19 (1.19-1.20) | 0.96 | 1.17 → 1.03 |
| conv MNIST 2000, single | Rust | 0.73 (0.71-0.81) | 0.74 (0.68-0.76) | 1.02 | 0.73 → 0.71 |
| conv MNIST 2000, single | numpy | 3.66 (3.62-3.70) | 3.55 (3.52-3.55) | 0.97 | 3.61 → 3.44 |
| dense MNIST, B = 32, 3 epochs | Rust | 3.39 (3.38-3.43) | 2.66 (2.65-2.67) | 0.78 | 2.86 → 2.29 |
| dense MNIST, B = 32, 3 epochs | numpy | 9.35 (9.26-9.83) | 4.04 (4.01-4.10) | 0.43 | 8.27 → 2.59 |

- **Stage 0 predicted the Rust saving.** Two passes at 0.157 s saved predicts 0.31 s for a
  one-epoch run; the B = 32 epoch lost 0.30 s. Four passes predict 0.63 s for three epochs; it
  lost 0.74 s. Rust single-example lost 0.47 s, noisier (its before range is 0.5 s wide).
- **numpy gains most**, 2.4 s of a one-epoch B = 32 run (2.66 predicted) and 5.3 s of three
  epochs, so Rust / numpy at dense B = 32 goes from 0.33 to 0.54: numpy's per-row pass was
  slower, so it lost more.
- **Rust conv is unchanged**, as it should be (it keeps the per-row pass); its B = 32 cell
  moved +5%, inside the 20% noise band. numpy conv gained 4% (12% from the loader), less than
  the 0.21 s stage 0 predicted for B = 32.
- **Left over:** a row-wise crate argmax (about 2.5% of a Rust B = 32 one-epoch run, under
  the bar), and batching Rust conv once candidate 4 makes its `forward_batch` cheaper per example.

The plan and stage 0, as recorded when the candidate was open:

`_training_accuracy` (`train.py`) calls `classify_state` once per training row, n + 1 times
over the training set for n epochs. Candidate 1 (done) removed the row conversion
(`classify_row` on the prepared matrix). What is left of one Rust pass on dense full MNIST
was estimated at about 1.24 - 0.90 = 0.34 s, mostly per-row call overhead
(Python dispatch, one small forward per layer). That could be the largest single cost left in
a dense epoch. The number is inferred from the candidate 1 table, not measured, and it
inherits the doubt about the conversion measured apart (see candidate 1). Candidate: a
`classify_batch` on the array networks that runs `forward_batch` over chunks of rows (from
candidate 1's prepared matrix) and takes the argmax per row, with `_training_accuracy` falling
back to `classify_state` for networks without it (the pure-Python ones and the ensembles). It
is a trainer change, not a kernel change, and must change no training result. Dense batched
forward rows equal the single-example forward exactly ("Kernel invariants"). Conv, pooling,
softmax and dropout (which must keep its inference behaviour) need a test that each batched
prediction equals `classify_state`'s, for every network class. Two caveats:
- **Conv gains nothing yet.** Conv `forward_batch` currently costs 1.5-2.1x per example at N =
  32 what single calls do (candidate 4), so a batched conv accuracy pass would be slower until
  candidate 4 is fixed. Dense networks can go first.
- **Stage 0:** time one accuracy pass per row against batched (chunks of 32 and 512), dense
  full MNIST and MNIST conv, both backends, from pre-converted inputs. Proceed only if it
  saves at least about 10% of an epoch after candidate 1.

**Stage 0 done (2026-09-24): go for dense on both backends and numpy conv, no-go for Rust
conv.** `python scripts/accuracy_pass_timing.py time` (one process per network, backend and
repeat, median of 5; each measure the median of 3 runs in its process; seed-0 weights,
`prepared_mnist` inputs). Seconds for one pass:

| network | backend | per row | batched 32 | batched 512 | epoch B = 32 | epoch single |
| --- | --- | --- | --- | --- | --- | --- |
| dense, 60000 rows | numpy | 1.50 (1.44-1.73) | 0.18 (0.17-0.19) | 0.14 (0.13-0.16) | 3.70 | 11.83 |
| dense, 60000 rows | Rust | 0.37 (0.36-0.46) | 0.21 (0.21-0.23) | 0.29 (0.29-0.32) | 1.18 | 1.92 |
| conv, 2000 rows | numpy | 0.27 (0.26-0.30) | 0.17 (0.16-0.18) | 0.20 (0.20-0.21) | 1.27 | 3.68 |
| conv, 2000 rows | Rust | 0.11 (0.10-0.11) | 0.13 (0.13-0.14) | 0.18 (0.18-0.19) | 0.60 | 0.74 |

The saving at chunk 32 as a share of an epoch (the epochs are the trainers given the prepared
dataset, so they include two passes). A one-epoch run has two passes; a long run about one
per epoch, so its share is one pass's saving over an epoch less one pass:

| network | backend | saved per pass | one-epoch, B = 32 | one-epoch, single | long run, B = 32 | long run, single |
| --- | --- | --- | --- | --- | --- | --- |
| dense | numpy | 1.33 | 72% | 22% | 60% | 13% |
| dense | Rust | 0.16 | 27% | 16% | 19% | 10% |
| conv | numpy | 0.11 | 17% | 6% | 11% | 3% |
| conv | Rust | -0.03 | -9% | -7% | -5% | -4% |

- **Chunk 32, not 512.** Rust at 512 is slower than at 32 (0.29 against 0.21 s; 512 x 784 x 30
  crosses the 8M-flop threading threshold, unexamined); numpy gains a little at 512 (0.14
  against 0.18) but 32 takes 97% of its saving, so one chunk size serves both.
- **The per-row Rust pass is 0.37 s**, near the 0.34 s estimated from candidate 1's table.
- **Rust's argmax is 15% of its batched pass** (0.03 of 0.21 s): the crate's `argmax` takes a
  vector, so the probe converts the output with `tolist` and takes each row's argmax in Python.
  A row-wise crate argmax would save about 2.5% of a one-epoch B = 32 run, under the bar.
- **Rust conv is slower batched**, as caveat 1 predicted, so it keeps the per-row pass until
  candidate 4 is fixed. Its forward-only pass (0.19 s) was even slower than forward plus argmax
  (0.13 s), in every process: see the allocator finding under "Other findings".
- **Predictions:** no batched prediction differed from the per-row one, in any cell (60000
  dense rows, 2000 conv rows, both chunk sizes). But **numpy's batched outputs are not
  bit-identical to its per-row ones**: `X @ W.T` against `W @ x` differs by 1 ULP (max abs
  2.2e-16) in 8273 of 50000 dense outputs and 2863 of 20000 conv outputs. So a numpy argmax
  can flip where two outputs are within an ULP (or a binary output within an ULP of 0.5),
  which could move the pocket-best epoch. Decided (2026-09-24): batch numpy anyway, tested for
  equal predictions, not claimed bit-identical; the suite's pinned results must hold. Rust
  dense rows are exact ("Kernel invariants").

**The dataset as one backend array** (candidate 1, optimization 5; crate #19, #372-#374,
2026-09-24). The trainers used to convert a tuple into an array on every call: each batch in
`learn_batch` and each row in `learn` and in every accuracy-pass `classify_state`. Now an array
network trains from a `PreparedDataset` (`indrajala_ml/prepared_dataset.py`): the training set
as one backend matrix plus its labels. The trainers prepare it once per run from the tuple list,
or take one the caller built (`prepared_mnist` loads MNIST straight into one). The networks read
rows through `learn_row`, `learn_batch_rows` and `classify_row` (numpy row views and fancy
indexing; the crate's new `Array.row` and `Array.take_rows`). `learn`/`learn_batch` now convert,
then call the same step as the row methods. The mini-batch trainer shuffles row indices, which
gives the same permutation for a seed. Training is unchanged bit for bit: every seeded pin
passes as before, and `tests/test_prepared_dataset.py` checks all 22 array network classes
step by step against the tuple path.

Preparing is a one-time cost per run: 0.42-0.43 s in Rust (`Array.from_rows`, which reads
tuples by index; `Array(tuples)` took 1.53 s and a first `from_rows` that iterated each row took
1.38-1.46 s) and 1.42-1.45 s in numpy (`np.array`; `np.fromiter` was 1.29-1.31). From the
MNIST binary, `prepared_mnist` builds no Python floats.

**The A/B.** One epoch through the trainers the demos use, from the tuple list (preparation
included) and from `prepared_mnist`, against the commit before the change (`96a1487`).
`scripts/prepared_dataset_timing.py`, median of 5 (ranges), one process per measurement, the
two sides run one after the other. Seconds:

| config | backend | before | after | after, from the loader | after / before |
| --- | --- | --- | --- | --- | --- |
| dense MNIST, B = 32 | Rust | 3.72 (3.68-3.84) | 1.69 (1.64-1.81) | 1.22 (1.19-1.29) | 0.45 |
| dense MNIST, B = 32 | numpy | 7.92 (7.56-8.33) | 5.37 (5.12-5.58) | 3.87 (3.71-4.19) | 0.68 |
| dense MNIST, single | Rust | 4.60 (4.48-4.80) | 2.57 (2.41-2.68) | 2.03 (1.97-2.35) | 0.56 |
| dense MNIST, single | numpy | 14.93 (14.43-15.64) | 13.76 (12.84-14.00) | 11.91 (11.61-12.40) | 0.92 |
| conv MNIST 2000, B = 32 | Rust | 0.73 (0.70-0.81) | 0.64 (0.64-0.66) | 0.63 (0.61-0.65) | 0.88 |
| conv MNIST 2000, B = 32 | numpy | 1.44 (1.39-1.68) | 1.36 (1.34-1.40) | 1.28 (1.26-1.29) | 0.94 |
| conv MNIST 2000, single | Rust | 0.84 (0.78-0.92) | 0.76 (0.73-0.86) | 0.80 (0.70-0.88) | 0.90 |
| conv MNIST 2000, single | numpy | 3.87 (3.67-4.06) | 3.89 (3.70-4.05) | 3.68 (3.65-4.02) | 1.01 |

- **The 70% estimate held up better than feared.** The Rust B = 32 epoch lost 2.03 s, and
  adding back preparation's 0.43 s gives about 2.46 s of conversion removed, against the
  2.7 s measured apart. The doubt raised below (conversion measured apart plus the Rust ops
  leaving no time for the step loop's Python) was worth about 10%, not most of the stake.
- **Conv gains 12% in Rust at B = 32**, as the 15 µs-a-row estimate predicted. Single-example
  conv is within noise in both backends.
- **numpy single-example gains least** (8%). Its per-row cost is mostly its own per-call work,
  not the conversion.
- **Longer runs gain more.** A one-epoch run pays preparation once for one epoch and two
  accuracy passes. Each further epoch saves its batch conversion and one accuracy pass's row
  conversion, and preparation isn't repeated.
- **Loading straight into the array** saves preparation and more (Rust B = 32 1.22 s against
  1.69). The demos still load tuples; switching one is a caller change, not measured here.
- Rust / numpy at B = 32 goes from 0.47 to 0.31 (0.32 from the loader): numpy converted more
  slowly, so it lost more time to conversion before.

**Before: the stake.** Both backends converted Python tuples to an array on every call (`pa.Array(list(state))` in
`RustArrayNetworkBase._forward`/`learn`, a list of rows in `learn_batch`, `np.array` in the
numpy `ArrayNetworkBase`): 15.3 µs for one 784-pixel MNIST row in Rust, 29.5-32.1 µs in numpy
(re-measured; see "Per-row cost" below). That
didn't change the ratio, but it is a fixed cost neither backend's maths can remove, and a
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

**Stage 0's go/no-go is met (batch-size-scaling study, #367, 2026-09-24).** Dense 784 -> 30 ->
10, one full-MNIST epoch of `train_backprop_network_mini_batch`, median of 5, one process per
measurement, `scripts/batch_size_timing.py time`. Seconds per epoch:

| B | backend | trainer epoch | step loop | one accuracy pass | batch conversion | row conversion (60000 rows) |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | numpy | 8.18 | 2.07 | 2.99 | 1.75 | 1.76 |
| 32 | Rust | 3.78 | 1.28 | 1.26 | 0.91 | 0.90 |
| 128 | numpy | 9.42 | 3.25 | 2.93 | 1.73 | 1.73 |
| 128 | Rust | 4.37 | 1.64 | 1.22 | 0.97 | 0.90 |
| 512 | numpy | 9.41 | 3.34 | 3.11 | 1.83 | 1.77 |
| 512 | Rust | 4.85 | 2.16 | 1.25 | 1.24 | 0.90 |
| 1024 | numpy | 9.54 | 3.21 | 3.07 | 1.90 | 1.77 |
| 1024 | Rust | 4.54 | 1.93 | 1.24 | 1.26 | 0.93 |

"Batch conversion" is `pa.Array`/`np.array` of every batch's list of rows, the first line of
`learn_batch`. "Row conversion" is one array per training row, as each `classify_state` of
an accuracy pass does. Both are measured apart from training, in the same process. A
one-epoch run has two accuracy passes. For Rust at B = 32, conversion comes to 0.91 + 2 x
0.90 = 2.7 s of the 3.78 s epoch: about 70%, against the 10% bar. For numpy at B = 32 it is
5.3 s of 8.18 s. Over a long run (about one pass per epoch) the Rust share is about the same:
0.91 + 0.90 = 1.8 s of about 2.5 s. In the step loop alone, conversion is 57-71% of Rust's
time; the Rust ops themselves are 0.37-0.40 s per epoch at every batch size. The rest of
stage 0's list (single-example, the conv networks, the A/B against pre-converted inputs) was
not measured and isn't needed for the decision. The conv networks' share is unmeasured; at
15 µs a row, the 6000 conversions of a timed MNIST conv epoch would be about 90 ms of 0.74 s
(12%). **Status: go; the largest stake measured in this document.** Most of each accuracy
pass is conversion too (0.90 of 1.24 s in Rust); what is left of the pass is candidate 2.

**The stake is likely smaller than 70%.** At B = 32, the conversion measured apart (0.91 s)
plus the Rust ops (0.37 s) already equal the whole 1.28 s step loop, leaving nothing for
the Python of 1875 steps, which can't be right. The same loop measured 1.59 s in candidate
7's profile run, so the two step-loop figures also disagree by 24%. The likeliest reading is
that converting rows in a tight loop apart from training costs more than it does inside the
loop; that is unmeasured. So the implementation's A/B against pre-converted inputs is the
real stake, not the 70%, and must be reported as such.

Batch conversion also grows with the batch size (Rust 0.91 s at B = 32, 1.24-1.26 s at 512
and 1024), though the rows converted per epoch don't change. It is part of why the Rust step
loop gets slower at larger batches at the same flops per epoch: from B = 32 to 512 the step
loop grows 0.88 s (1.28 to 2.16) and batch conversion 0.33 s, about 40% of it. The Rust ops
stay flat and there are 16 times fewer steps, so the rest is unexplained.
(`demo_batch_size_scaling` prints 1.42 s per epoch at B = 32 and 2.02 s at 1024.)

**Per-row cost, re-measured (2026-09-24).** A loop converting all 60000 training rows
(`to_array(list(state))`, median of 5 loops, one process per cell, two passes each) costs
15.3 µs a row in Rust and 29.5-32.1 µs in numpy, not the 49 and 55 µs this entry quoted
before. That earlier figure is unexplained. It isn't the loader: with freshly boxed floats (the loader before
#365) the same loop costs 17.3 µs in Rust and 31.5-31.6 µs in numpy. #365's shared pixel
floats therefore make Rust conversion about 12% cheaper. Over the roughly 180000 row
conversions of a batch-32 Rust epoch (one batch pass, two accuracy passes) that is about
0.36 s of 3.8 s. **Rust dense-MNIST epoch times from before #365 are not directly comparable
with later ones.** numpy's change is within noise.

**Dense `forward_batch`** (`matmul_nt`, #17), kept as a finding: formerly an open candidate, but
on one thread no gap is left. The 5.6x at batch 64 recorded earlier (numpy and Rust
interleaved) was the interleaving: in separate processes batch 64 was 1.8x numpy (1161-1164 against 625-653 µs), and unthreaded
Rust was linear in the batch at about 2x numpy per row. So the cost was the kernel: every row
of `X` re-read all of `W` (1.4 MB at 32 x 5408, past L2). #17 runs 4 rows of `X` against 2 rows
of `W` at a time, bit-identical. Two focused passes, old and new in separate processes, the
second with the build order reversed (µs, unthreaded):

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
save about 12 ms of 0.75 s. What is left is not the kernel. With default threading, 32 x
5408 is still 1.1-2.2x numpy at batch 32 (470-659 against 294-415 µs), which was put down to
`W` streaming in from L3 once per 4 rows of `X`. But on one thread each (2026-09-24, two
passes) Rust is faster at every op table shape: 479 against 563-575 µs at batch 32,
8421-8477 against 8937-9918 at batch 512, and 1217-1257 against 1271-1308 at 30 x 784, batch
512. So the batch-32 gap is numpy's OpenBLAS threading, as in candidate 6, and at batch 512
numpy's threads beat Rust's (3695-4110 against 4747-4898 µs at 32 x 5408), a threading
question (candidate 9). The idea recorded before, blocking over `k` so a panel of `W` stays
in L2 across all rows of `X` (storing and reloading each pair's 4-lane accumulator between
panels, bit-identical), would speed up a kernel that is already ahead, so it is not a candidate.

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
  close to 16 KB blocks but not better. The thread count wasn't recorded; the same product
  takes 31-32 ms on one thread and 10-11 on 8 today (see "Threading"), so these were presumably
  threaded, and the comparison holds only between the two settings.

## Other findings from the same measurements

- **glibc heap trimming can fault a Rust batch op's buffers back in on every call** (found in
  candidate 2's stage 0, 2026-09-24; not yet a candidate). Chained Rust conv `forward_batch` calls
  over the 2000-row MNIST subset in chunks of 32 took 0.20 s with 27000 minor faults a pass;
  with a `tolist` of each chunk's output between them (the Python objects pin the top of the
  heap) 0.14-0.15 s and 1000-1400 faults; with `MALLOC_TRIM_THRESHOLD_` and
  `MALLOC_MMAP_THRESHOLD_` at 1e9, 0.13-0.14 s and no faults either way. The likely mechanism is
  glibc returning the freed top of the heap (conv `cols` is 1.56 MB at N = 32) and faulting it
  in again on the next allocation. **Training pays it too, but only 3-6% of an epoch**
  (2026-09-24): one Rust mini-batch 32 epoch over the same subset, whole trainer call, each
  run in its own process, 5 runs each, default settings against both thresholds at 1e9:

  | architecture | faults, default | faults, raised | median s, default | median s, raised |
  |---|---|---|---|---|
  | conv | 14860 | 6763 | 0.676 | 0.645 (-4.6%) |
  | conv-pool-conv | 20732 | 7201-7466 | 0.969 | 0.918 (-5.3%) |
  | conv-conv-stride2 | 27661 | 7863-7921 | 0.947 | 0.890 (-6.0%) |

  Run ranges don't overlap at conv-pool-conv or conv-conv-stride2 and barely overlap at conv
  (0.659-0.689 against 0.639-0.669). The saving is about 3-4 µs per fault avoided. Single-example
  epochs aren't affected: 3200-4800 faults either way, and times within noise. So the
  fix (reused buffers in the crate, or the allocator's settings) is worth at most about 6% of a
  conv mini-batch epoch. Setting the thresholds is process-wide and would change numpy's
  allocations as well.

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
  affects benchmarks, the end-to-end demo included (see "Where things stand"). Rust's own
  thread count doesn't protect it (the 1-thread run above was slowed too), so every per-op
  ratio measured interleaved with a threaded numpy call is suspect, whatever the Rust op's
  size.
- **The conv demo's mini-batch runs barely train.** They reach about 10% accuracy at lr 0.5 in
  1-2 epochs, and lr 2, 4 and 8 don't fix every configuration. Their timings are valid; their
  accuracy columns are not informative.
- **On dense full MNIST the trainer's accuracy passes are over half the epoch.** In
  `train_backprop_network_mini_batch` the two `_training_accuracy` passes of a one-epoch run
  (60000 single-example `classify_state` calls each) take 52-67% of a Rust epoch and 62-73% of a
  numpy one, at B = 32 to 1024 (batch-size-scaling study, #367; the table is in candidate 1's
  record under "Completed").
  Over a long run there is about one pass per epoch, so the share is smaller. The share is
  largest at B = 32. About 70% of each pass was converting the row to an array, which candidate 1
  removed (#374). Candidate 2 (#379) batched the rest: the Rust B = 32 epoch lost another 19%,
  numpy's 50%.
- **Dense epochs, Rust / numpy end to end by batch size** (same run, medians of 5): 0.46 at
  B = 32 and 128, 0.52 at 512 and 0.48 at 1024. The step loop alone: 0.62, 0.50, 0.65, 0.60.
- **Loading full MNIST shares one float per pixel value** (#365). `load_mnist_dataset` reuses
  256 float objects instead of boxing 47 million. That took a training-set load from 1.9 GB to
  0.45 GB and from 5.1 s to 2.4 s, with identical values. Without it, 4 sweep workers don't fit
  in this machine's 5 GB. It also makes Rust row conversion about 12% cheaper (candidate 1), so
  Rust dense-MNIST epoch times from before #365 aren't directly comparable with later ones.
