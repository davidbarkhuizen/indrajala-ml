# Workplan: optimization 7, dense batch downstream and accumulate at short `k`

Candidate 1 in [`../optimizations.md`](../optimizations.md). At 32 x 5408, batch 32 these two
ops are the largest per-op gaps left: `downstream_batch` 581-654 µs against numpy's 232-246
(2.4-2.8x), `accumulate_gradient_batch` 976-1180 against 411-453 (2.2-2.9x). Both run on one
thread since #16. The rules in `../optimizations.md` apply: each stage is its own crate PR plus
a submodule bump PR here, measured before and after. A stage that measures no gain is closed
and its numbers recorded.

## Context

Both ops are `matmul_2d` products with `k` equal to the batch or the layer width, 32 here:

| op | crate path (`rust/src/fused.rs`) | product | extra work |
| --- | --- | --- | --- |
| `downstream_batch` | `layer_downstream_batch` → `matmul` | `delta_batch (32, 32) @ W (32, 5408)` | none |
| `accumulate_gradient_batch` | `layer_accumulate_gradient_batch` | `delta_batch.T (32, 32) @ X (32, 5408)` | `transpose()` copy, then `grad_w.combine_with_array(update, g + u)`, then `sum_axis0` for `grad_b` |

Each is 5.5M flops with a 1.4 MB output. `matmul_2d` zeroes `out` (`vec![0.0; m * n]`), picks
`rows_per_block = 16 KB / (k * 8)` (64 at `k` = 32, so all 32 rows in one block), and calls
`tiled_row_range` (`rust/src/linalg.rs`). The AVX2 path there, for each 16-column tile and each
row, runs 4 accumulators over `k` and stores them. `matmul_narrow` (every conv matmul) uses the
same `tiled_row_range` with one row per block.

End-to-end stake: MNIST, one `ConvSpec(3, 8)`, mini-batch 32 (the worst cell, 0.49) makes 62
calls of each per epoch. Closing both to numpy saves about 60 ms of a 0.75 s epoch (about 8%).
The same kernel also runs the 30 x 784 rows of the op table (`downstream_batch` 1.5-1.8x and
`accumulate_gradient_batch` 1.1-1.4x at batch 32, and both at batch 512), so dense MNIST
mini-batch gains too.

**Hypotheses, none measured yet.** Stage 0 decides between them:

1. **FMA latency.** Per row and tile the kernel has 4 independent chains of `k` dependent FMAs.
   Zen 2 has 2 FMA pipes with about 5 cycles of latency, so about 10 chains are needed to keep
   both busy. 4 chains cap the kernel at about 40% of FMA throughput. At 3-3.8 GHz that is
   roughly 230-290 µs for 5.5M flops, so this alone doesn't explain 600 µs, but it caps any
   fix of the others. numpy's 232 µs is about 24 GFLOP/s, which one core can reach only with
   more chains in flight. The fix is the one #17 made for `matmul_nt`: register tiles over
   several rows (for example 2 or 3 rows x 16 columns, 8 or 12 accumulators). Each output keeps
   its FMA chain, so it is bit-identical.
2. **Allocation and page faults.** A 1.4 MB `Vec` is above glibc's default mmap threshold. If
   each call gets a fresh mapping, its 338 pages fault on first touch (inside the kernel, or in
   the zeroing), at a few hundred ns each. glibc raises the threshold dynamically after the
   first such block is freed, so later calls may already reuse heap pages; only a count says.
   Accumulate allocates 1.4 MB twice (the update and the sum). Stage 0 of the threading workplan found first touch made no difference *between
   threads*; whether the faults cost anything at all was never measured.
3. **Accumulate's extra pass** (accumulate only). `combine_with_array` reads `grad_w` and the
   update and writes a third 1.4 MB array. Adding `grad_w` in the tile's store, `out = g +
   acc`, is the same single rounding of `g + u`, so it is bit-identical and removes the update
   buffer and the pass. The `transpose()` copy is 32 x 32, too small to matter (the closed
   `matmul-tn` branch measured removing a larger one at ±4%).
4. **The Python call boundary.** Wrapping the result and the `RustArray` round trip. Probably
   small at 600 µs, but it is subtracted first so the others are measured on the kernel alone.

## Stage 0: decompose the time (a local probe build, no crate PR)

All with the focused benchmark (loops of about 20 ms, median of 9, two passes), numpy and Rust
in separate processes, on current `main`. Use a scratch script, not a demo. At 32 x 5408 and 30
x 784, batch 32 and 512:

1. **The fused op against a bare `Array @ Array`** of the same product. The difference is the
   call boundary and, for accumulate, the transpose, combine and `sum_axis0`. Time those three
   separately too.
2. **Page faults.** `resource.getrusage(RUSAGE_SELF).ru_minflt` before and after each loop,
   divided by calls. `perf` is unavailable (no root), and this needs none. If there are about
   338 faults per output, test the cost directly in the probe build: the same kernel writing
   into a buffer allocated once and reused across calls, against a fresh `Vec` per call.
3. **FMA latency.** In the probe build, a 2-row x 16-column and a 3-row x 16-column variant of
   the tile loop, bit-identity checked against the current kernel with `==`. Also the kernel
   alone at `k` = 32 against `k` = 128 at equal flops (`(32, 128) @ (128, 1352)`): a latency-
   bound kernel gains from a longer `k` only through the tile setup, a memory-bound one from
   less output traffic per flop.
4. **numpy's side.** numpy at `OPENBLAS_NUM_THREADS=1` against its default. If numpy's 232 µs
   is multi-threaded, that changes the target for a single-threaded Rust kernel. The end-to-end
   ratio stays the acceptance number either way.

**Decision recorded at the end of stage 0**, in `../optimizations.md` candidate 1: µs per call
attributed to each hypothesis, and which of stages A-C to run, in order of µs saved. Skip any
stage that stage 0 shows can't save at least about 10% of the op.

## Stage A: multi-row register tiles in `tiled_row_range`

Hold a tile of R rows x 16 columns (4R accumulators) across all of `k`, R chosen from stage 0
(expected 2 or 3; 16 YMM registers bound it at 3 with the `b` loads and the broadcasts). Rows
left over (`rows % R`) run the current 1-row tile. The 4-wide and scalar column tails keep their
loops, with rows blocked the same way only if stage 0 shows it matters.

- **Scope.** `matmul_narrow` calls the same function with one row per block. Measure the conv
  ops too (forward, downstream, accumulate at 28x28, N = 32 and 512). If they don't gain or get
  slower, keep them on the 1-row tile by passing the tile height, the way `rows_per_block` is
  passed now.
- **Bit identity.** Each output is still one FMA chain, `k` increasing from 0.0. Extend
  `test_matrix_at_matrix_is_the_fma_chain_exactly` in the crate's `tests/test_linalg.py` so that
  `m` reaches every row remainder (1 to 2R + 1, not just 1 and 3), at the existing
  `TILE_WIDTHS`. `BIG_SHAPES` and the thread-count tests already cover the large shapes.
  Mutation (with `PYTHONDONTWRITEBYTECODE=1` and `-B`): give the second row of the tile the first
  row's `a` value at one `k`, and drop the last remainder row. Each must fail the tests.

## Stage B: fuse the gradient add into the kernel (accumulate only)

A `matmul_2d_add(a, b, c)` variant, or a `tiled_row_range` parameter, whose store writes `c +
acc` in place of `acc`. `layer_accumulate_gradient_batch` then allocates one output, not two, and
skips `combine_with_array`. Keep `sum_axis0` for `grad_b` as it is.

- **Bit identity.** `grad_w + (delta.T @ X)` today is `g + u` with `u` the finished chain, one
  rounding. The fused store is the same operation on the same values. Add a crate test that
  `layer_accumulate_gradient_batch`'s `grad_w` equals `grad_w + (delta.T @ X)` computed with
  separate crate calls, with `==` on `tolist()`, at `BIG_SHAPES` and the `TILE_WIDTHS` shapes,
  with `grad_w` containing `-0.0` and zeros (the signed-zero case the single-example test
  covers). There is no such exact test today, only an `rtol` one. Mutation: start the chain from
  `g` (a different rounding); the test must fail.

## Stage C: allocation (only if stage 0 finds page faults cost materially)

Options, in order of preference; the stage 0 numbers decide:

- **Don't zero outputs the kernel fully writes.** `matmul_2d` and `matmul_narrow` write every
  output, so `vec![0.0; m * n]` is a wasted pass. Allocate uninitialized and let the kernel's
  stores be the first touch. This needs `unsafe` (`Vec::with_capacity` and `set_len` after the
  write, or `MaybeUninit`) and a `// SAFETY:` argument that every element is written, which the
  existing exact tests check at every tile width. It removes a pass, not the faults.
- **Keep the pages mapped between calls.** For example raise glibc's mmap threshold for the
  process (`mallopt(M_MMAP_THRESHOLD, ...)` at module init) so large outputs come from the heap
  and are reused. It is process-wide and affects numpy in the same process, so measure numpy
  too. Reject it if it needs more than a few lines or a new dependency.
- **Not here:** reusing output buffers across calls through the Python API. It changes the
  immutable `RustArray` contract that every caller relies on.

## Measurement and acceptance for every stage

- **Per op:** the focused benchmark, old and new builds alternated, two passes, at every op
  table row in `../optimizations.md` (all use `matmul_2d` or `matmul_nt`; the `matmul_nt` rows
  must not move), plus the conv ops if `tiled_row_range` changed. Commit the crate change
  first, and switch builds with `git checkout main -- src/linalg.rs`, not a stash.
- **End to end:** one epoch of MNIST, one `ConvSpec(3, 8)`, dense 32, mini-batch 32, lr 0.5,
  the demo's 2000-row subset, from a snapshot of `randomized(...)` after `np.random.seed(0)`,
  `random.seed(0)` before each epoch. Builds alternated with the order swapped every run,
  one process per epoch, medians of 5. The same at mini-batch 512, and one dense MNIST epoch
  at batch 32 and 512. Then `demo_conv_rust_vs_vectorized_digit_recognition` for the ratio
  table. Run them in the background and give an ETA (the conv demo takes about 3 minutes).
- **Bit identity:** the full suite (`./cli build-rust && ./cli test`) passes with no pin
  changes. A pin that moves means the change isn't bit-identical, which is a bug here, since
  every stage is claimed bit-identical.
- **Update `../optimizations.md`:** the op table, the end-to-end table, candidate 1's entry,
  and a row in "Completed" or "Closed with no measured gain" per stage. Delete this workplan
  when the last stage closes, moving its lasting findings into `../optimizations.md`.

## Out of scope

- Threading these products. They run on one thread since #16 for measured reasons (see
  "Threading" in `../optimizations.md`).
- Changing any summation order. Every stage here keeps each output's FMA chain and rounding.
- Blocking over `k`: at `k` = 32 the `b` panel per tile is 4 KB and already stays in L1.
- `matmul_nt` (every `forward_batch`), which is candidate 2.
