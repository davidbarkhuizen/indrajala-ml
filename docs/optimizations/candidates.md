# Optimizations: candidates

Part of the optimization docs; the index is [../optimizations.md](../optimizations.md).

Future optimizations to investigate, and to implement if they pay. Ranked by the Rust time each
would save in configurations that are actually trained (the demos' single-example and batch-32
runs; batch 512 and up only in the batch-size-scaling study), then by how well the stake and the
fix are established. Stakes are shares of the epochs in [Current baseline](current-baseline.md).
When a candidate is implemented it moves to [Implemented](implemented.md); when it fails it moves
to [Rejected](rejected.md), with the reason in either case.

## Ranked

### 1. Conv accumulate with a large `cols`

`D @ cols` (`(O, N*P) @ (N*P, C*k*k)`) through `matmul_narrow`. `matmul_narrow` passes
`rows_per_block = 1` (a row of `D` is `N*P` doubles, past `matmul_2d`'s 16 KB rule), so
`tiled_row_range`'s 2-row tiles (crate #26) never run. Each of the `O` output rows makes one pass
over all of `k` per column tile (16-wide, then 4-wide, then scalar), and each pass reads that
tile's columns from every row of `cols`.

Two fixes, both bit-identical (every output stays one FMA chain in increasing `k`):

- **2-row blocks** (`rows_per_block` >= 2 from `matmul_narrow`): halves the 16-wide tile passes
  wherever `C*k*k` >= 16. Works whether or not `cols` fits in a cache, but not where training
  threads the op over rows (each thread already has 2 of the 8).
- **`k`-blocking**: a slab of `cols` rows (about 256 rows of 72 doubles, 147 KB, fits L2) serves
  every output row and column tile before the next slab; partial chains are stored to `out` and
  reloaded between slabs (a stored double is exact, so the chain's value is unchanged). Pays only
  where `cols` passes the 4 MB L3: at 13x13x8, N = 32 (`cols` 2.2 MB) it gave nothing
  ([Rejected](rejected.md#kernels)). Add a crate test comparing it bitwise against the unblocked kernel.

Where `cols` is large, per layer (MNIST 28x28, `ConvSpec(3, 8)` layers, 8 output rows):

| layer | P | `C*k*k` | tile passes per row | `cols`, N = 1 | N = 32 | N = 512 |
| --- | --- | --- | --- | --- | --- | --- |
| first conv (every demo network) | 676 | 9 | 2 x 4-wide + 1 scalar | 49 KB | 1.56 MB | 25 MB |
| conv-pool-conv, second conv (13x13x8 in) | 121 | 72 | 4 x 16-wide + 2 x 4-wide | 70 KB | 2.23 MB | 36 MB |
| conv-conv-stride2, second conv (26x26x8 in, stride 2) | 144 | 72 | same | 83 KB | 2.65 MB | 42 MB |
| conv-conv, second conv (26x26x8 in) | 576 | 72 | same | 332 KB | 10.6 MB | 170 MB |

**Stage 0 (measured).** Rust accumulate, `focused_benchmark.py --op accumulate --rust-threads 1`,
batched against its single calls (`demo_layer_op_timing`'s shapes; ranges over 2 passes):

| layer | single µs | N = 32 µs | vs 32 singles | N = 512 ms | vs 512 singles |
| --- | --- | --- | --- | --- | --- |
| first conv 28x28 | 27-28 | 1032-1080 | 1.15-1.24x | 61.8-62.0 | 4.3-4.5x |
| second conv 13x13x8 | 11.6-11.7 | 618-676 | 1.65-1.81x | 46.4-46.8 | 7.8x |
| second conv 26x26x8/2 | 13.3-14.0 | 740-769 | 1.65-1.80x | 57.4-61.3 | 8.2-8.6x |
| second conv 26x26x8 | 67-68 | 9973-10479 | 4.6-4.8x | 223-226 | 6.5x |

Inside the network (each call timed in a real MNIST mini-batch 32 epoch) the second conv runs
slower than isolated: 950 µs a call at 13x13x8 (2.6x its single calls), and 5.3 ms at 26x26x8
(2.5x), where the product (10.6M flops) is past the 8M threading threshold and splits its 8 rows
over threads. Accumulate (both layers) is 17-18% of the Rust mini-batch 32 epoch in all three
networks and 10-12% single-example (`epoch_op_profile.py`, profiled totals).

- **2-row blocks** (probe: `rows_per_block` from an env var in the accumulate, bit-identical at
  1, 2, 4 and 8; 4 best, 8 no better): about 40% off the isolated second-conv accumulate at N = 32
  and 512 (13x13x8 364-370 µs, 26x26x8/2 475-476, 26x26x8 5968-6158 on one thread), 20-30% off
  its single calls, nothing on the first conv. In the network: 13x13x8 950 → 750 µs a call, about
  13 ms of a 0.77 s mini-batch epoch (1.6%) and 0.7% single-example; conv-conv single-example
  about 40 ms of 2.0 s (1.8%). Threaded conv-conv (default threads) gains nothing (5.3-5.5 ms at 1
  and 4): each thread gets 2 rows. Below the gate's 3%.
- **`k`-blocking** would bring conv-conv's threaded 5.3 ms toward its single calls' 2.1 ms: at most
  about 0.2 s of its 2.37 s mini-batch epoch (8.5%), a ceiling, not a projection. Row threading
  makes every thread stream all of `cols`, so it needs the threads split over columns instead.
- **conv-conv trains on UCI digits** (single-example 0.85-0.86) but not reliably on MNIST at the
  demo's lr 0.5 (numpy collapsed to 0.11, Rust reached 0.52, 1-ULP control 15%); it trains at
  lr 0.2 (0.74). It is in the conv demo's `ARCHITECTURES` as the workload past the L3.

No demo trains conv at N = 512: the conv batch-size-scaling study (findings in
`batch_size_scaling.py`) found no trained B = 512 conv workload.

**Decision.** Both fixes fail the proposed gate (>= 1.3x its single calls and >= 3% of a trained
mini-batch 32 epoch) on the saving; the owner chose to implement both. Plan, one PR each:

- **2-row blocks.** `matmul_narrow` with `rows_per_block` > 1 for the accumulate (4 measured
  best; decide whether forward and downstream, with tall `a`, take it too). Crate PR, then a
  parent PR bumping `rust/` with the epoch profile against the old build (builds alternated).
- **`k`-blocking with column-split threading.** Slabs of `cols` rows sized to L2, partial chains
  stored and reloaded; threads over column tiles, not rows, so `cols` is streamed about once.
  Crate test comparing it bitwise against the unblocked kernel; measured on conv-conv at default
  threads, in the network. The candidate then moves to Implemented or Rejected.

## Deferred: threading past the threshold

Only the batch-size-scaling study (dense 30 x 784 at B ≥ 512, where the Rust ops are 0.4 s of a
2 s step loop) and conv-conv's second-conv accumulate at B = 32 (candidate 1) run products above
8M flops. Revisit only for a large-batch use case:

- **A persistent pool**, removing the 60-200 µs spawn cost. Whether warm workers also avoid the
  cold clock is untested; they would idle between calls in training unless they spin. rayon
  (`par_chunks_mut` keeps contiguous row blocks; measure its build time and per-call overhead) or
  a small hand-rolled pool (scoped borrows need `unsafe` lifetime erasure or copies).
- **The caller computes the first block** and spawns one thread fewer: in isolation at 32 x 5408,
  batch 32, 443-584 µs at 4 threads against 729-779 spawn-all. The cheapest threading change
  left, bit-identical, untried end to end.

## Leads

A possible gain whose stake nobody has measured. Each names the stage 0 that would measure it; a
lead that measures a stake worth having joins the ranking.

- **Batched conv evaluation without `cols`.** The batched conv forward writes a whole-batch `cols`
  (1.56 MB at N = 32) that inference never reads, which is why Rust conv's accuracy pass stays
  per row. The dense tail doesn't gain from batching either, so the stake is the pooled and
  strided networks' 0.023-0.028 s per pass, and only if it makes the pass a win there. Stage 0:
  a probe forward that skips `cols`, timed batched against per row on all three demo
  architectures (`scripts/accuracy_pass_timing.py`).
- **Batch-sized zero-fills that are fully overwritten.** `matmul_narrow`'s output (conv
  downstream's `dcols`, the accumulate's product), `deltas_by_position`, `deltas_by_channel`,
  `matmul_2d`'s output and `max_pool_forward_batch`'s `a` and `argmax` (the call's floor with both
  filled is 13 µs, against 102-112 µs for the 2x2 probe at batch 32). Conv downstream (1.2-1.3x
  its 32 single calls) and accumulate (1.1-1.2x) may share conv forward's former cause. col2im's
  `dx` needs its zeros (a scatter-add). Mind the first-touch gotcha. Stage 0: time each op's parts
  in place in a probe build.
- **glibc heap trimming in Rust conv mini-batch training.** Freed batch buffers at the top of the
  heap are returned to the OS and faulted back in on the next call. Raising
  `MALLOC_TRIM_THRESHOLD_` and `MALLOC_MMAP_THRESHOLD_` saved 4.6-6.0% of a conv mini-batch 32
  epoch (14.9-27.7k faults down to 6.8-7.9k, about 3-4 µs per fault avoided; single-example
  unaffected), measured before conv forward stopped allocating batch buffers. Fix would be buffers
  reused inside the crate. Stage 0: re-measure faults and epoch time on the current build.
- **Demos loading straight into the backend array.** Training from `prepared_mnist` skips
  preparation: dense Rust B = 32 epoch 1.22 s against 1.69 from the tuple list. The demos still
  load tuples; a caller change, unmeasured.

## Open questions

Unexplained effects a candidate's stage 0 may need settled:

- **numpy's layer op is faster than its own bare product** on one thread: `downstream_batch` at
  32 x 5408, batch 32, 573-578 µs against 850-897 for `delta_batch @ W` of the same shapes.
- **The Rust dense step loop slows with batch size at equal flops**: B = 32 → 512 added 0.88 s, of
  which batch conversion (since removed) was 0.33 s; the Rust ops stayed flat.
- **How much the conv demo's in-process interleaving slows Rust** (see
  [Measurement](measurement.md#gotchas)), and the end-to-end ratio against single-threaded numpy,
  estimated at 0.55-0.6 for MNIST conv mini-batch 32 but not measured side by side.
