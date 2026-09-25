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
  wherever `C*k*k` >= 16. Untested for the conv shapes (the `matmul_narrow` doc comment says so).
  Works whether or not `cols` fits in a cache.
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
| conv-conv (not trained yet), second conv (26x26x8 in) | 576 | 72 | same | 332 KB | 10.6 MB | 170 MB |

Measured so far: first conv at N = 512, one thread, 57-58 ms against 14-15 ms for 512 single
calls (3.8-4.0x), about 600 MB streamed per call; at N = 32, 1.1-1.2x its single calls, about
1-1.5% of the epoch. The second-conv accumulate has never been timed on its own; conv accumulate
(both layers) is 11% / 17% of the conv-pool-conv single-example / mini-batch 32 epoch.

**Stake in trained configurations is unmeasured.** No demo trains conv at N = 512: the conv
batch-size-scaling study (findings in `batch_size_scaling.py`) found no trained B = 512 conv
workload (the linear rule fails at momentum 0.0 and 0.9, the best capped rate stays 2.5 points
below the batch-32 band); a scratch probe put the threaded stake at about 7-12% of a B = 512
step. At B = 32 the demo's second-conv `cols` (2.2-2.7 MB) fits the L3, so only 2-row blocks
can pay there; the smallest trained workload past the L3 is a conv-conv network (second conv
`cols` 10.6 MB at B = 32), one line in the conv demo's `ARCHITECTURES`. Expect a few percent of
an epoch at best.

Plan (stage 0 first, one PR each, gate before any kernel work):

- **Stage 0 tooling.** `demo_layer_op_timing.py`'s `CONV_SHAPES` hard-codes one input channel and
  stride 1 (`conv_cases` builds `CONV_LAYERS[backend](side, side, 1, CONV_KERNEL_SIZE,
  CONV_CHANNELS)`). Give each shape an input-channel count and a stride (the layer constructors
  already take `input_channels` and `stride`), and add the second-conv shapes 13x13x8,
  26x26x8 stride 2 and 26x26x8. `scripts/focused_benchmark.py` picks the cases up.
- **Stage 0a, 2-row blocks on the demo's shapes.** `focused_benchmark.py --shape <second-conv
  shapes> --op accumulate --batch-sizes 32 512 --rust-threads 1 --passes 2`: batched against 32
  single calls. Then a probe build with `rows_per_block = 2` (and 4) in `matmul_narrow`, timed the
  same way (commit crate changes before switching builds). Refresh the conv-pool-conv and
  conv-conv-stride2 epoch shares with `scripts/epoch_op_profile.py` (the 17% predates the one-pass
  pool downstream; stride2 was never profiled).
- **Stage 0b, `k`-blocking past the L3.** Add `"conv-conv": [ConvSpec(3, 8), ConvSpec(3, 8)]` to
  the conv demo's `ARCHITECTURES`, check it trains (accuracy against the other architectures),
  time its second-conv accumulate batched against 32 single calls, and profile its epoch.
- **Gate** (proposed): implement a fix only if its op is >= 1.3x its single calls and the
  projected saving is >= 3% of a trained mini-batch 32 epoch. Otherwise record the measured
  numbers here and keep this candidate for large-batch use only.
- **Implementation.** 2-row blocks first, measured alone; then `k`-blocking if 0b passed. Crate
  PR with the bitwise test, parent PR bumping `rust/` with end-to-end demo ratios and epoch times
  (separate processes); the candidate moves to Implemented or Rejected.

## Deferred: threading past the threshold

Only the batch-size-scaling study trains products above 8M flops (dense 30 x 784 at B ≥ 512),
where the Rust ops are 0.4 s of a 2 s step loop. Revisit only for a large-batch use case:

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
