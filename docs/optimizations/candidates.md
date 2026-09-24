# Optimizations: candidates

Part of the optimization docs; the index is [../optimizations.md](../optimizations.md).

Future optimizations to investigate, and to implement if they pay. Ranked by the Rust time each
would save in configurations that are actually trained (the demos' single-example and batch-32
runs; batch 512 and up only in the batch-size-scaling study), then by how well the stake and the
fix are established. Stakes are shares of the epochs in [Current baseline](current-baseline.md).
When a candidate is implemented it moves to [Implemented](implemented.md); when it fails it moves
to [Rejected](rejected.md), with the reason in either case.

## Ranked

### 1. Dense `accumulate_gradient_batch`: fuse the gradient add into the kernel store

`accumulate_gradient_batch` at 32 x 5408, batch 32 builds the `(32, 32) @ (32, 5408)` product
through `matmul_2d`, then `grad_w + update` as a second 1.4 MB pass (140-160 µs), and keeps four
1.4 MB arrays live against a 4 MB L3 (a further 240 µs unattributed, probably that). On one thread
it is the one batch-32 dense op still behind numpy (1.2-1.3x); `downstream_batch`, the same
product without the add, is level or faster since the 2-row tiles (crate #26).

**Stake:** the extra pass alone is about 9-10 ms (1.5%) of the conv mini-batch 32 epoch (63
calls), more if the unattributed time goes with it; plus the 30 x 784 accumulate in dense MNIST
mini-batch.

**Fix:** a `matmul_2d_add` (or a `tiled_row_range` parameter) whose store writes `c + acc`. One
output allocated, no `combine_with_array`. `g + u` with `u` the finished chain is the same single
rounding, so bit-identical. Test: `==` against `grad_w + (delta.T @ X)` from separate crate calls,
at `BIG_SHAPES` and `TILE_WIDTHS`, with `-0.0` and zeros in `grad_w`. Mutation that must fail:
start the chain from `g`.

**Acceptance:** the dense accumulate rows in the baseline, then the old-against-new epoch op
profile; the full suite with no pin changes.

### 2. Dense `accumulate_gradient_batch` at long `k`, and a blocked transpose

At batch 512 (`k` = 512), on one thread, accumulate is 3.0-3.4x numpy at 32 x 5408 and 1.6-2.1x
at 30 x 784. Likely cause: `b`'s `k x 16` panel (64 KB at `k` = 512) no longer fits the 32 KB L1,
so 2-row tiles recover only part of it. Fix: block over `k` (or pack `b`'s panel), storing and
reloading the tile's accumulators between blocks so each output keeps its chain. The conv
accumulate candidate needs the same `k`-blocked `tiled_row_range`.

Separately, Rust's `transpose()` of `delta_batch` feeding this product takes 87-107 µs at batch
512 against numpy's 8-12 µs (a naive loop in `rust/src/array.rs` whose writes stride by `rows`),
about 9% of the 30 x 784 op. A blocked transpose is a copy, so trivially bit-identical, and the
cheaper thing to try first.

**Stake:** only batch 512 and up trains here (the batch-size-scaling study). Accumulate is 11-13%
of the dense step loop at every B, but those calls are threaded (12M+ flops) and already beat
numpy's one-thread time, so the saving is likely under 0.1 s an epoch. **Gate:** a one-thread
`k`-blocking probe must gain enough to leave a threaded saving above 5% of the step loop;
re-profile first.

### 3. Conv accumulate with a large `cols`

`D @ cols` (`(O, N*P) @ (N*P, C*k*k)`) through `matmul_narrow`: at 28x28, `ConvSpec(3, 8)`, N = 512,
one thread, 57-58 ms against 14-15 ms for 512 single calls. Each output row makes one pass over
`k` per column tile, and each pass streams all of `cols` (25 MB, past the L3): about 600 MB per
call. Fix: block over `k` so a slab of `cols` serves every row and column tile before the next;
the per-output order is unchanged, so bit-identical. `tiled_row_range`'s 2-row tiles (crate #26)
would first halve the passes if `matmul_narrow` passed 2-row blocks, where `C*k*k` >= 16 (they
only cover 16-wide column tiles). **Stake is small in trained configurations:** no demo trains conv at N = 512,
and at N = 32 (`cols` 1.56 MB) the op is only 1.1-1.2x its single calls, about 1-1.5% of the
epoch. Worth doing alongside the long-`k` candidate, which needs the same kernel change.

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
