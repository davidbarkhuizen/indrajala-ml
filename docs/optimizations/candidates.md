# Optimizations: candidates

Part of the optimization docs; the index is [../optimizations.md](../optimizations.md).

Future optimizations to investigate, and to implement if they pay. Ranked by the Rust time each
would save in configurations that are actually trained (the demos' single-example and batch-32
runs; batch 512 and up only in the batch-size-scaling study), then by how well the stake and the
fix are established. Stakes are shares of the epochs in [Current baseline](current-baseline.md).
When a candidate is implemented it moves to [Implemented](implemented.md); when it fails it moves
to [Rejected](rejected.md), with the reason in either case.

## Ranked

None ranked: each lead below needs a stage 0 first.

## Deferred: threading past the threshold

Only the batch-size-scaling study (dense 30 x 784 at B ≥ 512, where the Rust ops are 0.4 s of a
2 s step loop), conv-conv's second-conv accumulate at B = 32 (threaded over column chunks) and
conv-conv16's second conv and dense tail at B = 32 run products above 8M flops; conv-conv16 is
the batch-32 test case (it gains nothing end to end now; see Open questions). Revisit for a
large-batch use case, or once the effect on unthreaded ops is understood:

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
  a probe forward that skips `cols`, timed batched against per row on every demo
  architecture (`scripts/accuracy_pass_timing.py`).
- **Per-example second-conv downstream.** conv-conv's `conv_downstream_batch` at N = 32 takes
  8.0-8.2 ms a call in isolation against 2.4-3.5 ms for its 32 single calls, and 8 threads on
  its product change nothing; conv-conv16's (16 channels) takes 8.9 ms in training, so the cost
  follows the whole-batch `dcols` (`N*P x C*k*k`, 10.6 MB, past the L3), not `O`. The same
  per-example structure as conv forward would keep each example's `dcols` in L2, and leave
  independent examples to thread. Stake: 19-22% of both networks' mini-batch 32 runs. Stage 0:
  time the product and col2im apart, and a per-example probe, at N = 32.
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

- **Threading slows the unthreaded ops around it.** In conv-conv16 mini-batch 32 training
  (`op_call_timing.py`, `--rust-threads 1` against the default, alternated, 4 processes each) the
  threaded ops gain 0-9% a call (second-conv accumulate 3906-4068 against 4275-4356 µs, dense
  tail forward and accumulate 5-8%, second-conv downstream even), but the unthreaded ones lose
  more: first-conv forward 1199-1222 against 916-930 µs, second-conv forward 5308-5450 against
  4599-4669, first-conv accumulate 1216-1263 against 1008-1026. The run comes out even (2.83-2.99
  s against 2.81-2.90). conv-conv shows it too (first-conv accumulate 1096 against 930 µs). Clock
  (the all-core boost is lower), L3 eviction and scheduling are untested causes. Any batch-32
  threading candidate has to answer it first.

- **numpy's layer op is faster than its own bare product** on one thread: `downstream_batch` at
  32 x 5408, batch 32, 573-578 µs against 850-897 for `delta_batch @ W` of the same shapes.
- **The Rust dense step loop slows with batch size at equal flops**: B = 32 → 512 added 0.88 s, of
  which batch conversion (since removed) was 0.33 s; the Rust ops stayed flat.
- **How much the conv demo's in-process interleaving slows Rust** (see
  [Measurement](measurement.md#gotchas)), and the end-to-end ratio against single-threaded numpy,
  estimated at 0.55-0.6 for MNIST conv mini-batch 32 but not measured side by side.
