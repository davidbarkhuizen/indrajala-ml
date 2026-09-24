# Optimizations: future work

Part of the optimization record; the index is [../optimizations.md](../optimizations.md).

Work that isn't committed or ranked, in three kinds:

- **Deferred:** understood, but not worth doing now. Each names what would reopen it.
- **Leads:** a possible gain whose stake nobody has measured. Each says where it came from and
  what its stage 0 would measure. A lead gets a candidate number, and moves to
  [Open candidates](candidates.md), when its stage 0 measures a stake worth having; one that
  fails its stage 0 moves to [History](history.md).
- **Open questions:** effects the record calls unmeasured or unexplained, that a measurement
  could settle. They are not work items on their own, but a candidate's stage 0 may need one
  answered.

Ideas considered and ruled out stay as "out of scope" notes beside the design they belong to,
with their reasons, so they aren't proposed again.

## Deferred

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
     the cold-clock finding in [Threading](kernels.md#threading) argues against.
   - **Split by columns** when `b` is the larger operand, so each thread reads only its panel of
     `b`: skipped. Nothing pointed at `b` traffic; a register-only probe (no memory traffic)
     showed the same worker slowdown.

   Out of scope for threading: splitting over `k` (a cross-thread reduction changes summation
   order), threading pooling or elementwise ops, and releasing the GIL (the trainers are
   single-threaded Python, so there is nothing to overlap with).

## Leads

Candidate 10 used to group the first two; the number is retired.

- **Batched conv evaluation without `cols`** (from candidate 4's stage 2). The batched accuracy
  pass still loses in `conv_forward_batch` (15-25% more per example than single calls in that
  pass), which still writes a whole-batch `cols` (1.56 MB at N = 32) that evaluation never
  reads. `conv-infer-batch` (crate `ee432d0`) was closed on N = 1 evidence only. The dense tail
  doesn't gain from batching either (see stage 2), so the stake is the pooled and strided
  networks' 0.023-0.028 s per pass, and it is only worth having if it also makes the pass a win
  there. Stage 0: a probe forward that skips `cols`, timed as a batched pass against the per-row
  pass on all three demo architectures (`scripts/accuracy_pass_timing.py`).
- **Other batch-sized zero-fills that are fully overwritten** (from candidate 4's stage 0).
  `matmul_narrow`'s output (conv downstream's `dcols`, 1.56 MB at N = 32, and the accumulate's
  product), `deltas_by_position` and `deltas_by_channel` (1.38 MB each at N = 32),
  `matmul_2d`'s output and `max_pool_forward_batch`'s `a` and `argmax`. Conv downstream (1.2-1.3x
  its 32 single calls) and accumulate (1.1-1.2x) may share candidate 4's cause. col2im's `dx`
  does need its zeros (a scatter-add). Mind the finding in [Lessons](lessons.md) that the fill
  can be what brings the lines into cache. Stage 0: time each op's parts in place in a probe
  build, as candidate 4's stage 0 did.
- **glibc heap trimming in Rust conv mini-batch training** (from candidate 2's stage 0; the
  table is in [Lessons](lessons.md)). Raising both allocator thresholds saved 4.6-6.0% of a
  conv mini-batch 32 epoch, so reused buffers in the crate (or the allocator's settings, which
  are process-wide and change numpy's allocations too) are worth at most about 6%. That was
  measured before candidate 4 stopped allocating conv forward's batch buffers. Stage 0:
  re-measure the fault and time table on the current build.
- **A row-wise crate `argmax`** (from candidate 2's stage 0). The Rust batched accuracy pass
  converts its output with `tolist` and takes each row's argmax in Python: 15% of the pass,
  about 2.5% of a one-epoch dense B = 32 run, under the 5% bar.
- **Demos loading straight into the backend array** (from candidate 1). Training from
  `prepared_mnist` skips preparation: the dense Rust B = 32 epoch took 1.22 s against 1.69 s
  from the tuple list. The demos still load tuples; switching one is a caller change, not
  measured.

## Open questions

- **The tables in [Where things stand](status.md) predate recent candidates.** The end-to-end
  ratios are from before candidates 1-4, and the epoch profile from before candidates 3 and 4.
  A demo run with more repeats, and `scripts/epoch_op_profile.py`, would refresh them.
- **How much the demo's interleaving slows Rust.** The demo runs numpy then Rust in one process,
  so OpenBLAS's threads may still be spinning when each Rust run starts: at most the first
  100-500 ms of each run, unmeasured.
- **The end-to-end ratio against single-threaded numpy.** For MNIST conv mini-batch 32 it is
  estimated at 0.55-0.6 from runs in different sessions, not measured side by side.
- **numpy's layer op is faster than its own bare product** on one thread: `downstream_batch` at
  32 x 5408, batch 32 took 573-578 µs against 850-897 for `delta_batch @ W` of the same shapes,
  in the same runs (candidate 6's stage 0). No explanation found.
- **Why the Rust dense step loop slows with batch size at equal flops.** From B = 32 to 512 it
  grew 0.88 s, of which batch conversion was 0.33 s; the Rust ops stayed flat and there are 16
  times fewer steps, so the rest is unexplained (batch-size-scaling study, before candidate 1
  removed batch conversion).
- **The Rust accuracy pass at chunk 512.** It was slower than at chunk 32 (0.29 against 0.21 s);
  512 x 784 x 30 crosses the 8M-flop threading threshold, unexamined. Chunk 32 is what the
  trainers use.
