# Optimizations: future work

Part of the optimization record; the index is [../optimizations.md](../optimizations.md).

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

10. **Leads from candidate 4, unranked (each needs a stage 0).**
    - **Batched conv evaluation without `cols`.** Candidate 4's stage 2 found the batched
      accuracy pass still loses in `conv_forward_batch` (15-25% more per example than single
      calls in that pass), which still writes a whole-batch `cols` (1.56 MB at N = 32) that
      evaluation never reads. `conv-infer-batch` (crate `ee432d0`) was closed on N = 1 evidence
      only. The dense tail doesn't gain from batching either (see stage 2), so the stake is the
      pooled and strided networks' 0.023-0.028 s per pass, and it is only worth having if it also
      makes the pass a win there.
    - **Other batch-sized zero-fills that are fully overwritten.** `matmul_narrow`'s output (conv
      downstream's `dcols`, 1.56 MB at N = 32, and the accumulate's product), `deltas_by_position`
      and `deltas_by_channel` (1.38 MB each at N = 32), `matmul_2d`'s output and
      `max_pool_forward_batch`'s `a` and `argmax`. Conv downstream (1.2-1.3x its 32 single calls)
      and accumulate (1.1-1.2x) may share candidate 4's cause. col2im's `dx` does need its zeros
      (a scatter-add). Mind the finding under "Other findings" that the fill can be what brings
      the lines into cache.
