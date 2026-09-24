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

## Documents

- [Where things stand](optimizations/status.md): the end-to-end Rust / numpy ratios, the per-op
  table and where a profiled epoch spends its Rust time.
- [Open candidates](optimizations/candidates.md): committed work, ranked, with stage plans.
- [Future work](optimizations/future-work.md): deferred candidates and leads that still need a
  stage 0.
- [The kernels](optimizations/kernels.md): "Threading" (the policy and what it rests on) and
  "Kernel invariants" (each kernel's fixed summation order).
- [Method](optimizations/method.md): "How to measure" and "Rules for an optimization PR".
- [Lessons](optimizations/lessons.md): "Other findings from the same measurements", lessons
  that apply beyond one candidate.
- [History](optimizations/history.md): "Completed", every finished or closed change with its
  measurements, and the candidates as recorded when they were open.

## Done candidates

1. **The dataset as one backend array** (optimization 5): **done** (crate #19, #372-#374,
   2026-09-24). The dense MNIST Rust epoch at B = 32 went from 3.72 to 1.69 s. See
   "The dataset as one backend array" under "Completed". The entry stays here so the
   numbers below keep their meaning.

2. **A batched accuracy pass**: **done** (#378-#379, 2026-09-24). The trainers' accuracy pass
   runs `forward_batch` over 32-row chunks; the dense MNIST epoch at B = 32 went from 1.61 to
   1.31 s in Rust and 4.85 to 2.44 s in numpy. See "A batched accuracy pass" under
   "Completed". Rust conv keeps the per-row pass: after candidate 4 a batched pass still
   measured a tie for conv and slower for the pooled and strided networks (candidate 4's stage 2).

3. **Dense single-example `downstream` through the tiled kernel**: **done** (crate #20,
   2026-09-24). At 32 x 5408 it went from 43.8-48.1 to 26.2-26.4 µs (numpy 29.0-29.3), and the
   MNIST conv single-example Rust epoch lost about 45 ms (6%). See "Dense single-example
   downstream through the tiled kernel" under "Completed".

4. **Conv `forward_batch` one example at a time**: **done** (#383 stage 0, crate #21,
   2026-09-24). At N = 32 it went from 1351-1694 to 920-1010 µs (32 single calls: about 850), at
   N = 512 from 34.0-40.5 to 15.0-15.7 ms, and it saves 40-55 ms (5-6%) of the MNIST conv mini-batch
   32 epochs. See "Conv forward_batch one example at a time" under "Completed". Switching Rust
   conv to candidate 2's batched accuracy pass still doesn't pay (stage 2, closed; see there), so
   it keeps the per-row pass.
