# Optimizations: lessons

Part of the optimization record; the index is [../optimizations.md](../optimizations.md).

## Other findings from the same measurements

- **Zero-filling a buffer can be what brings it into cache** (candidate 4's stage 0,
  2026-09-24). Dropping the zero-fill of conv `forward_batch`'s buffers but keeping its loops
  made the strided ReLU scatter slower, not the op faster in full: 456-475 against 266-324 µs at
  N = 32, and 13-24 against 7.5 µs at N = 1, where the zeroed 43 KB `A` had been sitting in L1/L2.
  The fill had been paying for the lines' first touch. Removing it paid only once `A` was written
  in order (a gather from the product instead of a scatter into `A`). So when removing a
  zero-fill, check that whatever writes the buffer first writes it sequentially.
- **A bare product's benchmark can miss what the same product costs inside its op.** The conv
  forward's product alone (`--matmul 21632x9x8`) scaled linearly at N = 32 (297-318 µs against 32
  x 9.3), zeroed output and all, yet inside the op, after im2col had streamed 1.56 MB through the
  cache, the same call with its zeroed output took 462-670 µs. Time an op's parts in place (a
  probe build) before ruling one out on a bare benchmark.
- **Timing clocks up between paired processes.** In one `--malloc both` pass the single-example
  conv forward read 26-27 µs in the first process of a pair and 41-43 in the second, whichever
  setting ran second, twice; later passes read 26-28 for both. Take a surprising gap between
  two settings only after the order has been swapped.
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
  affects benchmarks, the end-to-end demo included (see [Where things stand](status.md)). Rust's own
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
  record in [History](history.md#candidate-1-the-dataset-as-one-backend-array)).
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
