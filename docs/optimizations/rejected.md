# Optimizations: rejected

Part of the optimization docs; the index is [../optimizations.md](../optimizations.md).

Ideas that were investigated and discarded, with why, so they aren't proposed again. Each says
what would reopen it, if anything. Crate branches named here are kept in `rust/`.

## Kernels

- **A transposed-left matmul for `accumulate_gradient_batch`** (crate branch `matmul-tn`), to
  skip copying `delta_batch.T`. Bit-identical, and within ±4% at every shape: the copy is batch
  x `M`, small next to the product, and since the transpose is blocked (see
  [Implemented](implemented.md#avoiding-copies-and-passes)) about 2% of the op at batch 512.
- **`k`-blocking the dense `accumulate_gradient_batch` at long `k`** (batch 512 and up, where
  `b`'s `k x 16` panel passes the L1). Failed its gate on the profile alone, with no probe: at B
  = 512 the op is 0.15-0.17 s of a 1.8 s step loop (8-9%), so a threaded saving above 5% needs
  it 2.4x faster, and on one thread the 30 x 784 product is already within 1.1-1.3x of
  OpenBLAS's (1440-1690 against 1264 µs). Even a 2x kernel saves about 4%. Reopen only for a
  use case that trains at large batch.
- **Other conv formulations** (crate branch `conv-forward-formulations-proto`). `W @ colsT` had
  the fastest forward at small `O` but a costlier backward, and lost the training step to
  `matmul_narrow` in 17 of 24 configurations. A direct kernel with no `cols` paid only at large
  `O` and N, and can't serve training, which needs `cols`.
- **`k`-blocking in `matmul_narrow` at small `cols`.** At 13x13x8, N = 32 (`cols` 0.3-2.2 MB) the
  old `k`-blocked `matmul_2d` gave nothing over `matmul_narrow` (+2%, +4%, -1%). It is only a
  candidate where `cols` passes the L3 (see [Candidates](candidates.md#1-conv-accumulate-with-a-large-cols)).
- **Row-block sizes for `matmul_2d` other than 16 KB of `a`.** 1-row blocks were 2-3x slower at
  `k` in the hundreds (`b`'s panel is reloaded per row); one block for all rows was close to 16
  KB but no better.
- **Other `matmul_nt` tile shapes.** 2 x 4, 3 x 3 and 2 x 2 were slower than or tied with 4 x 2.
- **`k`-blocking `matmul_nt`** so a panel of `W` stays in L2 across all rows of `X`. On one
  thread Rust's `forward_batch` is already faster than numpy at every measured shape; the
  remaining gap is numpy's threading, not the kernel.

## Threading

- **The 4M-flop threshold** (the original policy). It threaded the 32 x 5408, batch 32 products,
  which made the MNIST conv mini-batch 32 epoch 12% slower: in training each threaded call pays
  cold-clocked workers (see [Implemented](implemented.md#threading-policy)).
- **A 32-rows-per-thread floor**: made the conv mini-batch 512 epoch 21% slower.
- **Blaming threading for the dense batch gaps.** At batch 8 and 16 the same ops ran on one
  thread and were already 3.5-9.4x numpy; the kernel was the cause, fixed by register tiling.
- **Threading the per-example conv forward product.** At N = 512 whole-batch threading gave
  nothing (35.6-37.8 ms against 34-37 on one thread).
- **Splitting over `k`** (a cross-thread reduction changes the summation order), **threading
  pooling or elementwise ops** (too little work per call), and **releasing the GIL** (the
  trainers are single-threaded Python; there is nothing to overlap with).
- **Splitting by columns** so each thread reads only its panel of `b`: nothing pointed at `b`
  traffic, and a register-only probe showed the same worker slowdown.

## Allocation

- **Reusing output buffers through the Python API.** It would break the immutable `RustArray`
  contract every caller relies on.
- **Uninitialised or unzeroed outputs for the dense short-`k` products.** The real call path
  measured 0 page faults per call and zeroing a 1.4 MB output at 25-27 µs (4-5%); not worth an
  `unsafe` allocation. (Conv forward's zero-fills did matter; see
  [Implemented](implemented.md#conv-forward_batch-one-example-at-a-time).)
- **Raising glibc's allocator thresholds at module init.** Process-wide, so it would change
  numpy's allocations in the same process, and it is worth at most about 6% of a conv mini-batch
  epoch (see [Candidates](candidates.md#leads)).

## Training path

- **Batching Rust conv's accuracy pass** (as the dense networks do). Batched 32, it ties for the
  one-conv network and is 14-20% slower for conv-pool-conv and conv-conv-stride2 (0.186 against
  0.163 s, 0.169 against 0.141), because the batched conv forward writes a whole-batch `cols` that
  inference never reads, and the dense tail doesn't gain from batching either (Rust
  `forward_batch` at 32 x 5408, batch 32 is no cheaper per row than single calls). Reopen with a
  batched forward that skips `cols` (see [Candidates](candidates.md#leads)).
- **A forward-only conv path for evaluation at N = 1** (crate branch `conv-infer-batch`), skipping
  `cols`. At N = 1 `cols` is 48 KB and stays in cache; a whole-network pass was 1-8% slower. That
  evidence doesn't cover N = 32, which is the lead above.
- **Accuracy-pass chunks of 512 rows.** Rust dense is slower at 512 than at 32 (0.29 against 0.21
  s a pass; 512 x 784 x 30 crosses the threading threshold), and numpy gets 97% of its gain at 32.
- **A row-wise crate `argmax` for the batched pass.** 15% of the Rust batched pass, about 2.5% of
  a one-epoch dense B = 32 run: under the 5% bar.
