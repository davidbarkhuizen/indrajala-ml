# Workplan: optimization 3, conv forward

Order: stages A and B after optimization 1; stage C after optimization 4 (see
`optimizations.md`).

## Context

`conv_forward_batch` is 41% of profiled Rust training time single-example and 51% mini-batch
(MNIST conv-pool-conv, cProfile own time, evaluation passes included). It already beats numpy
(58 vs 81 µs), so the goal is to shrink the Rust share, not to close a gap. Per call it:

1. builds `cols (N·P, C·k·k)` (im2col);
2. computes `cols @ W.T` through `matmul`, with a materialized `W.T` (`O x C·k·k`, tiny);
3. scatters into channel-major `Z (N, O·P)`, adding `b`;
4. builds `A = relu(Z)` as a second output-sized array;
5. returns `(Z, A, cols)`. The Python layer keeps all three and, single-example, reshapes `Z`
   and `A` to 1D (two more copies).

Three independent candidates, in increasing risk order.

## Stage A: stop returning `Z` (bit-identical)

**Done** (indrajala-math-rust#9). `conv_forward_batch` returns `(A, cols)` and applies the ReLU
while it scatters into the output, so it builds one output-sized array instead of two. A new crate
test requires `A` to equal, bit for bit, the old two-pass result rebuilt from its parts: the
crate's own `cols @ W.T`, `+ b`, the channel-major scatter, then `np.maximum`. Swapping in
`matmul_nt` (a different summation order) fails all 14 of its cases. Also, an old build and the
new one gave identical `A` and `cols` over 33 cases: the crate's 7 test shapes and 28x28x1 k3 O8,
8x8x1 k3 O8, 13x13x8 k3 O16 and 26x26x8 k2 O8 s2, each at N = 1, 3 and 32. The ReLU-at-zero
tests now pin `a == [0, 2, 2, 3]` in both backends, plus the exact-zero delta.

Rust µs per call (`ConvRustArrayLayer` methods, 3x3 kernel, 8 channels, median of 9 loops), two
runs per build with the builds alternated:

| shape | op | old | new |
| --- | --- | --- | --- |
| 28x28 | `forward` | 56.8, 57.9 | 55.2, 55.4 |
| 28x28 | `forward_batch`, N = 1 | 53.3, 68.9 | 52.6, 52.0 |
| 28x28 | `forward_batch`, N = 32 | 2793, 2609 | 2703, 2482 |
| 28x28 | `forward_batch`, N = 512 | 47634, 48620 | 40232, 41968 |
| 8x8 | `forward` | 5.6, 5.6 | 5.2, 4.8 |
| 8x8 | `forward_batch`, N = 1 | 4.1, 4.1 | 4.3, 3.9 |
| 8x8 | `forward_batch`, N = 32 | 89.6, 118.4 | 102.7, 91.6 |

A small gain, clearest where the output is largest (N = 512: about 14%). The single-example
`forward` also drops the `Z` reshape (8x8: about 10%). At N = 1 and 32 the change is within the
run-to-run spread.

`Z` exists only to mirror `ConvArrayLayer.Z`. One test reads it: `tests/
test_conv_rust_array_layer.py:149` asserts `layer.z == [0.0, 2.0, 2.0, 3.0]` in the
ReLU-at-zero test. Nothing in the Rust backward path reads `Z`: `compute_hidden_delta` masks on
`A`/`a`.

**Crate PR:**

- `conv_forward_batch` returns `(A, cols)`. It writes `max(z, 0)` straight into one output
  buffer during the scatter, with no separate `Z` vector and no second pass.
- This is a breaking signature change to a `#[pyfunction]`. indrajala-ml is the only consumer,
  and the bump PR changes the one call site, so no compatibility shim.
- Crate test (`rust/tests/test_conv_ops.py`): `A` still matches the brute-force numpy reference
  exactly as before. Also check `A` against the old `relu(Z)` bit for bit: `max(z, 0)` on the
  same `z` is the same operation.

**Here:**

- `ConvRustArrayLayer.forward_batch`/`forward` drop `self.Z`/`self.z`.
- Rewrite the ReLU-at-zero test to assert the convention through what the layer exposes. The
  pre-activation that is exactly 0 gives `a == 0`, and a nonzero upstream delta at that position
  gives `delta == 0` there. That is what the test protects: the derivative at `z == 0`, not the
  stored `Z`. Check the numpy `ConvArrayLayer` version of the test says the same thing, so the two
  backends still test one convention.
- Update the `ConvRustArrayLayer` docstring: it no longer mirrors `ConvArrayLayer.Z`.

**Measure:** conv forward per call, single-example and batch 32, at 28x28 and 8x8.

## Stage B: forward-only path for evaluation (bit-identical)

**Closed, not merged: no measured gain.** Built as planned, kept for the record with no PR:
crate branch `conv-infer-batch` (`ee432d0`) and branch `opt3-stage-b-infer` here (`a14ddcb`).
`conv_infer_batch` fills one reused im2col row per output position and runs `axpy_row` over
`W.T`'s rows, `k` increasing, as `matmul_2d_row_range` does. Its crate test required exact
equality with `conv_forward_batch`'s `A` at the 7 test shapes (N = 1 and 3). It also covered 4
more cases, including one over `matmul`'s threading threshold and one over its blocking threshold.
Starting each sum from `b` instead of adding `b` at the end failed all 18 cases. The layer and
network tests (`infer` == `forward` exactly, a gradient or backward step after `infer` raises,
network `_forward` == the base forward loop exactly, predictions between `learn` steps change no
weight) passed, with 2570 passed in the full suite.

Rust µs per call, both paths in one build, interleaved, median of 3 rounds of 9 loops:

| layer | N | `forward` | `infer` | change |
| --- | --- | --- | --- | --- |
| conv 28x28, 8 channels | 1 | 53.6 | 55.0 | +3% |
| conv 28x28, 8 channels | 32 | 2280 | 1791 | -21% |
| conv 8x8, 8 channels | 1 | 5.0 | 5.0 | -1% |
| conv 8x8, 8 channels | 32 | 88.6 | 91.7 | +3% |
| pool 26x26, 8 channels, /2 | 1 | 18.9 | 18.2 | -4% |

The evaluation pass is single-example (`_forward`, one state at a time), so N = 1 is the case that
matters. A whole-network evaluation pass (MNIST shapes, dense 32, 200 states, median of 7,
interleaved, two runs) was slower with `infer`: conv +6.3%/+8.2%, conv-pool-conv +1.0%/+4.1%,
conv-conv-stride2 +4.1%/+2.3%. So the end-to-end demo was not run.

At N = 1 the skipped `cols` is 676 x 9 values, 48 KB, cheap to allocate and fill. The op's time
is its 6084 `axpy_row` calls on 8-wide rows (676 positions x 9 kernel values), about 8 ns each,
and `infer` makes the same calls. That per-row overhead is what stage C addresses. The one gain,
`infer_batch` at 28x28, N = 32 (-21%), is on a path nothing evaluates with. It does point at
something: batched `forward_batch` at N = 32 costs more per example than 32 single-example calls
(2280 vs 32 x 53.6 = 1715 µs), likely because its 1.5 MB `cols` falls out of cache. That's for
stage C's measurement to take into account, not measured here.

The trainers' per-epoch accuracy passes (`classify_state`/`predict_probabilities` via
`RustArrayNetworkBase._forward`) call `layer.forward`, which builds and keeps `cols` that
nothing reads.

**Crate PR:** `conv_infer_batch(W, X, b, geometry) -> A`. It skips storing `cols`: for each
output position it fills one reusable `C·k·k` scratch row and computes the `O` outputs straight
from it. For bit-identity, the per-output reduction must use the same order and FMA as
`matmul_2d_row_range` applied to that row. The simplest correct route is to call the same inner
`axpy_row` loop over a one-row `cols` slice, not to write a new reduction. Crate test: exact
equality with `conv_forward_batch`'s `A` over the full `tests/test_conv_ops.py` shape list.

**Here:**

- `ConvRustArrayLayer.infer(x)`/`infer_batch(X)` return `A` and set `self._cols = None`, so
  calling `accumulate_gradient` after an inference pass fails loudly instead of using stale
  columns. The same applies to `MaxPoolRustArrayLayer` (argmax is also backward-only). Pool
  inference can reuse `max_pool_forward_batch` and drop the argmax, unless a measurement shows
  that's worth a dedicated op.
- `ConvRustArrayMultiClassBackpropClassifierNetwork` overrides `_forward` to call `infer` on
  front-end layers and `forward` on dense ones. `learn`/`learn_batch` are untouched, so training
  keeps caching. This uses a separate method, not a training-mode flag, because the existing
  `_set_training_mode` hook is only set during `learn`. Making layer caching depend on it would
  make layer-level tests (forward then accumulate, no network) depend on network state.
- Tests: `infer` == `forward` exactly, for conv and pool layers; `accumulate_gradient` after
  `infer` raises; predictions of the conv network are unchanged, exactly, before and after
  (fixed weights, UCI rows).

**Measure:** end-to-end demo ratios. The per-op gain is the `cols` allocation, and the network
gain is proportional to how much of training is evaluation.

## Stage C: a formulation for small output channel counts (measure first)

**Done** (indrajala-math-rust#11): C2 with `cols` kept. im2col stays; `cols @ W.T` goes through a
new `linalg::matmul_narrow`, which keeps each output row's `O` running sums in AVX2 registers
across all of `k` (blocks of 16 columns, then 4, then a scalar tail) instead of one `axpy_row`
load/FMA/store pass per `k`. Every output is the same FMA chain, so it is bit-identical, and
`cols`, the backward ops and the Python layer are unchanged.

**Measurement step.** Prototypes on crate branch `conv-forward-formulations-proto` (`ef85831`,
kept for the record, no PR):
- C1: `W @ colsT`, as planned.
- C2: this kernel, with `cols` kept.
- C2-direct: the kernel with no `cols` (one scratch row per position).
- Both C1 backward options: transpose `colsT` back, then the existing `D @ cols`; or `(colsT @
  D_by_position).T`, which keeps every gradient element's FMA chain.

All were exactly equal to the current ops (`A`, `cols`/`colsT`, `grad_W`, `grad_b`) over 176
cases x 7 checks: 6 shapes including multi-channel and strided ones, `O` in {1, 3, 4, 6, 8, 16,
20, 32}, and N up to 130 (matmul's threaded and blocked paths). So the choice is on speed
alone. Rust µs per call, median of 7 loops, one build. A step is forward + accumulate, and C1's
step uses its faster backward:

| shape | O | N | forward now | C1 | C2 | C2-direct | accumulate now | C1: transpose back | C1: (colsT @ D).T | step now | step C1 (best backward) | step C2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 28x28x1 | 4 | 1 | 49.9 | 13.2 | 17.6 | 22.5 | 18.6 | 24.3 | 32.7 | 68.5 | 37.5 | 36.2 |
| 28x28x1 | 4 | 32 | 1687.6 | 488.8 | 566.9 | 703.0 | 661.1 | 944.9 | 1033.3 | 2348.7 | 1433.7 | 1228.0 |
| 28x28x1 | 8 | 1 | 66.5 | 19.6 | 22.6 | 28.1 | 36.2 | 42.0 | 40.6 | 102.7 | 60.2 | 58.7 |
| 28x28x1 | 8 | 32 | 1935.7 | 1135.8 | 1002.8 | 961.0 | 1319.0 | 1807.6 | 1468.0 | 3254.6 | 2603.8 | 2321.7 |
| 28x28x1 | 16 | 1 | 65.6 | 32.5 | 29.3 | 34.6 | 71.8 | 79.9 | 59.8 | 137.4 | 92.3 | 101.1 |
| 28x28x1 | 16 | 32 | 3633.5 | 3603.8 | 2603.2 | 2221.6 | 2883.7 | 3699.4 | 2791.7 | 6517.2 | 6395.4 | 5486.9 |
| 28x28x1 | 32 | 1 | 101.5 | 59.5 | 48.0 | 52.1 | 143.4 | 148.8 | 102.1 | 244.9 | 161.6 | 191.4 |
| 28x28x1 | 32 | 32 | 7567.3 | 6259.9 | 6894.0 | 5719.4 | 6342.3 | 6639.1 | 7911.7 | 13909.6 | 12899.1 | 13236.3 |
| 8x8x1 | 4 | 1 | 3.6 | 1.7 | 1.9 | 2.7 | 1.7 | 2.1 | 2.8 | 5.3 | 3.8 | 3.6 |
| 8x8x1 | 4 | 32 | 84.6 | 23.6 | 30.5 | 39.6 | 31.5 | 42.9 | 58.9 | 116.0 | 66.5 | 62.0 |
| 8x8x1 | 8 | 1 | 3.6 | 2.2 | 2.2 | 2.3 | 2.7 | 3.0 | 3.1 | 6.3 | 5.2 | 4.8 |
| 8x8x1 | 8 | 32 | 88.4 | 36.4 | 41.1 | 50.5 | 64.8 | 74.1 | 72.3 | 153.2 | 108.7 | 105.9 |
| 8x8x1 | 16 | 1 | 4.4 | 3.2 | 2.5 | 2.8 | 4.7 | 5.1 | 4.2 | 9.1 | 7.4 | 7.2 |
| 8x8x1 | 16 | 32 | 115.1 | 58.4 | 52.3 | 60.4 | 125.2 | 136.4 | 104.9 | 240.3 | 163.3 | 177.5 |
| 8x8x1 | 32 | 1 | 6.5 | 5.1 | 3.7 | 4.4 | 8.8 | 9.0 | 6.5 | 15.3 | 11.7 | 12.5 |
| 8x8x1 | 32 | 32 | 186.8 | 106.3 | 84.8 | 92.6 | 255.6 | 268.5 | 180.8 | 442.4 | 287.1 | 340.4 |
| 13x13x8 | 4 | 1 | 52.9 | 19.3 | 21.7 | 22.9 | 9.9 | 16.3 | 41.9 | 62.8 | 35.5 | 31.6 |
| 13x13x8 | 4 | 32 | 1725.0 | 679.0 | 705.1 | 712.7 | 311.8 | 792.8 | 1318.9 | 2036.8 | 1471.8 | 1016.9 |
| 13x13x8 | 8 | 1 | 55.9 | 28.7 | 31.6 | 33.0 | 19.2 | 25.2 | 45.7 | 75.1 | 53.9 | 50.8 |
| 13x13x8 | 8 | 32 | 1827.8 | 928.0 | 997.5 | 1016.4 | 614.8 | 1121.7 | 1418.8 | 2442.6 | 2049.7 | 1612.3 |
| 13x13x8 | 16 | 1 | 70.1 | 46.9 | 26.9 | 26.4 | 37.5 | 44.0 | 60.4 | 107.5 | 91.0 | 64.4 |
| 13x13x8 | 16 | 32 | 2074.8 | 1686.3 | 1309.2 | 1036.4 | 1165.4 | 1977.4 | 1604.4 | 3240.2 | 3290.7 | 2474.6 |
| 13x13x8 | 32 | 1 | 106.6 | 87.8 | 43.3 | 43.2 | 79.2 | 84.9 | 99.0 | 185.8 | 172.6 | 122.5 |
| 13x13x8 | 32 | 32 | 3293.0 | 2939.9 | 1895.7 | 1769.7 | 2088.8 | 3283.1 | 2664.9 | 5381.8 | 5604.8 | 3984.5 |

- C2 beats the current step at all 24 configurations.
- C1 has the fastest forward at small `O`. Transposing back is always slower than the current
  accumulate, and `(colsT @ D).T` is faster only at `O` ≥ 16. C1 beats C2 on the step in 7
  configurations, all at `O` ≥ 16, and it would change the cached layout.
- C2-direct (no `cols`) only pays at large `O` and N, and it can't serve training.

So C2 goes in for every `O`, with no dispatch. It never lost to the current op.

**Tests.** The stage A exact test rebuilds `A` with `Array @` (plain `matmul`), so it is now
`matmul_narrow`'s exact check against `matmul`. It gained 9 cases:
- output widths 1, 5, 16, 21 and 35, which cover every kernel path;
- 28x28 at N = 128, over the threading threshold;
- fan_in 800 x 48, over matmul's blocking threshold;
- 13x13x8 x 32 at N = 32.

Reversing the `k` order in the 16-block, the 4-block or the tail fails 5, 6 and 16 cases.
The full suite passed with the Rust conv pins unchanged (2550 passed).

**Before and after** (`conv_forward_batch`, µs, two runs per build, builds alternated):

| shape | O | N | old | new | change |
| --- | --- | --- | --- | --- | --- |
| 28x28x1 | 4 | 1 | 54.5, 55.1 | 20.5, 18.9 | -64% |
| 28x28x1 | 4 | 32 | 1540.5, 1586.1 | 892.5, 603.3 | -52% |
| 28x28x1 | 8 | 1 | 51.2, 52.8 | 24.2, 24.4 | -53% |
| 28x28x1 | 8 | 32 | 4746.5, 4494.0 | 3683.8, 3330.1 | -24% |
| 28x28x1 | 16 | 1 | 65.7, 65.2 | 32.0, 32.8 | -50% |
| 28x28x1 | 16 | 32 | 7629.8, 7407.2 | 6110.1, 6153.8 | -18% |
| 28x28x1 | 32 | 1 | 165.2, 102.6 | 119.2, 60.9 | -33% |
| 28x28x1 | 32 | 32 | 12457.2, 12944.7 | 9719.4, 10324.3 | -21% |
| 8x8x1 | 4 | 1 | 3.5, 3.6 | 1.9, 1.9 | -46% |
| 8x8x1 | 4 | 32 | 81.4, 84.0 | 30.9, 30.8 | -63% |
| 8x8x1 | 8 | 1 | 3.7, 3.7 | 2.3, 2.2 | -39% |
| 8x8x1 | 8 | 32 | 90.3, 91.6 | 42.9, 70.9 | -37% |
| 8x8x1 | 16 | 1 | 4.4, 4.4 | 2.7, 2.6 | -40% |
| 8x8x1 | 16 | 32 | 110.7, 113.6 | 113.9, 58.9 | -23% |
| 8x8x1 | 32 | 1 | 6.5, 6.6 | 4.2, 4.2 | -36% |
| 8x8x1 | 32 | 32 | 176.4, 179.0 | 207.5, 103.2 | -13% |
| 13x13x8 | 4 | 1 | 50.0, 51.6 | 22.3, 21.6 | -57% |
| 13x13x8 | 4 | 32 | 1732.7, 1631.5 | 683.0, 721.1 | -58% |
| 13x13x8 | 8 | 1 | 57.0, 57.2 | 31.8, 31.9 | -44% |
| 13x13x8 | 8 | 32 | 1795.0, 1942.1 | 1013.5, 1030.1 | -45% |
| 13x13x8 | 16 | 1 | 69.9, 70.4 | 27.1, 26.8 | -62% |
| 13x13x8 | 16 | 32 | 2053.9, 1690.8 | 1558.8, 1277.2 | -24% |
| 13x13x8 | 32 | 1 | 105.2, 106.7 | 46.6, 45.5 | -57% |
| 13x13x8 | 32 | 32 | 3120.4, 3236.1 | 1807.5, 1954.7 | -41% |

At N = 1 the new op takes 33-64% less time, and at N = 32 13-63% less. The large-N rows vary
more between runs, but every one improved. End to end, Rust/numpy (before is #335; test
accuracies and agreement columns are unchanged):

| case | before | after |
| --- | --- | --- |
| UCI conv, single / mini-batch | 0.16 / 0.25 | 0.15 / 0.22 |
| UCI conv-pool-conv, single / mini-batch | 0.08 / 0.11 | 0.08 / 0.09 |
| UCI conv-conv-stride2, single / mini-batch | 0.11 / 0.17 | 0.10 / 0.13 |
| MNIST conv, single / mini-batch | 0.31 / 0.87 | **0.26 / 0.67** |
| MNIST conv-pool-conv, single / mini-batch | 0.44 / 0.63 | **0.32 / 0.46** |
| MNIST conv-conv-stride2, single / mini-batch | 0.55 / 0.70 | **0.38 / 0.52** |

`conv_forward_batch` fell from 51% to 36% of profiled Rust conv-pool-conv time
(single-example), and from 51% to 38% (mini-batch).

**Found along the way, not acted on.** The same narrow-row pattern is in the other two conv
matmuls:
- `conv_accumulate_gradient_batch`'s `D @ cols` has output rows `fan_in` wide (9 for the first
  layer).
- `conv_downstream_batch`'s `D_by_position @ W` has output rows `fan_in` wide too.

At 28x28, O = 8, N = 1, accumulate now costs more than forward (36 vs 23 µs in the prototype
run). `matmul_narrow` would apply to both unchanged and keep their bits. This is an estimate, not
measured.

With `O = 8`, `cols @ W.T` runs `axpy_row` on 8-wide output rows, two AVX2 lanes. Loop overhead
is a large share. Two candidate formulations:

- **C1, `W @ colsT`:** build im2col transposed, `colsT (C·k·k, N·P)`, and compute `(O, N·P)` =
  `W @ colsT` with the existing `matmul`: `O` output rows of length `N·P`, long SIMD rows.
  **Summation order:** `out[o, np] = Σ_k W[o, k] · colsT[k, np]`, sequential over `k` with FMA.
  That's the same per-element order and rounding as the current `cols @ W.T`, so the forward
  result should be bit-identical. The crate test must confirm it. For `N = 1`, `(O, P)` is
  already channel-major, so the scatter disappears. For `N > 1` it becomes an `(O, N, P) -> (N,
  O, P)` permute that adds `b` and the ReLU.
  **Backward:** `conv_accumulate_gradient_batch` needs `D (O, N·P) @ cols`. With `colsT`
  cached, that's `D @ colsT.T`: either transpose `colsT` there (a copy, but only in training),
  or use optimization 1 stage C's `matmul_nt`, which changes the gradient's summation order. The
  stage C PR measures the first and only considers the second under the bit-changing protocol.
- **C2, a direct kernel:** for small `O` (≤ 16), a loop over output positions holding the `O`
  rows of `W` hot, with no im2col for the forward pass. It has to use the same FMA order to be
  bit-identical, and that's checked, not assumed.

**Steps:**

1. **Measurement first, no PR.** Prototype C1 and C2 as extra crate functions behind a local
   branch. Time them against the stage A `conv_forward_batch` at `O` = 4, 8, 16, 32, on 28x28
   and 8x8, with `N` = 1 and 32. Also time the backward-side cost C1 adds (the transposed cache).
   Record everything in `recommended-optimizations.md` under item 3, including a formulation
   that loses.
2. If one formulation wins on the whole training step (forward + accumulate), not just the
   forward op: **crate PR** replacing the internals of `conv_forward_batch` (stage B's
   `conv_infer_batch` was not merged), possibly dispatching on `O` if the crossover is clear in the data. Tests:
   exact equality with the previous op wherever the plan claims bit-identity, and the
   brute-force reference as before.
3. **Here:** bump, full suite, end-to-end demo. No Python change is expected unless the cached
   layout changes. `_cols` is opaque to Python, and only `conv_accumulate_gradient_batch` reads it.

If nothing wins, close stage C with the numbers. im2col-then-matmul stays, and the "is it the
right formulation" question in `rust-cnn.md` gets a measured answer.

## Out of scope

- Threading im2col itself, or the conv op across examples. The matmul inside already threads
  above its flop threshold. Revisit only if stage C's profile shows im2col dominating at
  mini-batch sizes.
- Any change to numpy's `ConvArrayLayer`. It is the parity reference.
