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
