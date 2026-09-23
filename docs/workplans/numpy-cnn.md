# Workplan: vectorized convolutional networks (numpy implementation)

## Context

Convolution and max pooling exist only in the pure-Python, one-object-per-node implementation
(`ConvLayer`, `MaxPoolLayer`, `ConvMultiClassBackpropClassifierNetwork`; see #308-#315). This
plan adds the numpy counterpart, parity-checked step by step against the pure-Python classes the
same way `VectorizedMultiClassBackpropClassifierNetwork` was (#165-#171).

Scope decisions:

- **numpy only.** The Rust implementation is planned separately. The layouts chosen here
  (kernel matrix shape, channel-major flattening, im2col column order) are chosen so the Rust
  classes can later mirror them and be parity-tested against numpy, as the dense Rust classes
  are today.
- **No performance measurement in this plan.** Timing waits until the Rust CNN exists; then
  numpy and Rust are timed against each other. The pure-Python implementation is never timed
  (README, Models section). It is the correctness and parity reference only.
- **Mirror the pure-Python feature set exactly:** 'valid' padding, stride, multi-channel input,
  ReLU conv layers, max pooling, a sigmoid dense tail and a one-vs-rest sigmoid output layer,
  plain SGD. No momentum/Adam/softmax/dropout conv variants (none exist in pure Python either).

## The existing numpy machinery, and what it assumes

`ArrayNetworkBase` (`array_network_base.py`) drives every numpy network generically over
`self.layers`: `forward`/`forward_batch`, then `compute_output_delta*` on the output layer, then
`layers[i].compute_hidden_delta*(layers[i + 1])` in reverse, then
`accumulate_gradient*(input_activation)` + `apply_accumulated_gradient(lr, batch_size)` per
layer. `train_linear_classifier_network` calls `learn()` (the single-example path), and the
mini-batch trainer calls `learn_batch()`, so **both paths are real and both need parity**.

Two assumptions break for conv/pool layers:

1. **Backprop to the previous layer reads `next_layer.W` directly.** `compute_hidden_delta`
   computes `next_layer.W.T @ next_layer.delta` and `compute_hidden_delta_batch` computes
   `next_layer.delta_batch @ next_layer.W`. That's 8 sites in 3 numpy files (`array_layer.py`,
   `relu_array_layer.py`, `dropout_array_layer.py`). It's correct only when the next layer is
   dense. The old multi-layer-CNN workplan counted 18 sites across 7 files, but that included
   the Rust layers, which are out of scope here. A dense layer never precedes a conv/pool layer,
   so none of these sites is actually wrong today. What's missing is a hook a conv/pool layer
   can call on *its* next layer, which may itself be conv or pool.
2. **Every layer has a dense `W`/`b`.** `ArrayNetworkBase.randomize`/`snapshot`/`restore`
   assume `(layer.W, layer.b)` with `W.shape == (size, previous_size)`. A conv layer's `W` is
   `(channel_count, fan_in)`, and a pool layer has no weights.

## Layout conventions (fixed here, mirrored by Rust later)

- **Activations** stay flat at the layer boundary: `(N, C·H·W)`, channel-major, flat index
  `c·H·W + r·W + col`. This matches `ConvLayer.nodes`/`MaxPoolLayer.nodes`, so a dense layer
  after the conv front end has the *same* weight matrix in both implementations. Conv/pool
  layers reshape to `(N, C, H, W)` internally.
- **Conv kernel matrix** `W.shape == (channel_count, in_channels·k·k)`, each row in
  (channel, kernel row, kernel col) order. That is exactly `ConvKernel.weights` for that output
  channel, so parity weight injection is `W[c] = kernel.weights`. `b.shape == (channel_count,)`.
- **im2col columns** `cols.shape == (N, P, in_channels·k·k)`, `P = out_h·out_w` in row-major
  output-position order, column order matching the `W` rows above.
- **Pool window slots** are numbered row-major `(pr, pc)`, matching
  `MaxPoolLayer._window_indices`. This matters for tie-breaking (below).

## Stage 1: a `downstream` hook on `ArrayLayer`

The numpy counterpart of pure-Python `BackpropLayer.downstream_sum`: the layer that *owns* the
weights says what gradient flows back to its input, instead of the upstream layer reaching
into `next_layer.W`.

- `ArrayLayer.downstream() -> self.W.T @ self.delta` and
  `ArrayLayer.downstream_batch() -> self.delta_batch @ self.W`, pure extractions of the
  existing expressions.
- Rewrite the 8 sites in `array_layer.py`/`relu_array_layer.py`/`dropout_array_layer.py` to call
  `next_layer.downstream()`/`downstream_batch()`. It's the same expression, so the result is
  bit-identical. One formula, and the conv/pool layers plug in without special cases.
  Rust layers are untouched (separate plan).
- Tests: the full existing numpy suite must pass unchanged (every numpy sibling's
  step-by-step parity test already pins this). Add a direct `downstream`/`downstream_batch`
  test against the hand-written expression.

## Stage 2: `ConvArrayLayer` (`conv_array_layer.py`)

A standalone class, not an `ArrayLayer` subclass. `ArrayLayer`'s `size`/`input_size` define its
`W` shape and gradient-accumulator shape, but a conv layer's `size` must mean its *flattened
output count* (`channel_count·out_h·out_w`, what the next layer chains from) while its `W` is
`(channel_count, fan_in)`. This mirrors `ConvLayer` not subclassing `BackpropLayer`. It
implements the same duck-typed surface `ArrayNetworkBase` drives.

Constructor: `(input_height, input_width, input_channels, kernel_size, channel_count, stride=1)`,
with the same assertions as `ConvLayer` and `out_h = (H - k) // stride + 1` (likewise `out_w`).

**Forward (batch path is primary):**

1. `X.reshape(N, C, H, W)` → `np.lib.stride_tricks.sliding_window_view(axis=(2, 3))`
   → `[:, :, ::s, ::s]` → transpose to `(N, out_h, out_w, C, k, k)` → reshape to
   `cols (N, P, C·k·k)` (this copy is the materialized im2col), cached for the backward pass.
   `sliding_window_view` needs numpy ≥ 1.20. The installed version is 1.21.5, and stage 2 pins
   the lower bound in `pyproject.toml`.
2. `Z = cols @ W.T + b` → `(N, P, channel_count)` → transpose `(0, 2, 1)` → reshape
   `(N, channel_count·P)` (channel-major). `A = np.maximum(0, Z)`.

**Backward:**

- `compute_hidden_delta_batch(next_layer)`:
  `delta_batch = next_layer.downstream_batch() * (A > 0)`. This is `relu_delta`'s own
  convention (derivative 0 at exactly z == 0, read from the activation).
- `compute_output_delta*` raise `NotImplementedError`, as `ReLUArrayLayer`/`ConvUnit` do.
- `accumulate_gradient_batch(_input)`: with `D = delta_batch.reshape(N, channel_count, P)`,
  `grad_W += einsum('nop,npk->ok', D, cols)` and `grad_b += D.sum(axis=(0, 2))`. Spatial
  positions are summed and batch rows are summed here, then averaged by `batch_size` in
  `apply_accumulated_gradient`, the same composition `ConvKernel` documents. It uses the
  `cols` cached by forward rather than rebuilding them from `_input` (forward always precedes
  it in `ArrayNetworkBase`), and says so in a comment.
- `downstream_batch()` (col2im, the vectorized form of `ConvLayer._fan_out`):
  `dcols = D.transpose(0, 2, 1) @ W` → `(N, P, C·k·k)` → reshape `(N, out_h, out_w, C, k, k)`.
  Then scatter-add into `dX (N, C, H, W)` with one strided slice-add per kernel offset:
  `dX[:, :, kr : kr + s·(out_h-1) + 1 : s, kc : ... : s] += dcols[..., kr, kc]` (transposed to
  `(N, C, out_h, out_w)`). That's `k²` vectorized adds, which handle overlapping receptive
  fields exactly (no `np.add.at`). Reshape to `(N, C·H·W)`.
- `apply_accumulated_gradient(lr, batch_size)`: the same formula as `ArrayLayer`'s.

**Single-example path:** `forward(x)`, `compute_hidden_delta(next_layer)` (via
`next_layer.downstream()`), `accumulate_gradient`, and `downstream()` are thin `N = 1`
wrappers over the batch path, keeping `self.a`/`self.delta` in sync with `self.A`/
`self.delta_batch`. That keeps one implementation of the maths. Any cost of this choice is
a question for the later numpy-vs-Rust timing work, not this plan.

**Tests (`tests/test_conv_array_layer.py`):** build a pure-Python `ConvLayer` over a
`StateLayer`, inject identical kernels (`kernel.weights = list(W[c])`), and check:

- forward parity, single and batch, over 1 and 3 input channels, stride 1 and 2, and a
  non-square input (`H != W`);
- batch-row independence: row `i` of `forward_batch` equals `forward` on example `i`;
- `downstream`/`downstream_batch` against `ConvLayer.downstream_sum(i)` for every input index,
  including stride > 1, where some inputs are read by no receptive field (gradient exactly 0);
- gradient-accumulation parity against `ConvKernel`'s accumulators after a multi-example batch;
- the ReLU-at-zero convention (a hand-built `z == 0` position gets zero delta);
- an **independent** finite-difference gradient check on the numpy layer alone (every kernel
  weight and bias, and the input gradient), so correctness doesn't rest only on agreeing with
  pure Python.

## Stage 3: `MaxPoolArrayLayer` (`max_pool_array_layer.py`)

Weight-free, with the same constructor shape as `MaxPoolLayer`
(`stride` defaults to `pool_size`, `channel_count == input_channels`).

- Forward: `(N, C, H, W)` → `sliding_window_view` → `[:, :, ::s, ::s]` → reshape
  `(N, C, out_h, out_w, p·p)`. `argmax(axis=-1)` is cached, and `A` comes from
  `take_along_axis`, flattened channel-major.
- **Ties:** `np.argmax` returns the first occurrence, and `PoolUnit` uses `values.index(max)`,
  also the first. Both number slots row-major `(pr, pc)`, so they agree. This matters because
  ties are *common*, not rare: after a ReLU conv layer, all-zero windows are routine. Exact
  zeros are bit-identical in both implementations, so tie parity is exact. A near-tie flipped
  by floating-point summation order is possible in principle (≈1e-15 relative), and a parity
  failure there would be recorded as that, not papered over with a looser tolerance.
- `compute_hidden_delta_batch(next_layer)`: `delta_batch = next_layer.downstream_batch()`
  (max is the identity on its winning input).
- `downstream_batch()`: one masked strided slice-add per slot `(pr, pc)`: add
  `delta · (argmax == slot)` into `dX[:, :, pr::s, pc::s]` (bounded to `out_h`/`out_w`). With
  overlapping windows (`stride < pool_size`) one input receives every delta it won, as
  `MaxPoolLayer.downstream_sum` does.
- The gradient hooks (`accumulate_gradient*`, `apply_accumulated_gradient`) are no-ops. There
  is no `W`/`b`.
- Single-example path: `N = 1` wrappers, as in stage 2.

**Tests (`tests/test_max_pool_array_layer.py`):** forward, argmax, and `downstream` parity
against `MaxPoolLayer` for non-overlapping and overlapping windows and multiple channels;
explicit tie tests (all-zero window, and a partial tie) pinning slot-0 selection in both
implementations; a numpy-only finite-difference check through conv → pool → conv.

## Stage 4: `ConvVectorizedMultiClassBackpropClassifierNetwork`

The numpy sibling of `ConvMultiClassBackpropClassifierNetwork`, named by the existing numpy
multiclass convention (`…VectorizedMultiClass…`).

- Subclasses `VectorizedMultiClassBackpropClassifierNetwork`, inheriting
  `predict_probabilities`/`classify_state`/`_target_array`/`_target_batch_array`, and from
  `ArrayNetworkBase` `_forward`/`learn`/`learn_batch`, all unchanged. They only iterate
  `self.layers` through the per-layer hooks stages 1-3 provide.
- Like the pure-Python conv class, it doesn't call `super().__init__()`. Its constructor
  `(input_height, input_width, conv_specs, dense_layer_sizes, class_count)` builds
  `self.layers = conv/pool layers + dense ArrayLayers + output ArrayLayer`. It chains
  `(height, width, channels)` through the specs exactly as the pure-Python class does, with the
  same validation (at least one `ConvSpec`, `validate_class_count`, `validate_layer_sizes`).
  The chaining loop is identical in both classes, so extract it into one shared helper rather
  than copying it.
- `randomize()`: layers in forward order. Conv layers use
  `fan_in_aware_random_layer(channel_count, k²·in_channels)` (the existing `array_layer.py`
  helper; its `(size, previous_size)` shape is exactly the conv `W` shape). Pool layers draw
  nothing. The dense tail's fan-in starts from the last conv/pool layer's flattened size.
- `snapshot()`/`restore()`: one entry per layer, `(W, b)` for conv and dense layers and an empty
  entry for pool layers, mirroring the pure-Python pool layer's `[]` snapshot.
- `save()`/`load()`: the same JSON envelope keys as the pure-Python conv class
  (`input_height`, `input_width`, `conv_layers`, `dense_layer_sizes`, `class_count`,
  `snapshot`), reusing `_spec_to_json`/`_spec_from_json` (promoted from private helpers in the
  pure-Python module to a shared location). No committed conv model file exists, so there's no
  compatibility cost.

**Tests (`tests/test_conv_vectorized_multiclass_backprop_model.py`)**, with a new
`matching_conv_array_backprop_networks` helper in `tests/helpers.py` (the conv counterpart of
`matching_array_backprop_networks`: identical injected weights in both networks, because the
two RNG streams aren't comparable):

- `predict_probabilities`/`classify_state` sweeps;
- `learn` parity **after every step** and `learn_batch` parity **after every batch** (the
  existing regression-gate pattern), for at least: one conv layer; conv → conv with stride 2;
  conv → pool → conv; conv → overlapping pool; a multi-channel second layer. Inputs are
  pixel-like `[0, 1]` values, including real UCI digits rows, so ReLU zeros and pooling ties
  actually occur rather than being tested only in isolation;
- **end-to-end reproduction of the pinned pure-Python result:** build the pure-Python network
  with `random.seed(0)` exactly as `test_trains_on_a_real_uci_digits_subset` does, copy its
  initial weights into the numpy network, train with the same `train_linear_classifier_network`
  call, and assert the same `best_training_accuracy == 0.9875`, `best_epoch_index == 10`, and
  test accuracy `0.925`. Only the numpy network needs training in the test. If floating-point
  order flips a single prediction, record that honestly instead of loosening the assertion
  silently;
- construction shape/validation, `randomized` (symmetry broken, usable), snapshot/restore, and
  save/load round trips, including a pooled network.

Also in this stage: update the README Models section (convolution and max pooling now exist in
the pure-Python and numpy implementations) and the `ConvMultiClassBackpropClassifierNetwork`
docstring's cross-reference.

## Deferred / out of scope

- **Rust CNN**: separate workplan. It mirrors this plan's layouts and is parity-tested against
  these numpy classes.
- **All performance measurement**: deferred until the Rust CNN exists, then numpy vs Rust.
  That includes the single-example `N = 1` wrapper cost and whether im2col-matmul is the right
  numpy formulation.
- **The open accuracy questions** from the multi-layer CNN work (does a second conv layer help
  on 8×8 digits, and pooling vs stride 2?): still unmeasured. Once stage 4 lands they could run
  on the numpy network as a pure accuracy measurement. That's a separate decision, not part of
  this plan.
- 'same' padding, conv variants (momentum/Adam/softmax/dropout/L2), and conv ensembles. None
  exist in the pure-Python reference either.
