# Workplan: convolutional networks in Rust

## Context

Convolution and max pooling exist in the pure-Python implementation (`ConvLayer`,
`MaxPoolLayer`, `ConvMultiClassBackpropClassifierNetwork`) and the numpy implementation
(`ConvArrayLayer`, `MaxPoolArrayLayer`, `ConvVectorizedMultiClassBackpropClassifierNetwork`,
#317-#320). The numpy classes fixed their layouts so that Rust could mirror them. This plan adds
the Rust counterpart, tested for parity against the numpy classes the way the dense Rust
classes are, and then runs the numpy-vs-Rust timing the numpy plan deferred.

Scope decisions:

- **Mirror the numpy feature set and layouts exactly.** That means 'valid' padding, stride,
  multi-channel input, ReLU conv layers, max pooling, a sigmoid dense tail, a one-vs-rest
  sigmoid output layer, and plain SGD. No momentum/Adam/softmax/dropout conv variants. Neither
  of the other implementations has them.
- **numpy is the parity reference.** Fused ops are checked against `ConvArrayLayer`/
  `MaxPoolArrayLayer` method by method, as `tests/test_*fused_layer_ops.py` does for the dense
  ops. Networks are checked step by step against the numpy conv network. The numpy network is
  itself pinned step by step to pure Python, and it runs the parity tests far faster.
- **Timing comes last, and it's numpy vs Rust only.** The pure-Python implementation is never
  timed (README, Models section). No performance claim is made before stage 4 measures it.

## Constraints from the existing crate

1. **`RustArray` is 1D or 2D only** (`Shape::Vector`/`Shape::Matrix`, and `array.rs` says
   "general N-dimensional machinery is deliberately not built here"). This plan keeps it that
   way. Every conv/pool tensor crosses the boundary as a matrix: activations and deltas
   `(N, C·H·W)`, the kernel matrix `(channel_count, C·k·k)`, the cached im2col columns
   `(N·P, C·k·k)`, and the pool argmax `(N, C·out_h·out_w)`. The 4D views exist only as index
   arithmetic inside the Rust functions.
2. **Arrays are rebound, not mutated.** Every fused op returns new arrays and the Python layer
   rebinds them (`RustArrayLayer` does this throughout). The conv ops follow the same pattern.
3. **Summation order is fixed, and it isn't numpy's.** `linalg.rs::dot_product` uses one
   canonical 4-lane grouping in both its scalar and AVX2 paths. The result is bit-identical
   across machines, but differs from numpy and from a sequential sum in the last few ULPs. The
   conv ops must use the existing `matmul`, not a hand-rolled reduction. That keeps them
   machine-independent, and parity against numpy is checked with `rtol`, as for the dense ops.
4. **Two repos.** Ops live in the `indrajala-math-rust` submodule (`rust/`). Layers, networks
   and cross-repo parity tests live here. So every stage that touches Rust is two PRs: a crate
   PR (ops, plus the crate's own numpy-only tests), then a PR here that bumps the submodule and
   adds the Python classes and parity tests. The crate README's rule applies: a change to its
   fused ops is also tested here before the bump is merged.

## Fused op design (`rust/src/conv.rs`, new)

**`ConvGeometry` (`#[pyclass]`, frozen).** It holds `input_height`, `input_width`,
`input_channels`, `kernel_size`, `stride`, and the derived `out_height`/`out_width`/
`positions`/`fan_in`. The constructor validates it, with the same checks as `ConvArrayLayer`.
Each Python layer builds one at construction and passes it to every call, so ops don't take
five loose ints and the shape arithmetic lives in one place. Pooling uses the same struct with
`kernel_size = pool_size`.

Conv ops. All of them are batch-only; the single-example path is described below.

- `conv_forward_batch(W, X, b, geometry) -> (Z, A, cols)`: build im2col `cols (N·P, C·k·k)`,
  row `n·P + p` in row-major output order, columns in `W`-row order. Then compute
  `cols @ W.T` with the existing `matmul`, and scatter the `(N·P, O)` result into channel-major
  `(N, O·P)` while adding `b`. `A = relu(Z)`. Returning `Z` too matches `ConvArrayLayer.Z`
  (used by the ReLU-at-zero test); if that costs anything, stage 4 will show it.
- `conv_downstream_batch(W, delta_batch, geometry) -> dX (N, C·H·W)`: gather `delta_batch` into
  `(N·P, O)`, compute `dcols = D @ W` with `matmul`, then col2im as a plain scatter-add loop
  over `(n, p, c, kr, kc)`. In Rust the loop is the natural form, and the overlap semantics are
  the same as numpy's k² slice-adds.
- `conv_accumulate_gradient_batch(delta_batch, cols, grad_W, grad_b, geometry) -> (grad_W,
  grad_b)`: gather `delta_batch` into `(O, N·P)`, then `grad_W += D @ cols` and
  `grad_b += row sums`.
- ReLU hidden delta: reuse the existing `array_relu_mask(downstream, A)`, whose derivative is
  0 at `z == 0`, the same convention as numpy.
- Apply: reuse `layer_apply_accumulated_gradient`. It doesn't depend on shape.

Pool ops:

- `max_pool_forward_batch(X, geometry) -> (A, argmax)`: a strict `>` scan over the window slots
  in row-major `(pr, pc)` order, so the first maximum wins, as with `np.argmax` and `PoolUnit`.
  `argmax` is an `f64` matrix of slot indices. The values are small exact integers, and this
  avoids a new integer array type.
- `max_pool_downstream_batch(delta_batch, argmax, geometry) -> dX`: send each window's delta
  to its winning input by scatter-add, so overlapping windows accumulate.

Dense downstream:

- `layer_downstream(W, delta)` / `layer_downstream_batch(W, delta_batch)`: `W.T @ delta` and
  `delta_batch @ W`, exposing `fused.rs::hidden_downstream*` without its shape check against
  the upstream activation.

## Stage 1: conv ops and `ConvRustArrayLayer`

**Crate PR:** `conv.rs` with `ConvGeometry` and the conv ops, plus `layer_downstream*` in
`fused.rs`. Register them in `lib.rs` and add them to the README layout table. The crate's own
tests (`rust/tests/test_conv_ops.py`, numpy only) check against a **brute-force numpy
reference written from the definition**: nested loops over output position, channel and kernel
offset, with no im2col. That makes this check independent of both implementations under test.
They also cover geometry validation errors and shape errors.

**Here:**

- `RustArrayLayer.downstream()`/`downstream_batch()` via `pa.layer_downstream*`. The dense
  Rust layers keep their fused `layer_*hidden_delta*` calls (which read `next_layer.W`)
  instead of being rewritten as in numpy stage 1. A dense layer never precedes a conv/pool
  layer, so those sites are correct. Splitting their fusion would add a boundary crossing to
  every dense backward step of the production backend in exchange for symmetry alone. Only the
  conv/pool layers call `next_layer.downstream*()`.
- `conv_rust_array_layer.py`: `ConvRustArrayLayer`, a standalone class with the same
  duck-typed surface as `ConvArrayLayer`. Its single-example methods reshape `x`/`delta` to
  `(1, n)` with `Array.reshape` and call the batch ops, the same `N = 1` wrapping numpy uses.
  (The reshape copies; stage 4 measures whether that matters.)
- `tests/test_conv_fused_layer_ops.py`: each op against the matching `ConvArrayLayer` method,
  over the same shape list as `tests/test_conv_array_layer.py` (1/2/3 channels, stride 1/2/3,
  non-square, kernel == input). It includes the stride-past-kernel case, where unread inputs
  get a gradient of exactly 0 in both implementations.
- `tests/test_conv_rust_array_layer.py`: layer-level parity against `ConvArrayLayer` (forward,
  downstream, accumulate, apply, single-example path == batch row), the ReLU-at-zero
  convention, and a finite-difference check on the Rust layer alone.

## Stage 2: pool ops and `MaxPoolRustArrayLayer`

**Crate PR:** the pool ops, tested against a brute-force numpy reference, including explicit
ties (an all-zero window and a partial tie, both resolving to the first slot).

**Here:**

- `max_pool_rust_array_layer.py`: `MaxPoolRustArrayLayer`, weight-free, with no-op gradient
  hooks and the same `N = 1` wrapping.
- `tests/test_max_pool_fused_layer_ops.py`/`tests/test_max_pool_rust_array_layer.py`: forward,
  argmax and downstream parity against `MaxPoolArrayLayer`, run over the same shape list as
  `tests/test_max_pool_array_layer.py` with continuous and tie-heavy inputs. Argmax must match
  **exactly**, not within a tolerance. Also a finite-difference check through conv → pool →
  conv on the Rust layers.

## Stage 3: `ConvRustArrayMultiClassBackpropClassifierNetwork`

Rust-only, so no crate PR.

- It subclasses `RustArrayMultiClassBackpropClassifierNetwork` without calling
  `super().__init__()`, and builds its front end with `conv_front_end.build_conv_front_end`,
  as both other conv networks do. It inherits `RustArrayNetworkBase`'s `_forward`/`learn`/
  `learn_batch` and the multiclass predict/classify unchanged. It overrides `randomize`
  (`pa.uniform` scoped to kernel fan-in, no draws for pool layers), `snapshot`/`restore` (a
  `()` entry for pool layers), and `save`/`load`.
- **The save format is the same as the numpy conv network's**: the same envelope keys and the
  same `(W, b)` layout per layer. A test must show that a numpy-saved model loads into Rust and
  gives the same predictions, and the reverse. Because the layouts are shared, this should
  work; the test is what establishes it.
- Tests (`tests/test_conv_rust_array_multiclass_backprop_model.py`), using a
  `matching_conv_numpy_rust_networks` helper (identical injected weights, because the two RNGs
  aren't comparable):
  - `learn` parity after every step and `learn_batch` parity after every batch, against the
    numpy conv network, on real UCI digits rows, over the same five architectures as
    `tests/test_conv_vectorized_multiclass_backprop_model.py`;
  - **end-to-end:** copy the seeded pure-Python initial weights into the Rust network (as the
    numpy test does), train with the same `train_linear_classifier_network` call, and assert
    the pinned 0.9875 / epoch index 10 / 0.925. Rust's dot-product grouping differs from
    numpy's, so a single flipped prediction is possible here. If it happens, record it as a
    measured difference and don't loosen the assertion silently;
  - construction/validation, `randomized` (symmetry broken, fan-in bounds, usable),
    snapshot/restore, and save/load round trips, pooled networks included.
- README Models section: convolution and max pooling now exist in all three implementations.

## Stage 4: numpy vs Rust timing

The comparison the numpy plan deferred. Only numpy and Rust are timed.

- A new demo (`demo_conv_rust_vs_vectorized_digit_recognition.py`, registered in
  `demos/registry.py`) times both backends with `demos/timing.py`'s `timed_train`, starting
  from **identical initial weights**, on UCI digits and an MNIST subset. It covers the
  single-example trainer (`train_linear_classifier_network`) and the mini-batch trainer
  (`train_backprop_network_mini_batch`), for at least one conv layer, conv → pool → conv, and a
  stride-2 second layer.
- Report each median over repeated runs, not a single run, plus the Rust/numpy ratio and
  whether both backends reached the same accuracy. (They should, up to the dot-product
  grouping.)
- The questions this answers, with results recorded in the PR and not decided in advance: is
  the `N = 1` single-example path (reshape plus batch op) a significant cost in either backend,
  and is im2col-matmul the right formulation in Rust? Any optimization that follows from the
  numbers is a separate decision after this stage, not part of this plan.

## Deferred / out of scope

- 'same' padding, conv variants (momentum/Adam/softmax/dropout/L2), and conv ensembles. None
  exist in any implementation.
- N-dimensional `RustArray`. The matrix-at-the-boundary convention above makes it unnecessary.
- The open accuracy questions from the multi-layer CNN work (does a second conv layer help on
  8×8 digits, and pooling vs stride 2?). They can run on either array network as a pure
  accuracy measurement, but that's a separate decision.
