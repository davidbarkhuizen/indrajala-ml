# Workplan: optimization 2, fuse the dense single-example gradient accumulation

Order: first after stage 0 (see `optimizations.md`).

## Context

`fused.rs::layer_accumulate_gradient` does two full-size passes and two full-size allocations:

```rust
let outer_product = outer(delta, input_activation)?;                          // alloc + pass 1
let new_grad_w = grad_w.combine_with_array(&outer_product, |g, o| g + o, "add")?; // alloc + pass 2
```

At 32 x 5408 the call measures 639 µs in Rust against numpy's 243 µs. The outer product alone
is 367 µs. Then `RustArrayLayer.apply_accumulated_gradient` makes a third pass over `W` (322 µs),
and `_reset_gradient_accum` allocates a fresh zero `grad_W` (a fourth pass). In single-example
`learn()` (`RustArrayNetworkBase.learn`), those four passes run for every layer on every step.

Every dense Rust layer that inherits `accumulate_gradient` from `RustArrayLayer` benefits:
plain, ReLU, softmax, dropout, L2, momentum and Adam. The conv network's dense tail is one of
them.

## Numerics: this can be bit-identical

The current code computes `g + (d * x)` with two roundings: the product, then the sum. A fused
loop that writes `grad_w[i, j] + delta[i] * x[j]` as a plain multiply then add (**not**
`mul_add`) produces exactly the same bits. `recommended-optimizations.md` suggests `axpy_row`,
but `axpy_row` uses FMA (one rounding). That would change the bits for a small gain, since the
cost here is allocation and memory passes, not arithmetic. So:

- **Stage A uses a plain multiply-add loop**, kept bit-identical. It's written so LLVM can
  auto-vectorize it: slices, no bounds checks in the inner loop. Separate `vmulpd` + `vaddpd`
  instructions keep the two roundings, so a vectorized loop stays bit-identical.
- An FMA variant is measured alongside it in the stage A PR, for the record. It's adopted only
  if it's materially faster, and in that case it follows the bit-changing protocol.

## Stage A: one-pass `layer_accumulate_gradient` (crate + here)

**Done** (indrajala-math-rust #5, bumped here). Bit-identical, as planned. The FMA variant was not
materially faster, so it wasn't adopted. At 32 x 5408, Rust `accumulate_gradient` went from 596 to
62 µs, and the single-example SGD step (accumulate + apply + reset) from 1308 to 463 µs. MNIST
one-conv single-example went from 1.21 to 0.75 Rust/numpy. Dense MNIST, one epoch, went from 0.645
to 0.423.

**Crate PR:**

- Rewrite `layer_accumulate_gradient` to allocate one output buffer and fill row `i` with
  `grad_w_row[j] + delta[i] * x[j]`. Keep the signature and the shape errors unchanged:
  `delta`/`x` 1D, `grad_w` of shape `(len(delta), len(x))`. It currently gets these checks from
  `outer` and `combine_with_array`, so they become explicit.
- `grad_b` is unchanged: it is already one small pass.
- Crate test (`rust/tests/test_linalg.py` has the formula tests): exact equality between the
  new op and `outer` + add composed from the crate's own primitives, over random shapes, with
  `x` containing zeros, negative zeros and subnormals.

**Here:** bump the submodule. No Python change. `tests/test_fused_layer_ops.py::
test_layer_accumulate_gradient_matches_array_layer` and every network parity test should pass
unchanged. Also add a test that pins the bit-identity at the indrajala-ml level: one
`RustArrayMultiClassBackpropClassifierNetwork` `learn()` step on the old submodule commit
produces weights recorded as a fixture, and the new commit must reproduce them exactly. A
cheaper equivalent is fine if the fixture turns out awkward: compare against an in-test
`pa.outer` + `+` composition.

**Measure:** the dense accumulate_gradient row at 32 x 5408 and at 30 x 784, and the
end-to-end conv and dense demos.

## Stage B: fused single-example SGD step (Python contract change; gated on stage A)

**Done** (indrajala-math-rust #6, bumped here). The dense MNIST epoch takes 15% less Rust time
(6.68 -> 5.65s, fused vs unfused on the same build), above the 10% bar, and the final weights
after 60000 steps are bit-identical. The per-layer step at 32 x 5408 went from 463 to 64 µs.
`sgd_step` also guards against a subclass overriding `accumulate_gradient`, not only
`apply_accumulated_gradient`, since the fused step bypasses both.

After stage A, a single-example step still allocates `grad_W`, reads it back in `apply`, and
allocates zeros in `_reset_gradient_accum`. At `batch_size=1` the accumulator is always fresh:
`RustArrayNetworkBase.learn` calls `accumulate_gradient` then `apply_accumulated_gradient` per
layer, and apply resets it. So the whole thing is `W -= lr * outer(delta, x)`.

- **Crate:** `layer_sgd_step(w, b, delta, x, learning_rate) -> (w, b)`. It computes
  `w[i, j] - scale * (0.0 + delta[i] * x[j])` in one pass with one allocation. The `0.0 +` is
  there on purpose: it keeps the result bit-identical to accumulate-into-zeros-then-apply,
  including the sign of zero. It can be dropped only if a test shows it makes no difference.
  Crate test: exact equality against `layer_accumulate_gradient` into zeros, then
  `layer_apply_accumulated_gradient(batch_size=1)`.
- **Here:** a layer-level method `sgd_step(input_activation, learning_rate)`.
  - `RustArrayLayer` implements it with `pa.layer_sgd_step`.
  - Subclasses whose `apply_accumulated_gradient` isn't plain SGD (momentum, Adam, L2) must
    *not* inherit the fused version. Give them the fallback: `accumulate_gradient` then
    `apply_accumulated_gradient(lr, 1)`. Enforce this with an explicit override on each, not
    an `isinstance` check, and add a test that fails if a new `RustArrayLayer` subclass
    overrides `apply_accumulated_gradient` without also overriding `sgd_step`.
  - Conv and pool layers get the fallback too. Their accumulate is already batch-shaped.
  - `RustArrayNetworkBase.learn`'s last loop calls `layer.sgd_step(...)`.
  - Only the Rust side changes. `ArrayNetworkBase` (numpy) keeps its loop: the aim is Rust
    speed, and numpy is the parity reference.
- **Tests:** every existing Rust network step-by-step parity test must pass unchanged, and a
  new test checks one `learn()` step exactly against the stage-A behaviour for each dense Rust
  variant.

**Measure:** accumulate + apply + reset, before and after, as one row in the stage 0 harness at
both dense shapes, plus the end-to-end demos. If stage B saves less than about 10% of the
dense single-example step time, close it with the numbers and don't merge. The contract change
isn't worth a marginal gain.

## Out of scope

- The batch path, `layer_accumulate_gradient_batch`. Its transpose is optimization 1 stage A.
- In-place mutation of `RustArray` from Python. The rebind-not-mutate pattern stays.
