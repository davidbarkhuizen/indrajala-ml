# Workplan: optimization 4, remove the N = 1 reshapes from the single-example conv path

Order: after optimization 3 stages A and B (see `optimizations.md`). Stage B was closed without
merging, so there is no `conv_infer_batch` or `infer` to cover below.

## Context

The Rust conv and pool ops are batch-only. The single-example methods of `ConvRustArrayLayer`
and `MaxPoolRustArrayLayer` wrap one example as `(1, n)` with `Array.reshape`, which copies
(`array.rs::reshape` clones the buffer). They also unwrap the results the same way. Per single
example, today:

| layer | method | reshapes |
| --- | --- | --- |
| conv | `forward` | `x` in; `A` out (`Z` until optimization 3 stage A) |
| conv | `downstream` | `delta` in; `dX` out |
| conv | `accumulate_gradient` | `delta` in |
| pool | `forward` | `x` in; `A` and `argmax` out |
| pool | `downstream` | `delta` and `argmax` in; `dX` out |

That's 3–8 µs per example in Rust: 27% overhead on an 8x8 UCI conv layer and 5.6% at 28x28.
`Array.reshape` accounts for about 4% of profiled Rust time (68,000 calls in one MNIST epoch).
numpy's `x[np.newaxis]` and `[0]` are views, so its overhead is 1.7–5.5%.

After optimization 3 stage A, conv `forward` no longer reshapes `Z`. What's left is removed
here.

## Approach: the ops accept a 1D vector as N = 1

Each conv/pool op takes its batch operands as either `(N, size)` or a 1D vector of length `size`
(meaning `N = 1`), and returns outputs in the same rank as its input. Rust already treats
`Shape::Vector` and `Shape::Matrix` as distinct tags over the same flat row-major buffer, so a
`(1, size)` matrix and a `size` vector have identical data. The ops change only in how they read
`n` and tag the result. That makes this **bit-identical by construction**; the tests still check
it.

An alternative was considered and rejected: a zero-copy `reshape` (a shape-only rebind). It would
need `RustArray` to share buffers (`Arc<Vec<f64>>` or views), which `array.rs` deliberately doesn't
do, and it would touch every op in the crate for a conv-only benefit.

## Stage A: rank-preserving conv and pool ops (crate + here)

**Crate PR:**

- `conv.rs`: replace `require_matrix(x, None, size, ...)` with a helper that accepts either
  rank and returns `(n, is_vector)`. `Shape::Vector(size)` gives `n = 1`. The existing error text
  is kept for any other shape.
- Every op that returns per-example arrays tags its output as a vector when its batch input was
  a vector: `conv_forward_batch` (`A`), `conv_downstream_batch` (`dX`), `max_pool_forward_batch` (`A`, `argmax`) and
  `max_pool_downstream_batch` (`dX`). `cols` stays `(P, C·k·k)`. It's internal, so its rank is
  irrelevant.
- `conv_accumulate_gradient_batch` accepts a vector `delta` (`n = 1`) and checks it against
  `cols`' row count.
- `max_pool_downstream_batch` requires `delta` and `argmax` to have the same rank.
- Keep the `_batch` names. Renaming every op would be churn with no gain. Update each doc
  comment to say it accepts a vector as N = 1.
- Crate tests (`rust/tests/test_conv_ops.py`, `test_max_pool_ops.py`): for every op and every
  shape in the existing lists, the vector-input call returns exactly (`==`) the flattened
  `(1, size)` result. Also test the rejection of mixed ranks and wrong vector lengths.

**Here:**

- `ConvRustArrayLayer.forward`/`downstream`/`accumulate_gradient` and
  `MaxPoolRustArrayLayer.forward`/`downstream` pass the 1D arrays straight through, with no
  `reshape`. Single-example `self.a`/`self.argmax` are the op outputs directly. Update both
  docstrings, which currently describe the `(1, n)` wrapping.
- Tests: the existing "single-example path == batch row" tests in
  `tests/test_conv_rust_array_layer.py` and `tests/test_max_pool_rust_array_layer.py` must pass
  unchanged. Tighten them to exact equality if they currently use `approx`: there's no reason
  for any difference now. All step-by-step network parity tests pass unchanged.

**Measure:** re-run the table in `recommended-optimizations.md` item 4 (single-example path vs
batch ops on pre-shaped input, one `ConvSpec(3, 8)` layer, UCI 8x8 and MNIST 28x28). The
expected result is Rust overhead near 0. Also count `Array.reshape` calls in a one-epoch MNIST
profile (expected: near 0 from the conv path), and run the end-to-end demo.

## Out of scope

- The dense single-example path. It already takes 1D arrays natively.
- Zero-copy `reshape` in general (see above).
