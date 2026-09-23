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

**Done** (indrajala-math-rust#10). As planned, with one addition: the pool layer test now also
asserts the single-example output shapes, since `assert_array_equal` would broadcast a `(1, n)`
result against a row and pass. The "single-example == batch row" tests were already exact. For
every op, the crate tests require the vector call to return exactly the flattened `(1, size)`
call's result, over 7 conv and 6 pool shapes. Tagging outputs as matrices regardless of input
rank fails all 19 of those cases.

Rust µs per call, single-example layer methods (3x3 kernel, 8 channels, 2x2 pool, median of 3
rounds of 9 loops), three runs per build with the builds alternated:

| op | old | new |
| --- | --- | --- |
| conv 8x8 `forward` | 4.77, 4.68, 4.69 | 3.87, 3.89, 3.90 |
| conv 8x8 `downstream` | 4.01, 4.03, 4.10 | 3.31, 3.32, 3.31 |
| conv 8x8 `accumulate_gradient` | 3.35, 3.33, 3.38 | 3.15, 3.07, 3.07 |
| conv 8x8 forward + downstream + accumulate | 12.52, 12.71, 12.35 | 10.33, 10.29, 10.28 |
| pool 6x6x8 `forward` | 2.68, 2.82, 2.79 | 1.54, 1.53, 1.54 |
| pool 6x6x8 `downstream` | 2.55, 2.54, 2.59 | 1.42, 1.44, 1.42 |
| conv 28x28 forward + downstream + accumulate | 149.3, 147.8, 145.7 | 141.5, 153.7, 143.0 |
| pool 26x26x8 `forward` | 18.47, 18.61, 18.80 | 15.98, 16.16, 16.19 |
| pool 26x26x8 `downstream` | 23.69, 24.13, 24.01 | 23.08, 23.30, 23.05 |

At 28x28, the saved copies (a few µs) are inside the run-to-run spread. The 28x28
`accumulate_gradient` measured about 2 µs slower in all three alternated runs (38.1-38.8 vs
39.9-40.7). Within one build, though, the op with a vector `delta` is as fast as or faster than
with a `(1, n)` one (39.0-39.9 vs 39.1-40.5), and reshape plus the matrix call adds about 1 µs.
So it's a build or process artifact, not the change.

The demo's N = 1 wrapping table (one `ConvSpec(3, 8)` layer, forward + downstream + accumulate,
single-example path vs batch ops on pre-shaped input), before (after optimization 3 stage A) and
after:

| dataset | Rust overhead before | after |
| --- | --- | --- |
| UCI digits 8x8 | 28.5% (15.1 vs 11.7 µs) | **0.3%** (11.2 vs 11.1 µs) |
| MNIST 28x28 | 5.6% (164.6 vs 155.8 µs) | **-0.5%** (163.4 vs 164.2 µs) |

`Array.reshape` calls counted with cProfile over 200 single-example `learn` steps, 200
`classify_state` calls and a mini-batch pass (MNIST shapes, conv-pool-conv): 0. The demo's
profile had 56,000 (4.0% of Rust time).

End-to-end single-example Rust/numpy ratios, before (#333) and after. Mini-batch is unchanged
within noise, as expected, since it never took the single-example path:

| case | before | after |
| --- | --- | --- |
| UCI conv | 0.17 | 0.16 |
| UCI conv-pool-conv | 0.11 | 0.08 |
| UCI conv-conv-stride2 | 0.14 | 0.11 |
| MNIST conv | 0.31 | 0.31 |
| MNIST conv-pool-conv | 0.49 | 0.44 |
| MNIST conv-conv-stride2 | 0.60 | 0.55 |

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
