# Workplan: optimization 1, stop materializing transposes in the dense ops

Order: after optimization 2 (see `optimizations.md`).

## Context

`RustArray::transpose` allocates and fills a full transposed copy. Three dense fused ops call it
on every invocation:

| site | product | per call |
| --- | --- | --- |
| `layer_downstream` (and so `hidden_downstream`, used by every `layer_*hidden_delta`) | `W.T @ delta` | single-example backward |
| `layer_accumulate_gradient_batch` | `delta_batch.T @ X` | mini-batch backward |
| `linear_preactivation_batch` (every `layer_*forward_batch`) | `X @ W.T` | mini-batch forward |

Only the first was measured: 327 µs at 32 x 5408, of which the transpose is 273 µs.
`matmul`'s existing vector @ matrix case (`delta @ W`) measured 41 µs. `conv_forward_batch`
also calls `w.transpose()`, but on the kernel matrix (8 x 9 in the demo), which is negligible.
It's left alone here; optimization 3 stage C revisits that product.

## Numerics: the three sites differ

Two orderings exist in `linalg.rs`. `dot_product` uses a 4-lane interleaved grouping (the matrix
@ vector case). `axpy_row` accumulates sequentially over `k` with FMA (vector @ matrix and
matrix @ matrix). Each site's current and transpose-free orderings:

- **`delta_batch.T @ X` (stage A): bit-identical.** Currently `matmul_2d(delta.T, X)` computes
  `out[i, :] += delta.T[i, k] * X[k, :]` sequentially over `k` with `axpy_row`. A transposed-left
  kernel does the same `axpy_row(out[i], delta[k, i], X[k])` calls in the same `k` order. It just
  reads `delta[k, i]` with a stride instead of from a copied row. Same operations, same bits.
- **`W.T @ delta` (stage B): changes bits.** It's currently `dot_product` per output element (4-lane
  grouping). `delta @ W` is `axpy_row`, sequential over `k`. The difference is a few ULPs
  (measured max 2.7e-15 at 32 x 5408).
- **`X @ W.T` (stage C): changes bits the other way.** Currently it's `matmul_2d(X, W.T)`,
  sequential `axpy_row` over `k`. A transpose-free kernel computes each output as
  `dot_product(X[n], W[j])`, both rows contiguous, which gives the 4-lane grouping. Keeping the
  sequential order without the transpose would mean strided access to `W`, which is the cost
  we're removing.

So stage A goes first and alone. B and C each follow the bit-changing protocol in
`optimizations.md`.

## Stage A: transposed-left matmul for `layer_accumulate_gradient_batch` (bit-identical)

**Closed, not merged: no measured gain.** Built as planned on crate branch `matmul-tn`
(`e59c511`, kept for the record, no PR): `matmul_2d` and `matmul_2d_row_range` take an `a`
stride pair, and `matmul_tn` passes `(1, M)`. Its crate tests pinned exact bit-identity against
`grad_W + delta_batch.T @ X` at 9 shapes covering every threading/blocking combination, plus 20
seeds with signed zeros and subnormals. Reversing the `k` order on the strided path failed 22 of
them.

Measured as old vs new builds swapped round by round (6 rounds of 15 x 1500-call loops, Rust µs
per call, median):

| shape | batch | old | new | change |
| --- | --- | --- | --- | --- |
| 32 x 5408 | 32 | 3010 | 3013 | +0.1% |
| 32 x 5408 | 512 | 22314 | 22262 | -0.2% |
| 30 x 784 | 32 | 190 | 197 | +3.7% |
| 30 x 784 | 512 | 3900 | 3749 | -3.9% |

The batch-1 rows and 10 x 30 moved by a few µs, also inside the noise. Every change is smaller
than the 10-25% spread between rounds. In hindsight the copy removed is `delta_batch.T`, batch x
`M` (32 x 32 or 512 x 30), which is small next to the matmul. The 273 µs measured transpose is
`W.T` in `layer_downstream`, which is stage B.

**Crate PR:**

- `linalg.rs`: `matmul_tn(a, b)` computes `a.T @ b` for `a (K, M)`, `b (K, N)` without
  materializing `a.T`. It reuses `matmul_2d`'s threading and blocking decisions (same
  thresholds, same row-range split over the `M` output rows). Each row range's inner loop is
  `for k: axpy_row(out[i], a[k*M + i], b[k])`, which keeps per-row `k` order increasing within
  and across blocks, exactly as `matmul_2d_row_range` does. The cleanest route is to generalize
  `matmul_2d_row_range` over an `a`-element accessor (a stride pair), not to copy it.
- `layer_accumulate_gradient_batch` calls `matmul_tn(delta_batch, X)`.
- Crate tests: exact equality `matmul_tn(a, b) == matmul(a.T, b)`, with shapes on both sides of
  the threading threshold (4M flops) and the blocking threshold (`b` > 256 KB). Include `M = 1`,
  `K = 1`, and non-multiples of 4 and 64.

**Here:** bump. Add a test that pins bit-identity of `pa.layer_accumulate_gradient_batch`
against a `pa`-composed `delta.T @ X + grad_W`. All parity tests pass unchanged.

**Measure:** the dense `accumulate_gradient_batch` row at batch sizes 1, 32 and 512, at both
dense shapes. Strided reads of `a` may cost something at large `M`. If a shape regresses,
record it and decide in the PR whether to gate `matmul_tn` by shape. Don't gate it
speculatively.

## Stage B: `layer_downstream` via vector @ matrix (changes bits)

**Done** (indrajala-math-rust#7). `layer_downstream` is `matmul(delta, w)`, and now rejects a 2D
`delta` (`delta @ W` would be a different product from `W.T @ delta` for one). A crate test pins
it bit-exactly to a pure-Python sequential FMA chain (`Fraction`-exact, correctly rounded), the
scalar path's definition, so the AVX2 path is shown to give the same bits. Putting back
`W.T @ delta` fails 5 of its 7 shapes (the other two have `M = 1`).

Bit-changing protocol: the full suite passed unchanged (2350), including the Rust conv pin
0.9875 / epoch 10 / 0.925, so no 1-ULP control was needed. Old vs new, 20 seeds per shape:

| shape (M x N) | op | max abs | median ULP | max diff in ULPs of the vector's largest element |
| --- | --- | --- | --- | --- |
| 30 x 784 | downstream | 2.66e-15 | 1 | 3 |
| 30 x 784 | hidden_delta | 6.66e-16 | 1 | 3 |
| 10 x 30 | downstream | 8.88e-16 | 1 | 2 |
| 10 x 30 | hidden_delta | 2.22e-16 | 0 | 2 |
| 32 x 5408 | downstream | 3.55e-15 | 1 | 4 |
| 32 x 5408 | hidden_delta | 8.88e-16 | 1 | 4 |

Raw max ULP reaches ~1e5, but only on entries near zero, where cancellation makes a ULP tiny.

Timing, Rust µs per call (median of 4 old/new A/B rounds):

| shape | op | numpy | before | after |
| --- | --- | --- | --- | --- |
| 32 x 5408 | downstream | 27.9 | 307.8 | 42.7 |
| 32 x 5408 | hidden_delta | 36.6 | 312.1 | 48.7 |
| 30 x 784 | downstream | 5.2 | 30.9 | 5.6 |
| 30 x 784 | hidden_delta | 9.0 | 31.5 | 5.9 |
| 10 x 30 | downstream | 1.7 | 0.9 | 0.5 |
| 10 x 30 | hidden_delta | 4.2 | 1.2 | 0.7 |

The end-to-end MNIST conv single-example case was already below 1 (0.47) after optimization 2.
Conv demo, MNIST single-example, Rust/numpy: conv 0.47 -> 0.38, conv-pool-conv 0.51 -> 0.49,
conv-conv-stride2 0.65 -> 0.61. Mini-batch doesn't use `layer_downstream`, and is unchanged.
Dense MNIST (784 -> 30 -> 10), one single-example epoch over all 60000 from identical weights,
median of 3 runs alternating old and new builds: Rust/numpy 0.370 -> 0.382, inside the noise
(numpy, which doesn't use the crate, was itself ~5% slower in the new-build slots). As expected:
at that shape stage B only touches the 10 x 30 `hidden_delta`, about 0.5 µs x 60000 = 0.03 s.
Rust test accuracy moved 0.9283 -> 0.9318 (numpy 0.9292), the rounding sensitivity a few-ULP
change is expected to cause.

**Crate PR:**

- `layer_downstream(w, delta)` computes `matmul(delta, w)` (the existing vector @ matrix case)
  and no longer calls `w.transpose()`.
- This changes `hidden_downstream`, so `layer_hidden_delta`, `layer_relu_hidden_delta` and
  `layer_dropout_hidden_delta` change too. They're all intended beneficiaries.
- Update the doc comment on `matmul`'s vector @ matrix case, which says it is "not exercised by
  the current class design". After this change it is on the hottest single-example path.
- Crate tests: `layer_downstream` against numpy `W.T @ delta` within `rtol` (as now), plus the
  scalar-vs-AVX2 bit-identity property. It already holds for `axpy_row`, so this needs no new
  mechanism, only a test that exercises it through `layer_downstream`.

**Here:**

- Bump. Run the full suite under the bit-changing protocol. The step-by-step parity tests for
  every dense Rust variant and the conv Rust network are the gate. The end-to-end pins most at
  risk are the Rust conv network's 0.9875 / epoch 10 / 0.925 and any pinned dense Rust
  end-to-end results.
- Record the old-vs-new max abs / max ULP difference of `layer_hidden_delta` at 30 x 784 and 10 x
  30 (dense production) and 32 x 5408 (conv tail).

**Measure:** the dense downstream row at 32 x 5408 (expect about 327 -> 41 µs if the ad hoc
number holds) and the hidden-delta rows at the dense production shape. Also the end-to-end
MNIST conv single-example case, the one case where Rust was slower than numpy (1.21). The PR
reports whether it crosses below 1.

## Stage C: transposed-right matmul for `linear_preactivation_batch` (changes bits; measure first)

Nobody has measured whether this transpose matters. At the dense production shape `W` is 30 x
784 (188 KB copy per `forward_batch`), and at the conv tail it's 32 x 5408.

- **First, a measurement-only step, no PR:** time `layer_forward_batch` with the transpose
  hoisted out (pre-transposed `W` passed in) against the current op at batch 1/32/512 on both
  shapes. If the transpose is under about 10% of the call at every shape, stop here and record
  the numbers in `recommended-optimizations.md`.
- If it's worth doing:
  - **Crate PR:** `matmul_nt(a, b)` computes `a @ b.T` for `a (M, K)`, `b (N, K)` as
    `out[m, n] = dot_product(a[m], b[n])`, threaded over `M` rows with `matmul_2d`'s threshold.
    `linear_preactivation_batch` uses it. Tests: within `rtol` of numpy, plus the property that
    the result is bit-identical to `M`·`N` independent `matmul(W, x)` matrix @ vector calls. That
    property is useful: it makes single-example `layer_forward` and one-row `layer_forward_batch`
    agree exactly, which they currently don't guarantee.
  - **Here:** bump, and apply the bit-changing protocol. Add a test that `forward_batch` on a
    one-row batch equals `forward` exactly, for each dense Rust variant.
  - The `dot_product` path may be slower than `axpy_row` for some shapes: few long dot products
    against many short axpys. The PR reports every shape measured, and the op is adopted only
    if it wins at the production shapes.

## Out of scope

- A general strided or view-based `RustArray`. Each case here is a dedicated kernel, in line
  with the crate's "no N-dimensional machinery" rule.
- `conv.rs`'s own transpose of the kernel matrix (see optimization 3 stage C).
