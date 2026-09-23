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

**Done** (indrajala-math-rust#8). `linear_preactivation_batch` calls `matmul_nt(x, w)`, which
shares `matmul_2d`'s threading decision through a new `for_each_row_range`. Crate tests pin every
row of each `layer_*forward_batch` (sigmoid, ReLU, softmax, eval-mode dropout) bit-exactly to the
single-example op on that row, at 11 shapes including three over the threading threshold. The
old `matmul(x, w.T)` fails 35 of them: every one with `K >= 4`, plus the rejection of a 1D `X`,
which the old code accepted through the vector @ matrix case. Here,
`tests/test_rust_array_layer_forward_batch.py` pins the same property for every dense Rust layer
class, including the one-row batch.

Measurement step, same binary, `W.T` hoisted out vs the current op (Rust µs per call, median of
9 interleaved loops; `W.T alone` times the copy by itself):

| shape | batch | current | hoisted | `W.T` alone | saved |
| --- | --- | --- | --- | --- | --- |
| 32 x 5408 | 1 | 331.5 | 60.2 | 276.1 | 82% |
| 32 x 5408 | 32 | 1875.1 | 1387.8 | 325.8 | 26% |
| 32 x 5408 | 512 | 10556.7 | 10539.3 | 325.1 | 0% |
| 30 x 784 | 1 | 31.5 | 8.6 | 23.1 | 73% |
| 30 x 784 | 32 | 277.8 | 251.6 | 23.2 | 9% |
| 30 x 784 | 512 | 2202.7 | 2093.5 | 34.1 | 5% |
| 10 x 30 | 1 | 1.2 | 0.7 | 0.6 | 43% |
| 10 x 30 | 32 | 9.7 | 9.3 | 0.6 | 4% |
| 10 x 30 | 512 | 140.0 | 139.7 | 0.7 | 0% |

Well over the 10% bar at batch 1 on every shape, so the stage went ahead.

`matmul_nt` against the old op, both built into one binary and interleaved (Rust µs per call,
median of 9 loops, `layer_forward_batch`):

| shape | batch | numpy | old | new | change |
| --- | --- | --- | --- | --- | --- |
| 32 x 5408 | 1 | 31.6 | 330.3 | 58.4 | -82% |
| 32 x 5408 | 32 | 520.1 | 2073.5 | 920.0 | -56% |
| 32 x 5408 | 512 | 3710.0 | 16711.9 | 7689.4 | -54% |
| 30 x 784 | 1 | 11.2 | 36.4 | 9.2 | -75% |
| 30 x 784 | 32 | 59.8 | 294.2 | 270.7 | -8% |
| 30 x 784 | 512 | 1161.0 | 4760.8 | 4054.3 | -15% |
| 10 x 30 | 1 | 7.2 | 1.3 | 0.7 | -45% |
| 10 x 30 | 32 | 11.5 | 10.0 | 7.2 | -28% |
| 10 x 30 | 512 | 68.0 | 146.1 | 107.8 | -26% |

It wins at every shape, so the fear below that few long dot products lose to many short axpys
didn't hold. It beats hoisting the copy out too (920 vs 1388 µs at 32 x 5408, batch 32). At 30 x
784, batch 32, both Rust columns are still ~4.5x numpy: that gap is in the kernel, not the
transpose (see `recommended-optimizations.md`).

Bit-changing protocol: the full suite passed unchanged (2439), including the Rust conv pin 0.9875
/ epoch 10 / 0.925, so no 1-ULP control was needed. Old vs new `layer_forward_batch`, 20 seeds
per shape:

| shape | batch | max abs | median ULP | max diff in ULPs of the row's largest element |
| --- | --- | --- | --- | --- |
| 32 x 5408 | 1 | 6.55e-15 | 12 | 59 |
| 32 x 5408 | 32 | 2.20e-14 | 12 | 198 |
| 30 x 784 | 1 | 1.67e-15 | 3 | 15 |
| 30 x 784 | 32 | 2.16e-15 | 3 | 20 |
| 10 x 30 | 1 | 2.22e-16 | 0 | 2 |
| 10 x 30 | 32 | 2.22e-16 | 0 | 2 |

Larger than stage B's, because the old op summed up to 5408 products in one sequential chain.
Against an 80-bit `longdouble` reference the new op is the more accurate one: mean absolute error
is 10-30% lower at every shape above, and the max is comparable (4.9e-15 old vs 5.3e-15 new at
32 x 5408 batch 1, 1.4e-14 vs 9.8e-15 at batch 32).

End to end. Dense MNIST (784 -> 30 -> 10), one mini-batch epoch (batch 32) over all 60000 from
identical weights and shuffle order, 3 rounds alternating old and new builds: Rust 5.67 -> 5.38 s
(median), Rust/numpy 0.678 -> 0.637. Test accuracy was 0.9001 for numpy and for both Rust builds.
The conv demo's ratios are unchanged within noise (MNIST mini-batch: conv 0.89 -> 0.89,
conv-pool-conv 0.62 -> 0.65, conv-conv-stride2 0.73 -> 0.73). Its dense tail is 32 x 32 x 10 there,
and `layer_forward_batch` is 1.6% of the profiled mini-batch time, behind `conv_forward_batch` at
52%.

The plan as written:

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
