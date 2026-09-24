# Recommended optimizations

Findings from the Rust CNN work (`docs/workplans/rust-cnn.md`, stage 4) that point at
optimizations. Stage 4 only measures. Whether to act on any of this is a separate decision, and
each item below would need its own before/after measurement and must keep the numpy parity tests
passing.

## How the numbers were measured

- **End-to-end:** `python -m indrajala_ml.demos.demo_conv_rust_vs_vectorized_digit_recognition`.
  This is the median of 5 interleaved runs from identical initial weights, on this machine (8
  logical cores). The demo reproduces every end-to-end number here.
- **Per-op:** single-example layer methods called 300 times in a loop, median over 5 loops, in
  µs per call. The shapes are the ones the demo's MNIST networks use: a `ConvSpec(3, 8)` layer on
  28x28 input, and the 32 x 5408 dense layer that follows it. The table below came from ad hoc
  scripts. `python -m indrajala_ml.demos.demo_layer_op_timing` now measures the same way and
  reproduces it, each row to within about 10%. It also covers the batch ops (see "Batch ops"
  below).

Rust / numpy wall-clock ratio (below 1 means Rust is faster):

| architecture | UCI digits, single | UCI digits, mini-batch | MNIST subset, single | MNIST subset, mini-batch |
| --- | --- | --- | --- | --- |
| conv | 0.29 | 0.26 | **1.21** | 0.94 |
| conv-pool-conv | 0.12 | 0.13 | 0.57 | 0.66 |
| conv-conv-stride2 | 0.15 | 0.18 | 0.73 | 0.71 |

The one case where Rust is slower (MNIST, one conv layer, single-example) is not caused by the
conv layer. Per op, the Rust conv layer matches or beats numpy's. The cost is in the wide dense
layer after it:

| op, per call | numpy | Rust |
| --- | --- | --- |
| conv forward (28x28, 8 channels) | 81 µs | 58 µs |
| conv downstream | 58 µs | 51 µs |
| conv accumulate_gradient | 41 µs | 40 µs |
| dense forward (32 x 5408) | 29 µs | 58 µs |
| **dense downstream** | 34 µs | **327 µs** |
| **dense accumulate_gradient** | 243 µs | **639 µs** |
| dense apply_accumulated_gradient | 539 µs | 322 µs |

## 1. Dense downstream: stop transposing `W` on every call (high value, small change)

**Stage A (`delta_batch.T @ X`) closed, not merged: no measured gain** (see
`workplans/optimization-1-dense-transposes.md`). **Stage B (`layer_downstream`) done**, as
proposed below: `downstream` went from 308 to 43 µs at 32 x 5408 and from 31 to 5.6 µs at 30 x
784, with results within 4 ULPs (of each vector's largest element) of before. **Stage C
(`X @ W.T` in `forward_batch`) done**: `forward_batch` went from 330 to 58 µs at 32 x 5408 batch
1 and from 2074 to 920 µs at batch 32, and every row of a batched forward is now bit-identical
to the single-example forward. The figures below are from before stage B.

`fused.rs::layer_downstream` computes `matmul(&w.transpose(), delta)`. `RustArray::transpose`
allocates and fills a full transposed copy of `W`. At 32 x 5408 that copy alone measures
**273 µs of the 327 µs call**. `matmul` already has a vector @ matrix case (`delta @ W`, the same
product), which measures **41 µs**: 8x faster, and close to numpy's 34 µs. Its result agrees to
within 2.7e-15.

- **Where it matters:** the first dense layer after a conv front end, whose `W` is wide. That
  layer's `downstream()` runs on every single-example backward step. `hidden_downstream`, which
  every `layer_*hidden_delta` uses (plain, ReLU, dropout), goes through the same transpose. So
  the dense production networks pay it too, though at their usual shapes (`W` of 10 x 30 behind a
  784 -> 30 hidden layer) the copy is small.
- **Numerics:** the vector @ matrix path sums sequentially through `axpy_row`, not through
  `dot_product`'s 4-lane grouping. It is still bit-identical between the scalar and AVX2 paths,
  so still machine-independent. But the Rust results change by a few ULPs, and every
  `rtol`-based parity test should be re-run to confirm it.
- **Same pattern elsewhere:** `linear_preactivation_batch` (`X @ W.T`) and
  `layer_accumulate_gradient_batch` (`delta_batch.T @ X`) also materialize a transpose per call.
  These weren't measured separately. A transposed-operand matmul variant would remove all three.

## 2. Dense accumulate_gradient: fuse the outer product into the add (high value)

**Done, both stages, bit-identical** (see `workplans/optimization-2-dense-accumulate-gradient.md`).
Stage A (one-pass accumulate) took `accumulate_gradient` from 596 to 62 µs at 32 x 5408. Stage B
(the fused single-example SGD step, `sgd_step`) took the whole per-layer step (accumulate +
apply + reset) from 1308 to 64 µs. The figures below are from before either.

`layer_accumulate_gradient` builds `outer(delta, x)` as a new 32 x 5408 array, then
`combine_with_array` allocates a second one for `grad_W + outer`. The outer product alone
measures **367 µs** of the **624 µs** call. One pass that writes
`grad_W[i, j] + delta[i] * x[j]` into a single output buffer would remove one full-size
allocation and one full pass. That is an `axpy_row` per row, which is already AVX2+FMA. Note that
`grad_W` is rebuilt as zeros on every single-example step (`_reset_gradient_accum`), so writing
`delta[i] * x[j]` directly when the accumulator is known to be fresh would drop a second pass.
That, though, is a Python-side contract change.

## 3. Conv forward: the largest share of Rust conv training time (measure first)

In a cProfile of Rust training on MNIST conv-pool-conv, `conv_forward_batch` is **41%** of
profiled time single-example and **51%** mini-batch. That is its own time, and it includes the
extra forward passes the trainers run to evaluate accuracy each epoch. It is already faster than
numpy's conv forward (58 vs 81 µs), so this is about the Rust share, not the gap to numpy.
**The first candidate (don't return `Z`) is done** (see `workplans/optimization-3-conv-forward.md`
stage A): bit-identical, and a small gain, about 14% at N = 512 and within noise at N = 1 and 32.
**The second (skip `cols` when not training) was built and closed** (stage B): at N = 1 `cols`
is only 48 KB, and a single-example evaluation pass without it measured 1-8% slower, not faster.
The op's cost is its per-row `axpy_row` calls on 8-wide rows, the third candidate.
**The third (small output channel counts) is done** (stage C). im2col-then-matmul measured as the
right formulation, but `matmul`'s row kernel wasn't. A register-blocked `matmul_narrow` for `cols
@ W.T` is bit-identical and takes 33-64% less time at N = 1. The same pattern in the conv
accumulate and downstream matmuls has since been fixed the same way (indrajala-math-rust#12).
The candidates as first written:

- **Don't return `Z`.** `conv_forward_batch` builds `Z`, then a separate `A = relu(Z)`, then
  returns both. `Z` only exists to match `ConvArrayLayer.Z`, which a single ReLU-at-zero test
  reads. Returning only `A` and `cols` drops one output-sized allocation and copy per call.
- **Skip `cols` when not training.** Evaluation forward passes (`classify_state` /
  `predict_probabilities`) never call `accumulate_gradient`. For them the cached im2col array is a
  pure cost: N·P x C·k·k, 6084 values for a 28x28 layer. A forward-only variant could fuse
  im2col, the matmul and the scatter without keeping `cols`.
- **Small output channel counts.** With `O = 8`, `cols @ W.T` runs `axpy_row` over rows of only
  8 values, 2 AVX2 lanes' worth, so the per-row loop overhead is a large share. For `O` this
  small, a direct loop that holds `W` in registers, or computing `W @ cols.T` (O rows of N·P,
  long rows for the SIMD path), may beat im2col-then-matmul. This is the plan's "is
  im2col-matmul the right formulation in Rust?" question. The measured answer so far is that it
  already beats numpy, and the remaining cost is concentrated in this op.

## 4. Single-example conv path: the N = 1 reshape (low-to-medium value)

**Done** (see `workplans/optimization-4-single-example-conv-path.md`): the conv and pool ops take
a 1D vector as N = 1, so the single-example path makes no `reshape` calls. Rust's overhead is now
0.3% on UCI digits (was 27-28.5%) and -0.5% on MNIST (was 5.6%). The figures below are from
before.

The Rust conv and pool layers wrap a single example as `(1, n)` with `Array.reshape`, which copies.
Measured on one `ConvSpec(3, 8)` layer (forward + downstream + accumulate_gradient), comparing
the single-example path against calling the batch ops on pre-shaped `(1, n)` input:

| dataset | numpy overhead | Rust overhead |
| --- | --- | --- |
| UCI digits 8x8 | 5.5% (111 vs 105 µs) | **27.1%** (13.8 vs 10.8 µs) |
| MNIST 28x28 | 1.7% (198 vs 195 µs) | 5.6% (158 vs 150 µs) |

In Rust it is about 3–8 µs per example: one input reshape plus the reshapes of `a`/`z`, `delta`
and `dX`, each a Python call and a copy. It matters most at small spatial sizes, where the op
itself is cheap. `Array.reshape` is about 4% of profiled Rust time (68,000 calls in one MNIST
epoch). Options:

- let the conv/pool ops accept a 1D vector as N = 1 and return 1D outputs, so the Python layer
  never reshapes;
- or stop returning `Z` (see 3), which removes one of the reshapes directly.

numpy's `x[np.newaxis]` and `[0]` are views, which is why its overhead is small.

## 5. Mini-batch conversion from Python lists (low value, shared by both backends)

`learn_batch` and `_forward` build a backend array from Python tuples on every call.
`pa.Array(list(state))` for one 784-pixel MNIST row measures 49 µs, and numpy's `np.array` 55 µs.
Both backends pay this, so it doesn't change the ratio, but it is a fixed per-example cost that
neither the Rust nor the numpy maths can remove. Keeping the dataset as one pre-built array and
slicing batches from it would remove it. That changes the trainer interface
(`train.py` takes lists of `(state, label)` tuples), so it is a larger decision than 1–4.

## Batch ops: Rust is slower than numpy at the wide dense shapes

Found by the first run of `demo_layer_op_timing`, which added the batch ops. At batch 32 and 512,
on the 32 x 5408 and 30 x 784 dense shapes, most Rust batch ops take 1.3x to 11x numpy's time
(numpy's matmul calls OpenBLAS). For example, at batch 32, 32 x 5408: `forward_batch` 2296 vs
567 µs, `downstream_batch` 2985 vs 272 µs, `accumulate_gradient_batch` 3904 vs 755 µs. At batch 1,
`forward_batch` at 32 x 5408 is 312 vs 32 µs, which is mostly the `W.T` copy (optimization 1
stage C, since done: 58 µs). Why the end-to-end mini-batch ratios above don't show a gap this large hasn't been
measured. Nothing here is acted on yet: optimization 1 removes the
transposes, and the numbers the run gives after it will show how much of the gap remains. (They
did: see "Remaining candidates" below for the numbers after optimization 1.)

Optimization 1 stage A ruled out one cause: removing the `delta_batch.T` copy from
`accumulate_gradient_batch` left it unchanged (within ±4%, the spread between runs is 10-25%).
That copy is only batch x `M`, tiny next to the matmul. So that op's gap is in the matmul itself
or the separate `grad_W +` pass. One untested guess: at batch 32, 32 x 5408 the matmul is ~5.5M
flops, just over `matmul_2d`'s 4M threading threshold, so it pays thread spawns for little work.

Optimization 1 stage C removed the `W.T` copy from `forward_batch`. At 30 x 784, batch 32 it is
still 271 vs 60 µs, 4.5x numpy, so that gap is in the kernel. An estimate, not measured: each
`dot_product` has one 4-lane accumulator, so every FMA waits on the one before it (about 4
cycles). At 752K multiply-adds, 188K dependent FMAs is about 250 µs at 3 GHz, which is the
measured time. Computing 4 outputs at once (4 rows of `W` against one `x`), each with its own
accumulator in `dot_product`'s grouping, would hide that latency and keep every output's bits.
The same kernel would speed up the single-example `W @ x` too, and has to be used for both, to
keep the batch-row = single-example property.

**Done, and the estimate held** (indrajala-math-rust#13). `dot_products_into` runs 8, then 4,
then 2 rows of `W` against one vector at once, each in `dot_product`'s exact grouping. The matrix
@ vector case and `matmul_nt` share it, so it is bit-identical everywhere and batch rows still
equal the single-example forward. New crate tests pin every output to an exact `Fraction`-FMA
emulation of the grouping; they also pass on the old build.

Rust µs per call, before → after:

| shape | op | before | after |
| --- | --- | --- | --- |
| 30 x 784 | forward | 9.0 | 3.7 |
| 30 x 784 | batch 32 | 259.8 | 92.9 (numpy about 60) |
| 32 x 5408 | forward | 57.1 | 19.7 (numpy about 31) |
| 32 x 5408 | batch 32 | 1021 | 672 |
| 32 x 5408 | batch 512 | 7338 | 4841 |

At 30 x 784 and 16 x 784, batch 512 barely moves (1418 → 1350 µs). It is threaded, and memory
traffic dominates there rather than FMA latency. Reusing each loaded `W` block across several
rows of `X` would target that; not tried.

Dense MNIST 784 -> 30 -> 10, one 60000-example epoch, identical weights, median of 3, builds
alternated (Rust seconds; numpy's own time varied 15.2-17.2 s between the runs):
- Single-example: Rust 5.49 → 5.06 s.
- Mini-batch 32: Rust 5.45 → 4.69 s, ratio 0.665 → 0.557.

Test accuracies were identical between the builds.

## Remaining candidates (as of 2026-09-24)

What is still open after optimizations 1-4 and the dense dot-product fix. Candidate 1's kernel
part is done (indrajala-math-rust#14); nothing else below has been measured as a fix. The per-op
numbers come from `demo_layer_op_timing` on 2026-09-24 unless marked otherwise, Rust vs numpy µs
per call.

1. **Dense batch downstream and accumulate at the wide shapes** (the "Batch ops" section above,
   with numbers after optimization 1):

   | shape | op | batch | numpy | Rust | Rust/numpy |
   | --- | --- | --- | --- | --- | --- |
   | 32 x 5408 | `downstream_batch` | 32 | 296 | 2800 | 9.5x |
   | 32 x 5408 | `accumulate_gradient_batch` | 32 | 716 | 3911 | 5.5x |
   | 32 x 5408 | `hidden_delta_batch` | 32 | 1533 | 3588 | 2.3x |
   | 30 x 784 | `downstream_batch` | 512 | 452 | 3777 | 8.4x |
   | 30 x 784 | `accumulate_gradient_batch` | 512 | 776 | 3906 | 5.0x |

   These are the largest remaining gaps to numpy. The only candidate cause on record is the
   untested guess above: `matmul_2d` starting threads just over its 4M-flop threshold. The odd
   shape of the numbers fits it: at 32 x 5408, `downstream_batch` is 9.5x numpy at batch 32 but
   1.4x at batch 512. First step: profile one op at batch 32 with threading forced off, then on.

   **Kernel part done** (indrajala-math-rust#14). Re-measured with a focused benchmark (20 ms
   loops, median of 9), the gaps were larger than the harness showed: `downstream_batch` at
   32 x 5408, batch 32 was 13.9x numpy. The threading guess was not the main cause. At batch 8
   and 16 the same op stays under the threshold, runs on one thread, and was already 3.5x and
   9.4x. The kernel was: `matmul_2d` added one row of `b` into the output row per `k`
   (`axpy_row`), so each output row was loaded and stored `K` times, and at 5408 wide it is 43
   KB. `matmul_2d` now holds 16-column output tiles in registers across all of `k`, in row blocks
   of about 16 KB of `a`. That is `matmul_narrow`'s kernel with row blocks added, and the two now
   share it. It is bit-identical.

   Rust µs per call, before → after (two passes each, builds alternated; this machine's spread
   between passes is 20-30%):

   | shape | op | batch | before | after | numpy |
   | --- | --- | --- | --- | --- | --- |
   | 32 x 5408 | `downstream_batch` | 16 | 1184-1220 | 514-528 | 127-146 |
   | 32 x 5408 | `downstream_batch` | 32 | 2883-3001 | 3005-3075 | 228-258 |
   | 32 x 5408 | `downstream_batch` | 512 | 19717-22825 | 14413-15990 | 11652-13609 |
   | 32 x 5408 | `accumulate_gradient_batch` | 8 | 621-1008 | 444-471 | 202-239 |
   | 32 x 5408 | `accumulate_gradient_batch` | 512 | 26548-31629 | 15345-17118 | 12891-14674 |
   | 30 x 784 | `downstream_batch` | 32 | 226-1604 | 78 | 45-414 |
   | 30 x 784 | `hidden_delta_batch` | 32 | 212-242 | 82-96 | 79-115 |
   | 30 x 784 | `accumulate_gradient_batch` | 32 | 189-204 | 94-96 | 77-93 |

   End to end, dense MNIST 784 -> 30 -> 10 mini-batch 32, one epoch, median of 3, two runs per
   build: Rust 4.54-4.66 → 4.34-4.51 s. The conv demo is unchanged except MNIST `conv`
   mini-batch, Rust/numpy 0.59 → 0.54. Test accuracies are identical between the builds.

   **Still open: threading.** Above the 4M-flop threshold the gain mostly disappears. At
   32 x 5408, batch 32, `downstream_batch` is still about 3000 µs, against about 550 µs for the
   same product on one thread. Measured on the new kernel with threading forced on:
   - **Starting threads costs 100-200 µs per call.** A 64x64x64 matmul takes 28 µs on 1 thread
     and 120-200 µs on 2-8.
   - **Splitting by rows makes every thread stream all of `b`.** 32 x 5408 at batch 32 takes
     about 550 µs on 1 thread and 550-630 µs on 8.
   - **Large products still scale.** `(32, 512) @ (512, 5408)` goes from 32 ms on 1 thread to
     6-7 ms on 8.

   Candidates: split by columns when `b` is the larger operand, a persistent thread pool, or a
   higher threshold. Any of them keeps every output's bits, since each output is still computed
   by one thread. `matmul_nt` (every `forward_batch`) and `matmul_narrow` (the conv ops) use
   the same `for_each_row_range` threading, so a fix there applies to them too (see 2).

2. **Dense `forward_batch` at large batches is memory-bound.** The dot-product fix left batch 512
   at 30 x 784 and 16 x 784 barely changed (1418 → 1350 µs, focused benchmark). Each row of `X`
   re-reads all of `W`. Candidate: a register block of 2-4 rows of `X` against the same `W`
   rows, so each `W` load serves several outputs. Each output keeps `dot_product`'s grouping, so
   this is bit-identical.

   Re-measured on 2026-09-24 (focused benchmark, 20 ms loops, median of 9): 32 x 5408 is 1.2x
   numpy at batch 32 (682-971 µs) but 5.6x at batch 64 (3395 vs 608), and 30 x 784 is 3.4x at
   batch 512. The jump from batch 32 to 64 is 3.5-5x the time for twice the work, and both are
   past the threading threshold. `matmul_nt` splits `X`'s rows across threads, so every thread
   streams all of `W` (1.4 MB at 32 x 5408). This is likely the threading problem in 1, not only
   memory traffic. Measure with threading off first.

3. **Conv `forward_batch` at N = 32 costs more than 32 single-example calls.** Measured at 28x28
   (2280 vs 1715 µs, optimization 3 stage B, before stage C). The likely cause, unmeasured: the
   1.5 MB `cols` falls out of cache between im2col and the matmul. Candidate: im2col and multiply
   one block of output positions at a time, keeping `cols` for the backward pass. Re-measure
   first, since stage C changed the matmul.

4. **Conv accumulate with a large `cols` has no cache blocking.** `matmul_narrow` gave no gain on
   `conv_accumulate_gradient_batch` at 13x13x8, N = 32 (+2%, +4%, -1% at O = 4, 8, 32), where
   `cols` is 0.3-2.2 MB and `matmul_2d`'s old `k`-blocking made up for its row overhead
   (`workplans/optimization-3-conv-forward.md`, stage C follow-up). Since
   indrajala-math-rust#14 both functions share one kernel, `tiled_row_range`, and that
   `k`-blocking is gone. `matmul_narrow` uses 1-row blocks. Moving it to `matmul_2d`'s 16 KB
   row blocks made this op slower at 28x28, N = 512 (35-39 vs 30-32 ms), so row blocking
   isn't the fix. Candidate: block over `k` with the tile's accumulators kept across `k` blocks
   (stored and reloaded, not reset). The per-output `k` order is unchanged, so it is
   bit-identical.

5. **`max_pool_forward_batch` is now the second-largest Rust conv op**, about 10% of profiled
   conv-pool-conv training (single-example: 0.10 s of 1.0 s, 6000 calls). It has never been
   examined. It does no arithmetic, so the likely costs are the per-window index arithmetic and
   the separate `argmax` output.

6. **Dense single-example `downstream` at 32 x 5408 is still 1.5x numpy** (50 vs 33 µs), after
   optimization 1 stage B took it from 308 to 43 µs. It is `delta @ W` through `axpy_row` over
   32 rows 5408 wide, so it is bound by streaming `W` (1.4 MB) rather than by FMA latency. It
   affects the conv-tail single-example step only. Low value.

7. **The per-op harness's batch rows are noisy.** At batch 32 and 512 it runs 10 calls per loop.
   On 2026-09-24 it reported 30 x 784 `forward_batch` at batch 512 as 3469 µs, where a focused
   benchmark (median of 9 loops of about 20 ms each) measured 1350. Before acting on any batch
   row above, re-measure with more calls per loop, or raise the harness's batch call count.
   Candidate 1's re-measure found it can also understate a gap: 13.9x numpy where the harness
   had 9.5x.

Optimization 5 (the dataset as one array) is still open as well, above.

## Not optimizations, but found in the same measurements

- **Single-example training runs diverge between backends, as they do within numpy.** In the
  demo, numpy and Rust networks trained from the same weights sometimes end up classifying only
  71–83% of test rows the same. The cause is not a Rust bug. Starting numpy against itself with
  one weight nudged by 1 ULP diverges just as much (73–81%). Both pairs stay within about 1e-15
  through the first UCI epoch, then both jump to differences of about 0.7 in the second. Long
  single-example runs at learning rate 0.5 are chaotically sensitive to rounding. Any future
  optimization that changes summation order (1, 2 and 3 all do) will shift these end-of-training
  numbers the same way. It should be judged by the step-by-step parity tests (per-step
  agreement to about 1e-15), not by end-of-run accuracy matching exactly.
- **The demo's mini-batch runs barely train.** They reach about 10% accuracy with lr 0.5 in 1–2
  epochs, and lr 2, 4 and 8 don't fix every configuration. The mini-batch timings are valid; the
  accuracy columns for them are not informative.
