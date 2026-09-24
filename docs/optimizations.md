# Optimizations

Performance work on the Rust backend (`rust/`, the `indrajala-math-rust` crate) against numpy,
the benchmark it is compared with. It started from the Rust CNN timing (#317-#320 and the Rust
CNN stages), which found where Rust was slower than numpy or slower than it needed to be. Each
item is measured before and after, and must keep every parity test passing.

Two open items have a full workplan of their own: candidate 1,
[`workplans/optimization-6-matmul-threading.md`](workplans/optimization-6-matmul-threading.md),
and candidate 7,
[`workplans/optimization-5-dataset-as-array.md`](workplans/optimization-5-dataset-as-array.md).
Everything else is below.

## Where things stand (2026-09-24)

Rust / numpy wall-clock ratio end to end (below 1 means Rust is faster). The conv rows are from
`demo_conv_rust_vs_vectorized_digit_recognition` after indrajala-math-rust#14:

| architecture | UCI digits, single | UCI digits, mini-batch | MNIST subset, single | MNIST subset, mini-batch |
| --- | --- | --- | --- | --- |
| conv | 0.13 | 0.17 | 0.21 | 0.54 |
| conv-pool-conv | 0.08 | 0.09 | 0.31 | 0.44 |
| conv-conv-stride2 | 0.10 | 0.12 | 0.36 | 0.48 |

At the start, MNIST conv single-example was 1.21 (Rust slower) and conv-pool-conv mini-batch was
0.66. Dense MNIST 784 -> 30 -> 10, one 60000-example epoch: Rust/numpy about 0.33
single-example and 0.49-0.53 mini-batch 32.

Per op, the single-example ops are all at or better than numpy. The large gaps left are all in
batch ops **past `matmul`'s 4M-flop threading threshold**. Rust / numpy µs per call, two passes
of the focused benchmark (see "How to measure"), after #14:

| shape | op | batch | numpy | Rust | Rust/numpy |
| --- | --- | --- | --- | --- | --- |
| 32 x 5408 | `downstream_batch` | 32 | 238-258 | 3005-3075 | 12-13x |
| 32 x 5408 | `accumulate_gradient_batch` | 32 | 449-453 | 3084-3306 | 7x |
| 32 x 5408 | `forward_batch` | 512 | 5732-8715 | 12097-13058 | 1.4-2.3x |
| 30 x 784 | `downstream_batch` | 512 | 477-834 | 2982-3877 | 4-8x |
| 30 x 784 | `accumulate_gradient_batch` | 512 | 949-969 | 3477-3832 | 4x |
| 30 x 784 | `forward_batch` | 512 | 805-1453 | 2968-4290 | 2-5x |
| 30 x 784 | `downstream_batch` | 32 | 45 | 78 | 1.7x |
| 30 x 784 | `accumulate_gradient_batch` | 32 | 88-93 | 94-96 | 1.0-1.1x |

## Open candidates, in priority order

1. **Threading in `for_each_row_range`** (`rust/src/linalg.rs`). Every threaded matmul goes
   through it: `matmul_2d` (dense batch downstream and accumulate), `matmul_nt` (every
   `forward_batch`) and `matmul_narrow` (the conv ops). Measured on the tiled kernel with
   threading forced on:
   - **Starting threads costs 100-200 µs per call.** `std::thread::scope` spawns fresh threads
     on every call. A 64x64x64 matmul takes 28 µs on 1 thread and 120-200 µs on 2-8.
   - **Splitting by rows makes every thread stream all of `b`.** At 32 x 5408, batch 32,
     `downstream_batch` takes about 550 µs on 1 thread and 550-630 µs on 8. With the threads
     it is about 3000 µs in the op table above.
   - **Large products still scale.** `(32, 512) @ (512, 5408)` goes from 32 ms on 1 thread to
     6-7 ms on 8. 30 x 784 at batch 512 goes from 1.3-1.7 ms to 0.8 ms on 4 threads (no better
     on 8).

   Candidates: split by columns when `b` is the larger operand, a persistent thread pool, or a
   higher threshold. Each output is still computed by one thread under any of them, so every
   output keeps its bits. Measure each against a single-thread baseline at the shapes in the
   table above. This machine has 4 cores / 8 threads, and the threshold is machine-dependent.
   Workplan: [`workplans/optimization-6-matmul-threading.md`](workplans/optimization-6-matmul-threading.md).

2. **Dense `forward_batch` at large batches** (`matmul_nt`). 32 x 5408 is 1.2x numpy at batch 32
   (682-971 µs) but 5.6x at batch 64 (3395 vs 608): 3.5-5x the time for twice the work, and
   past the threading threshold. So it is likely candidate 1 first. Measure with threading off
   before anything else. Once that is settled, the remaining cost is memory traffic: each row of
   `X` re-reads all of `W`. A register block of 2-4 rows of `X` against the same `W` rows would
   let each `W` load serve several outputs. Each output keeps `dot_product`'s grouping, so it
   is bit-identical.

3. **Conv `forward_batch` at N = 32 costs more than 32 single-example calls.** Measured at 28x28
   (2280 vs 1715 µs, before the `matmul_narrow` kernel). The likely cause, unmeasured: the 1.5 MB
   `cols` falls out of cache between im2col and the matmul. Candidate: im2col and multiply one
   block of output positions at a time, keeping `cols` for the backward pass. Re-measure first:
   the kernel has changed since.

4. **Conv accumulate with a large `cols`.** `D @ cols` goes through `matmul_narrow`, which gave
   no gain at 13x13x8, N = 32 (+2%, +4%, -1% at O = 4, 8, 32; `cols` 0.3-2.2 MB), where the old
   `matmul_2d` had `k`-blocking. That blocking is gone since #14. Moving the op to `matmul_2d`'s
   16 KB row blocks made it slower at 28x28, N = 512 (35-39 vs 30-32 ms), so row blocking isn't
   the fix. Candidate: block over `k` and store and reload the tile's accumulators between `k`
   blocks rather than resetting them. The per-output `k` order is unchanged, so it is
   bit-identical.

5. **`max_pool_forward_batch`** is the second- or third-largest Rust conv op, about 10% of
   profiled conv-pool-conv training (0.10 s of 0.9 s single-example, 6000 calls). It has never
   been examined. It does no arithmetic, so the likely costs are the per-window index arithmetic
   and the separate `argmax` output.

6. **Dense single-example `downstream` at 32 x 5408 is 1.5x numpy** (50 vs 33 µs, per-op
   harness). It is vector @ matrix, `delta @ W` through `axpy_row`: 32 load/FMA/store passes
   over a 43 KB output row, the pattern #14 removed from `matmul_2d`. The same product as a
   one-row `downstream_batch` goes through the tiled kernel and measured 29 vs numpy's 36 µs.
   Candidate: route vector @ matrix through `tiled_row_range` as a one-row matrix. It is the
   same FMA chain, so bit-identical, and the crate tests that compare the two would need
   another reference. Affects the conv-tail single-example step only. Low value.

7. **The dataset as one backend array** (optimization 5, low value). Both backends convert Python
   tuples to an array on every call: `pa.Array(list(state))` for one 784-pixel MNIST row is 49
   µs, numpy's `np.array` 55 µs. That doesn't change the ratio, but it is a fixed cost neither
   backend's maths can remove, and it is a larger share now the maths is faster. It changes the
   trainer interface, so its workplan starts with a go/no-go measurement:
   [`workplans/optimization-5-dataset-as-array.md`](workplans/optimization-5-dataset-as-array.md).

## How to measure

- **Per op, quick survey:** `python -m indrajala_ml.demos.demo_layer_op_timing`. It times every
  dense and conv layer method, single-example and batch 1/32/512, numpy and Rust interleaved,
  300 calls per loop (300 // batch for batch ops, at least 10), median of 5 loops. **Its batch
  rows are noisy and can be off either way**: it reported 30 x 784 `forward_batch` at batch 512
  as 3469 µs where a focused benchmark measured 1350, and `downstream_batch` at 32 x 5408, batch
  32 as 9.5x numpy where a focused benchmark measured 13.9x. Use it to find candidates, not to
  judge them.
- **Per op, focused:** time the op in loops of about 20 ms, median of 9, numpy and Rust
  interleaved loop by loop. Two passes per build at least. This is the number to quote.
- **End to end:** `python -m indrajala_ml.demos.demo_conv_rust_vs_vectorized_digit_recognition`
  (about 3 minutes; median of 5 interleaved runs from identical initial weights, UCI digits and
  a 2000-row MNIST subset, single-example and mini-batch 32, plus a cProfile of Rust time by
  op). For a dense op change, also one epoch of dense MNIST 784 -> 30 -> 10 from identical
  weights, single-example and mini-batch 32, median of 3.
- **Old vs new builds:** alternate the builds (old, new, old, new) and run each benchmark on
  both. Builds take about 6 s (`./cli build-rust`). Commit the crate change before switching,
  and switch with `git checkout main -- src/linalg.rs` and back, not a stash.
- **This machine** (Ryzen 7 3700U laptop, 4 cores / 8 threads, 512 KB L2 per core, 4 MB L3)
  varies 20-30% between passes, sometimes more. A background IDE made a first measurement
  unusable once. Treat changes under about 20% as noise unless both passes agree, and re-check a
  surprising result with the build order reversed.
- **Never time pure Python.** It is for correctness and parity only.

## Rules for an optimization PR

- **Two repos.** Crate changes land in `indrajala-math-rust` (`rust/`) first, with its own
  numpy-only tests. Then a PR here bumps the submodule and runs the full suite
  (`./cli build-rust && ./cli test`). Neither merges until both pass.
- **Bit-identical claims are tested, not assumed.** A change claimed as bit-identical gets a
  crate test that pins the new op with exact equality (`==` on `tolist()`, not `approx`), at
  shapes that reach every kernel path, threading and blocking threshold. The test must also pass
  on the old build, and a mutation (a second rounding, a changed start value or order) must
  fail it. If exact equality fails, the change is bit-changing.
- **Bit-changing protocol.** For a change to summation order:
  1. The existing `rtol` parity tests (`tests/test_*fused_layer_ops.py`, the step-by-step
     network parity tests) must pass unchanged.
  2. If an end-to-end pinned result moves (for example the Rust conv network's 0.9875 /
     epoch 10 / 0.925), don't loosen it silently. Compare it with a 1-ULP control: nudge one
     initial weight by 1 ULP on the *old* code. If the pin moves by a similar amount, it is
     rounding sensitivity (see "Other findings"); update the pin and record the control in
     the PR. If not, treat it as a bug.
  3. Record the max abs and max ULP difference from the old op at the benchmark shapes.
- **Measure first, and close what doesn't pay.** Each stage is its own PR, merged before the
  next starts. A stage that measures no gain is closed with its numbers recorded here, not
  merged. Every PR quotes the before/after per-op rows for the ops it touches and the end-to-end
  ratios.

## Kernel invariants

Every Rust result is bit-identical between the scalar and AVX2+FMA paths, so a training run
doesn't depend on the machine. Each kernel has one fixed summation order:

- **Matrix @ vector and `matmul_nt` (`X @ W.T`, every forward):** `dot_product`'s grouping.
  Lane `j` of a 4-lane accumulator sums indices `j, j+4, ...` by FMA, then
  `(l0 + l1) + (l2 + l3)`, then the `k % 4` tail sequentially. `dot_products_into` runs 8, then
  4, then 2 rows at once in that grouping. A batched forward's rows equal the single-example
  forward exactly.
- **Matrix @ matrix (`matmul_2d`, `matmul_narrow`) and vector @ matrix:** one FMA chain per
  output, `k` increasing from 0.0. `tiled_row_range` holds 16-column output tiles in registers
  across all of `k`: `matmul_2d` in row blocks of about 16 KB of `a`, `matmul_narrow` one row at
  a time. Vector @ matrix goes through `axpy_row` with the same chain.
- **Threading** splits output rows across threads, so each output is computed by one thread and
  the thread count can't change any value.
- These orders differ from numpy's in the last few ULPs, so parity with numpy is checked with
  `rtol`. Crate tests pin each order exactly against a `Fraction`-emulated FMA reference.

## Completed

Crate PR numbers are `indrajala-math-rust`'s. "Bit-identical" means every output kept its bits.

| change | crate PR | result |
| --- | --- | --- |
| One-pass dense `accumulate_gradient` (no `outer` temporary) | #5 | 596 → 62 µs at 32 x 5408; bit-identical |
| Fused single-example SGD step, `layer_sgd_step` (a Python contract change: `sgd_step` on the Rust layers, with a fallback for momentum/Adam/L2/conv) | #6 | per-layer step 463 → 64 µs at 32 x 5408; dense MNIST epoch -15%; bit-identical |
| `layer_downstream` as `delta @ W`, no `W.T` copy | #7 | 308 → 43 µs at 32 x 5408; bits changed (within 4 ULPs) |
| `matmul_nt` for `X @ W.T` in `forward_batch`, no `W.T` copy | #8 | 330 → 58 µs at 32 x 5408, batch 1; batch rows now equal the single-example forward exactly |
| Conv forward stops returning `Z` | #9 | about 14% at N = 512, noise at N = 1 and 32; bit-identical |
| Conv and pool ops take a vector as N = 1, so no reshapes | #10 | Rust N = 1 overhead 28.5% → 0.3% (UCI conv layer); bit-identical |
| `matmul_narrow`: register-held output rows for conv `cols @ W.T` | #11 | conv forward 33-64% less at N = 1, 13-63% at N = 32; MNIST conv-pool-conv mini-batch 0.63 → 0.46; bit-identical |
| Conv downstream and accumulate via `matmul_narrow` | #12 | downstream 4-56% less, accumulate 16-43% less in 21 of 24 configurations; bit-identical |
| `dot_products_into`: 8/4/2 rows at once in `dot_product`'s grouping | #13 | 30 x 784 forward 9.0 → 3.7 µs, batch 32 260 → 93 µs; bit-identical |
| `matmul_2d` register-tiled, sharing `matmul_narrow`'s kernel | #14 | unthreaded dense batch downstream/accumulate 0.1-0.6x their old time; mini-batch dense MNIST epoch about -4%; bit-identical |

Closed with no measured gain, kept as findings:

- **Transposed-left matmul for `accumulate_gradient_batch`** (crate branch `matmul-tn`,
  `e59c511`). Bit-identical, and within ±4% at every shape. The copy it removed was
  `delta_batch.T`, batch x `M`, small next to the matmul.
- **Forward-only conv path for evaluation** (crate branch `conv-infer-batch`, `ee432d0`), which
  skipped storing `cols`. At N = 1 `cols` is only 48 KB, and a whole-network evaluation pass was
  1-8% slower with it.
- **Conv formulations** (crate branch `conv-forward-formulations-proto`, `ef85831`). `W @ colsT`
  had the fastest forward at small `O`, but its backward cost more, and it lost the training step
  to the `matmul_narrow` kernel in 17 of 24 configurations. A direct kernel with no `cols` only
  paid at large `O` and N, and can't serve training.
- **The threading guess.** The dense batch gaps were first blamed on `matmul_2d` threading
  just past its threshold. At batch 8 and 16 the same ops ran on one thread and were already
  3.5-9.4x numpy; the kernel was the main cause (#14). Threading is a real but separate cost
  (candidate 1).
- **Row blocks for `matmul_2d`.** 1-row blocks were 2-3x slower than 16 KB blocks where `K` is
  in the hundreds (`(32, 512) @ (512, 5408)`: 40-47 vs 15-16 ms). One block for all rows was
  close to 16 KB blocks but not better.

## Other findings from the same measurements

- **Single-example training is chaotically sensitive to rounding.** numpy and Rust networks
  trained from the same weights can end up classifying only 71-83% of test rows the same. numpy
  against itself, with one weight nudged by 1 ULP, diverges just as much (73-81%). Both pairs
  stay within about 1e-15 through the first UCI epoch, then jump to differences of about 0.7 in
  the second. So a bit-changing optimization shifts end-of-training numbers the same way. Judge
  it by the step-by-step parity tests (per-step agreement to about 1e-15), not by end-of-run
  accuracy.
- **The conv demo's mini-batch runs barely train.** They reach about 10% accuracy at lr 0.5 in
  1-2 epochs, and lr 2, 4 and 8 don't fix every configuration. Their timings are valid; their
  accuracy columns are not informative.
