# Optimizations: history

Part of the optimization record; the index is [../optimizations.md](../optimizations.md).

The audit trail: every finished or closed change with its measurements, and each candidate as
recorded while it was open. Newest work first. Current numbers are in
[Where things stand](status.md); a number replaced there moves to
[Superseded numbers](#superseded-numbers) with the PR that replaced it. A candidate that closes
moves here from [Open candidates](candidates.md) or [Future work](future-work.md), and keeps its
row in the index's candidate table.

## Old names

Four workplans ran separately before this record was one document, and were folded in:

| old name | now |
| --- | --- |
| optimization 5, the dataset as one backend array | candidate 1 (done) |
| optimization 6, threading | finished (crate #15, #16); its findings are in [Threading](kernels.md#threading), and what it left open is candidate 9 |
| optimization 7, dense batch ops at short `k` | candidate 6 (stage 0 done, #354); its long-`k` finding is candidate 7 |
| the batch-size-scaling study (#365-#368) | finished; its timing stage measured candidates 1 and 7 and the trainers' accuracy passes on full MNIST |

## Crate changes, by PR

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
| `set_matmul_threading` override, and tests that the thread count can't change any kernel's bits | #15 | measurement and test infrastructure; no default behaviour changed |
| Threading threshold 4M → 8M flops, policy in `matmul_thread_count` | #16 | MNIST conv mini-batch 32 epoch 0.879 → 0.770 s (-12%); batch 512 and dense MNIST unchanged; bit-identical |
| `Array.from_rows`, `row` and `take_rows`, for the dataset as one backend array (candidate 1; the Python side is #373-#374) | #19 | dense MNIST Rust epoch at B = 32 3.72 → 1.69 s, single-example 4.60 → 2.57; training bit-identical |
| `matmul_nt` in 4 x 2 register tiles (4 rows of `X` against 2 rows of `W`; 2 x 4, 3 x 3 and 2 x 2 measured slower or tied) | #17 | unthreaded dense `forward_batch` 0.6-0.9x its old time (second pass: 32 x 5408, batch 32: 771-916 → 545-555 µs; 30 x 784, batch 32: 102-110 → 76-77); epochs within noise; bit-identical |
| A batched training-set accuracy pass: `classify_rows`, `forward_batch` over 32-row chunks (candidate 2) | - | dense MNIST epoch at B = 32 1.61 → 1.31 s in Rust, 4.85 → 2.44 in numpy; predictions equal, pinned results unchanged |
| Vector @ matrix (single-example `layer_downstream`) through `tiled_row_range` as a one-row product, `axpy_row` removed (candidate 3) | #20 | 43.8-48.1 → 26.2-26.4 µs at 32 x 5408 (numpy 29.0-29.3); MNIST conv single-example Rust epoch about -6%; bit-identical |
| Conv `forward_batch` one example at a time: im2col, the product into one reused `(P, O)` buffer, `A` appended in order; no zeroed batch buffers (candidate 4) | #21 | N = 32 1351-1694 → 920-1010 µs, N = 512 34.0-40.5 → 15.0-15.7 ms; `conv_forward_batch` in the MNIST conv B = 32 epoch 204-221 → 161-179 ms; bit-identical |

## Candidate 4: conv forward_batch one example at a time

**Conv forward_batch one example at a time** (candidate 4; #383 stage 0, crate #21,
2026-09-24). `conv_forward_batch` runs im2col, the product and the bias + ReLU per example:
each example's rows are appended to `cols` (still kept whole for the backward pass), its `P`
rows of `cols @ W.T` go through `tiled_row_range` (`matmul_narrow`'s kernel, now `pub(crate)`)
into one `(P, O)` buffer reused for every example, and its row of `A` is appended in
channel-major order. Nothing batch-sized is zero-filled, and `A` is written in order instead of
scattered. The product no longer threads; at N = 512 threading had given nothing (35.6-37.8 ms
against 34-37 on one thread). No `unsafe`. The existing exact tests (`A` against the crate's
matmul, bias, scatter and ReLU; `cols` against the brute-force definition, over every kernel
path) pass unchanged, and bias after the max (30 failures), the product from the previous
example's rows (22), the wrong channel in the gather (29) and one `cols` value 1 ULP off in the
last example (6) each failed them. Crate suite 894 passed; full suite passed with no pin moved.

Focused benchmark, builds alternated old, new, old, new, default threads and one thread, glibc's
allocator defaults and raised thresholds (`--malloc both`); µs per call:

| N | old | new |
| --- | --- | --- |
| 1 | 26.3-28.5 | 25.7-26.6 |
| 32 | 1351-1694 | 920-1010 |
| 512 | 34.0-40.5 ms | 15.0-15.7 ms |

The new op faults 676 times a call at N = 512 with glibc's defaults (0 before), at no cost in
time (the raised-threshold rows are the same).

- **In the epoch** (the demo's 2000 MNIST rows, one epoch, cProfile via `rust_op_breakdown`, 3
  processes per build, builds alternated old, new, new, old, old, new; the mini-batch counts
  include the 4000 or 8000 per-row accuracy-pass calls):

  | architecture, trainer | `conv_forward_batch` old | new | profiled epoch old | new |
  | --- | --- | --- | --- | --- |
  | conv, B = 32 | 204-221 ms | 161-179 ms | 0.691-0.712 s | 0.631-0.674 s |
  | conv-pool-conv, B = 32 | 414-427 ms | 358-362 ms | 0.926-0.954 s | 0.883-0.895 s |
  | conv, single | 172-222 ms | 155-169 ms | 0.821-0.913 s | 0.794-0.853 s |
  | conv-pool-conv, single | 359-367 ms | 340-348 ms | 0.815-0.837 s | 0.802-0.818 s |

- **Epoch time could not resolve it**: `prepared_dataset_timing.py --configs "conv B=32" "conv
  single"`, 9 runs per build in calls alternated old, new, new, old, old, new, read the Rust B =
  32 epoch 0.645 → 0.609 s (-5.6%), but numpy's, which the change can't touch, moved -5.1% in
  the same calls. The profile above measures the op directly.

**Stage 2, Rust conv's batched accuracy pass: closed, no gain** (2026-09-24, on the new build).
`scripts/accuracy_pass_timing.py`, which now covers all three demo architectures, 9 processes
per cell (an earlier 5-repeat run, disturbed by other load, agreed); seconds per pass, 0
predictions differing anywhere:

| Rust network | per row | batched 32 | batched faster in |
| --- | --- | --- | --- |
| conv | 0.104 (0.095-0.127) | 0.103 (0.098-0.111) | 6 of 9 |
| conv-pool-conv | 0.163 (0.158-0.180) | 0.186 (0.184-0.219) | 0 of 9 |
| conv-conv-stride2 | 0.141 (0.139-0.153) | 0.169 (0.161-0.196) | 0 of 9 |

(numpy's batched pass saves 0.05-0.20 s on each.) A cProfile of one pass by crate op puts the
loss in `conv_forward_batch` itself: per row against batched, 47.8 against 55.4 ms (conv), 109.5
against 136.0 (conv-pool-conv), 120.5 against 147.2 (conv-conv-stride2); then
`max_pool_forward_batch` (31.5 against 37.0) and `take_rows` (3-4 ms). The batched op still
writes a whole-batch `cols` (1.56 MB at N = 32) that inference never reads, while single calls
reuse the same small hot buffers. So the lead is a forward that skips `cols` in batched
evaluation: `conv-infer-batch` was closed on N = 1 evidence (48 KB there), which doesn't cover
N = 32. The override in `ConvRustArrayMultiClassBackpropClassifierNetwork.classify_rows` stays.

Two more things the stage 2 runs showed:
- **Batched evaluation doesn't gain in the dense tail either.** In the conv network's pass the
  5408 -> 32 -> 10 tail took 43.7 ms batched (`layer_forward_batch`) against 40.2 ms per row
  (`layer_forward`), cProfile's per-call overhead favouring the batched side. That matches the
  per-op table in [Where things stand](status.md): Rust `forward_batch` at 32 x 5408, batch 32, is 470-659 µs, 15-21 µs a row,
  no cheaper than a single-example forward. So a no-`cols` conv forward alone would leave the
  pass near a tie for the one-conv network.
- **Chunks of 512 are worse still in Rust**: batched 512 took 0.096 (conv), 0.232
  (conv-pool-conv) and 0.262 s (conv-conv-stride2) against per-row 0.104, 0.163 and 0.141; numpy
  at 512 also lost its gain on conv-conv-stride2 (0.425 against 0.384). 32 stays the chunk size.

The candidate and its stage 0, as recorded when it was open:

**Conv `forward_batch` costs more per example than single-example calls.** Re-measured
2026-09-24 at 28x28, `ConvSpec(3, 8)`, Rust on one thread (focused benchmark, two passes):
26.5-28.7 µs for one example, 1365-1790 µs at N = 32 (1.5-2.1x the 848-918 of 32 single calls)
and 36.2-38.4 ms at N = 512 (2.5-2.8x). Before the `matmul_narrow` kernel it was 2280 against
1715 µs at N = 32 (1.3x), so the gap has widened as the single-example path got faster. The
other conv batch ops cost less extra at N = 32: downstream 1020-1023 µs against 800-867 for 32
calls (1.2-1.3x), accumulate 1037-1106 against 899-934 (1.1-1.2x; candidate 8 at N = 512). In
the MNIST conv mini-batch 32 epoch the 63 batch calls took about 96 ms (profiled, the 4000
accuracy-pass calls taken out at 27 µs each), so at the single-example rate they would save
about 40 ms, 5% of the worst end-to-end cell. It is also what would let conv networks gain
from candidate 2 (Rust conv keeps the per-row accuracy pass until then).

**Stage 0 (2026-09-24): the cause is zero-filling and cache traffic past L2, not page faults
and not the matmul. Go, with a per-example pipeline.** Measured with a local probe crate build
(`probe/cand4-stage0` in `rust/`, not pushed) that runs the op's work with a timer between its
parts, frees its buffers in the real call path's order, and checks its `A` and `cols` against
`conv_forward_batch` bit for bit:
- **Faults are ruled out for the isolated op.** The real op faults 0 times a call at N = 32 and
  512 (`focused_benchmark.py`), and `--malloc raised` (no heap trimming) leaves its time
  unchanged: 1400-1422 against 1401-1447 µs at N = 32. The chained-call faults under "Other
  findings" are a separate, smaller effect (3-6% of an epoch).
- **The matmul scales linearly at N = 32:** bare `21632x9x8` 297-318 µs against 32 x 9.3.
  (Not at N = 512: 11.0-12.8 ms against 4.8.)
- **The parts, µs at N = 32, fault-free** (both allocator thresholds raised, where the probe
  and the real op agree; with glibc's defaults the probe faulted 195-490 times a call where
  the real op faults 0-20, the warning in [How to measure](method.md#how-to-measure) about a
  probe's allocation pattern):

  | part | now at N = 32 | 32 x N = 1 |
  |---|---|---|
  | zeroed `cols` (1.56 MB) | 217-235 | 32 |
  | im2col | 248-279 | 240 |
  | matmul, with its zeroed 1.38 MB output | 462-670 | 373 |
  | zeroed `A` (1.38 MB) | 177-232 | 27 |
  | ReLU scatter | 266-324 | 237 |

  The three `vec![0.0; ...]` buffers are each overwritten in full; at N = 1 they sit in L1/L2
  and zeroing them is free.
- **Three designs, all bit-identical to the current op** (probe totals, µs, fault-free, two
  passes; single calls 26-29 µs, so 32 x 830-930 and 512 x 13.3-14.8 ms):

  | design | N = 1 | N = 32 | N = 512 |
  |---|---|---|---|
  | now (zeroed buffers) | 28.5-31.1 | 1403-1736 | 30.4-32.3 ms |
  | original loops into uninitialised buffers | 34.1-44.3 | 1182-1209 | 23.3-23.4 ms |
  | no zeroing: `cols` appended, `A` gathered in output order | 26.0-26.6 | 1006-1071 | 17.9-19.8 ms |
  | the same, one example at a time (im2col, matmul, gather) | 27.0-27.9 | 979-983 | 15.7-15.9 ms |

  Just dropping the zero-fill is not enough: the strided scatter then pays for fetching `A`'s
  lines itself (456-475 µs at N = 32, and 13-24 against 7.5 µs at N = 1). Writing `A`
  sequentially (reading `by_position` with stride `O`) fixes that. Running the pipeline one
  example at a time keeps each 48 KB slab of `cols` and the 43 KB `P x O` product hot, and
  reuses one product buffer; it is the best at every N and brings N = 512 near the single
  calls. `cols` is still kept whole for the backward pass.
- **Stake:** about 420-500 µs of the 1400 at N = 32, so about 26-32 ms (4%) of the conv
  mini-batch 32 epoch's 63 calls, plus whatever candidate 2's batched accuracy pass then gains
  for Rust conv (to measure: the blocked op is still 30.6 against 26-28 µs per example, so the
  pass would gain only from the per-call overhead of the other layers).
- **Open for the fix:** a per-example matmul runs on one thread, so a threaded `forward_batch`
  at large N would lose its row threading (per-block, or threads over examples, would keep
  it). The same zeroed-output pattern is in `matmul_narrow` itself, so downstream (1.2-1.3x)
  and accumulate (1.1-1.2x) may share part of the cause; col2im's scatter-add does need its
  zeroed buffer.

## Candidate 3: dense single-example downstream through the tiled kernel

**Dense single-example downstream through the tiled kernel** (candidate 3; crate #20,
2026-09-24). The vector @ matrix case of `matmul` (`layer_downstream`'s `delta @ W`) calls
`tiled_row_range` with one row, the kernel `matmul_2d` and `matmul_narrow` use; `axpy_row`, its
only user, is gone. The crate test that compared matrix @ matrix rows with vector @ matrix now
compares the kernel with itself, so vector @ matrix is pinned against the `Fraction` FMA chain
directly, at every column-tile width and at 32 x 5408, 30 x 784 and 10 x 30. The pins passed on
the old build; rounding twice in the vector @ matrix case failed 26 of them. Crate suite 894
passed; full suite 3138 passed with no pin moved.

Focused benchmark, builds alternated old, new, old, new, each in its own process; µs per call:

| shape | old | new | numpy |
| --- | --- | --- | --- |
| 32 x 5408 (conv tail) | 43.8-48.1 | 26.2-26.4 | 29.0-29.3 |
| 30 x 784 | 5.8-6.4 | 3.4-3.5 | 5.6 |
| 10 x 30 | 0.6 | 0.5 | 1.8 |

- **In the epoch** (MNIST conv single-example, the demo's 2000 rows; cProfile via
  `rust_op_breakdown`, 3 runs per build, builds alternated twice): the 2000 `layer_downstream`
  calls took 86.2-90.4 ms old and 42.4-55.7 ms new, the 44 ms predicted.
- **Epoch time** (`prepared_dataset_timing.py --configs "conv single"`, Rust medians): with 9
  repeats, new build first, 0.744 → 0.700 s and 0.765 → 0.716 (about -6%). An earlier pass of 5
  repeats, old first, read 0.776 → 0.655 and 0.695 → 0.706, inside the noise; a 44 ms saving
  on a 0.7 s epoch needs more repeats than 5.
- **The demo** after the change read MNIST conv single-example at 0.20 (0.26 before), but its
  mini-batch cells, which this doesn't touch, moved by as much (conv 0.49 → 0.60), so the table
  in [Where things stand](status.md) keeps its numbers.

The candidate, as recorded when it was open:

**Dense single-example `downstream` at 32 x 5408 was 1.5-1.6x numpy.** First seen in the
interleaved per-op harness (50 vs 33 µs); in separate processes (focused benchmark, two
passes, 2026-09-24) it holds: 42.7-52.0 µs against 28.1-33.4. It is vector @ matrix, `delta
@ W` through `axpy_row`: 32 load/FMA/store passes over a 43 KB output row, the pattern #14
removed from `matmul_2d`. The same product as a one-row `downstream_batch` goes through the
tiled kernel and took 21.4-22.1 µs against numpy's 28.3-28.8 in the same runs. In the MNIST
conv single-example epoch its 2000 calls took 88 ms of 0.80 s (11%, profiled), so the tiled
kernel's rate would save about 44 ms (5%). Dense MNIST's only downstream is the 10 x 30
layer's, so this is the conv tail's. Candidate: route vector @ matrix through
`tiled_row_range` as a one-row matrix. It is the same FMA chain, so bit-identical, and the
crate tests that compare the two would need another reference. It ranks above candidate 4,
whose stake is about the same, because the cause and the fix are both known and the change
is small.

## Candidate 2: a batched accuracy pass

**A batched accuracy pass** (candidate 2; #378 stage 0, #379, 2026-09-24). The trainers'
`_training_accuracy` now calls `classify_rows(prepared)` on the array networks: both bases run
`forward_batch` over chunks of `CLASSIFY_CHUNK_ROWS` = 32 rows (`prepared_dataset.py`) and the
shape classes' `_classify_output_batch` takes each row's argmax (`np.argmax(axis=1)`; in Rust
`tolist` and a Python first-max, since the crate's `argmax` takes a vector) or 0.5 threshold.
The Rust conv network overrides it with the per-row loop, since its batched forward is slower
(below). `accuracy()` in `multiclass_evaluate.py` is unchanged. Tests
(`tests/test_prepared_dataset.py`): for all 22 classes, `classify_rows` equals `classify_row`
row for row over 70 rows (three chunks, the last partial) and the prepared accuracy pass equals
the tuple one; both dropout classes classify in inference mode (numpy's draws nothing from
`np.random`; Rust's at drop probability 0.5 predicts as `classify_row` does); the Rust row
argmax breaks ties, signed zeros and NaNs as `pa.argmax` does; the batched binary threshold is
strictly above 0.5 on both backends. Seven source mutations (last-max argmax, `>=` thresholds,
a dropped or truncated chunk, float labels, misaligned predictions) each failed them. The full
suite passed with every pinned training result unchanged.

**The A/B.** `scripts/prepared_dataset_timing.py` (which gained `--epochs`), median of 5 (ranges),
one process per measurement, against `main` at #378 on `PYTHONPATH`; one-epoch runs before then
after, three-epoch runs after then before. Seconds:

| config | backend | before | after | after / before | from the loader, before → after |
| --- | --- | --- | --- | --- | --- |
| dense MNIST, B = 32 | Rust | 1.61 (1.59-1.71) | 1.31 (1.31-1.32) | 0.81 | 1.15 → 0.89 |
| dense MNIST, B = 32 | numpy | 4.85 (4.81-5.04) | 2.44 (2.42-2.45) | 0.50 | 3.49 → 1.03 |
| dense MNIST, single | Rust | 2.58 (2.29-2.78) | 2.11 (2.06-2.30) | 0.82 | 2.00 → 1.89 |
| dense MNIST, single | numpy | 12.96 (12.01-13.98) | 9.97 (9.69-10.85) | 0.77 | 11.61 → 8.44 |
| conv MNIST 2000, B = 32 | Rust | 0.61 (0.61-0.64) | 0.64 (0.63-0.64) | 1.05 | 0.59 → 0.61 |
| conv MNIST 2000, B = 32 | numpy | 1.24 (1.24-1.35) | 1.19 (1.19-1.20) | 0.96 | 1.17 → 1.03 |
| conv MNIST 2000, single | Rust | 0.73 (0.71-0.81) | 0.74 (0.68-0.76) | 1.02 | 0.73 → 0.71 |
| conv MNIST 2000, single | numpy | 3.66 (3.62-3.70) | 3.55 (3.52-3.55) | 0.97 | 3.61 → 3.44 |
| dense MNIST, B = 32, 3 epochs | Rust | 3.39 (3.38-3.43) | 2.66 (2.65-2.67) | 0.78 | 2.86 → 2.29 |
| dense MNIST, B = 32, 3 epochs | numpy | 9.35 (9.26-9.83) | 4.04 (4.01-4.10) | 0.43 | 8.27 → 2.59 |

- **Stage 0 predicted the Rust saving.** Two passes at 0.157 s saved predicts 0.31 s for a
  one-epoch run; the B = 32 epoch lost 0.30 s. Four passes predict 0.63 s for three epochs; it
  lost 0.74 s. Rust single-example lost 0.47 s, noisier (its before range is 0.5 s wide).
- **numpy gains most**, 2.4 s of a one-epoch B = 32 run (2.66 predicted) and 5.3 s of three
  epochs, so Rust / numpy at dense B = 32 goes from 0.33 to 0.54: numpy's per-row pass was
  slower, so it lost more.
- **Rust conv is unchanged**, as it should be (it keeps the per-row pass); its B = 32 cell
  moved +5%, inside the 20% noise band. numpy conv gained 4% (12% from the loader), less than
  the 0.21 s stage 0 predicted for B = 32.
- **Left over:** a row-wise crate argmax (about 2.5% of a Rust B = 32 one-epoch run, under
  the bar), and batching Rust conv once candidate 4 makes its `forward_batch` cheaper per example.

The plan and stage 0, as recorded when the candidate was open:

`_training_accuracy` (`train.py`) calls `classify_state` once per training row, n + 1 times
over the training set for n epochs. Candidate 1 (done) removed the row conversion
(`classify_row` on the prepared matrix). What is left of one Rust pass on dense full MNIST
was estimated at about 1.24 - 0.90 = 0.34 s, mostly per-row call overhead
(Python dispatch, one small forward per layer). That could be the largest single cost left in
a dense epoch. The number is inferred from the candidate 1 table, not measured, and it
inherits the doubt about the conversion measured apart (see candidate 1). Candidate: a
`classify_batch` on the array networks that runs `forward_batch` over chunks of rows (from
candidate 1's prepared matrix) and takes the argmax per row, with `_training_accuracy` falling
back to `classify_state` for networks without it (the pure-Python ones and the ensembles). It
is a trainer change, not a kernel change, and must change no training result. Dense batched
forward rows equal the single-example forward exactly ([Kernel invariants](kernels.md#kernel-invariants)). Conv, pooling,
softmax and dropout (which must keep its inference behaviour) need a test that each batched
prediction equals `classify_state`'s, for every network class. Two caveats:
- **Conv gains nothing yet.** Conv `forward_batch` currently costs 1.5-2.1x per example at N =
  32 what single calls do (candidate 4), so a batched conv accuracy pass would be slower until
  candidate 4 is fixed. Dense networks can go first.
- **Stage 0:** time one accuracy pass per row against batched (chunks of 32 and 512), dense
  full MNIST and MNIST conv, both backends, from pre-converted inputs. Proceed only if it
  saves at least about 10% of an epoch after candidate 1.

**Stage 0 done (2026-09-24): go for dense on both backends and numpy conv, no-go for Rust
conv.** `python scripts/accuracy_pass_timing.py time` (one process per network, backend and
repeat, median of 5; each measure the median of 3 runs in its process; seed-0 weights,
`prepared_mnist` inputs). Seconds for one pass:

| network | backend | per row | batched 32 | batched 512 | epoch B = 32 | epoch single |
| --- | --- | --- | --- | --- | --- | --- |
| dense, 60000 rows | numpy | 1.50 (1.44-1.73) | 0.18 (0.17-0.19) | 0.14 (0.13-0.16) | 3.70 | 11.83 |
| dense, 60000 rows | Rust | 0.37 (0.36-0.46) | 0.21 (0.21-0.23) | 0.29 (0.29-0.32) | 1.18 | 1.92 |
| conv, 2000 rows | numpy | 0.27 (0.26-0.30) | 0.17 (0.16-0.18) | 0.20 (0.20-0.21) | 1.27 | 3.68 |
| conv, 2000 rows | Rust | 0.11 (0.10-0.11) | 0.13 (0.13-0.14) | 0.18 (0.18-0.19) | 0.60 | 0.74 |

The saving at chunk 32 as a share of an epoch (the epochs are the trainers given the prepared
dataset, so they include two passes). A one-epoch run has two passes; a long run about one
per epoch, so its share is one pass's saving over an epoch less one pass:

| network | backend | saved per pass | one-epoch, B = 32 | one-epoch, single | long run, B = 32 | long run, single |
| --- | --- | --- | --- | --- | --- | --- |
| dense | numpy | 1.33 | 72% | 22% | 60% | 13% |
| dense | Rust | 0.16 | 27% | 16% | 19% | 10% |
| conv | numpy | 0.11 | 17% | 6% | 11% | 3% |
| conv | Rust | -0.03 | -9% | -7% | -5% | -4% |

- **Chunk 32, not 512.** Rust at 512 is slower than at 32 (0.29 against 0.21 s; 512 x 784 x 30
  crosses the 8M-flop threading threshold, unexamined); numpy gains a little at 512 (0.14
  against 0.18) but 32 takes 97% of its saving, so one chunk size serves both.
- **The per-row Rust pass is 0.37 s**, near the 0.34 s estimated from candidate 1's table.
- **Rust's argmax is 15% of its batched pass** (0.03 of 0.21 s): the crate's `argmax` takes a
  vector, so the probe converts the output with `tolist` and takes each row's argmax in Python.
  A row-wise crate argmax would save about 2.5% of a one-epoch B = 32 run, under the bar.
- **Rust conv is slower batched**, as caveat 1 predicted, so it keeps the per-row pass until
  candidate 4 is fixed. Its forward-only pass (0.19 s) was even slower than forward plus argmax
  (0.13 s), in every process: see the allocator finding in [Lessons](lessons.md).
- **Predictions:** no batched prediction differed from the per-row one, in any cell (60000
  dense rows, 2000 conv rows, both chunk sizes). But **numpy's batched outputs are not
  bit-identical to its per-row ones**: `X @ W.T` against `W @ x` differs by 1 ULP (max abs
  2.2e-16) in 8273 of 50000 dense outputs and 2863 of 20000 conv outputs. So a numpy argmax
  can flip where two outputs are within an ULP (or a binary output within an ULP of 0.5),
  which could move the pocket-best epoch. Decided (2026-09-24): batch numpy anyway, tested for
  equal predictions, not claimed bit-identical; the suite's pinned results must hold. Rust
  dense rows are exact ([Kernel invariants](kernels.md#kernel-invariants)).

## Candidate 1: the dataset as one backend array

**The dataset as one backend array** (candidate 1, optimization 5; crate #19, #372-#374,
2026-09-24). The trainers used to convert a tuple into an array on every call: each batch in
`learn_batch` and each row in `learn` and in every accuracy-pass `classify_state`. Now an array
network trains from a `PreparedDataset` (`indrajala_ml/prepared_dataset.py`): the training set
as one backend matrix plus its labels. The trainers prepare it once per run from the tuple list,
or take one the caller built (`prepared_mnist` loads MNIST straight into one). The networks read
rows through `learn_row`, `learn_batch_rows` and `classify_row` (numpy row views and fancy
indexing; the crate's new `Array.row` and `Array.take_rows`). `learn`/`learn_batch` now convert,
then call the same step as the row methods. The mini-batch trainer shuffles row indices, which
gives the same permutation for a seed. Training is unchanged bit for bit: every seeded pin
passes as before, and `tests/test_prepared_dataset.py` checks all 22 array network classes
step by step against the tuple path.

Preparing is a one-time cost per run: 0.42-0.43 s in Rust (`Array.from_rows`, which reads
tuples by index; `Array(tuples)` took 1.53 s and a first `from_rows` that iterated each row took
1.38-1.46 s) and 1.42-1.45 s in numpy (`np.array`; `np.fromiter` was 1.29-1.31). From the
MNIST binary, `prepared_mnist` builds no Python floats.

**The A/B.** One epoch through the trainers the demos use, from the tuple list (preparation
included) and from `prepared_mnist`, against the commit before the change (`96a1487`).
`scripts/prepared_dataset_timing.py`, median of 5 (ranges), one process per measurement, the
two sides run one after the other. Seconds:

| config | backend | before | after | after, from the loader | after / before |
| --- | --- | --- | --- | --- | --- |
| dense MNIST, B = 32 | Rust | 3.72 (3.68-3.84) | 1.69 (1.64-1.81) | 1.22 (1.19-1.29) | 0.45 |
| dense MNIST, B = 32 | numpy | 7.92 (7.56-8.33) | 5.37 (5.12-5.58) | 3.87 (3.71-4.19) | 0.68 |
| dense MNIST, single | Rust | 4.60 (4.48-4.80) | 2.57 (2.41-2.68) | 2.03 (1.97-2.35) | 0.56 |
| dense MNIST, single | numpy | 14.93 (14.43-15.64) | 13.76 (12.84-14.00) | 11.91 (11.61-12.40) | 0.92 |
| conv MNIST 2000, B = 32 | Rust | 0.73 (0.70-0.81) | 0.64 (0.64-0.66) | 0.63 (0.61-0.65) | 0.88 |
| conv MNIST 2000, B = 32 | numpy | 1.44 (1.39-1.68) | 1.36 (1.34-1.40) | 1.28 (1.26-1.29) | 0.94 |
| conv MNIST 2000, single | Rust | 0.84 (0.78-0.92) | 0.76 (0.73-0.86) | 0.80 (0.70-0.88) | 0.90 |
| conv MNIST 2000, single | numpy | 3.87 (3.67-4.06) | 3.89 (3.70-4.05) | 3.68 (3.65-4.02) | 1.01 |

- **The 70% estimate held up better than feared.** The Rust B = 32 epoch lost 2.03 s, and
  adding back preparation's 0.43 s gives about 2.46 s of conversion removed, against the
  2.7 s measured apart. The doubt raised below (conversion measured apart plus the Rust ops
  leaving no time for the step loop's Python) was worth about 10%, not most of the stake.
- **Conv gains 12% in Rust at B = 32**, as the 15 µs-a-row estimate predicted. Single-example
  conv is within noise in both backends.
- **numpy single-example gains least** (8%). Its per-row cost is mostly its own per-call work,
  not the conversion.
- **Longer runs gain more.** A one-epoch run pays preparation once for one epoch and two
  accuracy passes. Each further epoch saves its batch conversion and one accuracy pass's row
  conversion, and preparation isn't repeated.
- **Loading straight into the array** saves preparation and more (Rust B = 32 1.22 s against
  1.69). The demos still load tuples; switching one is a caller change, not measured here.
- Rust / numpy at B = 32 goes from 0.47 to 0.31 (0.32 from the loader): numpy converted more
  slowly, so it lost more time to conversion before.

**Before: the stake.** Both backends converted Python tuples to an array on every call (`pa.Array(list(state))` in
`RustArrayNetworkBase._forward`/`learn`, a list of rows in `learn_batch`, `np.array` in the
numpy `ArrayNetworkBase`): 15.3 µs for one 784-pixel MNIST row in Rust, 29.5-32.1 µs in numpy
(re-measured; see "Per-row cost" below). That
didn't change the ratio, but it is a fixed cost neither backend's maths can remove, and a
larger share now the maths is faster. The trainers (`train.py`) take `list[tuple[state,
label]]`, shuffle it in Python and pass tuples through, a contract shared with the
pure-Python networks, the ensembles and the sweeps.

**Stage 0, go/no-go, fixed before measuring:** measure the conversion's share of one epoch,
single-example and mini-batch 32, dense 784 -> 30 -> 10 and the conv networks, both backends
(cProfile own time of `Array.__new__`/`np.array`, plus an A/B against pre-converted inputs
through a test-only path), and what the per-epoch `_training_accuracy` passes spend
re-converting every row. Proceed only if the conversion is at least about 10% of epoch time
for some production configuration; otherwise record the numbers here and close.

**Design, if it goes ahead:** add a fast path, keep the existing interface. A
`PreparedDataset` built once per run holds the states as one backend matrix, the labels as a
list and its backend; the two array network bases get `prepare_dataset(rows)`, and
`learn_row`/`learn_batch_rows` that share their bodies with `learn`/`learn_batch` (which
become convert-then-call wrappers, so the paths can't drift). numpy row slices are free
views; a Rust slice copies, so a `gather_rows(matrix, indices)` crate op only if that copy
measures material. The trainers shuffle an index list in place of the tuple list, which
consumes the RNG identically (`shuffle` on a list of the same length makes the same
permutation), so every seeded end-to-end pin must stay exact, with a test that the same seed
gives the same visiting order on both paths. Tests: `learn_row` gives exactly the weights of
`learn`, step by step, for every numpy and Rust network class, enumerated from a registry so a
new class can't be missed. Out of scope: changing the `load_*` functions, and anything the
trainers compute.

**Stage 0's go/no-go is met (batch-size-scaling study, #367, 2026-09-24).** Dense 784 -> 30 ->
10, one full-MNIST epoch of `train_backprop_network_mini_batch`, median of 5, one process per
measurement, `scripts/batch_size_timing.py time`. Seconds per epoch:

| B | backend | trainer epoch | step loop | one accuracy pass | batch conversion | row conversion (60000 rows) |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | numpy | 8.18 | 2.07 | 2.99 | 1.75 | 1.76 |
| 32 | Rust | 3.78 | 1.28 | 1.26 | 0.91 | 0.90 |
| 128 | numpy | 9.42 | 3.25 | 2.93 | 1.73 | 1.73 |
| 128 | Rust | 4.37 | 1.64 | 1.22 | 0.97 | 0.90 |
| 512 | numpy | 9.41 | 3.34 | 3.11 | 1.83 | 1.77 |
| 512 | Rust | 4.85 | 2.16 | 1.25 | 1.24 | 0.90 |
| 1024 | numpy | 9.54 | 3.21 | 3.07 | 1.90 | 1.77 |
| 1024 | Rust | 4.54 | 1.93 | 1.24 | 1.26 | 0.93 |

"Batch conversion" is `pa.Array`/`np.array` of every batch's list of rows, the first line of
`learn_batch`. "Row conversion" is one array per training row, as each `classify_state` of
an accuracy pass does. Both are measured apart from training, in the same process. A
one-epoch run has two accuracy passes. For Rust at B = 32, conversion comes to 0.91 + 2 x
0.90 = 2.7 s of the 3.78 s epoch: about 70%, against the 10% bar. For numpy at B = 32 it is
5.3 s of 8.18 s. Over a long run (about one pass per epoch) the Rust share is about the same:
0.91 + 0.90 = 1.8 s of about 2.5 s. In the step loop alone, conversion is 57-71% of Rust's
time; the Rust ops themselves are 0.37-0.40 s per epoch at every batch size. The rest of
stage 0's list (single-example, the conv networks, the A/B against pre-converted inputs) was
not measured and isn't needed for the decision. The conv networks' share is unmeasured; at
15 µs a row, the 6000 conversions of a timed MNIST conv epoch would be about 90 ms of 0.74 s
(12%). **Status: go; the largest stake measured in this document.** Most of each accuracy
pass is conversion too (0.90 of 1.24 s in Rust); what is left of the pass is candidate 2.

**The stake is likely smaller than 70%.** At B = 32, the conversion measured apart (0.91 s)
plus the Rust ops (0.37 s) already equal the whole 1.28 s step loop, leaving nothing for
the Python of 1875 steps, which can't be right. The same loop measured 1.59 s in candidate
7's profile run, so the two step-loop figures also disagree by 24%. The likeliest reading is
that converting rows in a tight loop apart from training costs more than it does inside the
loop; that is unmeasured. So the implementation's A/B against pre-converted inputs is the
real stake, not the 70%, and must be reported as such.

Batch conversion also grows with the batch size (Rust 0.91 s at B = 32, 1.24-1.26 s at 512
and 1024), though the rows converted per epoch don't change. It is part of why the Rust step
loop gets slower at larger batches at the same flops per epoch: from B = 32 to 512 the step
loop grows 0.88 s (1.28 to 2.16) and batch conversion 0.33 s, about 40% of it. The Rust ops
stay flat and there are 16 times fewer steps, so the rest is unexplained.
(`demo_batch_size_scaling` prints 1.42 s per epoch at B = 32 and 2.02 s at 1024.)

**Per-row cost, re-measured (2026-09-24).** A loop converting all 60000 training rows
(`to_array(list(state))`, median of 5 loops, one process per cell, two passes each) costs
15.3 µs a row in Rust and 29.5-32.1 µs in numpy, not the 49 and 55 µs this entry quoted
before. That earlier figure is unexplained. It isn't the loader: with freshly boxed floats (the loader before
#365) the same loop costs 17.3 µs in Rust and 31.5-31.6 µs in numpy. #365's shared pixel
floats therefore make Rust conversion about 12% cheaper. Over the roughly 180000 row
conversions of a batch-32 Rust epoch (one batch pass, two accuracy passes) that is about
0.36 s of 3.8 s. **Rust dense-MNIST epoch times from before #365 are not directly comparable
with later ones.** numpy's change is within noise.

## Dense forward_batch (crate #17)

**Dense `forward_batch`** (`matmul_nt`, #17), kept as a finding: formerly an open candidate, but
on one thread no gap is left. The 5.6x at batch 64 recorded earlier (numpy and Rust
interleaved) was the interleaving: in separate processes batch 64 was 1.8x numpy (1161-1164 against 625-653 µs), and unthreaded
Rust was linear in the batch at about 2x numpy per row. So the cost was the kernel: every row
of `X` re-read all of `W` (1.4 MB at 32 x 5408, past L2). #17 runs 4 rows of `X` against 2 rows
of `W` at a time, bit-identical. Two focused passes, old and new in separate processes, the
second with the build order reversed (µs, unthreaded):

| shape | batch | old | new |
| --- | --- | --- | --- |
| 32 x 5408 | 32 | 635-916 | 490-903, then 545-555 |
| 32 x 5408 | 128 | 2621-3708 | 2339-2769 |
| 32 x 5408 | 512 | 10452-15660 | 8233-13628 (9331-10204 in the second pass) |
| 30 x 784 | 32 | 95-110 | 69-98 (76-77) |
| 30 x 784 | 512 | 1537-2109 | 1208-1478 |

**Threading on the new kernel** (default against unthreaded, new build, both passes):
threading still pays at 32 x 5408 from batch 128 (1455-2016 against 2339-2769 µs) and at
batch 512 (6226-7672 against 8233-13628), and a little at 30 x 784, batch 512 (1038-1275
against 1208-1478). At batch 64 (11M flops, threaded) the two overlap (887-1121 against
998-1361). With default threading, 30 x 784 at batch 512 went from 1472-1655 to 1038-1275.

**End to end**, one epoch, old against new build, one process per epoch, builds
alternated with the order swapped every run, medians of 6 (seconds):

| epoch | old | new |
| --- | --- | --- |
| MNIST conv, mini-batch 32 | 0.751 (0.710-0.856) | 0.743 (0.706-0.841) |
| MNIST conv, mini-batch 512 | 0.870 (0.787-0.878) | 0.843 (0.774-0.870) |
| dense MNIST, batch 32 | 4.170 (3.967-4.365) | 4.094 (4.007-4.337) |
| dense MNIST, batch 512 | 5.814 (5.774-6.054) | 6.000 (5.658-6.114) |

All within noise, as the arithmetic predicts: MNIST conv mini-batch 32's 62 forward calls
save about 12 ms of 0.75 s. What is left is not the kernel. With default threading, 32 x
5408 is still 1.1-2.2x numpy at batch 32 (470-659 against 294-415 µs), which was put down to
`W` streaming in from L3 once per 4 rows of `X`. But on one thread each (2026-09-24, two
passes) Rust is faster at every op table shape: 479 against 563-575 µs at batch 32,
8421-8477 against 8937-9918 at batch 512, and 1217-1257 against 1271-1308 at 30 x 784, batch
512. So the batch-32 gap is numpy's OpenBLAS threading, as in candidate 6, and at batch 512
numpy's threads beat Rust's (3695-4110 against 4747-4898 µs at 32 x 5408), a threading
question (candidate 9). The idea recorded before, blocking over `k` so a panel of `W` stays
in L2 across all rows of `X` (storing and reloading each pair's 4-lane accumulator between
panels, bit-identical), would speed up a kernel that is already ahead, so it is not a candidate.

## Closed with no measured gain

Kept as findings:

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
  (see [Threading](kernels.md#threading)).
- **Row blocks for `matmul_2d`.** 1-row blocks were 2-3x slower than 16 KB blocks where `K` is
  in the hundreds (`(32, 512) @ (512, 5408)`: 40-47 vs 15-16 ms). One block for all rows was
  close to 16 KB blocks but not better. The thread count wasn't recorded; the same product
  takes 31-32 ms on one thread and 10-11 on 8 today (see [Threading](kernels.md#threading)), so these were presumably
  threaded, and the comparison holds only between the two settings.

## Superseded numbers

Values the current-state files quoted before a later change replaced them, newest first.

- **Dense MNIST end to end, Rust / numpy** (one 60000-example epoch, single-example and
  mini-batch 32): 0.20 and 0.33 after candidate 1, before candidate 2 (#379); 0.31 and 0.47
  before candidate 1.
- **Conv `forward_batch` in the MNIST conv mini-batch 32 epoch:** its 63 batch calls took about
  1.5 ms each before candidate 4 (crate #21), against 0.86 ms for 32 single-example calls.
- **Dense single-example `downstream` at 32 x 5408:** 1.5-1.6x numpy before candidate 3 (crate
  #20), and 11% of the MNIST conv single-example epoch.
- **The per-op table, measured interleaved.** An earlier version of the table was measured with
  numpy and Rust interleaved in one process and read 12-13x, 7x, 4-8x and 4x on the 32 x 5408
  batch-32 `downstream_batch`, `accumulate_gradient_batch` and `forward_batch` rows and the 30 x
  784 batch-512 `downstream_batch` row (another interleaved run read 13.9x on the first); that
  was numpy's OpenBLAS threads taking cores from Rust (see [Lessons](lessons.md)).
- **The conv demo end to end, at the start of this work:** MNIST conv single-example was 1.21
  (Rust slower) and conv-pool-conv mini-batch was 0.66.
