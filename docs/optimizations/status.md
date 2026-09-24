# Optimizations: where things stand

Part of the optimization record; the index is [../optimizations.md](../optimizations.md).

## Where things stand (2026-09-24)

Rust / numpy wall-clock ratio end to end (below 1 means Rust is faster), from
`demo_conv_rust_vs_vectorized_digit_recognition` after indrajala-math-rust#16. After #17 every
cell was within 0.05 of these (MNIST mini-batch 0.47, 0.43, 0.48), which is run-to-run noise:

| architecture | UCI digits, single | UCI digits, mini-batch | MNIST subset, single | MNIST subset, mini-batch |
| --- | --- | --- | --- | --- |
| conv | 0.12 | 0.17 | 0.26 | 0.49 |
| conv-pool-conv | 0.09 | 0.09 | 0.30 | 0.41 |
| conv-conv-stride2 | 0.11 | 0.12 | 0.35 | 0.46 |

At the start, MNIST conv single-example was 1.21 (Rust slower) and conv-pool-conv mini-batch was
0.66. Dense MNIST 784 -> 30 -> 10, one 60000-example epoch: Rust/numpy about 0.21
single-example and 0.54 mini-batch 32 since candidate 2 (#379; 0.20 and 0.33 before it, 0.31 and
0.47 before candidate 1), both backends' epochs much shorter (see candidates 1 and 2 under
"Completed"). Candidate 2 raised the B = 32 ratio because numpy's accuracy pass gained more.

Two caveats on these ratios. numpy runs with OpenBLAS's default threading, which slows its own
training: its MNIST conv mini-batch 32 epoch took 1.22-1.32 s at `OPENBLAS_NUM_THREADS=1`
against 1.39-1.56 s by default (optimization 7 stage 0), so against single-threaded numpy that
cell would be nearer 0.55-0.6 (an estimate from runs in different sessions, not measured side
by side). And the demo runs numpy then Rust in one process, so each Rust run starts while
OpenBLAS's threads may still be spinning (see "Other findings"); that can only slow Rust, by at
most the first 100-500 ms of each run, and is unmeasured.

Per op, the single-example ops are all at or better than numpy since candidate 3 (dense
`downstream` at 32 x 5408 was 1.5-1.6x, now 0.9x). The gaps left are in batch ops. Rust / numpy µs per
call, focused benchmark (see "How to measure"), numpy and Rust in separate processes, Rust
with the default threading (threshold 8M flops since #16). The last two columns put both
backends on one thread (`--openblas-threads 1 --rust-threads 1`, 2026-09-24: optimization 7
stage 0, and the `forward_batch` rows in a later pass of two):

| shape | op | batch | numpy | Rust | Rust/numpy | numpy, 1 thread | Rust, 1 thread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 x 5408 | `downstream_batch` | 32 | 232-246 | 581-654 | 2.4-2.8x | 573-578 | 610-625 |
| 32 x 5408 | `accumulate_gradient_batch` | 32 | 411-453 | 976-1180 | 2.2-2.9x | 807-871 | 1040-1083 |
| 32 x 5408 | `downstream_batch` | 512 | 11753-11777 | 10648-11847 | 0.9-1.0x | 12236-12361 | 14746-15199 |
| 32 x 5408 | `accumulate_gradient_batch` | 512 | 12926-13493 | 6811-11720 | 0.5-0.9x | 9425-10298 | 31253-31958 |
| 32 x 5408 | `forward_batch` | 512 | 3695-4110 | 4747-4898 | 1.2-1.3x | 8937-9918 | 8421-8477 |
| 32 x 5408 | `forward_batch` | 32 | 294-415 | 470-659 | 1.1-2.2x | 563-575 | 479 |
| 30 x 784 | `downstream_batch` | 512 | 416-945 | 1193-1401 | 1.3-3.4x | 1153-1466 | 1383-1560 |
| 30 x 784 | `accumulate_gradient_batch` | 512 | 714-776 | 997-1336 | 1.3-1.9x | 1269-1436 | 2335-2715 |
| 30 x 784 | `forward_batch` | 512 | 750-1027 | 1038-1275 | 1.0-1.7x | 1271-1308 | 1217-1257 |
| 30 x 784 | `downstream_batch` | 32 | 46-50 | 73-85 | 1.5-1.8x | 71-84 | 75-86 |
| 30 x 784 | `accumulate_gradient_batch` | 32 | 70-73 | 82-97 | 1.1-1.4x | 90-102 | 84-93 |

The first two rows (5.5M flops) run on one thread since #16; their Rust numbers are from then.
The two 32 x 5408 batch-512 rows are from optimization 7 stage 0's default-threading run (88M
flops, threaded).
The `forward_batch` rows are after #17, except 32 x 5408 at batch 512, whose threaded time moved
between 4.6 and 9.8 ms from run to run in the same session, so it keeps its earlier numbers. The
batch-512 Rust numbers are partly warm-clock numbers (see "Threading"). An earlier version of
this table was measured with numpy and Rust interleaved in one process and read 12-13x, 7x,
4-8x and 4x on the 32 x 5408 batch-32 `downstream_batch`, `accumulate_gradient_batch` and
`forward_batch` rows and the 30 x 784 batch-512 `downstream_batch` row (another interleaved run
read 13.9x on the first); that was numpy's OpenBLAS threads
taking cores from Rust (see "Other findings"). numpy's numbers are with OpenBLAS's default
threading, which it uses even at 5.5M flops. On one thread each, the batch-32 rows are level
or within 1.3x (32 x 5408 accumulate, its extra pass; candidate 6), Rust's `forward_batch` is
level or faster at every row, and `downstream_batch` at batch 512 (`k` = 32 or 30) is 1.2x at
32 x 5408 and 0.9-1.4x, overlapping, at 30 x 784. The large gap that remains is at batch 512
with a long `k`: accumulate (`k` = 512) is 3.0-3.4x numpy at 32 x 5408 and 1.6-2.1x at 30 x 784
(see candidate 7).

The per-op tables compare against numpy; the profile of a whole epoch finds other costs. Rust
time by op, cProfile over one MNIST epoch (2000 rows), 2026-09-24:

- **About a quarter of each timed MNIST conv epoch is not training.** The trainers run
  `_training_accuracy` (`train.py`) over every training row before the first epoch and after each
  one, one example at a time. In the conv mini-batch 32 epoch that is 4000 of the 4063
  `conv_forward_batch` calls and all 8000 `layer_forward` calls, about 0.19 s of 0.74 s; the
  single-example epoch is similar (4000 of its 6000 conv forwards). numpy runs the same passes,
  so the ratios include them, and every stake quoted below as a share of an epoch is a share of
  this timed epoch. That overstates the passes for longer runs: the trainers run n + 1 passes
  for n epochs, so a one-epoch run has two and a long run about one per epoch.
- **Conv MNIST, mini-batch 32** (0.74 s): `conv_forward_batch` 27% (its 63 batch calls about 1.5
  ms each, against 0.86 ms for 32 single-example calls; about 0.95 ms since candidate 4),
  `conv_accumulate_gradient_batch` 10.5%, dense `accumulate_gradient_batch` 9.5% and
  `downstream_batch` 6.7% (candidate 6), `forward_batch` 6.3%.
- **Conv MNIST, single-example** (0.80 s): `layer_sgd_step` 20%, `conv_forward_batch` 20%,
  `layer_forward` 19%, dense `downstream` 11% (5-7% since candidate 3), conv accumulate 7%.
- **Conv-pool-conv MNIST**: `conv_forward_batch` 41-42% in both trainers, `max_pool_forward_batch`
  10-11% (candidate 5).
