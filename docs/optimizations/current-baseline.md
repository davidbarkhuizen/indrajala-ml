# Optimizations: current baseline

Part of the optimization docs; the index is [../optimizations.md](../optimizations.md).

Where Rust stands against numpy on the current code: the numbers candidates are ranked against.
Current values only; a change that moves one replaces it here (the old value stays in its PR).
How each number is measured is in [Measurement](measurement.md).

## End to end

Rust / numpy wall-clock ratio (below 1 means Rust is faster), from
`demo_conv_rust_vs_vectorized_digit_recognition`:

| architecture | UCI digits, single | UCI digits, mini-batch | MNIST subset, single | MNIST subset, mini-batch |
| --- | --- | --- | --- | --- |
| conv | 0.17 | 0.73 | 0.19 | 0.54 |
| conv-pool-conv | 0.11 | 0.46 | 0.26 | 0.44 |
| conv-conv-stride2 | 0.13 | 0.53 | 0.35 | 0.51 |

Dense MNIST 784 -> 30 -> 10, one 60000-example epoch: about 0.21 single-example and 0.54
mini-batch 32.

Caveats:

- **The conv table is one demo run** (each cell the median of 5) on the current build, after the
  one-pass pool downstream (crate #25). The UCI mini-batch runs take 0.06-0.12 s in all, so fixed
  per-run costs dominate their ratios; read them as noisy. The dense ratios are current.
- **numpy runs with OpenBLAS's default threading, which slows its own training** (its MNIST conv
  mini-batch 32 epoch: 1.22-1.32 s at `OPENBLAS_NUM_THREADS=1` against 1.39-1.56 s by default).
  Against single-threaded numpy that cell would be nearer 0.55-0.6 (estimated, not measured side
  by side).
- **The demo interleaves numpy and Rust in one process**, which can only slow Rust (see
  [Measurement](measurement.md#gotchas)); unmeasured.

## Per op

Every single-example op is at or better than numpy. The gaps are in batch ops. µs per call,
focused benchmark, separate processes; Rust with the default threading (8M-flop threshold). The
last two columns put both on one thread:

| shape | op | batch | numpy | Rust | Rust/numpy | numpy, 1 thread | Rust, 1 thread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 x 5408 | `downstream_batch` | 32 | 232-246 | 581-654 | 2.4-2.8x | 573-578 | 610-625 |
| 32 x 5408 | `accumulate_gradient_batch` | 32 | 411-453 | 976-1180 | 2.2-2.9x | 807-871 | 1040-1083 |
| 32 x 5408 | `forward_batch` | 32 | 294-415 | 470-659 | 1.1-2.2x | 563-575 | 479 |
| 32 x 5408 | `downstream_batch` | 512 | 11753-11777 | 10648-11847 | 0.9-1.0x | 12236-12361 | 14746-15199 |
| 32 x 5408 | `accumulate_gradient_batch` | 512 | 12926-13493 | 6811-11720 | 0.5-0.9x | 9425-10298 | 31253-31958 |
| 32 x 5408 | `forward_batch` | 512 | 3695-4110 | 4747-4898 | 1.2-1.3x | 8937-9918 | 8421-8477 |
| 30 x 784 | `downstream_batch` | 32 | 46-50 | 73-85 | 1.5-1.8x | 71-84 | 75-86 |
| 30 x 784 | `accumulate_gradient_batch` | 32 | 70-73 | 82-97 | 1.1-1.4x | 90-102 | 84-93 |
| 30 x 784 | `downstream_batch` | 512 | 416-945 | 1193-1401 | 1.3-3.4x | 1153-1466 | 1383-1560 |
| 30 x 784 | `accumulate_gradient_batch` | 512 | 714-776 | 997-1336 | 1.3-1.9x | 1269-1436 | 2335-2715 |
| 30 x 784 | `forward_batch` | 512 | 750-1027 | 1038-1275 | 1.0-1.7x | 1271-1308 | 1217-1257 |

Reading it: most of the default-threading gap at batch 32 is numpy's OpenBLAS threading; on one
thread each, the batch-32 rows are level or within 1.3x (32 x 5408 accumulate, its extra pass),
and Rust's `forward_batch` is level or faster everywhere. The large one-thread gap left is
accumulate at batch 512 (`k` = 512): 3.0-3.4x at 32 x 5408 and 1.6-2.1x at 30 x 784. The batch-512
Rust numbers are threaded and partly warm-clock numbers.

Conv ops at 28x28, `ConvSpec(3, 8)`, one thread, against 32 or 512 single-example calls:
`forward_batch` 920-1010 µs at N = 32 and 15.0-15.7 ms at N = 512 (single calls 26-29 µs);
downstream 1.2-1.3x and accumulate 1.1-1.2x their single calls at N = 32, accumulate 3.8-4.0x at
N = 512.

Max-pool ops, PoolSpec(2) on 26x26x8 (the conv-pool-conv MNIST layer), ReLU-like input, one
thread each: forward 4.1-4.2 µs single against numpy's 95, 112-114 at batch 32 against 1570-1630;
downstream 5.7-6.3 single against 48-50, 166-177 or 373-394 at batch 32 (two modes between
processes) against 617-735. At 6x6x8 (UCI digits) the Rust forward is 1.0 µs single and
downstream 0.9-1.1.

## Where a Rust epoch spends its time

cProfile of Rust time by op over one MNIST-subset epoch (2000 rows). Profiled before
single-example downstream through the tiled kernel and the per-example conv forward; those ops'
current shares are in parentheses, the rest is not re-profiled
(`scripts/epoch_op_profile.py` refreshes it):

- **Conv, mini-batch 32** (0.74 s): `conv_forward_batch` 27% (its 63 batch calls about 0.95 ms
  each now), `conv_accumulate_gradient_batch` 10.5%, dense `accumulate_gradient_batch` 9.5%,
  `downstream_batch` 6.7%, `forward_batch` 6.3%.
- **Conv, single-example** (0.80 s): `layer_sgd_step` 20%, `conv_forward_batch` 20%,
  `layer_forward` 19%, dense `downstream` (5-7% now), conv accumulate 7%.
- **Conv-pool-conv** (0.67-0.71 s single-example, 0.76-0.79 s mini-batch 32, current build):
  `conv_forward_batch` 47% / 45%, `conv_accumulate_gradient_batch` 11% / 17% (profiled before
  the one-pass pool downstream took about 4% off each run), `max_pool_forward_batch` 3% / 4%,
  `max_pool_downstream_batch` 1.7% / 2.3%.
- **About a quarter of a one-epoch conv run is the accuracy passes**, not training: the trainers
  run n + 1 per-row passes for n epochs (Rust conv keeps the per-row pass). Stakes quoted as a
  share of an epoch are shares of this timed one-epoch run, which overstates the passes for
  longer runs.
- **Dense full MNIST** (784 -> 30 -> 10): the Rust ops are about 0.4 s of every epoch at any batch
  size; `accumulate_gradient_batch` is 11-13% of the step loop and `forward_batch` 7-10%. The rest
  is Python.
