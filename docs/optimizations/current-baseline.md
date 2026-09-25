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
| conv | 0.18 | 0.64 | 0.18 | 0.49 |
| conv-pool-conv | 0.11 | 0.46 | 0.27 | 0.41 |
| conv-conv-stride2 | 0.15 | 0.53 | 0.35 | 0.48 |
| conv-conv | 0.17 | 0.53 | 0.36 | 0.44 |

Dense MNIST 784 -> 30 -> 10, one 60000-example epoch: about 0.21 single-example and 0.54
mini-batch 32.

Caveats:

- **The conv table is one demo run** (each cell the median of 5) on the current build. The UCI
  mini-batch runs take 0.05-0.18 s in all, so fixed per-run costs dominate their ratios; read
  them as noisy.
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
| 32 x 5408 | `downstream_batch` | 32 | 223-229 | 470-472 | 2.1x | 510-565 | 465-483 |
| 32 x 5408 | `accumulate_gradient_batch` | 32 | 436-450 | 860-871 | 1.9-2.0x | 713-778 | 793-810 |
| 32 x 5408 | `forward_batch` | 32 | 294-415 | 470-659 | 1.1-2.2x | 563-575 | 479 |
| 32 x 5408 | `downstream_batch` | 512 | 11785-12838 | 10530-10877 | 0.8-0.9x | 11570-13657 | 13361-13654 |
| 32 x 5408 | `accumulate_gradient_batch` | 512 | 12742-13818 | 6479-8509 | 0.5-0.7x | 9636-10499 | 27818-28065 |
| 32 x 5408 | `forward_batch` | 512 | 3695-4110 | 4747-4898 | 1.2-1.3x | 8937-9918 | 8421-8477 |
| 30 x 784 | `downstream_batch` | 32 | 45 | 65-67 | 1.5x | 72-79 | 70-77 |
| 30 x 784 | `accumulate_gradient_batch` | 32 | 68-86 | 69 | 0.8-1.0x | 96-105 | 74-82 |
| 30 x 784 | `downstream_batch` | 512 | 420-502 | 842-848 | 1.7-2.0x | 1332-1508 | 1199-1351 |
| 30 x 784 | `accumulate_gradient_batch` | 512 | 756-759 | 832-931 | 1.1-1.2x | 1340-1383 | 1506-1820 |
| 30 x 784 | `forward_batch` | 512 | 750-1027 | 1038-1275 | 1.0-1.7x | 1271-1308 | 1217-1257 |

Reading it: most of the default-threading gap at batch 32 is numpy's OpenBLAS threading; on one
thread each, Rust is level or faster at batch 32 except 32 x 5408 accumulate (1.0-1.1x), and
Rust's `forward_batch` is level or faster everywhere. The large one-thread gap left is accumulate
at batch 512 (`k` = 512): 2.6-2.9x at 32 x 5408 and 1.1-1.4x at 30 x 784. The batch-512 Rust
numbers are threaded and partly warm-clock numbers.

Conv ops at 28x28, `ConvSpec(3, 8)`, one thread, against 32 or 512 single-example calls:
`forward_batch` 920-1010 µs at N = 32 and 15.0-15.7 ms at N = 512 (single calls 26-29 µs);
downstream 1.2-1.3x and accumulate 1.1-1.2x their single calls at N = 32, accumulate 3.8-4.0x at
N = 512 (measured before `matmul_long_k`, which took the N = 32 accumulate to 0.83x). The
second-conv accumulate per call is in [Implemented](implemented.md).

Max-pool ops, PoolSpec(2) on 26x26x8 (the conv-pool-conv MNIST layer), ReLU-like input, one
thread each: forward 4.1-4.2 µs single against numpy's 95, 112-114 at batch 32 against 1570-1630;
downstream 5.7-6.3 single against 48-50, 166-177 or 373-394 at batch 32 (two modes between
processes) against 617-735. At 6x6x8 (UCI digits) the Rust forward is 1.0 µs single and
downstream 0.9-1.1.

## Where a Rust epoch spends its time

cProfile of Rust time by op over one MNIST-subset epoch (2000 rows), current build,
`scripts/epoch_op_profile.py` (3 processes each; profiled totals, single-example / mini-batch 32):

- **Conv** (0.70-0.71 / 0.56-0.61 s): single-example `layer_sgd_step` 22-23%,
  `conv_forward_batch` 21-22%, `layer_forward` 21-22%, conv accumulate 7%, dense `downstream` 7%;
  mini-batch `conv_forward_batch` 25-30%, `layer_forward` (the accuracy pass) 14-17%, conv
  accumulate 11-12%, dense `accumulate_gradient_batch` 9-10%, `forward_batch` 8-9%,
  `downstream_batch` 7%.
- **Second-conv networks**: conv-pool-conv (0.69-0.77 / 0.73-0.78 s) `conv_forward_batch` 45-54%
  / 46-52%, `conv_accumulate_gradient_batch` 9-11% / 13-15%, `conv_downstream_batch` 5-6% / 8-9%,
  `max_pool_forward_batch` 4% mini-batch; conv-conv-stride2 (0.70-0.71 / 0.78-0.81 s) forward
  53-55% / 49-53%, accumulate 10-11% / 14-15%, downstream 6% / 14-16%; conv-conv (1.93-2.03 /
  2.29-2.32 s) forward 50-55% / 51-52%, downstream 9% / 21-22%, accumulate 7-8% / 10%.
- **About a quarter of a one-epoch conv run is the accuracy passes**, not training: the trainers
  run n + 1 per-row passes for n epochs (Rust conv keeps the per-row pass). Stakes quoted as a
  share of an epoch are shares of this timed one-epoch run, which overstates the passes for
  longer runs.
- **Dense full MNIST** (784 -> 30 -> 10): the Rust ops are about 0.4 s of every epoch at any batch
  size; `accumulate_gradient_batch` is 11-13% of the step loop and `forward_batch` 7-10%. The rest
  is Python.
