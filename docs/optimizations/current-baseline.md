# Optimizations: current baseline

Part of the optimization docs; the index is [../optimizations.md](../optimizations.md).

Where Rust stands against numpy on the current code: the numbers candidates are ranked against.
Current values only; a change that moves one replaces it here (the old value stays in its PR).
How each number is measured is in [Measurement](../measurement.md).

## End to end

Rust / numpy wall-clock ratio (below 1 means Rust is faster), from
`demo_conv_rust_vs_vectorized_digit_recognition`:

| architecture | UCI digits, single | UCI digits, mini-batch | MNIST subset, single | MNIST subset, mini-batch |
| --- | --- | --- | --- | --- |
| conv | 0.15-0.16 | 0.63-0.64 | 0.20-0.21 | 0.53-0.54 |
| conv-pool-conv | 0.09-0.10 | 0.43-0.44 | 0.29 | 0.45-0.46 |
| conv-conv-stride2 | 0.12 | 0.53-0.54 | 0.36-0.37 | 0.52-0.53 |
| conv-conv | 0.15-0.16 | 0.59-0.60 | 0.36-0.37 | 0.48 |
| conv-conv16 | 0.16-0.17 | 0.56 | 0.38-0.39 | 0.46 |

Dense MNIST 784 -> 30 -> 10, one 60000-example epoch (`scripts/prepared_dataset_timing.py time`,
medians of 5): about 0.21 single-example (Rust 2.06-2.17 s, numpy 9.65-10.65 s) and 0.53
mini-batch 32 (Rust 1.28-1.30 s, numpy 2.39-2.41 s).

Caveats:

- **The conv table is two demo runs** (each cell the median of 5; a range where the two runs
  differ). The UCI mini-batch runs take 0.04-0.18 s in all, so fixed per-run costs dominate their
  ratios; read them as noisy.
- **numpy runs with OpenBLAS's default threading, which slows its own training**: its MNIST conv
  mini-batch 32 epoch (`prepared_dataset_timing.py`, medians of 5) takes 0.94-0.95 s at
  `OPENBLAS_NUM_THREADS=1` against 1.18-1.19 s by default, and Rust's 0.55-0.58 s. Against
  single-threaded numpy that cell is 0.58-0.60.
- **The demo interleaves numpy and Rust in one process**, which can only slow Rust (see
  [Measurement](../measurement.md#7-gotchas)); unmeasured.

## Per op

Every single-example op is at or better than numpy. The gaps are in batch ops. µs per call,
focused benchmark, separate processes; Rust with the default threading (8M-flop threshold). The
last two columns put both on one thread:

| shape | op | batch | numpy | Rust | Rust/numpy | numpy, 1 thread | Rust, 1 thread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 x 5408 | `downstream_batch` | 32 | 375 | 559-592 | 1.5-1.6x | 563 | 550-586 |
| 32 x 5408 | `accumulate_gradient_batch` | 32 | 574-579 | 820-821 | 1.4x | 738-752 | 789-844 |
| 32 x 5408 | `forward_batch` | 32 | 385-393 | 584-585 | 1.5x | 607-637 | 567-579 |
| 32 x 5408 | `downstream_batch` | 512 | 11527-11708 | 7946-8000 | 0.7x | 9939-9969 | 15052-15121 |
| 32 x 5408 | `accumulate_gradient_batch` | 512 | 13526-13576 | 7271-8771 | 0.5-0.6x | 10098-10122 | 28294-28430 |
| 32 x 5408 | `forward_batch` | 512 | 5246-5269 | 6770-6983 | 1.3x | 9688-9793 | 9668-9685 |
| 30 x 784 | `downstream_batch` | 32 | 61-62 | 77 | 1.2-1.3x | 77-78 | 76-77 |
| 30 x 784 | `accumulate_gradient_batch` | 32 | 90-91 | 83 | 0.9x | 95 | 82-83 |
| 30 x 784 | `downstream_batch` | 512 | 645-666 | 913-940 | 1.4-1.5x | 1320-1338 | 1484-1745 |
| 30 x 784 | `accumulate_gradient_batch` | 512 | 939-945 | 975-984 | 1.0x | 1373-1384 | 1808-1815 |
| 30 x 784 | `forward_batch` | 512 | 1030-1037 | 960-977 | 0.9x | 1384-1393 | 1372-1546 |

Reading it: at default threading Rust trails at batch 32 on the conv tail (1.4-1.6x), and most of
that is numpy's OpenBLAS threading: on one thread each, Rust is level or faster at batch 32
except 32 x 5408 accumulate (1.05-1.15x). The large one-thread gaps are at batch 512: accumulate
2.8x at 32 x 5408 and 1.3x at 30 x 784, and downstream 1.5x at 32 x 5408 and 1.1-1.3x at
30 x 784. The batch-512 Rust numbers are threaded and partly warm-clock numbers. The one-thread
32 x 5408 `downstream_batch` at batch 512 reads 15.1 ms here, against 12-13 ms for the same
kernel under an earlier Python/pyo3 stack in alternated runs on this machine, while `perf`
counts the same instructions per call for both builds and times a direct call at 13-15 ms in
both; the cause is open.

Conv ops at 28x28, `ConvSpec(3, 8)`, one thread each: Rust `forward_batch` 1046-1052 µs at N =
32 and 16.3-16.4 ms at N = 512 (single calls 28-29 µs; numpy 2.30-2.35 ms and 34.3-34.4 ms).
Against 32 or 512 single-example calls, Rust's downstream is 1.2-1.3x at N = 32 and 1.6-1.7x at
N = 512, and accumulate 1.1x and 1.5x.

Max-pool ops, PoolSpec(2) on 26x26x8 (the conv-pool-conv MNIST layer), ReLU-like input, one
thread each: forward 4.1-4.3 µs single against numpy's 95-99, 122-125 at batch 32 against
1694-1728; downstream 6.5-6.8 single against 46-47, 202 or 434-445 at batch 32 (two modes between
processes) against 527-560. At 6x6x8 (UCI digits) the Rust forward is 0.8 µs single and
downstream 0.9.

## Where a Rust epoch spends its time

cProfile of Rust time by op over one MNIST-subset epoch (2000 rows), current build,
`scripts/epoch_op_profile.py` (3 processes each; profiled totals, single-example / mini-batch 32):

- **Conv** (0.74-0.76 / 0.60-0.61 s): single-example `conv_forward_batch` 23%, `layer_sgd_step`
  22-23%, `layer_forward` 22%, conv accumulate 8%, dense `downstream` 7-8%; mini-batch
  `conv_forward_batch` 29%, `layer_forward` (the accuracy pass) 15-16%, conv accumulate 12%, dense
  `accumulate_gradient_batch` 9%, `forward_batch` 8%, `downstream_batch` 7-8%.
- **Second-conv networks**: conv-pool-conv (0.71-0.73 / 0.77 s) `conv_forward_batch` 51% / 50%,
  `conv_accumulate_gradient_batch` 10% / 14%, `conv_downstream_batch` 5% / 9%,
  `max_pool_forward_batch` 4% mini-batch; conv-conv-stride2 (0.72-0.73 / 0.81-0.82 s) forward
  55-56% / 52-53%, accumulate 10% / 14-15%, downstream 6% / 13-14%; conv-conv (1.99-2.04 /
  2.39-2.44 s) forward 54-55% / 55%, downstream 9% / 18%, accumulate 8% / 10%.
- **conv-conv16** (3.58-3.63 / 2.74-2.81 s), the network whose batch-32 products cross the
  threading threshold (second conv 21.2M flops, dense tail 9.4M): single-example
  `conv_forward_batch` 29-30%, `layer_sgd_step` 24%, `layer_forward` 18%, conv downstream and
  accumulate 8% each; mini-batch forward 43-44%, downstream 17-18%, accumulate 11-12%,
  `layer_forward` 8-9%.
- **About a quarter of a one-epoch conv run is the accuracy passes**, not training: the trainers
  run n + 1 per-row passes for n epochs (Rust conv keeps the per-row pass). Stakes quoted as a
  share of an epoch are shares of this timed one-epoch run, which overstates the passes for
  longer runs.
- **Dense full MNIST** (784 -> 30 -> 10, `scripts/batch_size_timing.py profile`): the Rust ops
  are about 0.33 s of every epoch at batch 512 and 1024; `accumulate_gradient_batch` is 7-9% of
  the profiled step loop and `forward_batch` 7%. The rest is Python.
