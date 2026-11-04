# Workplan: the crate's performance, kernels and threads as one plan

**Status: unified 2026-11-04 at the owner's direction; no crate code yet.** The owner: "1. review the
current situation as it stands 2. re-analyse the solution and approach 3. combine the threading plan
with the i7 plan and create one unified workplan". This replaces the multithreading workplan
(drafted 2026-11-02) and the i7 outline (`docs/i7-9700k-rust-optimization.md`, 2026-10-08); both are
in git history at `8e342ab`. The IDs D1-D7, O1, O2 and K1-K22 keep their meanings, so older
references still read; what the re-analysis changed is marked. The owner delegates the decisions;
none is pending.

**The short version.** Both plans ordered the work by the largest ratio to numpy, on shapes from
October. Measured today, the largest ratio isn't where the time goes, the first kernel fix helps
transformer training more than the conv tail it was planned for, and most of stage −1's prerequisites
can wait. So: one fix first (K7), a shape census to choose what follows, the machine-derived
parameters only where a measurement needs them, and threads after that, judged on epochs.

## The IDs, in time order

| ID | When | Short label | What it is |
|---|---|---|---|
| D1 | 2026-11-02, revised 2026-11-04 | the kernels first | single-thread kernel work before threading, one PR and A/B each; revised: only K1 is a prerequisite, the rest ordered by epoch share (D9) |
| D2 | 2026-11-02 | bits never depend on threads | threads only partition outputs (rows, column chunks, examples, heads); no reduction is split, so every output is one chain in today's order at every thread count |
| D3 | 2026-11-02 | a std-only pool | a small pool in the core, no dependency: workers parked between calls, work claimed in chunks, lazy and fork-aware, nesting run inline |
| D4 | 2026-11-02 | one thread budget a process | `INDRAJALA_THREADS` and `set_threads(n)`: the crate never uses more; default the physical cores; the runner's workers get 1 unless a job asks |
| D5 | 2026-11-02, revised 2026-11-04 | the machine read once | `CpuInfo` read once a process, thread counts and slab sizes derived from it, every override hook kept; revised: built when a kernel first needs it (K8), with per-core caches (D11) |
| D6 | 2026-11-02 | epochs decide | a threading change is judged on epochs and study runs at training's thread count, not on isolated loops, which flatter threading |
| D7 | 2026-11-02 | one plan, sub-stages | the items are stages here, each its own crate PR, bump and A/B, not a workplan each |
| O1 | 2026-11-02 | at the end | the owner: multithreading after roadmap steps 14 and 15 and the work after them. Reached: steps 1-15 are done |
| O2 | 2026-11-02 | `litshe` from its response | `litshe`'s thermal profile with the cooler is based on its measured temperature response |
| D8 | 2026-11-04 | one workplan | this file replaces the threading plan and the i7 outline |
| D9 | 2026-11-04 | order by epoch share | what comes next is chosen from where a backdrop's time goes, not from the ratio to numpy |
| D10 | 2026-11-04 | slabs, then packing | long-`k` products run in `k` slabs (K7a), panels packed where `b`'s stride passes a page (K7b); `matmul_long_k` folds into `matmul_2d` |
| D11 | 2026-11-04 | per-core caches | `CpuInfo` gives each cache's size per core, so a hybrid machine has an honest figure |
| D12 | 2026-11-04 | jebel claims, laptops guard | a speedup is claimed on jebel; the Ryzen and stone check that a change costs them nothing |
| D13 | 2026-11-04 | backdrops at production scale | the backdrops gain a larger transformer, so small models don't decide a kernel |

## What the review found (2026-11-04)

All on `jebel` at ml `8e342ab`, crate `9fe5ced`, except where a column names another machine. Only the
A/Bs of the prototype below went through `ab.py`; the rest are one-off readings, for choosing the
work. **No claim rests on them:** each stage measures its own (measurement.md §9). The probe rows,
the profiles and the scripts are in `pyramidon`'s
`~/data/indrajala-ml/runs/blocking-probe-2026-11-04/` (gitignored); the first readings' raw benchmark
output was lost in clean-up, and their numbers are given here. Rerunning them takes minutes.

### The October gaps are mostly still there

Rust / numpy. October's are the i7 outline's; today's are two passes, medians.

| shape | op | batch | threads | October | today |
|---|---|---:|---|---:|---:|
| 32 x 5408 | `accumulate_gradient_batch` | 512 | 1 | 3.93 | 3.9 |
| 32 x 5408 | `accumulate_gradient_batch` | 512 | default | 1.25 | 1.25 |
| 30 x 784 | `accumulate_gradient_batch` | 512 | 1 | 1.49 | 1.55 |
| 32 x 5408 | `forward_batch` | 512 | 1 | 1.55 | 1.43 |
| 32 x 5408 | `forward_batch` | 32 | 1 | 1.34 | 1.31 |
| 32 x 5408 | `forward_batch` | 32 | default | 2.56 | 2.5 |
| 32 x 5408 | `downstream_batch` | 32 | default | 3.09 | 3.0 |
| 30 x 784 | `downstream_batch` | 512 | default | 1.85 | 1.9 |

**Rust still wins every epoch** (one run each, Rust / numpy): conv B=32 0.72 (October 0.81), conv
B=512 0.87 (0.94), dense B=32 and B=512 0.44 (0.45). So these are lost ground on single ops, not a lag
in training. A sequence epoch at 256 windows is 0.73.

### Where the time goes now

`perf record`, share of samples, Rust. The conv row is from `epoch_op_profile` (its cProfile total).

| backdrop | largest |
|---|---|
| sequence training, 2 layers, `d` 64 | `tiled_row_range` **56%**, exp 5%, memmove 5%, `matmul_nt` 5%, layer norm 4%, transpose 3% |
| attention, `T` 1024 | tiled attend and backward 33%, exp 19%, `tiled_row_range` 18%, memmove 5%, transpose 5%, memset 4% |
| attention, `T` 64 | `tiled_row_range` 34%, exp 12%, tiled attend and backward 13%, transpose 6%, memmove 5% |
| window, linear, learn | `tiled_row_range` 39%, `blas_thread_server` 9%, layer norm 5%, `matmul_nt` 5% |
| window, family, learn | `tiled_row_range` 18%, `blas_thread_server` 11%, layer norm 8%, transpose 6%, `matmul_nt` 6%, memmove and calloc 7% |
| window, family, long test's forward | `matmul_nt` 16%, layer norm 15%, linear product 11%, tiled attend 6%, exp 6% |
| conv, mini-batch 32 | `conv_forward_batch` 33%, `layer_forward` 24%, `conv_accumulate` 10% |
| conv-conv, mini-batch 32 | `conv_forward_batch` **64%**, `conv_accumulate` 9%, `layer_forward` 7% |

Three things the plans didn't have. The window program spends 78-99.8% of its time in the crate
(`--breakdown`), so crate work reaches the studies. **`blas_thread_server` at 6-11% is an idle
OpenBLAS thread spinning during Rust training**: it costs a core, not this thread's time, and it's
what the benchmark's default environment does, not the studies' (they run `OPENBLAS_NUM_THREADS=1`).
And a profile names a function, not its shapes: nothing here says which product shapes the 56% is.
That's K25.

### Item 1's cause: confirmed, and it isn't about the conv tail

The outline blamed `b` read 8 times, and it was right. `perf stat`, 100 calls of `(32, 512) @ (512,
5408)`: 3.16M cache misses a call, against 2.77M predicted (346k lines x 8 reads).

| variant | cycles | cache misses | dTLB misses | instructions |
|---|---:|---:|---:|---:|
| today | 7.33G | 316M | 4.4M | 5.07G |
| all rows a block | 3.24G | 54M | 0.6M | 5.06G |
| panels packed | 2.42G | 51M | 0.6M | 5.24G |
| `k` slabs of 64 | 2.48G | 31M | 0.6M | 5.19G |

The instructions don't move; the wait does. The wider finding is in the 16 KB rule itself:
`rows_per_block = 16 KB / (K * 8)` is **1 once `K` passes 1,024**, so the 2-row register tile never
runs and every row re-reads `b`. Every token-wise gradient is that shape, `(d, N*T) @ (N*T, d)`. So
K7 is the transformers' fix as much as the conv tail's.

**A prototype on real training.** `matmul_2d` with `K` > 64 in slabs of 64, all rows a block, resuming
(about 25 lines; `proto.sh` in the evidence directory), as a wheel through `ab.py run --old-crate 9fe5ced
--new-crate WHEEL`, `ONNONO`, tier 2's pass count:

| benchmark | run | result |
|---|---|---|
| `sequence_training_timing --windows 1024` | `2026-11-04-proto-kslab-sequence` | epoch **−12.4%** (2.049 to 1.795 s), consistent; numpy within noise |
| `attention_timing` | `...-attention` | backward **−15.1%** (`T` 64 causal), −9.5% (`T` 64 packed), −7.6% and −6.0% (`T` 256); 8 rows within noise, max 3.7% |
| `prepared_dataset_timing` | `...-epochs` | conv and dense, B=32 and 512: all within noise, max 3.6% |
| `window_recurrent_timing` | `...-window` | all within noise, max 0.6%; its control, a 256 x 256 x 256 product, −45.1%, as it's on the changed path |
| golden run | | **bit-identical, 164 networks** (with the wheel's `.so`, `d1e0184c3543`, confirmed loaded) |

So K7 gains on transformers, costs nothing elsewhere, and does nothing for the window program, whose
products are short in `k`. It's the one thing here already known to clear the 5% bar.

### The candidates, on four kinds of core

`scripts/measurement/blocking_probe.rs`: both kernels copied (AVX2 tiles, the same chains), every
variant checked bit for bit against today's (**all 1,144 rows on these four, plus 192 in the first
jebel round, identical**). Its copies match the crate within a few percent where it's calibrated:
15.2 ms against 15.1-15.4, 364 µs against 364-392, 44 against 45; 6.1 ms against 6.0-6.1. One thread.
Ratio to today's; `slab` is 64 rows, `pack` packs panels in slabs of 128.

| `a @ b`, m x k x n | jebel today, µs | jebel slab / pack | Ryzen slab / pack | stone P slab / pack | stone E slab / pack |
|---|---:|---:|---:|---:|---:|
| 32 x 512 x 5408 | 15187 | 0.34 / 0.31 | 0.38 / 0.34 | 0.33 / 0.33 | 0.37 / 0.45 |
| 30 x 512 x 784 | 856 | 0.60 / 0.56 | 0.76 / 0.71 | 0.52 / 0.57 | 0.78 / 0.83 |
| 64 x 256 x 1024 | 1877 | 0.99 / **0.37** | 0.61 / **0.39** | 0.83 / 0.71 | 0.98 / 0.77 |
| 8 x 18432 x 64 | 2752 | **0.18** / 0.29 | **0.14** / 0.22 | **0.23** / 0.34 | **0.22** / 0.34 |
| 8 x 2048 x 64 | 216 | **0.21** / 0.30 | **0.33** / 0.54 | **0.34** / 0.52 | **0.51** / 0.65 |
| 32 x 32 x 5408 | 243 | 1.06 / 0.96 | 0.97 / 1.04 | 1.02 / 1.03 | 1.01 / 1.08 |
| 30 x 32 x 784 | 31 | 1.07 / 1.01 | 0.98 / 0.95 | 1.02 / 1.06 | 1.00 / 1.06 |

The slab's size hardly matters (packed 64 to 512 within about 20% on the first three rows on every machine;
October's table had 64 best or tied at narrow `n`), so K7 needs no machine parameter. Packing wins
where `b`'s row stride passes a 4 KB page (1024 and up) and loses at narrow `n`, where slabs alone
keep the stream sequential. Its buffer is 16 KB. At `K` 32 both are within 8% of today's: the
slab and all-rows variants are one algorithm there, so slab's 1.07 is this probe's noise at that size.

`a @ b.T` (`matmul_nt`, `k` slabs storing each output's 4 lanes). Ratio to today's, by slab:

| m x k x n | machine | today, µs | 256 | 512 | 1024 | 2048 |
|---|---|---:|---:|---:|---:|---:|
| 512 x 5408 x 32 | jebel (L2 256 KB) | 6125 | 0.64 | 0.64 | 0.88 | 0.91 |
| | Ryzen (512 KB) | 9987 | 0.92 | 0.90 | 0.98 | 0.96 |
| | stone P (1.28 MB) | 4501 | 0.86 | 0.83 | 0.90 | 0.99 |
| | stone E (2 MB, 4 cores) | 7368 | 1.01 | 1.01 | 1.01 | 0.99 |
| 32 x 5408 x 32 | jebel | 364 | 0.58 | 0.60 | 0.85 | 0.87 |
| | Ryzen | 595 | 0.84 | 0.82 | 0.89 | 0.89 |
| | stone P | 257 | 0.80 | 0.80 | 0.92 | 0.99 |
| | stone E | 430 | 1.01 | 1.01 | 1.03 | 1.00 |
| 512 x 784 x 30 | jebel | 692 | 0.69 | 0.68 | 0.98 | 0.98 |
| | Ryzen | 1278 | 0.92 | 0.89 | 0.95 | 0.95 |
| | stone P | 462 | 1.10 | 1.03 | 1.00 | 1.00 |
| | stone E | 1040 | 0.99 | 0.97 | 1.01 | 1.01 |
| 32 x 784 x 30 | jebel | 44 | 0.67 | 0.65 | 0.96 | 0.96 |
| | Ryzen | 74 | 0.91 | 0.89 | 0.96 | 0.96 |
| | stone P | 27 | 1.04 | 1.01 | 1.00 | 1.00 |
| | stone E | 61 | 0.98 | 0.98 | 1.00 | 1.00 |

This one does depend on the machine: jebel's edge falls exactly where a slab of `W` (`n * s * 8`)
reaches its L2, and stone's P-cores lose up to 10% slabbing a `W` that already fits. A **candidate
rule, to be tested in K8:** slab only when `W` is past half the L2, with a slab of `W` of `min(L2 / 4,
128 KB)`. On each row it takes the best cell or one close to it, except the Ryzen's at 784, where it
forgoes about 7% by not slabbing. Against numpy, K8 would turn the conv tail's 1.43x
into about 0.9x on jebel. At present shapes it's only 5% of a sequence epoch, so it may not clear the
bar until D13's larger backdrop.

**Threads, 8 on jebel, against numpy's own threads.**

| product | today, row split | candidate | numpy, default threads |
|---|---:|---:|---:|
| 32 x 512 x 5408 | 2126 µs | 1670 µs, column split and packed | 1830 µs |
| `.T` 512 x 5408 x 32 | 1470 µs | 980 µs, slabs | 1060-1160 µs |
| `.T` 32 x 5408 x 32 (below the threshold today) | 153 µs | 153 µs | 145 µs |
| 30 x 32 x 784 (below it) | 87 µs | - | 23 µs |

The kernels and the column split reach numpy's threaded time with today's spawn cost. Below the
threshold the spawn still dominates (K15). These are isolated loops, which D6 says flatter threads.

## Re-analysis: what changes

| both plans said | now | why |
|---|---|---|
| Order by ratio to numpy | Order by epoch share (D9) | Rust wins every epoch; the largest ratio isn't the largest cost |
| K7 is the conv tail's | K7 first, for transformers too | −12.4% on a sequence epoch, from the `K` > 1,024 path |
| K2-K4 (placement) first | When a read needs them | they settle 2% rows, and the next gains are 6% and up; K4's rows are under the bar |
| K5 (baseline) before the kernels | After them, as the closing baseline | the kernels move every number; today's readings say the gaps stand |
| `CpuInfo` before K7 | Built for K8 | K7's slab is the machine-independent 64 |
| Window program: threads only | Also its own kernels, from the census | it spends up to 99.8% in the crate, K7 didn't reach it |
| "L2" is one number | Per-core sizes (D11) | stone's cores differ by 2.5x per core |
| One slab in rows (`LONG_K_BLOCK`, 64) derived from L2 | Slabs of `W` in bytes, K8 only | 64 rows is 37 KB at `n` 72, 2.77 MB at `n` 5408; only K8 is sensitive |
| The laptop must not lose | The laptops guard (D12) | the owner, 2026-11-04: pyramidon, stone and klip (after its affine pass) as required |
| Small models as backdrops | A larger one too (D13) | at `d` 64 `W` stays in L2, so K8 and threads look smaller than production would see |

## Where threading pays, and where it doesn't

Carried from the threading plan, and still the frame for stages 8-12.

**Threading today.** `rust/core/src/linalg.rs` threads three products, and `nn/conv.rs` one op, each by
`std::thread::scope`, spawning OS threads on every call. `matmul_2d` and `matmul_nt` split output rows
(`for_each_row_range`); `matmul_long_k` splits column chunks, so each thread reads only its part of
`b`; conv's downstream splits examples. `matmul_thread_count`: 1 thread below 8M flops, else
`min(available_parallelism, 8, rows)`; the 8M was tuned on the Ryzen, and `available_parallelism`
counts SMT threads (8 on its 4 cores). Every split is by output, so results are bit-identical at
every thread count. Not threaded: the tiled attention, the window program, conv's forward, every
row-wise op, the decoder's step.

**What a thread costs** (`thread_cost_probe.py`, 2026-10-29): about 10 µs a thread on jebel and 19 on
`litshe`, plus a first one's 20-28. At b32 the spawns cost more than the product. A pool would turn
that into a split of the product's own time (jebel about 8 µs at four threads). On `litshe` four cores
contend for one memory bus: at b512 the crate gains 11% from four threads, OpenBLAS loses 67%.

**What training sees (measurement.md, Gotchas):** in training the cores idle between calls, so a
threaded call starts on cold, clocked-down cores; in isolated loops 8 threads ran 2.3-3.2x faster than
1 on 1.6-5.5M-flop products, yet lowering the threshold to 2M or 4M moved no epoch by more than 1.6%.
OpenBLAS's spinning workers slow a Rust call that follows a numpy product (1.56x at the conv tail,
batch 512). So threading's payoff is uncertain until it is read on epochs (D6).

**Where it pays.** Sweeps and batches are already parallel by process (8 workers on jebel, the runner
across klip and stone), so threads add little to their throughput. They pay where cores sit idle
beside a single job: one long run (a 1,024-character context run, an existence probe on a short
wall-clock budget), a run whose memory lets fewer processes fit than cores (klip's 5.3 GB), decoding
latency, and `litshe`'s 4 cores once it can hold all-core load. D4's budget keeps the two kinds from
oversubscribing each other.

## The machines (D12)

| machine | CPU | cores / threads | L2 per core, L3 | role here |
|---|---|---|---|---|
| jebel | i7-9700K (Coffee Lake) | 8 / 8, no SMT | 256 KB, 12 MB | **speedup claims:** every tier 2 A/B, PL1 65 W |
| klip, after the affine pass | Ryzen 7 3700U | 4 / 8, SMT | 512 KB, 4 MB | **guard:** a change costs the Ryzen nothing; physical against logical cores. 5.3 GB, <= 85 C, under the thermal guard |
| pyramidon | Ryzen 7 3700U | 4 / 8, SMT | 512 KB, 4 MB | **guard** until klip is free, one job at a time under an address-space cap, none beside heavy development: its A/As are noisier |
| stone | i7-1255U (Alder Lake, hybrid) | 2 P + 8 E, 12 threads | P 1.28 MB, E 2 MB across 4 (512 KB each); 12 MB | **guard:** hybrid scheduling, static splits leave the fast cores idle. The sandbox stands (no GitHub, bundles, `~/labour`); it needs a profile and noise rules (K27). Available weekdays 18:00-07:00 and weekends unless the owner says otherwise |
| `litshe` | Pi 5, Cortex-A76 | 4 / 4 | 512 KB, 2 MB | aarch64 parity and timings, when on; all-core once the active cooler is fitted (stage 12) |

## The items

| # | item | source | now |
|---|---|---|---|
| K1 | `ab.py run --old-crate X --new-crate X` installs neither wheel, timing the venv's crate on both sides | From the A/B harness | **stage 1**: a measurement fault |
| K25 | a shape census: which kernel, at which `m x k x n`, how many flops, for a backdrop | new (D9) | **stage 1**: a cargo feature, off in releases |
| K26 | backdrops at production scale: `sequence_training_timing` and `attention_timing` at a larger `d` | new (D13) | **stage 1** |
| K27 | stone's profile and noise rules, and the Ryzen's refreshed; `ab.py` from a bundle's checkout on stone | new (D12) | **stage 1** |
| K7 | `matmul_2d` for long `k`: slabs that resume each chain | i7 item 1 | **stage 2** (K7a), **stage 3** (K7b, packing) |
| K23 | accumulate without its transpose: `delta.T @ X` read through `delta` directly (`nn/dense.rs:192`, 3-6% of the backdrops) | new | **stage 4**, against K11 |
| K11 | an AVX2 transpose that doesn't lose to 4K aliasing (the measured kernels on the crate's `avx2-transpose` branch) | From the crate kernels | **stage 4**, only for what K23 leaves |
| K24 | fewer copies and allocations in the crate's ops (memmove, memset, calloc: 5-10% of the window and attention backdrops) | new | **stage 4**, sites from the census |
| K9 | conv on 28x28 inputs, `conv_forward_batch` first (64% of a conv-conv epoch) | i7 item 5 | **stage 5**: diagnosis, then a fix if there's a cause |
| K12 | the attention kernels: tiled attend and backward, 33% at `T` 1024 | From the window program; From multi-head attention | **stage 5** |
| K28 | the window program's own kernels: short-`k` products, layer norm, the linear product | new, from the profile | **stage 5**, from the census |
| K10 | narrow `k`: more independent outputs in flight per tile; 16-18% behind OpenBLAS on `litshe`, and short-`k` shapes dominate the window | From the crate kernels | **stage 5**, on x86 and on `litshe` when on |
| K6 | `CpuInfo` | i7 outline, Kernels and parameters | **stage 6** (D5, D11), for K8 and the thread counts |
| K8 | `matmul_nt` with `W` past the L2: slabs of `W` in bytes, the 4 lanes stored | i7 item 2 | **stage 7**, if the census and D13's backdrop put it over the bar |
| K13 | a batched crate decode step: `attention_step` over an `(N, c)` cache with the keys' segments | From the sequence task (generation) | **stage 7** |
| K15 | threaded scaling against OpenBLAS, a pool, the threshold re-swept | i7 item 4; From the crate kernels | **stages 8-10** |
| K14 | conv's forward spread over examples | i7 item 3 | **stage 11** |
| K16 | threads in the window program, examples split inside one call | From the window program (D8 (b)) | **stage 11** |
| K2 | x86 placement: loops aligned to 32 bytes (`-align-loops=32`), timed as a full baseline | From the crate kernels; From the benchmark machine baseline | **when a read needs it**: first when a claim sits near 2% |
| K3 | `litshe`'s placement: a cheaper pin (loops at 32 bytes, functions only) | From the crate kernels (D7) | **with K2**, on `litshe` |
| K4 | small unexplained rows: `learn linear k1 s16 m16 B64 W19` +4.2% since #1137; #1174's loader +2.3%; #1175's numpy causal forward +2.5% | research separation; context-length leftovers | **with K2**: under the bar |
| K5 | the per-op baseline predates the crate's later work | i7 outline | **stage 13**: the closing baseline |
| K17 | an SVE or AVX-512 `exp` and transpose | From the crate kernels; From aarch64 | not in this plan: no machine with them |
| K18 | a bits-changing kernel (fused or reassociated arithmetic) | From the window program (D3 (b)) | not in this plan: it needs the owner's approval and a golden re-record; D2 keeps the bits |
| K19 | musl's golden run | From aarch64 | not in this plan: a check, if musl becomes a target |
| K20 | the numpy side: the attention backward's 10%, the tiled numpy backward at `T` 64, numpy's decode step | From multi-head attention; context-length leftovers; From the sequence task | not in this plan: numpy expressions, in next-steps |
| K21 | the forward context's 1.1% on Rust's dense B=32 epoch | From segments | not in this plan: accepted (segments D10) |
| K22 | checkpointing the window program's forward | From the window program (D5 (c)) | not in this plan: memory, not speed |

## Decisions

Each settled, with its options.

- **D1. The kernels first. Settled: (a), revised.**
  - (a) *Chosen.* Kernels before threads, but only K1 as a prerequisite; the rest by epoch share. Pros:
    threads multiply what a kernel costs, so a thread's gain is read against kernels with no known
    gap; the gains arrive first. Cons: threading's own payoff waits.
  - (b) As drafted, all of stage −1 in order. Pros: settled, and no measurement fault is left behind.
    Cons: placement and a full baseline come before a 12% gain ready to ship.
  - (c) Interleave. Cons: a kernel change and a threading change in the same rows can't be told apart.
- **D2. Bits never depend on the thread count. Settled: (a).**
  - (a) *Chosen.* Partition outputs only: rows, column chunks, examples, (example, head) pairs, query
    blocks. No reduction is split across threads. Pros: the golden run holds at every thread count and
    on every machine; tests check bits at 1, 2, 3 and the default threads. Cons: a reduction over a
    long axis (the batch's sum in a weight gradient) can't be split, so its product threads over its
    outputs instead, as `matmul_long_k` does.
  - (b) A fixed reduction tree, independent of the thread count. Pros: splits reductions too. Cons: new
    bits everywhere it applies and a golden re-record; today's partitions already cover the products.
- **D3. The pool. Settled: (a).**
  - (a) *Chosen.* A std-only pool in the core: workers started on first use, parked between calls (a
    short bounded spin, then sleep, so it doesn't steal cores from numpy as OpenBLAS's spin steals them
    from us); a call hands it one job and its chunk count; workers claim chunks from an atomic counter
    (dynamic, so stone's E-cores and P-cores even out); the caller works too; a worker's panic is
    carried to the caller; the pool re-created when the process id changes (a fork), and a call from
    inside a worker runs inline. Pros: the core keeps no dependencies; control over spinning, fork and
    chunking, the three things that bite here; scoped borrows as today. Cons: about 200-300 lines of
    concurrency to get right; tested under `miri` and a stress test, and `loom` if its model fits.
  - (b) rayon. Pros: mature, work-stealing, scoped. Cons: the core's first dependency; its global pool
    hangs in a forked child (a known hazard with Python's `fork`); its spin policy isn't ours.
  - (c) Spawn per call, the threshold re-tuned. Cons: the spawn cost is what was measured; no gain at b32.
- **D4. One thread budget a process. Settled: (a).**
  - (a) *Chosen.* `INDRAJALA_THREADS` (read once) and `set_threads(n)` (Python-callable), the crate's
    ceiling; default the physical cores (D5), not `available_parallelism`'s logical ones. The batch
    runner and the studies' sweeps set 1 a worker unless a job declares threads, and refuse when workers
    x threads exceeds a host's cores. Pros: no oversubscription when 8 study processes each open a pool;
    one rule for every caller. Cons: every launcher has to set it (they're few, and a check catches the rest).
  - (b) Default 1, opt in. Pros: safe. Cons: a user of the library gets no threads unasked; the production
    target is the other way.
- **D5. The machine read once. Settled: (a), revised.**
  - (a) *Chosen.* `CpuInfo` in the core, read once, built when K8 first needs it; sizes and counts come
    from it, each override hook kept; the threading threshold stays measured per machine, since it
    depends on clock ramp and the pool's cost, not cache sizes. A formula has to reproduce today's
    constants wherever they were found best. Pros: one codebase, no branches per chip; no parsing until
    it's used. Cons: sysfs parsing to test on four machines.
  - (b) Before K7, as drafted. Pros: one mechanism from the start. Cons: K7's slab doesn't need it, so it
    delays the first gain for nothing.
- **D6. Epochs decide. Settled: (a).**
  - (a) *Chosen.* A threading change is judged on training epochs and study runs at the thread count
    training uses, beside the per-op rows; per-op rows say where, epochs say whether. Pros: isolated
    loops flatter threading (cold cores). Cons: longer A/Bs.
- **D7. One plan, sub-stages. Settled: (a).**
  - (a) *Chosen.* The items as stages of this plan, each its own PRs and A/B, findings written here.
    Pros: one place for the order and the dependencies. Cons: a long plan.
- **D8. One workplan. Settled: (a).**
  - (a) *Chosen.* This file replaces both; the outline's evidence and mechanisms are here, and it's
    deleted. Pros: one order, one set of IDs, no two documents to keep true (they already disagreed on
    workplans and machines). Cons: a long file; the outline's October tables live only in the machine
    profile and git history until K5 replaces them.
  - (b) Keep the outline as background, linked from here. Pros: nothing moves; a short plan. Cons: its
    stale conclusions stay readable as current.
  - (c) Two plans, kernels and threads, linked. Pros: each shorter. Cons: D1's order is what ties them,
    and it would then live in neither.
- **D9. Order by epoch share. Settled: (a).**
  - (a) *Chosen.* Stages after K7 are chosen from the census and the profiles on the backdrops: sequence,
    attention, window, conv, dense, and D13's larger model. An item must be able to clear about 5% of an
    epoch in a trained configuration on one of them, with its op's ratio to numpy as a tie-break. Pros:
    work follows real time; studies and production both benefit. Cons: a large per-op gap on a rare shape
    can wait (the conv tail's forward may).
  - (b) By the ratio to numpy, as the outline. Pros: simple, comparable. Cons: it put the conv tail first
    while transformers spent half their time on one fixable path.
  - (c) The window program alone, as the studies' own workload. Pros: most direct to the research. Cons:
    the crate is a library with a production target, and the same kernels serve both.
- **D10. Slabs, then packing. Settled: (a).**
  - (a) *Chosen.* K7a: past `K` 64, all rows a block and `k` in slabs of `LONG_K_BLOCK`, resuming, in
    `matmul_2d`, with `ADD` added once at the end (one rounding, as now); `matmul_long_k` becomes a caller
    of the same path, its column split kept for threaded long products. K7b: packed panels where `b`'s
    stride passes a page. Pros: a 25-line version is proved on real training and the golden run; one
    code path instead of two (structure first); each step is separately measured; K7b's buffer is 16 KB.
    Cons: two PRs; K7a alone leaves 64 x 256 x 1024's gain for K7b.
  - (b) Packing everywhere at once. Pros: a single rule. Cons: it loses to slabs at narrow `n` on all four
    cores (0.29 against 0.18), and `K` 32 by up to 8%.
  - (c) A shape test in the dispatch, sending some shapes to `matmul_long_k`, as the outline proposed. Pros:
    little new code. Cons: `matmul_long_k` has no `ADD`, and two paths to the same arithmetic stay.
- **D11. Per-core caches. Settled: (a).**
  - (a) *Chosen.* `CpuInfo` reports each level's size divided by the cores sharing it, and the smallest
    across cores: L2 512 KB on stone (E), not its P-cores' 1.28 MB, nor sysfs's first answer. The candidate
    rule gives stone's cores what they measure best or within noise. Pros: a hybrid's slabs fit every
    core a chunk can land on; no scheduling dependence. Cons: a P-core's cache is underused; the data has
    it at 0.80-0.83 anyway.
  - (b) One number from `cpu0`. Pros: simplest. Cons: E-cores get slabs sized for 2.5x their share.
  - (c) Per-core-type slabs, chosen where a thread runs. Pros: best fit. Cons: a syscall or `cpuid` per call,
    and migration between them; nothing measured asks for it.
- **D12. jebel claims, laptops guard. Settled: (a).**
  - (a) *Chosen.* A speedup is claimed on jebel only, by §9. A kernel or threading stage also runs one A/B
    on a Ryzen and one on stone, one thread, as a guard: no consistent slowdown beyond their noise rules,
    no claim. Replaces the outline's "A change for the i7 must not cost the laptop". Pros: three cache
    hierarchies check each parameter; K7's probe already paid for itself there. Cons: slower stages;
    a laptop's noise can hide small costs.
  - (b) Claims on every machine. Pros: broad claims. Cons: their noise isn't characterised, and it isn't
    worth characterising to a claim's standard.
  - (c) jebel alone. Pros: fastest. Cons: `CpuInfo` tested on one cache hierarchy designs for one; the owner
    has offered the others.
- **D13. Backdrops at production scale. Settled: (a).**
  - (a) *Chosen.* `sequence_training_timing` and `attention_timing` take `--d`, and a run at `d` 512 and
    `T` 256 joins the backdrops; no study trains it, only the timing does. Pros: a production model's `W`
    is past every L2 here (an FFN `W` of 512 x 2048 is 8.4 MB), so K8 and threads get judged where they
    would matter. Cons: a slower benchmark; run sparingly.
  - (b) Stay with today's models. Pros: no new cells. Cons: toy sizes decide a design (the production
    target).
- **Settled by the owner (2026-11-02):**
  - **O1. Where this sits. Settled: at the end,** after steps 14 and 15 and the work after them. Steps 1-15
    are done; the certificate note's draft and the two notes under consideration aren't this plan's and
    run beside it, since a crate PR needs jebel, not the owner.
  - **O2. `litshe`'s limits with the active cooler. Settled:** "we base the profile on the measured
    temperature response, when the cooler is available". Until then the rule stands (65 C ideal, 70 C
    hard, single-threaded passes), and stage 12 sets the profile from the measured curve.

## Stages

Each is a crate PR where the crate changes, then a "Bump rust/" PR and its A/Bs (tier 2 for a claimed
gain). Each ends at the 5% bar: an item whose next step can't clear 5% of an epoch stops, its reason
recorded here. Every kernel stage keeps the bits, adds crate tests at shapes that reach its paths, and
lands its NEON side too (bits checked by the committed Rust reference on CI; timed on `litshe` when
it's on).

- **Stage 0. This workplan,** with `scripts/measurement/blocking_probe.rs`.
- **Stage 1. Measurement.**
  - K1: `ab.py` builds and installs a named crate on both sides, or refuses.
  - K25: a `census` cargo feature on the core records `(kernel, m, k, n, calls, flops)` and writes it at
    exit; a build for measuring only, never in a wheel we ship. Run over the backdrops; the shares go here.
  - K26: `--d` on the two benchmarks (D13).
  - K27: stone's profile (`machine_profile.py profile --out`), an A/A for its noise rules and the Ryzen's;
    `ab.py` run on stone from a bundle's checkout, no GitHub.
- **Stage 2. K7a (D10).** Crate, then the bump. Expect the prototype's: −12.4% on a sequence epoch,
  −6% to −15% on attention's backward. Guards on a Ryzen and on stone (K27).
- **Stage 3. K7b (D10).** Packed panels past a page's stride. Target: 64 x 256 x 1024's 0.37, the conv tail
  at numpy's time. Its threaded form (column split with packing) waits for stage 10.
- **Stage 4. Copies (K23, K11, K24).** Accumulate without its transpose first; K11 only for what remains.
  Allocation sites from the census.
- **Stage 5. By share: conv forward (K9), attention (K12), the window's kernels (K28), narrow `k` (K10).**
  Each profiled (`perf_region.py`) then fixed or closed, in the order the census gives. The window's come
  first if the shares tie, since it's where the studies spend their time.
- **Stage 6. `CpuInfo` (K6, D5, D11).** Read once (Linux sysfs and `cpuid` on x86, sysfs on aarch64), the
  machine profiles' fields, per-core sizes. A new bottom layer `sys` beside `num` (`core/tests/layers.rs`
  needs it added). Tier 1: no slower, as nothing reads it yet but the next stage.
- **Stage 7. K8, K13, if over the bar.** K8 tests the candidate rule on jebel, a Ryzen and stone's P and E
  cores, at D13's larger backdrop. K13: a batched decode step.
- **Stage 8. Thread tools.** `thread_cost_probe.py` extended to a pool (call cost at 1-8 threads, cold and
  warm), a scaling sweep per op (threads x shape), oversubscription (N processes x T threads on one host),
  and an epoch timer at a set thread count through `ab.py` (`--rust-threads` exists). On jebel, stone and klip.
- **Stage 9. The budget and the pool (D3, D4).** `CpuInfo`'s physical cores, `INDRAJALA_THREADS`,
  `set_threads`; the runner and sweep helpers set 1 a worker and refuse oversubscription. The pool in the
  core, with its tests (every chunk run once, panic carried, fork recreates, nesting inline, `miri`, a stress
  test at random chunk counts); no caller yet. Tier 0.
- **Stage 10. The existing sites on the pool (K15).** `for_each_row_range`, the column split, conv's
  downstream; bits at 1, 2, 3 and the default threads; long products split by column, with stage 3's packing;
  the threshold re-swept per machine on epochs (D6). Tier 2 on jebel, guards on the laptops.
- **Stage 11. New threaded sites, by measured share.** Conv's forward over examples (K14); the tiled attend
  over (example, head) and query blocks; the window program over examples (K16); the batched decode over rows
  (after K13); row-wise ops only where a profile shows them worth a call's cost.
- **Stage 12. numpy beside the crate, and `litshe`.** Whether `OPENBLAS_NUM_THREADS` should follow the crate's
  budget in mixed runs (the spin gotcha, and the 6-11% above), documented either way. `litshe` all-core with
  the cooler (O2): its temperature response measured at 1, 2 and 4 busy cores, each to its plateau or to the
  hard limit, sampled each second under the thermal guard at 70 C; its profile set from that curve; then its
  scaling, its share of the memory bus and the threshold for aarch64.
- **Stage 13. Close.** The closing baseline (K5, tier 3): jebel's per-op table and epochs at main, replacing
  `machine_profiles/i7-9700k.md`'s tables, with the Ryzen's columns. The README's threading section (the budget,
  the bits rule, the pool), measurement.md (threaded A/Bs, the tools), K2-K4 closed or moved to next-steps'
  leftovers, the crate's next release noted, the workplan retired.

## Checks

- Every stage keeps the golden run byte-identical, on x86_64 and aarch64 (the committed Rust reference on
  both CI runners); threaded tests run at several thread counts.
- A/Bs on jebel under the machine lock (one at a time with the studies); klip's only after its affine pass,
  under its thermal guard; nothing beside heavy development on pyramidon, under an address-space cap;
  stone by bundle, in its hours.
- The probe's numbers choose work; only `ab.py` runs make claims.

## Not in this plan

- GPU backends (excluded, for now).
- Changing any summation order (K18, D2 (b)).
- Threads in numpy or pure Python (numpy's own BLAS threads are stage 12's reading only).
- SVE and AVX-512 kernels (K17), until a machine with them is in hand.
