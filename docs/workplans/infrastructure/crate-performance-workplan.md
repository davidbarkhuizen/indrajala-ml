# Workplan: the crate's performance, kernels and threads as one plan

**Status: unified 2026-11-04 at the owner's direction; in progress, and the owner's "work through the plan" (2026-11-04) stands.** The owner: "1. review the
current situation as it stands 2. re-analyse the solution and approach 3. combine the threading plan
with the i7 plan and create one unified workplan". This replaces the multithreading workplan
(drafted 2026-11-02) and the i7 outline (`docs/i7-9700k-rust-optimization.md`, 2026-10-08); both are
in git history at `8e342ab`. The IDs D1-D7, O1, O2 and K1-K22 keep their meanings, so older
references still read; what the re-analysis changed is marked. The owner delegates the decisions;
none is pending.

**The short version.** Both plans ordered the work by the largest ratio to numpy, on shapes from
October. Measured, the largest ratio isn't where the time goes, the first kernel fix helps
transformer training more than the conv tail it was planned for (K7a: −47.9% on a `d` 512 epoch), and
most of stage −1's prerequisites can wait. So: one fix first (K7), a shape census to choose what follows,
the machine-derived parameters only where a measurement needs them, and threads after that, judged on
epochs. **Stages 0-3 are done, and stage 4's first step.**

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

### Stage 2's results (K7a, crate #101)

Tier 2 on jebel, crate `690bce8` against `5a19b18` (the merged tree), `ONNONO`:

| benchmark | run (`2026-11-04-`) | result |
|---|---|---|
| sequence epoch, `d` 512, 8 heads | `k7a-sequence-d512` | **−47.9%** (18.54 to 9.66 s), numpy within noise |
| attention backward, `d` 512, `T` 256 | `k7a-attention-d512` | **−39.7%** (0.300 to 0.181 s) |
| sequence epoch, `d` 64 | `k7a-sequence` | **−12.4%** (2.048 to 1.793 s), numpy within 0.0% |
| attention backward, `d` 64 | `k7a-attention` | −12.2% at `T` 64, −7% at 256, −4% to −5% at 1,024; 19 rows within noise |
| accumulate at batch 512, one thread | `k7a-focused-1` | conv tail **−65.9%** (15.49 to 5.28 ms), attention −63.7%, dense 30 x 784 −39.1%; 32 rows within noise |
| the same, default threads | `k7a-focused` | attention −24.6%, dense −18.1%, conv tail −14.5% |
| conv and dense epochs | `k7a-epochs` | all 16 within noise, max 3.0% |
| window program | `k7a-window` | 5 rows within noise, max 3.4%; **`learn family` +1.0%, consistent, small**; its control, a 256 cube, −46.9% |

The golden run is bit-identical on jebel and pyramidon. **Guards, no slowdown:** stone (one thread: conv
tail −64.5%, sequence −9.6%) and the Ryzen on pyramidon (one thread: conv tail −56.9%, attention −51.0%;
`--skip-profile`, its noise rules still to refresh). `learn family`'s +1.0% is an atomic read and a branch
added to millions of `(3, 8)` products, or a build's shift; stage 4 replaces that path, so it's left there.

### Stage 3's results (K7b, crate #102)

**K7a alone had a cost its guard didn't reach.** The guards' shapes had small `m`. At many rows it left `a`'s
page-sized stride, and on the Ryzen `(2048, 512) @ (512, 512)` and `(2048, 256) @ (256, 64)` ran up to 1.7x
slower than before K7a (the probe, not an A/B). It was on `main` for about an hour, and K7b removes it.

**What the probe found** (`blocking_probe_large_m.rs` in the evidence directory; jebel, a Ryzen and stone's
P-cores, all bit-identical): D10's page-stride test was the wrong one. Packing lost below 64 rows and won from
64 rows at every `n`, so the rule is the row count (`PACKED_ROWS`): from 64 rows in a chunk, each slab of 128 rows
of `b` is packed, and 64 rows of `a` at a time run through every tile. Below that, K7a's slabs stand.

Tier 2 on jebel, crate `6a8190c` against `7b9878f` (the merged tree), `ONNONO`:

| benchmark | run (`2026-11-04-`) | result |
|---|---|---|
| sequence epoch, `d` 512 | `k7b-sequence-d512` | **−22.2%** (9.63 to 7.49 s; with K7a, 18.54 to 7.49, from two A/Bs), numpy within noise |
| attention backward, `d` 512, `T` 256 | `k7b-attention-d512` | **−20.8%** (0.184 to 0.146 s) |
| products, one thread | `k7b-products` | 512 x 2048 x 2048 **−68.8%**, 2048 x 512 x 2048 −67.7%, 2048 x 2048 x 512 −65.1%, 2048 x 512 x 512 −54.7%, 2048 x 2048 x 64 −50.3%, 64 x 256 x 1024 −46.7%, 2048 x 256 x 64 −25.8%; 13 rows within noise |
| sequence `d` 64, attention `d` 64, per-op rows, conv and dense epochs, window | `k7b-sequence`, `-attention`, `-focused`, `-epochs`, `-window` | **all within noise,** max 0.7%, 3.2%, 2.0%, 2.4%, 0.3% |

The golden run is bit-identical on jebel and pyramidon. **Guards, against the crate before K7a (`690bce8`):** no
Rust row slower. The Ryzen: 2048 x 512 x 512 −55.5%, 2048 x 512 x 2048 −71.3%, 2048 x 2048 x 512 −92.5%, sequence
within noise (max 6.2%). Stone: 2048 x 512 x 512 −41.2%, 512 x 2048 x 2048 −94.1%, sequence −10.6%; one pass
shifted −4.0%, so that run's small rows are unbalanced, and it wasn't extended, the rows read being far past it.

### Stage 4's results (K12, K28; crate #103)

**The first gain to reach the window program.** Tier 2 on jebel, crate `7fffc49` against `a4c4379` (the merged
tree but for `CpuInfo` and its lint fix, which nothing reads yet), `ONNONO`:

| benchmark | run (`2026-11-04-`) | result |
|---|---|---|
| window, `k1 s4 m16 B32 W12`, learn | `k12-window` | **−20.7%** |
| window, `k4 s4 m16 B32 W5`, learn | | **−11.1%** |
| window, family, learn | | **−9.7%** (0.023 to 0.020 s) |
| window, family, a long test's forward | | **−8.1%** (3.407 to 3.131 s) |
| window, linear, learn and forward | | within noise, max 1.0% |
| attention, `d` 64 | `k12-attention` | backward −5.5% (`T` 64), −3.1% (`T` 256); forward −4.1%, −2.0%; 20 rows within noise |
| sequence epoch, `d` 64 and `d` 512 | `k12-sequence`, `-d512` | −2.8% (1.797 to 1.746 s); `d` 512 within noise |
| per-op rows, one thread | `k12-focused` | attention forward −3.0% and −1.8%; 34 rows within noise |

K7a's +1.0% on `learn family` is gone. The golden run is bit-identical on jebel and pyramidon. **Guard, stone:** no
slower, the same four rows −8.6% to −19.3%. `CpuInfo` (K6) is in too (crate #104, #105), read by nothing yet.

### Stage 5's results (K6, K8; crate #106)

`matmul_nt` now takes `W` in slabs of `k`, resuming each output's four lanes, whose order is the bits'. The rule
is the candidate's and one more line: none if `W` fits half the L2, else a slab of `min(L2 / 4, 128 KB) / (n * 8)`
rows, and **none if that is under 64 rows**. Bits hold and the golden run is unchanged.

The minimum is the run's finding. The first version made a `d` 512 epoch **5x slower** (7.56 -> 38.93 s, forward
+362%): the slabs came to 12 rows, and the lanes stored between them to 33.5 MB, which cost far more than the slab saves.
The probe had never tried that shape, nor had the first run's targets; only the `d` 512 backdrop (D13) saw it. So
**every A/B of a kernel carries the `d` 512 backdrop**, on the guards too.

Passes ONNONO, old 4dba70e vs new ce5a267, the corrected rule:

| Machine | Benchmark | Result |
|---|---|---|
| jebel | conv tail forward (32 x 5408), one thread | b32 -32.7%, b512 -29.1% |
| | 30 x 784 forward, one thread | dense -21.6% / -23.1%, relu -24.4% / -24.3% |
| | same, default threads | conv tail -32.8% / -33.1%; dense -23.0% / -11.0% |
| | dense epoch | B=32 -7.8%, B=512 -6.0% (loader passes -8.5%, -8.6%) |
| | `d` 512 epoch, `d` 512 attend, `d` 64 epoch, window | all within noise (max 0.6%, 0.7%, 0.2%, 4.4%) |
| klip | conv tail forward | b32 -8.6%, b512 -5.1% |
| | `d` 512, sequence, window | within noise |
| stone | conv tail forward | b32 -16.8%, b512 -7.7%, both inside their spread (its smallest L2 share is 512 KB, so this `W` slabs and 30 x 784's doesn't) |
| | `d` 512, epochs, window | within noise |

jebel's `conv tail downstream_batch b512` at default threads read +6.6% in both runs, and within noise at one
thread. It runs `matmul_2d`, whose instructions K8 left alone; klip reads -8.0% on it. So it is where the moved code
landed, not a kernel, and it gets its next look when stage 10 replaces the threaded sites.

What this leaves: at `d` 512, `matmul_nt` is a third of the epoch and K8 doesn't reach it, since a 12-row slab loses.
Blocking its rows as well as `k` (the lanes held per output, rather than for all of them at once) is untried; a probe
goes in stage 7's order by the fresh census.

### Stage 6's results so far (K23; crate #107)

`matmul_transposed_add(c, d, b)` is `c + d.T @ b` with no copy of `d`, read through a row and a `k` step, for a `b`
under 512 wide. From 512 it keeps the copy. Bits hold (a new test against `matmul_add` of the transpose, on each side
of the slab, packing and threading thresholds), and the golden run is unchanged.

The guards decided both limits on the way:

1. The first version sent every shape through a zeroed product buffer. Conv tail b32 lost 30-35% on klip and stone,
   a 1.38 MB allocation and its page faults. ADD now writes in place up to the slab, as `matmul_add` did.
2. Then a `d` 512 epoch lost 3.1% and its backward 6.7% on jebel, and gained 3% on stone and klip. At `m` 512 `d`'s
   stride is 4096 bytes, so a slab's multipliers fall in one L1 set and a few L2 sets, more than jebel's 4-way 256 KB
   L2 holds (the mechanism is inferred from the stride and the caches; only the timings were measured). The conv tail
   at b512 also lost 10% on klip. Both have a wide `b`, where the copy is about `1 / n` of the work, so there's little
   to win.

Passes ONNONO, old 10aa3e5 vs new 3bb2a40:

| Machine | Benchmark | Result |
|---|---|---|
| jebel | attention accumulate (16 x 32), one thread | b32 -34.8%, b512 -33.2% |
| | same, default threads | b32 -34.3%, b512 -46.6% |
| | sequence epoch, `d` 64 | -8.2% |
| | window learn | linear -6.0%, family -5.2% |
| | `d` 512 epoch, `d` 512 attend | within noise (max 0.8%, 0.7%) |
| | dense and conv epochs | within noise; `epoch, loader` at B=512 +3.4% (dense), +2.5% (conv) |
| stone | attention accumulate | -32.2%, -30.8% |
| | `d` 512, epochs, window | within noise |
| klip | attention accumulate | -21.8%, -24.6% |
| | sequence epoch | -5.9% |
| | window learn | -3.2% to -4.6%; forward family +1.4% |
| | `d` 512 epoch | +1.0% |
| | conv tail accumulate b512 | +1.8% |

The `epoch, loader` rows repeated across two builds, on a smaller share than the epochs' gains, and the plain epoch
rows don't show them. They stay on record, not chased.

Jebel moved to kernel 7.0.0-42 during this stage (a reboot). Each A/B's two sides share a kernel, so this reading
holds, but A/Bs from before it don't compare with those after, and K5's baseline records the new kernel.

The rest of stage 6 closes under the bar, with K23 in. A flat profile on jebel (`perf record`, `perf_region.py`'s
machine, with call stacks from its branch records):

| Backdrop | transpose | memmove | memset | malloc family |
|---|---|---|---|---|
| attention | 3.9% | 3.7% | 3.0% | under 1% |
| window | 1.5% | 4.3% | 1.8% | about 5% |
| `d` 512 (both backends) | 0.55% | 6.0% | 0.8% | under 1% |

**K11** closes: at most 3.9%, and only part of that could be won. **K24** closes: no one site passes 1.4%
(`matmul_nt`'s zeroed output 1.38% of the window, `tiled_backward`'s tile copies 1.11% of attention, `scores_into`
0.92%, `token_rows`' reshape 0.67% of the window). The largest are `calloc` zeroing, which only uninitialized memory
would remove, and copies that need views `Array` hasn't got. The allocator's 5% of the window is spread over
many sites. Neither is reopened unless a site grows.

### Stage 7's first step (K28; crate #108)

The same profile put two things in the window's top three that no earlier reading had named: `layer_norm::normalize`
14.3% and `linear_preactivation_batch` 9.8%, ahead of the attend. Neither was a kernel's arithmetic:

- `normalize` allocated a `Vec` a row and ran its elementwise steps scalar. Only its two folds are chains; the rest
  takes four lanes (AVX2 with no FMA, since a fused `g * xh + b` would round once).
- `linear_preactivation_batch` added `b` into a new array with a push per value. It now adds in place, and
  `broadcast_row`, which others use, is an iterator that vectorizes.

Bits hold, the golden run is unchanged. Passes ONNONO, old 238a017 vs new 18ce111:

| Machine | Benchmark | Result |
|---|---|---|
| jebel | window forward | family -22.6%, linear -15.6% |
| | window learn | family -8.7%, linear -7.2% |
| | attention forward (16 x 32), one thread | b32 -16.6%, b512 -12.9% |
| | sequence epoch, `d` 64 | -7.0% |
| | `d` 512 epoch, `d` 512 attend forward | -2.9%, -4.8% |
| | dense, conv epochs | within noise |
| stone | window forward | family -26.0%, linear -16.3% |
| | window learn | family -11.3%, linear -9.0% |
| | `d` 512, epochs | within noise |
| klip | window forward family | -17.6% |
| | window learn | family -5.7%, linear -5.6% |
| | sequence epoch | -5.1% |
| | `d` 512 | within noise |

The cheapest change of the plan, and among its largest for the window. The kernels' share had hidden it: **read a
flat profile before a kernel**, as a census names only what it instruments.

One row against: attention's accumulate at b32 and one thread reads +6.8% and +5.3% on jebel (twice), and not on
stone, klip, or at default threads, with b512 -3.1%. K28 didn't touch that kernel, so it is taken as placement (K2).

Jebel rebooted itself mid-run at 14:34 on 2026-11-04, an unclean shutdown about an hour into kernel 7.0.0-42, with
nothing logged. One sequence A/B was lost and rerun; the run after it, on the new boot, passed the machine check.

### What the census says (K25, on stone, with K7a in)

Seconds by kernel, one worker. A site's time holds what it calls: the attends hold `exp`, `transpose` and
their products, so they aren't additive with those. Per-call rows in the evidence directory's `census/`.

| backdrop | largest shapes and kernels |
|---|---|
| sequence, `d` 64 | attend 0.86 s; `matmul_2d` 0.52, `matmul_nt` 0.32, exp 0.26, transpose 0.20 |
| sequence, `d` 512 | **`matmul_2d` 3.05** (2048 x 2048 x 512, 2048 x 512 x 2048, 512 x 2048 x 2048); **`matmul_nt` 1.83** (2048 x 2048 x 512); attend 0.50 |
| window family, learn | **attend 1.02** at `(192, 2 heads, d_k 8)`; `(3, 8)` tile products and transposes: 1.5M-2M calls each |
| window family, long test's forward | **attend 3.0** of 5.4, 61,500 calls at 46 µs; 4.2M `(3, 8)` products and as many transposes |
| window linear, learn | `matmul_nt` and tiny `matmul_2d`; the 0.044 s at 256 cube is the benchmark's own control |
| conv, mini-batch 32 | `conv_forward` 0.17; **`dot_products` 32 x 5408** 0.13 (the accuracy passes, 16 µs a row); `matmul_2d` 32 x 32 x 5408 0.05 |
| dense, B=512 | **`matmul_nt` 32 x 784 x 30, 0.21, half the time recorded** (the accuracy passes); 512 x 784 x 30 0.09; `matmul_2d` 0.08 |

So the order changed again (D9): **the attend** comes before the copies, as most of the window program's time
goes through its per-tile allocations, and **K8** clears the bar after all, on dense's accuracy passes and at
`d` 512, where the probe's 0.65 applies.

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
into about 0.9x on jebel. The census below puts it over the bar. (K8 kept this rule and added a minimum slab: stage 5's results, above.)

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
| klip | Ryzen 7 3700U, CPU boost off | 4 / 8, SMT | 512 KB, 4 MB | **guard:** a change costs the Ryzen nothing; physical against logical cores. Its own profile. 5.3 GB, <= 85 C, under the thermal guard (`~/code/thermal-guard.sh`) |
| pyramidon | Ryzen 7 3700U | 4 / 8, SMT | 512 KB, 4 MB | development; a guard only when klip is taken, under an address-space cap, beside no heavy job: its A/As are noisier |
| stone | i7-1255U (Alder Lake, hybrid) | 2 P + 8 E, 12 threads | P 1.28 MB, E 2 MB across 4 (512 KB each); 12 MB | **guard:** hybrid scheduling, static splits leave the fast cores idle. The sandbox stands (no GitHub, bundles, `~/labour`); it needs a profile and noise rules (K27). Available weekdays 18:00-07:00 and weekends unless the owner says otherwise |
| `litshe` | Pi 5, Cortex-A76 | 4 / 4 | 512 KB, 2 MB | aarch64 parity and timings, when on; all-core once the active cooler is fitted (stage 12) |

## The items

| # | item | source | now |
|---|---|---|---|
| K1 | `ab.py run --old-crate X --new-crate X` installed neither wheel, timing the venv's crate on both sides | From the A/B harness | **done**: a named crate is built and timed |
| K25 | a shape census: which kernel, at which `m x k x n`, how long, for a backdrop | new (D9) | **done** (crate #100): `shape_census.py`, its readings under stage 1 |
| K26 | backdrops at production scale: `sequence_training_timing` and `attention_timing` at a larger `d` | new (D13) | **done**: `--features`, `--heads` |
| K27 | stone's profile and noise rules, and the Ryzen's refreshed; `ab.py` from a bundle's checkout on stone | new (D12) | **done**: `i7-1255u.json`, `ryzen7-3700u-klip.json` |
| K7 | `matmul_2d` for long `k`: slabs that resume each chain | i7 item 1 | **done**: K7a (crate #101), K7b (crate #102), results below |
| K23 | accumulate without its transpose: `delta.T @ X` read through `delta` directly (`nn/dense.rs:192`, 3-6% of the backdrops) | new | **done (crate #107)**, below a `b` 512 wide: attention accumulate -33% to -47%, sequence epoch -8.2% (stage 6's results) |
| K11 | an AVX2 transpose that doesn't lose to 4K aliasing (the measured kernels on the crate's `avx2-transpose` branch) | From the crate kernels | **closed**: under the bar with K23 in, 3.9% of attention at most (stage 6's results) |
| K24 | fewer copies and allocations in the crate's ops (memmove, memset, calloc: 5-10% of the window and attention backdrops) | new | **closed**: no site past 1.4% (stage 6's results) |
| K9 | conv on 28x28 inputs, `conv_forward_batch` first (64% of a conv-conv epoch) | i7 item 5 | **stage 7**: diagnosis, then a fix if there's a cause |
| K12 | the attention kernels: tiled attend and backward, 33% at `T` 1024 | From the window program; From multi-head attention | **first step done** (crate #103); the rest from a fresh census, stage 7 |
| K28 | the window program's own kernels: short-`k` products, layer norm, the linear product | new, from the profile | **second step done (crate #108)**: layer norm and the dense add, window forward -22.6%; the rest by profile, stage 7 |
| K10 | narrow `k`: more independent outputs in flight per tile; 16-18% behind OpenBLAS on `litshe`, and short-`k` shapes dominate the window | From the crate kernels | **stage 7**, on x86 and on `litshe` when on |
| K6 | `CpuInfo` | i7 outline, Kernels and parameters | **done (crate #103, #105, #106)** (D5, D11), for K8 and the thread counts |
| K8 | `matmul_nt` with `W` past the L2: slabs of `W` in bytes, the 4 lanes stored | i7 item 2 | **done (crate #106)**: conv tail forward -33%, 30 x 784 -23%, dense epochs -6% to -8%; `d` 512 not reached (stage 5's results) |
| K13 | a batched crate decode step: `attention_step` over an `(N, c)` cache with the keys' segments | From the sequence task (generation) | **stage 7**, if over the bar |
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
  - (a) *Chosen.* `sequence_training_timing` and `attention_timing` take `--features` and `--heads`, and a
    run at `d` 512 and 8 heads (`T` 256 in `attention_timing`, the corpus's windows in training) joins the backdrops; no study trains it, only the timing does. Pros: a production model's `W`
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
  - K1: `ab.py` builds and installs a named crate on both sides. *Done.*
  - K25: a `census` cargo feature records `(kernel, m, k, n, calls, nanoseconds)`, each process's written
    at exit; never in a wheel we ship, and without it all 794 of the crate's functions are unchanged. *Done:*
    `scripts/measurement/shape_census.py`.
  - K26: `--features` and `--heads` on the two benchmarks (D13). *Done.*
  - K27: stone's profile and noise rules, from six A/As at `8eeda72`, `ab.py` run there from a bundle's
    checkout (working-notes, Machines). *Done:* no row of 123 consistent by chance, passes shifted up to
    1.9%, a 5.9% median spread at one thread: a guard, as D12 has it. klip (the Ryzen, CPU boost off): three A/As,
    no row of 54 consistent by chance, shifts up to 0.7%, a 1.1% median spread at one thread, far quieter than
    pyramidon, so it's the Ryzen's guard from here.
- **Stage 2. K7a (D10).** *Done:* crate #101. Results below.
- **Stage 3. K7b (D10, refined).** *Done:* crate #102. Results below.
- **Stage 4. The attend (K12, K28).** *First step done* (crate #103, results below): the tiled passes reuse
  their tiles' buffers. What's left in the attend comes from a fresh census, in stage 7.
- **Stage 5. `CpuInfo` and K8 (K6, D5, D11).** Read once (Linux sysfs and `cpuid` on x86, sysfs on aarch64), the
  machine profiles' fields, per-core sizes, in a new bottom layer `sys` beside `num` (`core/tests/layers.rs`
  needs it added). Then `matmul_nt`'s slabs: the candidate rule tested on jebel, a Ryzen and stone's P and E cores. **Done**: the rule with a 64-row minimum (stage 5's results).
- **Stage 6. Copies (K23, K11, K24). Done:** K23 merged, K11 and K24 closed under the bar.
  Accumulate without its transpose first; K11 only for what remains.
  Allocation sites from the census.
- **Stage 7. By share: conv forward (K9), the attend's remainder and the window's kernels (K12, K28), narrow
  `k` (K10), the batched decode step (K13).** Each profiled
  (`perf_region.py`) then fixed or closed, in the order a fresh census gives.
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
