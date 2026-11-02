# Workplan: multithreading in the crate, after its kernel loose ends

**Status: drafted 2026-11-02 at the owner's direction; no code yet.** The owner: "write up a detailed
workplan for multi-threading, on this basis. 1. I have ordered an active-cooler for the rpi5
(litshe), so that we can test multi-threading there 2. we should probably review the other
outstanding maths-kernel related work, compile a list of that, and do it BEFORE we start the actual
multi-threading work - as I guess stage -1 of the multithreading plan - do the detailed analysis,
then write up the workplan". The owner delegates the decisions (each below with its options' pros
and cons); the two left to the owner (O1, O2) were settled the same day.

This is the "coming multithreading work" the crate kernels plan's D3 deferred to (the owner,
2026-10-29: "multithreading in general comes in the near future"; `git show
7046c98:docs/workplans/infrastructure/crate-kernels-workplan.md`), with "one threading design, made
once with every kernel and the training loop in view". It takes over the threading items of
[i7-9700k-rust-optimization.md](../../i7-9700k-rust-optimization.md) (items 3 and 4), of
next-steps.md's "From the crate kernels" (the pool, the threshold) and "From the window program"
(threads in the program, D8 (b)).

## The IDs, in time order

| ID | When | Short label | What it is |
|---|---|---|---|
| D1 | 2026-11-02 | the kernels first | stage −1 does every outstanding single-thread kernel item that threading would build on or be measured against, one PR (and A/B) each, before stage 2 |
| D2 | 2026-11-02 | bits never depend on threads | threads only partition outputs (rows, column chunks, examples, heads); no reduction is split, so every output is one chain in today's order at every thread count |
| D3 | 2026-11-02 | a std-only pool | a small pool in the core, no dependency: workers parked between calls, work claimed in chunks, lazy and fork-aware, nesting run inline |
| D4 | 2026-11-02 | one thread budget a process | `INDRAJALA_THREADS` and `set_threads(n)`: the crate never uses more; default the physical cores; the runner's workers get 1 unless a job asks |
| D5 | 2026-11-02 | the machine read once | `CpuInfo` (physical cores, caches, ISA) read once per process; thread counts and slab sizes derived from it, every override hook kept |
| D6 | 2026-11-02 | epochs decide | a threading change is judged on epochs and study runs at training's thread count, not on isolated loops, which flatter threading |
| D7 | 2026-11-02 | one plan, sub-stages | stage −1's items are sub-stages here, each its own crate PR, bump and A/B, not a workplan each |
| O1 | 2026-11-02 | after step 13 | the owner: this plan runs after roadmap step 13 (FFN activations) and before steps 14 and 15 |
| O2 | 2026-11-02 | `litshe` from its response | the owner: `litshe`'s thermal profile with the cooler is based on its measured temperature response |

## Where things stand (read 2026-11-02 from the code, next-steps.md and the retired plans)

**Threading today.** `rust/core/src/linalg.rs` threads three products, and `conv.rs` one op, each by
`std::thread::scope`, spawning OS threads on every call:

- `matmul_2d` and `matmul_nt` split output rows (`for_each_row_range`); `matmul_long_k` splits output
  column chunks, so each thread reads only its part of `b`; conv's downstream splits examples.
- `matmul_thread_count`: 1 thread below 8M flops, else `min(available_parallelism, 8, rows)`. The
  8M was tuned on the Ryzen laptop; `available_parallelism` counts SMT threads (8 on a 4-core
  3700U); an override hook (`set_matmul_threading`) for tests and benchmarks.
- Every split is by output, so results are bit-identical at every thread count (the doc comments
  say so, and the golden run is recorded at the default).
- Not threaded: the tiled attention, the window program (the window-recurrent loop), conv's
  forward, every row-wise op (layer norm, softmax, the activations), the decoder's step.

**What a thread costs (the crate kernels plan, stage 3, 2026-10-29,
`scripts/measurement/thread_cost_probe.py`):** a scoped spawn costs about 10 µs a thread on jebel
and 19 on `litshe`, plus a first one's 20-28. At b32 the spawns cost more than the product (jebel:
`(32, 30) @ (30, 784)` 31.6 µs on one thread, 60.2 on four; numpy 41.0 and 22.9). A pool would turn
that into a split of the product's own time (jebel about 8 µs at four threads). On `litshe` four
cores contend for one memory bus: at b512 the crate gains 11% from four threads, OpenBLAS loses 67%.

**What training sees (measurement.md, Gotchas):** in training the cores idle between calls, so a
threaded call starts on cold, clocked-down cores; in isolated loops 8 threads ran 2.3-3.2x faster
than 1 on 1.6-5.5M-flop products, yet lowering the threshold to 2M or 4M moved no epoch by more
than 1.6%. OpenBLAS's spinning workers slow a Rust call that follows a numpy product (1.56x at the
conv tail, batch 512). So the payoff of threading is uncertain until it is read on epochs (D6).

**Where threading pays, and where it doesn't.** Sweeps and batches are already parallel by process
(8 workers on jebel, the runner across klip and stone), so threads add little to their throughput.
They pay where cores sit idle beside a single job: one long run (a 1,024-character context run, an
existence probe on a short wall-clock budget), a run whose memory lets fewer processes fit than
cores (klip's 5.3 GB), decoding latency (generation), and `litshe`'s 4 cores once it can hold all-core
load. D4's budget is what keeps the two kinds from oversubscribing each other.

**The machines.**

| machine | CPU | cores / threads | role here |
|---|---|---|---|
| jebel | i7-9700K (Coffee Lake) | 8 / 8, no SMT | the benchmark machine: every A/B, PL1 65 W |
| `litshe` | Pi 5, Cortex-A76 | 4 / 4 | aarch64 parity and timings; all-core work once the active cooler is fitted (stage 7) |
| stone | i7-1255U (Alder Lake, hybrid) | 2 P-cores with SMT + 8 E-cores, 12 threads | hybrid scheduling: static splits leave the fast cores idle |
| klip, pyramidon | Ryzen 7 3700U | 4 / 8, SMT | SMT: physical against logical cores; pyramidon development only |

## Stage −1: the outstanding kernel work, done first (D1)

Compiled from next-steps.md ("From the crate kernels", "From aarch64", "From the window program",
"From layer norm and attention", "From multi-head attention", "From generation", "From the sequence
task", "From the benchmark machine baseline", "From segments", "From the research separation"), the
i7 outline, the retired crate kernels plan, the context-length plan's leftovers, and the crate's
unmerged branches. Every item keeps the bits (blocking, tiling, copies, dispatch; never a new
summation order) and the golden run byte-identical.

**Classified:**

| # | item | source | verdict |
|---|---|---|---|
| K1 | `ab.py run --old-crate X --new-crate X` installs neither wheel, timing the venv's crate on both sides | From the A/B harness | **−1a**: a measurement fault; every stage after uses `ab.py` |
| K2 | x86 placement still moves rows through the uop cache: loops aligned to 32 bytes (`-align-loops=32`), timed as a full baseline | From the crate kernels; From the benchmark machine baseline (#609) | **−1a**: threading's A/Bs read few-percent rows, which placement now moves |
| K3 | `litshe`'s placement isn't pinned: a cheaper pin (loops at 32 bytes, functions only) is untried | From the crate kernels (D7) | **−1a**, on `litshe` |
| K4 | small unexplained rows: `learn linear k1 s16 m16 B64 W19` +4.2% since #1137; #1174's loader +2.3%; #1175's numpy causal forward +2.5% | research separation; context-length leftovers | **−1a**: reread after K2; closed as placement or named |
| K5 | the per-op baseline predates the crate's later work (conv per example, NEON, the vector `exp`, one codegen unit, tiled attention) | i7 outline ("re-measure a row before acting") | **−1b**: re-measure jebel's per-op table at main, one thread and default |
| K6 | `CpuInfo`: physical cores, L1/L2/L3, ISA, read once; constants derived from it | i7 outline, "Kernels and parameters from the machine" | **−1c** (D5): K7 and K8's slabs and stage 4's thread counts come from it |
| K7 | `matmul_2d` for few-row, long-`k` products: `k` slabs that resume each chain, so `b` is read once (conv tail `accumulate_gradient_batch` b512 3.93x numpy on one thread) | i7 item 1 | **−1d**, if K5 still shows the gap |
| K8 | `matmul_nt` with `W` past the L2: `k`-blocked so a slab of `W` stays in L2 (`forward_batch` 1.2-1.55x) | i7 item 2 | **−1e**, if K5 still shows it |
| K9 | conv on 28x28 inputs: which op lost ground (forward, downstream, accumulate at b1, b32, b512) | i7 item 5 | **−1f**: per-op diagnosis; a fix only with a cause |
| K10 | the narrow-`k` gap on `litshe`: 16-18% behind OpenBLAS on one thread, all of it backend stalls on serial FMA chains (`perf stat`, 2026-10-30); more independent outputs in flight per tile at narrow `k` | From the crate kernels | **−1g**, on `litshe` (single-core, needs no cooler) |
| K11 | an AVX2 transpose that doesn't lose to 4K aliasing: gather a block into a contiguous buffer before writing (the measured kernels on the crate's `avx2-transpose` branch) | From the crate kernels (stage 1, closed) | **−1h** |
| K12 | the attention kernels: the window program's attention window gains least (14% forward, 31% training), "a faster multi-head kernel"; the tiled attend timed per block at one thread | From the window program; From multi-head attention | **−1i**: profile first (`perf_region.py`), then the kernel; stage 5 threads it |
| K13 | a batched crate decode step: `attention_step` over an `(N, c)` cache with the keys' segments, one call a batch, not a row | From the sequence task (generation) | **−1j**: the decoder's batches are stage 5's natural split |
| K14 | conv's forward spread over examples | i7 item 3 | **threading**: stage 5 |
| K15 | threaded scaling against OpenBLAS, a pool, the threshold re-swept | i7 item 4; From the crate kernels | **threading**: stages 3-4 |
| K16 | threads in the window program, examples split inside one call | From the window program (D8 (b)) | **threading**: stage 5 |
| K17 | an SVE or AVX-512 `exp` and transpose | From the crate kernels; From aarch64 | **out**: no machine with them (the Pi 5's A76 has no SVE; no AVX-512 here) |
| K18 | a bits-changing kernel (fused or reassociated arithmetic) | From the window program (D3 (b)) | **out**: needs the owner's approval and a golden re-record; D2 keeps the bits |
| K19 | musl's golden run | From aarch64 | **out**: a check, not a kernel; if musl becomes a target |
| K20 | the numpy side: the attention backward's 10% (stage 3's blocks), the tiled numpy backward 43-47% slower at `T = 64` (#1171), numpy's cached decode step overhead-bound | From multi-head attention; context-length leftovers; From the sequence task | **out**: numpy expressions, not crate kernels; threading doesn't touch numpy. They stay in next-steps |
| K21 | the forward context's 1.1% on Rust's dense B=32 epoch | From segments | **out**: accepted (segments D10); revisited only if an A/B needs it |
| K22 | checkpointing the window program's forward | From the window program (D5 (c)) | **out**: memory, not speed; no study needs it |

**The sub-stages, in order.** Each is a crate PR (`indrajala-math-rust`), a "Bump rust/" PR here and
an A/B on jebel (and on `litshe` for aarch64 code), tier 2 for a claimed gain (measurement.md §9).
Each ends at the 5% bar: an item whose next step can't clear 5% in an epoch stops (the i7 outline's
rule).

- **−1a. Measurement first (K1-K4).** `ab.py` builds and installs a named crate on both sides or
  refuses (K1). x86 loops aligned to 32 bytes in `.cargo/config.toml`, timed as a full baseline on
  jebel, kept if it holds untouched rows still (`kernel_code_diff.py`) at an acceptable size cost
  (K2); the cheaper pin on `litshe` the same way (K3). Then K4's rows reread.
- **−1b. The baseline at main (K5).** jebel's per-op table (`op_call_timing.py`, `focused_benchmark.py`
  at one thread and default) and the conv and dense epochs, replacing
  [machine_profiles/i7-9700k.md](../../machine_profiles/i7-9700k.md)'s tables in one PR (next-steps,
  "Rerunning the baseline"). It decides which of K7-K9 are still gaps; a closed gap is recorded and
  dropped.
- **−1c. `CpuInfo` (K6).** Read once (Linux sysfs and `cpuid` on x86, sysfs on aarch64), with the
  machine profiles' fields; today's constants reproduced on jebel and the laptop wherever the
  benchmark machine plan found them best (the outline's rule). Tier 1: no slower.
- **−1d, −1e, −1f. The products and conv (K7-K9),** as the i7 outline describes, each only if −1b
  still shows its gap.
- **−1g. The narrow-`k` tile on aarch64 (K10).** `perf stat` before and after on `litshe`; the order
  of every chain unchanged, more outputs per tile.
- **−1h. The AVX2 transpose (K11).**
- **−1i. The attention kernel (K12).** `perf_region.py` on the tiled attend's forward and backward
  per block, then whatever it names.
- **−1j. The batched decode step (K13).**

## Decisions

- **D1. The kernels first. Settled: (a)** (the owner's direction).
  - (a) *Chosen.* Stage −1 before any threading. Pros: threading multiplies whatever a kernel costs,
    so its gains are read against kernels that no longer carry known gaps; the measurement faults
    (K1-K4) would otherwise blur every threading A/B. Cons: threading's own payoff waits.
  - (b) Interleave. Cons: a kernel change and a threading change in the same rows can't be told
    apart.
- **D2. Bits never depend on the thread count. Settled: (a).**
  - (a) *Chosen.* Partition outputs only: rows, column chunks, examples, (example, head) pairs, query
    blocks. No reduction is split across threads. Pros: the golden run holds at every thread count
    and on every machine, as now; tests check bits at 1, 2, 3 and the default threads. Cons: a
    reduction over a long axis (the batch's sum in a weight gradient) can't be split, so its product
    threads over its outputs instead, as `matmul_long_k` does.
  - (b) A fixed reduction tree (chunks of a fixed size, partial sums combined in a fixed order),
    independent of the thread count. Pros: splits reductions too. Cons: new bits everywhere it
    applies, a golden re-record (the owner's approval), and today's partitions already cover the
    products.
- **D3. The pool. Settled: (a).**
  - (a) *Chosen.* A std-only pool in the core: workers started on first use, parked between calls
    (a short bounded spin, then sleep, so it doesn't steal cores from numpy as OpenBLAS's spin steals
    them from us); a call hands it one job and its chunk count; workers claim chunks from an atomic
    counter (dynamic, so stone's E-cores and P-cores even out); the caller works too; a worker's
    panic is carried to the caller; the pool re-created when the process id changes (a fork), and a
    call from inside a worker runs inline (no nesting). Pros: the core keeps no dependencies (its
    crates.io promise of the same bits everywhere stays small to audit); control over spinning, fork
    and chunking, the three things that bite here; scoped borrows as today. Cons: about 200-300
    lines of concurrency to get right; tested under `miri` and a stress test, and `loom` if its
    model fits.
  - (b) rayon. Pros: mature, work-stealing, scoped. Cons: the core's first dependency; its global
    pool hangs in a forked child (a known hazard with Python's `fork`; forkserver lowers it, since
    the server process needn't have imported the crate, but nothing enforces that); its spin policy
    isn't ours.
  - (c) Spawn per call, the threshold re-tuned. Cons: the spawn cost is what stage 3 measured; no
    gain at b32.
- **D4. One thread budget a process. Settled: (a).**
  - (a) *Chosen.* `INDRAJALA_THREADS` (read once) and `set_threads(n)` (Python-callable), the crate's
    ceiling; default the physical cores (D5), not `available_parallelism`'s logical ones. The batch
    runner and the studies' sweeps set 1 a worker unless a job declares threads, and refuse when
    workers x threads exceeds a host's cores. Pros: no oversubscription when 8 study processes each
    open a pool; one rule for every caller. Cons: every launcher has to set it (the runner and the
    sweep helpers are few, and a check in them catches the rest).
  - (b) Default 1, opt in. Pros: safe. Cons: a user of the library gets no threads unasked; the
    production target is the other way.
- **D5. The machine read once. Settled: (a)** (the i7 outline's mechanism).
  - (a) *Chosen.* `CpuInfo` in the core, every size and count derived from it, each override hook
    kept; the threading threshold stays measured per machine, since it depends on clock ramp and the
    pool's cost, not cache sizes. Pros: one codebase, no branches per chip. Cons: sysfs parsing to
    test on four machines.
- **D6. Epochs decide. Settled: (a).**
  - (a) *Chosen.* A threading change is judged on training epochs and study runs at the thread
    count training uses, beside the per-op rows; per-op rows say where, epochs say whether. Pros:
    measurement.md's lesson (isolated loops flatter threading; cold cores). Cons: longer A/Bs.
- **D7. One plan, sub-stages. Settled: (a).**
  - (a) *Chosen.* Stage −1's items as sub-stages of this plan, each its own PRs and A/B. Pros: one
    place for the order and the dependencies; the outline's "a workplan each" would make ten small
    plans. Cons: a long plan; each sub-stage still writes its findings here.
- **Settled by the owner (2026-11-02):**
  - **O1. Where this sits in the roadmap. Settled: (a)** ("agreed"). The primitives roadmap is at step 12 (finishing) and
    step 13 (FFN activations, stage 3 next), with 14 and 15 after; both want jebel's A/Bs. Options:
    (a) after step 13 and before 14 and 15; (b) after step 15; (c) stage −1 now, between step 12's
    end and step 13's stage 3. I'd take (a): step 13 is already under way and its crate change
    (an activation-parameterised fused family) is better done before −1i touches the same code's
    neighbours; 14 (grouped-query attention) and 15 then build on a threaded attention.
  - **O2. `litshe`'s limits with the active cooler. Settled:** "we base the profile on the measured
    temperature response, when the cooler is available". Until then the rule stands (65 C ideal,
    70 C hard, single-threaded passes; the owner, 2026-10-18), and stage 7 sets the profile from
    the measured curve.

## Stages

In order: stage 0, then −1, then 1 to 8.

- **Stage 0. This workplan.**
- **Stage −1. The kernel work** (above), −1a to −1j.
- **Stage 1. Thread measurement tools** (a durable tool, built here): `thread_cost_probe.py` extended to a
   pool (call cost at 1-8 threads, cold and warm), a scaling sweep per op (threads x shape),
   oversubscription (N processes x T threads on one host), and an epoch timer at a set thread
   count through `ab.py` (`--rust-threads` exists). Run on jebel, stone and klip; `litshe` after
   stage 7's soak.
- **Stage 2. `CpuInfo`'s thread fields and the budget (D4, D5):** physical cores, `INDRAJALA_THREADS`,
   `set_threads`; the runner and sweep helpers set 1 a worker and refuse oversubscription.
- **Stage 3. The pool (D3)** in the core: its tests (every chunk run once, panic carried, fork recreates,
   nesting inline, `miri`, a stress test at random chunk counts); no caller yet. Tier 0.
- **Stage 4. The existing threaded sites on the pool:** `for_each_row_range`, `matmul_long_k`'s column
   chunks, conv's downstream; bits at 1, 2, 3 and the default threads; the threshold re-swept per
   machine (it may fall far below 8M with no spawn cost), on epochs (D6). Tier 2 on jebel.
- **Stage 5. New threaded sites,** each its own PR and A/B, in order of measured share: conv's forward over
   examples (K14); the tiled attend over (example, head) and query blocks; the window program over
   examples (K16); the batched decode step over rows (after −1j); row-wise ops only where a
   profile shows them worth a call's cost.
- **Stage 6. numpy beside the crate:** whether `OPENBLAS_NUM_THREADS` should follow the crate's budget in
   mixed runs (the spin gotcha), measured on the conv demo; documented either way.
- **Stage 7. `litshe` all-core, with the active cooler (O2):** its temperature response measured
   (temperature against time at 1, 2 and 4 busy cores, each to its plateau or to the hard limit,
   sampled each second under the thermal guard at 70 C); `litshe`'s profile set from that curve
   (how many cores, for how long, at what clock) and recorded in its machine profile and memory;
   then its scaling, its share of the memory bus (the b512 result) and the threshold for aarch64.
- **Stage 8. Docs:** the README's threading section (the budget, the bits rule, the pool), measurement.md
   (threaded A/Bs, the tools), next-steps' leftovers, the workplan retired.

## Checks and machines

- Every stage keeps the golden run byte-identical, on x86_64 and aarch64 (the committed Rust
  reference on both CI runners); threaded tests run at several thread counts.
- A/Bs on jebel under the machine lock (one at a time with the studies); aarch64 code on `litshe`,
  within its thermal limits; stone and klip for scaling and oversubscription readings, not for timing
  claims (their noise isn't characterised).
- Nothing runs on pyramidon beyond development and unit tests.

## Not in this plan

- GPU backends (excluded, for now).
- Changing any summation order (K18, D2 (b)).
- Threads in numpy or pure Python (numpy's own BLAS threads are stage 6's reading only).
- SVE and AVX-512 kernels (K17), until a machine with them is in hand.
