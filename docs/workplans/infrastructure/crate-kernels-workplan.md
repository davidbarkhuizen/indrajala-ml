# Workplan: the crate's kernel loose ends (from the aarch64 plan)

**Status: D1-D5 settled (2026-10-29) for extensibility, speed and production. Stage 1 closed (no
consistent gain on x86: 4K aliasing); stages 2-4 under way.**

The aarch64 plan (retired: `git show 6b9c6fe:docs/workplans/infrastructure/aarch64-workplan.md`)
made the crate bit-identical across x86_64 and aarch64 and gave the Pi its NEON kernels. It left
four speed questions, taken up here (the threading one measured, not built: D3): an AVX2 transpose, conv `accumulate_gradient`'s unexplained
slowdown, the narrow-`k` gap to numpy, and a vector `exp`. Every change keeps the bits (the aarch64
plan's D1 and D5): each is a copy, a reordering of work between threads, or a correctly rounded
function, never a new summation order.

## Where each stands (read from the code and the old plan's records, 2026-10-29)

1. **The transpose on x86.** `RustArray::transpose` (`rust/src/array.rs`) copies 8 x 8 blocks;
   aarch64 moves 2 x 2 sub-blocks through `vzip1q_f64`/`vzip2q_f64` (`transpose_block_neon`, crate
   #71: 37-58% faster on `litshe`), x86 still runs the scalar `transpose_block`. jebel's baseline
   put the crate's transpose at 1.87x numpy's (stage 6).
2. **Conv `accumulate_gradient`, +4-11%.** In the alignment probe (`2026-10-19-align-probe`, crate
   504a1f6 against b900778, every function aligned to 64 bytes) it stayed +4-11% slower, +1-8%
   unaligned, though the change timed (the conv downstream per example, crate #65) doesn't touch it:
   it runs `matmul_long_k` (`rust/src/linalg.rs`), the column-chunk threaded, `k`-slabbed product.
   The old plan left it "a question for the next kernel's A/B"; none came.
3. **The narrow-`k` gap.** Dense 30 x 784 `downstream_batch` b32 (`(32, 30) @ (30, 784)`, `k` 30)
   takes 127 µs on `litshe` against numpy's 58. That compares the crate on one thread with numpy on
   four: at one thread each, 128 against 108 µs (the old plan's stage 6), so the kernel's own gap is
   about 18%, the rest threads. The crate starts OS threads on every threaded call
   (`std::thread::scope` in `for_each_row_range`, `matmul_long_k`, conv's per-example downstream),
   where OpenBLAS keeps a pool: so the threading threshold sweep (8M flops down to 250k) found no
   threshold that gains, since a call's thread starts cost more than they save at these sizes.
   The tile's shape isn't it either (seven shapes timed, none wins).
4. **`exp`.** The crate's correctly rounded `exp` (`rust/src/cr_exp.rs`, CORE-MATH's `cr_exp`) runs
   one value at a time through `cr_exp::map`. It cost the forward passes that call it on jebel:
   dense `forward_batch` +17% to +32% (sigmoid), conv `forward` +9% to +20%, attention
   `forward_batch` +6.8% (`2026-10-17-cr-exp-focused`); accepted then as the price of one
   definition, "won back by a vector `exp`". Its call sites: `ufuncs::exp`, softmax's rows
   (`ufuncs.rs`), sigmoid (`fused.rs`), the recurrent state's soft codes (`recurrent.rs`), batch
   norm (`batch_norm.rs`). Its fast path rounds about 689 in 690 calls; the rest take the accurate
   path.

## Decisions

Every option's pros and cons, as settled.

- **D1: the vector `exp` runs CORE-MATH's fast path in lanes; a lane whose rounding test fails
  takes the scalar `exp`.** Settled (a).
  - (a) The fast path (range reduction, table lookup, polynomial, the `ub == lb` test) in NEON's 2
    lanes and AVX2's 4, then the scalar `exp_body` for each lane that failed. Pros: the same bits
    by construction, since both are correctly rounded (the reason for D4 then); tested against the
    scalar `exp` bit for bit, beside its exact-reference tests; the rare path stays one proved code.
    Cons: a table gather (two loads per lane on NEON, `_mm256_i64gather_pd` on AVX2); a branch per
    vector, taken about once in 345 NEON vectors.
  - (b) The whole vector redone in scalar when any lane fails. Pros: simpler. Cons: wastes the
    other lanes' work, 4x as often on AVX2; no simpler to prove.
  - (c) A new vector algorithm (a different reduction or polynomial). Pros: maybe faster. Cons: a
    new proof of correct rounding to write; (a) reuses CORE-MATH's.
- **D2: one slice kernel, `exp` over a slice into a slice; callers' arithmetic around it in their
  own passes.** Settled (a).
  - (a) `cr_exp::exp_into(&[f64], &mut [f64])`, the vector kernel behind one name, x86's AVX2 and
    FMA detected once per call as the matmul kernels do; softmax subtracts the row's maximum first,
    sigmoid computes `1 / (1 + e)` after, each a plain loop. Pros: one kernel to test; the
    surrounding `+ - /` are correctly rounded lane by lane, so the split can't change a bit; every
    call site moves to it. Cons: an extra pass over each buffer (in cache at these sizes).
  - (b) A generic `map` that vectorises its closure. Pros: one pass. Cons: Rust can't vectorise an
    arbitrary closure around an intrinsic kernel; each caller's kernel written out by hand.
- **D3: stage 3 measures the thread starts' share of the gap; a pool is left to the coming
  multithreading work.** Settled (a). The owner (2026-10-29): multithreading in general comes in
  the near future.
  - (a) Measure only: the cost of a threaded call's spawns on both machines, and how much of the
    gap it explains, recorded as that work's input. Pros: one threading design, made once with
    every kernel and the training loop in view; nothing built here that it would replace. Cons:
    the gap stays open until then.
  - (b) A std-only pool in the crate now (workers parked between calls, the existing partitions
    handed to it, so no output's value can move). Pros: every threaded kernel gains now. Cons: a
    second design beside the coming one, or a constraint on it; about a hundred lines of
    concurrency (shutdown, a worker's panic, nesting) likely redone.
  - (c) rayon now. Pros: mature. Cons: as (b), plus a dependency and its own global pool, chosen
    before the general design.
- **D4: the AVX2 transpose moves 4 x 4 sub-blocks** of the 8 x 8 blocks, through
  `_mm256_unpacklo_pd`/`_mm256_unpackhi_pd` and `_mm256_permute2f128_pd`, behind the AVX2
  detection the crate's other x86 kernels use; odd edges copied one value at a time, as NEON's.
  Settled. Pros: the NEON kernel's structure; a copy, so bit-identical. Cons: none found.
- **D5: conv `accumulate_gradient`'s slowdown is diagnosed before anything is changed.** Settled:
  first whether it persists at the current crate, aligned, against 504a1f6 (the probe's old
  side); if it does, a bisection between the two commits and `perf stat` counters on `litshe`
  (cycles, cache misses) if its kernel exposes them, else the probe's timings per `k` slab; if it
  doesn't persist, it is recorded as placement and closed. Pros: no change without a cause; the
  probe is cheap (single-core, minutes). Cons: may end with "not found", which is an answer.

- **D6: release builds compile in one codegen unit** (`[profile.release] codegen-units = 1`).
  Settled (a), 2026-10-29, from stage 2's findings.
  - (a) One unit. Pros: what LLVM inlines no longer depends on how rustc splits the crate, so a
    change in one file can't move another's code (stage 2's cause, and the 10-90% swings in
    untouched rows on jebel); no special case in the code; on `litshe` 38 rows faster, 3 slower.
    Cons: slower release builds; on jebel the old `cr_exp::map` path (sigmoid, softmax) +16-37%,
    a path stage 4 removes, so the two are timed together.
  - (b) A wrapper never inlined around `matmul_long_k`'s column range, on aarch64 only (crate #75).
    Pros: recovers the accumulate fully (aligned, every shape at #64's time); x86's code
    byte-identical. Cons: a special case, and the next change can move the units again: the
    wrapper itself took a `Zip` out of `max_pool_downstream_batch` (pool 6x6 +54%).
  - (c) Leave it. Pros: nothing to do. Cons: every crate change keeps moving unrelated kernels.

- **D7: a kernel's speed doesn't depend on where the linker puts it.** Settled (a), 2026-10-29
  (the owner chose to pin the build over shipping stage 4 as it stood, or stopping).
  - (a) The crate's `.cargo/config.toml`: on x86_64 no branch crosses or ends on a 32-byte boundary
    (`-x86-branches-within-32B-boundaries`, the mitigation for the Skylake-derived cores' JCC
    erratum: jebel is an i7-9700K); on aarch64 loops and functions aligned to 64 bytes. With D6,
    and checked by `scripts/measurement/kernel_code_diff.py`, which compares two builds' machine code
    function by function. Pros: an A/B moves only the rows whose code changed (on jebel,
    the vector `exp` against D6 and D7 alone: the `exp` rows faster, 263 within noise, none slower;
    on `litshe` 259 within noise, one +2.3%); layout only, so no bit moves. Cons: `.text` 2% larger
    on x86 (padding).
  - (b) Explicit inline attributes on each hot kernel. Pros: pins what is inlined. Cons: with D6
    the kernels' code already holds still (the tool: the vector `exp` changes only the functions
    it touches); the swings that remained came from placement, which attributes don't fix.
  - (c) Read every A/B through aligned controls. Pros: no build change. Cons: doubles every run, and
    function alignment alone left the x86 swings (stage 1's aligned run).

## Stages

One PR per stage; a crate stage lands in `indrajala-math-rust` first, then a "Bump rust/" PR. Every
crate change passes the crate's tests, the committed Rust reference
(`data/refactoring/golden_run_rust.json`) and the golden run on both machines, and its kernel's
bits are pinned by a test at shapes reaching every path (`docs/measurement.md`, §9). Timings on
`litshe` keep its thermal limits (65 C ideal, 70 C hard; single-threaded passes) and on `jebel`
its setup. Probes are kept as repo tools (`scripts/measurement/`).

### Stage 1: the AVX2 transpose (D4)

The kernel in the crate, its tests (every shape class: multiples of 8, odd edges, 1-row and
1-column), and the crate README's stale path to the old plan fixed in the same PR. Tier 2 on jebel
(`focused_benchmark`, the transpose and `accumulate_gradient_batch` rows the target); `litshe`
unaffected (x86 code only).

Done when the bump merges with its A/B.

**Closed 2026-10-29: no consistent gain; not merged** (crate #73, ml #1087, both left unmerged;
the kernels on the crate's `avx2-transpose` branch).

- **The kernel as D4 has it** (source quads outermost; crate 37da6f1). Tier 2 on jebel,
  `2026-10-29-bump-rust-avx2-transpose-focused_benchmark` (ONNONO):
  - transposes of 32 rows were 18-36% faster (`(32, 30).T` 0.83 to 0.56 µs), and dense and conv-tail
    `accumulate_gradient_batch` b32 6.6-8.1% faster;
  - but dense 10 x 30 `transpose` b512 (`(512, 10).T`) was +52% (4.0 to 6.1 µs), attention 16 x 32
    `accumulate_gradient_batch` b32 +22% (203 to 246 µs), and dense `accumulate_gradient_batch`
    b512 +4-5%.
- **Not placement.** The same A/B with every function aligned to 64 bytes
  (`2026-10-29-avx2-transpose-aligned`, crate tags `probe/align-10dfec9` and
  `probe/align-avx2-transpose`; the transpose, `accumulate_gradient_batch` and `sum_axis0` rows):
  `sum_axis0` flat, the transpose +49% at `(512, 10)` and attention's accumulate +13% (b32) and
  +4.8% (b512) still there. Column quads outermost (crate 3276b06,
  `2026-10-29-avx2-transpose-column-first`) changed nothing.
- **The cause: 4K aliasing.** Timed directly on jebel (each build's wheel, `Array.T`, best of 9),
  the AVX2 kernel loses only where the output's rows lie a multiple of 4 KB apart:
  `(512, 30)` 14.2 against 21.8 µs and `(1024, 10)` 8.1 against 12.2, but `(520, 30)` 11.6
  against 11.2 and `(504, 10)` level. Its loads interleave with stores to four output rows whose
  addresses share their low 12 bits, which an Intel core reads as a dependence.
- **A variant** (both row quads of a block loaded before either is stored, each output row's 8
  values stored together; crate 20c1d5d), timed the same way, interleaved over 3 rounds: against
  the scalar block, `(512, 10)` -8%, `(512, 32)` -8% and `(32, k)` -18% to -35%, but `(32, 784)` +8%,
  `(32, 5408)` +15% and `(520, 30)` +17%. Still not a gain on every shape.
- **What it would take:** a transpose that never interleaves its loads with stores to rows 4 KB
  apart, such as one that gathers a block into a contiguous buffer before writing it out. A design
  of its own, not D4's; it goes to next-steps.md when this plan retires. The NEON kernel is
  unaffected (`litshe`: every transpose 37-58% faster, crate #71).
- The crate README's stale path to the aarch64 plan, which stage 1's PR fixed, moves to stage 4's
  crate PR.

### Stage 2: conv `accumulate_gradient` (D5)

The probe on `litshe` as D5 says; its findings recorded here; a fix, if the cause is real and
fixable, as stage 2b in its own crate PR (tier 2 on `litshe`).

Done when the findings merge.

**Findings (2026-10-29): the cause is how rustc splits the crate; the fix is one codegen unit (D6).**

- **It persists.** `2026-10-29-align-probe-10dfec9` on `litshe`: 10dfec9 against 0a6910b (#64),
  both with every function aligned to 64 bytes (crate tags `probe/align-10dfec9`, the old probe's
  a722c7c): conv `accumulate_gradient` +4.2% to +11.2% on 5 of 6 shapes, numpy's rows flat. The
  old probe compared #64 with #65 directly, so there was nothing to bisect.
- **Not the source.** #65 left the accumulate's code alone; it removed `matmul_narrow`, another
  caller of the tiled kernel. The symbols show what changed: before, `matmul_long_k`'s column
  closure was a function of its own (1,488 bytes) with the kernel inlined into it; after, the closure
  is inlined into `matmul_long_k` (2,164 bytes) and the kernel (`tiled_row_range_neon::<2>`) is a
  call.
- **A wrapper restores it** (crate #75): the closure through a generic `#[inline(never)]` function on
  aarch64 compiles back to the 1,488-byte shape, and aligned (`2026-10-29-long-k-wrapper-aligned`)
  every shape is at #64's time (26x26x8 72.3 to 65.0 µs, #64 65.1). x86's `.text` stays
  byte-identical to 10dfec9's. But unaligned (`2026-10-29-long-k-wrapper-litshe`) pool 6x6
  `downstream_batch` b32 was +54%: `max_pool_downstream_batch`, its source untouched, now calls
  `Zip::new` instead of inlining it. **The mechanism:** rustc splits the crate into 16 codegen units,
  LLVM can't inline across them, and which unit a generic lands in moves with any change.
- **The same on jebel.** Every crate change in this plan moved untouched rows there: the transpose
  +60-94% (a named function in `linalg.rs`), conv `forward` +3-20% (the vector `exp`), pool
  -11% to -32%.
- **One codegen unit** (crate #76, `2026-10-29-codegen-units-1-litshe` and `-jebel`):
  - on `litshe`, 38 rows faster and 3 slower: conv `accumulate_gradient` -7% to -8% (the loss
    recovered) and dense `downstream`/`hidden_delta` -11% to -15%; pool 6x6 `downstream_batch` b32
    +21%; conv-tail `sgd step` +65%, an allocator mode, not code (the kernel's instructions are
    identical; the new side's loops range 116-285 µs and the fastest equals the old median);
  - on jebel, 20 faster and 11 slower: pool `downstream` -17% to -32%; dense `forward_batch` +35% and
    attention `forward_batch` +16%, the `cr_exp::map` path stage 4 replaces.

  So D6 (a), evaluated with stage 4, against 10dfec9 on both machines; #75 closes.

### Stage 3: the narrow-`k` gap: threads first (D3)

1. A probe of a threaded call's cost on both machines (a repo tool): an empty scoped spawn of
   1-3 threads, and the target product at 1, 2 and 4 threads (`--rust-threads`), against numpy at
   the same counts (`--openblas-threads`).
2. The findings recorded here, as the multithreading work's input (D3): the spawn cost per call,
   the products it keeps single-threaded, what a pool would gain on each.
3. The single-thread 18% recorded as found (one-thread `perf stat` if `litshe` has it); a kernel
   change for it only with a measured cause, in its own PR.

Done when the findings merge.

**Findings (2026-10-29), for the multithreading work (D3).** `scripts/measurement/thread_cost_probe.py`
(`scoped_spawn_cost.rs` beside it: `std::thread::scope` spawning and joining N threads that do no
work, as the crate's threaded calls do), on each machine's main crate (10dfec9), µs per call; the
outputs in `pyramidon:~/data/indrajala-ml/runs/probes/2026-10-29-thread-cost/`.

| | jebel | `litshe` |
| --- | --- | --- |
| an empty scoped spawn, 1 / 2 / 3 / 4 threads | 20.5 / 29.9 / 42.2 / 51.8 | 28.5 / 46.9 / 67.1 / 86.3 |
| `(32, 30) @ (30, 784)`, crate at 1 / 2 / 4 threads | 31.6 / 48.7 / 60.2 | 126.7 / 107.3 / 152.3 |
| the same, numpy at 1 / 2 / 4 OpenBLAS threads | 41.0 / 23.0 / 22.9 | 109.4 / 57.5 / 56.9 |
| `(512, 30) @ (30, 784)`, crate at 1 / 2 / 4 threads | 478 / 388 / 277 | 2,451 / 2,473 / 2,169 |
| the same, numpy at 1 / 2 / 4 threads | 499 / 290 / 168 | 2,073 / 2,685 / 3,463 |

- **A thread costs about 10 µs on jebel and 19 µs on `litshe`, plus a first one's 20-28.** The
  crate spawns on every threaded call, so at b32 the threads cost more than the product (jebel: 31.6
  µs on one thread, 60.2 on four). That is why the threading threshold stays at 8M flops (the
  aarch64 plan's sweep found nothing lower that gains).
- **A pool would gain:** at b32, the product's own share split (jebel about 8 µs at four threads
  against numpy's 23; `litshe` about 32 against 57) instead of a spawn per call; at b512, the spawns'
  50-90 µs on calls of 0.3-2.5 ms, a few percent. Products below the threshold (every dense layer
  at b32) only thread with a pool.
- **One thread, the kernel against OpenBLAS:** the crate is faster on jebel (31.6 against 41.0 µs;
  478 against 499 at b512) and 16-18% slower on `litshe` (126.7 against 109.4; 2,451 against 2,073),
  the gap the old plan measured. Its cause is not measured: `perf` isn't usable on `litshe`
  (`perf_event_paranoid` 4, and raising it is the owner's step), so it stays a question for the
  multithreading work, which will time the kernels again.
- **`litshe`'s memory bandwidth bounds b512:** the crate gains 11% from four threads, and OpenBLAS
  loses 67% (2,073 to 3,463 µs): four cores contend for one memory bus.

### Stage 4: the vector `exp` (D1, D2)

1. `exp_into` with NEON and AVX2 fast paths; tested bit for bit against the scalar `exp` on the
   inputs of `cr_exp`'s own tests (every binade, CORE-MATH's hard cases and their neighbours, the
   overflow and underflow edges, NaN and the infinities) and on 2^24 random inputs per machine,
   and the failing lanes counted.
2. Every call site moved to it (D2), one crate PR.
3. Tier 2 on both machines (`focused_benchmark`'s sigmoid, softmax and attention rows, and the
   window program's `learn`): against the scalar `exp`, and against `2026-10-17-cr-exp-focused`'s
   old side, to read how much of the C library's speed is back.

Done when the bump merges with both machines' A/Bs.

## After this plan

- An SVE or AVX-512 `exp` and transpose follow D1 and D4 when a machine with them is in hand.
- The multithreading work takes stage 3's findings (D3); a pool there opens the threading
  threshold to smaller products, re-swept on each machine.
