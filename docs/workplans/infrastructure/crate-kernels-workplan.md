# Workplan: the crate's kernel loose ends (from the aarch64 plan)

**Status: D1-D5 settled (2026-10-29) for extensibility, speed and production; no stage started.
`litshe` is powered off: the owner powers it on before stage 2.**

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

### Stage 2: conv `accumulate_gradient` (D5)

The probe on `litshe` as D5 says; its findings recorded here; a fix, if the cause is real and
fixable, as stage 2b in its own crate PR (tier 2 on `litshe`).

Done when the findings merge.

### Stage 3: the narrow-`k` gap: threads first (D3)

1. A probe of a threaded call's cost on both machines (a repo tool): an empty scoped spawn of
   1-3 threads, and the target product at 1, 2 and 4 threads (`--rust-threads`), against numpy at
   the same counts (`--openblas-threads`).
2. The findings recorded here, as the multithreading work's input (D3): the spawn cost per call,
   the products it keeps single-threaded, what a pool would gain on each.
3. The single-thread 18% recorded as found (one-thread `perf stat` if `litshe` has it); a kernel
   change for it only with a measured cause, in its own PR.

Done when the findings merge.

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
