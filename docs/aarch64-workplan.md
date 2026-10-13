# Workplan: aarch64 (a Raspberry Pi 5), and one set of bits on every machine

**Status: D1-D3 settled by the owner (2026-10-13); D4-D9 settled (2026-10-13) for extensibility,
speed and production; D10 settled by the owner (2026-10-13): the Pi is `litshe`, on wired Ethernet; its active
cooler and NVMe drive are on the way. Not started. The demo split
([demos-split-workplan.md](demos-split-workplan.md)) is done, so the Pi needs no matplotlib.**

indrajala-ml and the crate run on aarch64, tested and timed on a Raspberry Pi 5 (8 GB). The crate
stays **bit-identical across architectures**: a Rust-backend network trains to the same bits on
jebel, pyramidon and the Pi. numpy is not held to that. Then the crate's aarch64 kernels are made
to compete with numpy on the Pi, keeping those bits.

## Why

- **Only x86_64 has ever built or run the crate.** On aarch64 every kernel takes the scalar
  fallback, which no machine has run for real.
- **Most production arm64 is glibc Linux** (Ubuntu, Debian, Graviton and Ampere servers), and
  Apple silicon is the same instruction set. A library that only one architecture has run isn't
  a production library.
- **Reproducibility across machines is a production property.** The crate is nearly there by
  construction (below): making it a checked claim costs little and turns the golden run's Rust
  entries into a reference any machine can check against, not a per-machine file.

## Where the bits stand (read from the code, 2026-10-13)

- **Products already follow one order on every path.** `rust/src/linalg.rs`: four accumulator
  lanes, each updated by a fused multiply-add (`mul_add`), combined as `(l0 + l1) + (l2 + l3)`
  (`combine_lanes`), the tail added in order (`dot_product_tail`). The AVX2 kernels and the scalar
  fallback compute exactly that, by design and by the crate's tests. IEEE 754's `+ - * /`,
  `sqrt` and fused multiply-add are correctly rounded on x86_64 and aarch64 alike, and Rust
  never contracts `a * b + c` on its own, so the same order gives the same bits on both.
- **Threading doesn't change a product's bits**: rows or 16-wide column chunks are split between
  threads, never one sum (`matmul_long_k`'s comment; `set_matmul_threading` and
  `rust/tests/test_linalg.py`). The Pi has 4 cores, jebel 8.
- **`layer_norm.rs` and `batch_norm.rs` use only correctly rounded operations**, every sum a left
  fold.
- **Where the crate can still differ between machines:**
  1. **`exp`**: 6 call sites (`ufuncs.rs` 2, `recurrent.rs` 2, `fused.rs` 1, `batch_norm.rs` 1);
     sigmoid and tanh are built from it. `f64::exp` calls the C library's, which isn't
     correctly rounded: glibc on x86_64 picks an FMA-compiled build at run time, aarch64's is
     another build, musl another library. A rare last-bit difference is likely; one is enough
     to change a training run.
  2. **`powi`**: Adam's bias corrections (`fused.rs:330-331`) call LLVM's `powi`, whose
     evaluation order is the compiler's, not ours.
  3. **`max` with signed zeros and NaN**: relu's `v.max(0.0)` (`ufuncs.rs:49`, `fused.rs:527`,
     `fused.rs:537`, `conv.rs:244`, `batch_norm.rs:48`) and softmax's `fold(f64::NEG_INFINITY,
     f64::max)` (`ufuncs.rs:130`). Rust leaves which zero `max(-0.0, 0.0)` returns unspecified,
     and x86's `maxsd` and ARM's `fmaxnm` answer differently; NaN's sign differs too.
  4. **Future kernels**, if they change the order (wider lanes, across-lane sums): a rule (D5),
     not physics.
- **numpy differs between machines and isn't claimed**: its products are OpenBLAS's, whose
  kernels and summation order depend on the CPU model (jebel's Haswell kernel, the tests'
  Sandybridge notes, the Pi's ARM kernels), and GCC contracts multiply-adds in numpy's own C loops
  on aarch64 where baseline x86_64 can't. Even on x86 the tests compare numpy's products with the
  crate's "within rounding", not by bits.
- **The golden run**: 124 networks, 41 Rust, 41 numpy, 42 pure Python
  (`data/refactoring/golden_run.json`, per machine, untracked).

## Decisions

Every option's pros and cons, as settled.

- **D1: the crate is bit-identical across architectures; numpy is not** (a'). Settled by the
  owner (2026-10-13).
  - (a') The crate's results are the same bits on every machine: own `exp` (D4), explicit rules
    for `powi`, `max` and NaN (D6), kernels in the canonical order (D5). Pros: a Rust-backend
    run reproduces on any machine; the Pi's golden run checks against jebel's; mostly true
    already. Cons: an `exp` to build and prove; less freedom in a kernel's order; x86's golden
    run changes once (§8: more correct, better structured).
  - (a) Exact on x86_64, a tolerance elsewhere. Pros: kernels free to pick any order; no `exp`
    work. Cons: gives up a property the crate nearly has; every machine's runs differ; a bound to
    choose and defend per op.
  - (c) No cross-architecture parity. Pros: most freedom. Cons: as (a), and no check of ARM
    against x86 at all.
- **D2: Ubuntu 26.04 LTS arm64 on the Pi.** Settled by the owner (2026-10-13), to keep the
  variables to one.
  - Ubuntu 26.04. Pros: the same glibc, Python 3.14 and compilers as pyramidon and jebel, so a
    difference is the architecture's; what most production arm64 runs; the same scripts. Cons:
    snapd and unattended-upgrades to hold (as on jebel); heavier than Alpine.
  - Alpine 3.24 (Python 3.14.8). Pros: minimal and quiet; tests musl. Cons: two variables at
    once (the architecture and the C library); musl's `malloc` slows numpy, skewing the
    comparison in the crate's favour; none of the machine scripts carry over.
  - Raspberry Pi OS (Debian 13). Pros: the best hardware support. Cons: Python 3.13; Pi-only.
  - musl is tested instead in CI (stage 4).
- **D3: submodules, not PyPI, until 1.0.** Settled by the owner (2026-10-13);
  [pypi-release-workplan.md](pypi-release-workplan.md) is deferred, its stage 2 CI matrix taken
  up here (stage 4) without publishing.
- **D4: the crate's `exp` is correctly rounded.** Settled (a).
  - (a) Correctly rounded binary64 `exp` (a fast path good to well under half an ulp and a rare
    exact path, Ziv's method; CORE-MATH's published algorithm and proof are the model, its
    licence checked before any port). Pros: the result is defined by mathematics, not by our
    code, so any future implementation (a NEON or AVX-512 vector `exp`, another language, a GPU)
    can match it by construction and be tested against an exact reference; the most correct
    choice. Cons: the most work; the rare exact path is slow (its rate measured, expected
    around one call in thousands or fewer); a proof to follow and test.
  - (b) A fixed algorithm from correctly rounded operations only (a range reduction and a
    polynomial, as glibc's), deterministic but about 0.5 ulp. Pros: simpler, faster, no slow
    path. Cons: the definition is our code: a vector version must replicate it operation by
    operation, and "is it right" has no external answer.
  - (c) Keep the C library's. Pros: nothing to do. Cons: breaks D1.
  The exact reference for the tests is Python's integers (`fractions`, no new dependency):
  every test input's `exp` to more than enough bits, then rounded.
- **D5: the canonical order is a rule for every kernel**, written into the crate's README and
  each kernel's comment: products in four `mul_add` lanes combined `(l0 + l1) + (l2 + l3)`, tails
  in order; reductions as left folds; no operation whose result the hardware chooses. A NEON
  kernel holds the four lanes in two 2-wide registers. Settled. Pros: one definition for every
  architecture, already the code's; NEON's 32 registers leave room for the row tiling's
  independent chains. Cons: rules out a faster order a future kernel might want; the cost is
  measured in stage 7, and an order change would be a D1 revisit, never a silent one.
- **D6: `max`, relu, NaN and `powi` get explicit rules.** Settled.
  - relu is `if v > 0.0 { v } else if v.is_nan() { v } else { 0.0 }` (NaN propagates as numpy's
    `maximum` does; `-0.0` and `+0.0` give `+0.0`); softmax's row maximum is a left fold with the
    same comparison. The numpy and pure-Python layers adopt the same rule where their result
    could differ (only at signed zeros and NaN), checked against both machines' numpy first.
    Pros: deterministic everywhere, and the three implementations agree. Cons: a one-time
    change at signed zeros, which the golden run may show.
  - Adam's `beta^t` by square-and-multiply in a fixed order, written out; the numpy and Python
    optimizers likewise (they call the C library's `pow` today). Pros: deterministic. Cons: none
    found; `t` is small.
  - The alternative, leaving these to the hardware and compiler: Pros: nothing to do. Cons:
    breaks D1 on exactly the inputs that are hardest to debug.
- **D7: the Rust golden entries become one committed, machine-independent reference.** Settled.
  - (a) After stage 3, `data/refactoring/golden_run_rust.json` (41 entries, committed, a few MB
    at most: checked before committing), checked by `golden_training_run.py check` on every
    machine and in CI on x86_64 and aarch64 runners; numpy and Python entries stay per machine.
    Pros: the cross-machine claim is checked on every PR, not only when someone runs it on two
    machines. Cons: CI time for 41 trainings (measured; dropped to the smallest entries if it
    exceeds a few minutes); a re-record is a reviewed diff.
  - (b) Keep every entry per machine and compare files by hand. Pros: no new file. Cons: the
    claim is checked only when someone remembers to.
- **D8: a timing setup for the Pi in `ab.py`'s terms**: a fixed frequency policy (governor and
  maximum clock), the throttling state read before and after each pass (the Pi firmware's
  throttle flags, `vcgencmd get_throttled` or its sysfs equivalent) with a throttled pass
  refused, snaps and unattended-upgrades held, and a machine profile beside
  `docs/machine_profiles/i7-9700k.json`. No power cap: the Pi has no RAPL. Settled. Pros: the
  same protocol, so Pi timings are as trustworthy as jebel's. Cons: `machine_profile_capture.py`
  reads x86 sources (`intel_pstate`, RAPL): it grows an ARM branch selected by what the machine
  exposes, not by its name.
- **D9: the Pi is a timing machine like jebel**, with studies and development elsewhere.
  Settled. Pros: one role per machine; its timings stay clean. Cons: it can't take a study's
  overflow without breaking that (a later decision if needed).
- **D10: the machine.** Settled by the owner (2026-10-13): **`litshe`**, on wired Ethernet
  (steadier ssh, no radio interrupts while timing), with a fixed address or DHCP reservation
  (jebel's mDNS name sometimes fails to resolve). Storage: an NVMe drive (being set up), not an
  SD card. Cooling: an active cooler, to be bought; until it is fitted litshe runs in a passive
  case, which is enough for stages 1-4 (throttling slows a run but never changes its bits) and
  not for stage 5 onward. The cooler, the M.2 board and the case must fit together. An ssh key
  for the agent as on jebel.

## Stages

One PR per stage; a crate stage lands in `indrajala-math-rust` first, then a "Bump rust/" PR.

### Stage 0: the machine (owner)

Ubuntu Server 26.04 LTS arm64 on the Pi 5 from SSD or NVMe, the cooler fitted, the user and ssh
key, and the name D10 gives (`litshe`). The agent then checks `uptime`, `free -m`, `vcgencmd
get_throttled` (or sysfs) and the kernel over ssh.

### Stage 1: bring-up, and the measurement that decides how much stage 2 has to do

1. `./cli setup`, `./cli build-rust` (rustup and maturin as on jebel), the full `./cli test`.
   Record every failure; the exact parity tests are expected to pass (one machine, one C
   library, the scalar fallback).
2. Record the Pi's golden run; compare it with jebel's entry by entry
   (`golden_training_run.py compare A B`, a small new mode in this stage's PR: the first
   difference per entry, grouped by backend).
3. Per-op probes of the four sources above: `exp` over a large sample of inputs on both
   machines (bits compared), relu and softmax at `-0.0` and NaN, `powi` at Adam's `t` range.
4. The findings in this workplan: which Rust entries already match, and which source broke
   each one that doesn't.

Done when the findings are merged.

### Stage 2: the crate's own rules (crate PR, then the bump)

1. D4's `exp`, with tests against the exact reference: random inputs across the whole domain,
   the hard cases near overflow, underflow and halfway points, and the slow path's rate.
2. D6's rules in the crate; the numpy and Python layers' and optimizers' matching changes.
3. Timed on jebel (tier 1: the sigmoid, softmax and attention benchmarks, `--order ONNO`); a
   slower `exp` is weighed against D1, not hidden.
4. The golden run re-recorded on every machine (§8, approved by D1), archived.

Done when the Pi's Rust entries equal jebel's bit for bit.

### Stage 3: the committed Rust reference (D7)

`golden_run_rust.json`, its check mode, and CI's check on the x86_64 runner.

### Stage 4: CI on aarch64 and musl

From [pypi-release-workplan.md](pypi-release-workplan.md)'s stage 2, without publishing: the
crate's CI builds and tests on `ubuntu-24.04-arm` (glibc) and in Alpine containers on both
architectures (musl); indrajala-ml's CI runs its tests on `ubuntu-24.04-arm` and D7's check
there.

### Stage 5: the Pi's timing setup (D8)

`benchmark_machine_setup.sh` and `install_benchmark_setup.sh` for the Pi (or one script that reads
the machine), the machine profile, `ab.py remote --host litshe`, and an A/A run to measure the Pi's
noise, as stage 2 of the benchmark machine workplan did for jebel.

### Stage 6: the baseline, crate against numpy on the Pi

`focused_benchmark.py` with both backends over every op and shape, and the conv comparison end to
end, on the Pi and (for the same table) on jebel. The result is a table of crate-over-numpy
ratios per op on each machine, ordering stage 7's work. Expected from the code: matmul furthest
behind (scalar against OpenBLAS's NEON kernels); small ops, where the Python call dominates,
possibly ahead already.

### Stage 7 onward: NEON kernels, one per PR, in the baseline's order

Each under D5, behind `#[cfg(target_arch = "aarch64")]` (NEON is part of the aarch64 baseline,
so no run-time detection), with:
- the crate's tests and D7's reference passing bit for bit on the Pi;
- a tier 2 timing on the Pi (the full protocol); jebel's timing unaffected (tier 1, since x86 code
  doesn't change);
- where the canonical order costs speed against a freer one, the cost measured and recorded.

## After this plan

- Other aarch64 machines (Graviton in CI or cloud, Apple silicon) are checked by D7's reference
  alone.
- SVE (not on the Pi 5's Cortex-A76) and AVX-512 kernels follow D5.
- numpy's own cross-machine behaviour stays unclaimed; if it is ever wanted, it is a separate
  question about OpenBLAS.
