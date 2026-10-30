# Workplan: publish indrajala-math-rust on PyPI and crates.io

**Status: planned (2026-10-30). The owner asked for one plan covering both indexes before the PyPI
work starts. It replaces the PyPI-only plan (`pypi-release-workplan.md`, reopened the same day for
0.1.0), which was folded in here. The first release is 0.1.0 on both indexes. Stages run 1 to 7 in
order. Stage 0, the owner's accounts, is done just before stage 7. The retired aarch64 workplan
already made the crate's CI test glibc and musl on x86_64 and aarch64 (dev builds, not wheels).**

The crate repo ([indrajala-math-rust](https://github.com/davidbarkhuizen/indrajala-math-rust), the
`rust/` submodule) ships two packages from one Cargo workspace:

- **PyPI: `indrajala-math-rust`**, imported as `indrajala_math_rust`, as today. Anyone can run
  `pip install indrajala-math-rust` on Linux, macOS or Windows and get a tested prebuilt wheel, with
  no Rust toolchain needed.
- **crates.io: `indrajala-math`**, a pure-Rust library crate (`rlib`, no pyo3) holding the array
  type and the kernels. A Rust user adds `indrajala-math = "0.1"`. The Python bindings become a thin
  crate over it.

Every push and PR builds and tests on every platform we ship. A `v*` tag does the same, then
publishes both packages. Nothing is published unless the same workflow run built and tested it on
its own platform.

## Why

- **Anyone can use it.** Today the crate installs only from a source checkout, with rustup and
  maturin.
- **Rust users can use the kernels directly.** Today the kernels are reachable only through Python:
  pyo3 types are woven through 17 of the 19 source files.
- **The split is good structure in its own right.** Arithmetic separated from the Python boundary
  can be tested with `cargo test`, documented on docs.rs, and reused by later bindings (the
  CLAUDE.md rule: structure first).
- **Other platforms test what has only been assumed.** macOS and Windows have never built the crate.
- **Each change is measured and recorded.** abi3, the split, and the manylinux build could each
  change per-call cost, so each is timed against the current build before it is adopted
  ([measurement.md](../../measurement.md)).

## Where things are now (crate 4770503)

- `rust/Cargo.toml`: one package `indrajala_math_rust` 0.1.0, MIT, `crate-type = ["cdylib"]`,
  `pyo3 = "0.29"` (`extension-module`, `multiple-pymethods`), and `codegen-units = 1` in the release
  profile. The comment there records why: inlining in this crate is sensitive. crate #65 left a kernel
  as a call (conv +4-11% on `litshe`), and its fix took a `Zip` out of a pool kernel (+54%). The
  split has to respect this, since a call across crates is not inlined unless the callee is
  `#[inline]` or LTO is on.
- 10,114 lines in 19 files. Only `cr_exp.rs` and `scalar.rs` are free of pyo3. The other 17 mix
  kernels with `#[pyclass]`/`#[pymethods]`/`Bound<...>` code. There are 6 pyclasses (`Array`,
  `ConvGeometry`, `SeedSequence`, `Generator`, `RecurrenceStep`, `WindowProgram`). Every function
  borrows its arrays (`&RustArray`), so no call copies an array at the boundary.
- `unsafe` appears 42 times: 29 in `linalg.rs` and 9 in `cr_exp.rs` (SIMD intrinsics), 2 each in
  `array.rs` and `random.rs`. A public safe Rust API must not let a caller reach undefined behaviour
  through any of them.
- AVX2/FMA is gated with `#[cfg(target_arch = "x86_64")]` and picked at runtime with
  `is_x86_feature_detected!`, falling back to scalar code. aarch64 has NEON kernels, bit-identical
  to the x86_64 paths.
- The tests are Python only: about 3,000 op tests against numpy, plus stubtest and pyright on
  `indrajala_math_rust.pyi`. The crate has no `cargo test` suite. indrajala-ml holds the
  fused-op parity tests and the golden run.
- `rust/pyproject.toml` has a name, version, `license = "MIT"` and `requires-python = ">=3.14"`, but
  no description, authors, URLs or classifiers. The version is written in both files.
  `rust-toolchain.toml` pins 1.98.1.
- The crate CI has a glibc job (rustfmt, clippy, ruff, `maturin develop --release`, stubtest,
  pyright, pytest) and a musl job (an Alpine container), each on `ubuntu-latest` and
  `ubuntu-24.04-arm`. No wheels are built, and nothing runs on macOS or Windows.
- indrajala-ml builds the crate from the submodule (`./cli build-rust`, maturin 1.15.0, run in
  `rust/`). The import name `indrajala_math_rust` and the module's API must not change for it.
- On 2026-10-30, every candidate name returned 404 on both indexes (it is free):
  `indrajala-math-rust`, `indrajala-math` and `indrajala` on PyPI; `indrajala-math`,
  `indrajala-math-rust` and `indrajala` on crates.io. A name is claimed only by its first upload.

## Decisions

Settled by Claude (the owner delegates workplan choices). Each records the options considered.

- **What is published (D1): the crate, on both indexes; indrajala-ml itself is not.**
  - *The crate on both (chosen).* Pro: it is the reusable piece. Con: crates.io first needs the
    split (D2).
  - *indrajala-ml on PyPI too.* Pro: the whole library becomes installable. Con: the research still
    lives in it (the studies-repo workplan moves it out first), and its API is unsettled. It is
    left for a later plan, after the studies move.
- **Workspace layout (D2): one repo, a Cargo workspace with two members.**
  `indrajala-math` (`core/`, `rlib`, published) holds the array type, the kernels, the RNG and the
  window program, with no pyo3. `indrajala-math-py` (`python/`, `cdylib`, `publish = false`) holds
  only the pyclasses: newtypes over the core's types, argument conversion and error mapping. maturin
  builds `python/`, and `rust/pyproject.toml` stays at the root with `manifest-path`, so
  `./cli build-rust` keeps working from `rust/`.
  - *Two repos.* Con: every kernel change would need two PRs and a pin.
  - *One crate with a `python` cargo feature.* Pro: smallest change. Con: the published crate would
    still carry pyo3 code under `cfg`, and the boundary stays mixed into every file.
  - *The workspace (chosen).* Pro: a clean boundary, one PR per change. Con: the largest move.
- **Names (D3):** crates.io `indrajala-math` (the `-rust` suffix says nothing on crates.io). PyPI
  `indrajala-math-rust` (unchanged: the suffix tells Python users what is inside, and indrajala-ml's
  import stays). The repo keeps its name.
- **One version for both, from the workspace (D4).** `[workspace.package] version` is the only
  version. maturin reads it (`dynamic = ["version"]` in pyproject), and one `v<version>` tag
  publishes both.
  - *Separate versions.* Pro: a Python-only fix doesn't bump the Rust crate. Con: two tag schemes,
    and a wheel's version would no longer name the kernels it contains.
  - *One version (chosen).* Pro: one number names the code everywhere. Con: an occasional empty
    release on one index.
- **Order (D5): the split before the first release of either package.**
  - *PyPI first (the old plan), split later.* Pro: wheels sooner. Con: the CI would be restructured
    twice, the manylinux wheel timed twice, and the published wheel would be re-architected right
    after 0.1.0.
  - *Split first (chosen).* Pro: one CI layout, one timing of the final build, and the first release
    is the structure we keep. Con: the first wheel waits for the split. No user is waiting on 0.1.0.
  - Stage 2a is a feasibility probe. If the split can't keep the kernels' speed, crates.io drops out
    (recorded here) and the PyPI stages run unchanged on today's single crate.
- **Cross-crate inlining (D6): decided by the stage 2a measurement, not assumed.** The candidates:
  `lto = "fat"` in the workspace release profile (closest to today's one-crate build), or `#[inline]`
  on the core's hot entry points. Either way, the published `rlib` is built the way a Rust user's
  build will build it, and that is the build the A/B times.
- **Testing (D7): the Python op suite stays the numerical reference; the core gets its own
  `cargo test`.** The 3,000 numpy tests run through the bindings, which call the core, so they test
  the core's arithmetic. The core adds tests of its contract (shapes, errors, edge cases), doctests on
  every public item, and a small bit-fixture test per kernel family. The fixtures are the expected
  bits, written by the Python suite, so `cargo test` alone catches a moved bit on each platform.
  - *Port the suite to Rust.* Con: two copies of 3,000 tests, kept in step by hand.
  - *Only the Python suite.* Con: the crates.io package would ship untested by its own tooling.
- **The public Rust API (D8): 0.x, explicit, documented.** `#![warn(missing_docs)]`, a deliberate
  `pub` surface (internals `pub(crate)`), and errors as a `Result` with the crate's own error type,
  never a panic on bad input. The Python bindings map that error to the same Python exceptions as
  today. There are no stability promises before 1.0, stated in both READMEs. ndarray interop is not
  in 0.1.0.
- **Trusted publishing on both indexes (D9).** PyPI and TestPyPI use OIDC with
  `pypa/gh-action-pypi-publish`, which also generates PEP 740 attestations. crates.io uses its
  trusted publishing (`rust-lang/crates-io-auth-action`). crates.io's trusted publishing is
  configured on an existing crate, so the first publish (0.1.0) uses a short-lived, scoped API token
  held in the `crates-io` environment's secret, which is deleted afterwards. Re-check both indexes'
  current docs at stage 0. Each index has its own GitHub environment (`testpypi`, `pypi`,
  `crates-io`), and `pypi` and `crates-io` require the owner as reviewer.
- **Tags and rehearsal (D10).** `v<version>` publishes to PyPI and crates.io. `testpypi-v<version>`
  rehearses PyPI only. crates.io has no test index, so every PR runs `cargo publish --dry-run`
  (packaging plus a build from the packaged sources). The release job fails unless the tag equals
  the workspace version and the tagged commit is on `main`. Neither index takes a version twice, even
  after a yank or delete, so a failed rehearsal means a version bump.
- **Publish order within a release (D11): crates.io, then PyPI**, both only after the whole test
  matrix passes. Each publish job is idempotent: it skips a version the index already has. A failure
  after the first index succeeded is fixed by re-running the failed job, never by retagging.
- **Never ship an untested artefact; CI minutes don't limit this plan.** indrajala-ml keeps building
  from the submodule: its parity and timing work is tied to exact crate commits, and a release is a
  snapshot.

## Pitfalls to design around

- **The golden run is bit-exact.** abi3, the split and LTO move no arithmetic (Rust doesn't contract
  to FMA unasked), so `golden_training_run.py check` must pass bit-identical at every stage. A
  difference means the change reached arithmetic, and it is explained, not accepted within a
  tolerance.
- **Timing.** The baselines come from a local `maturin build --release` on `jebel`, without abi3
  ([machine_profiles/i7-9700k.md](../../machine_profiles/i7-9700k.md)). Stage 1 changes the Python
  boundary, stage 2 changes where code lives across crates, and stage 4 changes where the published
  wheel is built. Each is measured with `ab.py`, both builds committed and alternated. `litshe`'s
  rows go through `kernel_code_diff.py` (measurement.md §7: placement moves untouched code).
- **The split must not copy at the boundary.** The pyclasses wrap the core's types by value
  (newtypes) and pass borrows through, as today.
- **`unsafe` behind a public API.** Every `pub` function that reaches an intrinsic checks its
  preconditions (lengths, alignment, CPU features) itself, or is marked `unsafe` with a `# Safety`
  section. Stage 2c audits all 42 uses, and `cargo miri test` runs on the scalar paths.
- **The sdist must build:** it needs `rust-toolchain.toml`, `Cargo.lock`, the whole workspace, the
  stub and LICENSE. The `.crate` package needs README and LICENSE inside `core/`, and no
  path-dependency without a version.
- **Relative links break on PyPI and crates.io.** Both READMEs use absolute GitHub URLs.
- **The runner labels are GitHub's to change.** Before stage 4, check that each exists (an Intel
  macOS runner, `windows-11-arm`, `ubuntu-24.04-arm`). A platform with no native runner is dropped.

## Stages

Each stage is one or more PRs in the crate repo, each followed by a "Bump rust/" PR here with
`./cli build-rust && ./cli test` and the golden run green.

### Stage 0: accounts (the owner, no code; done just before stage 7)

1. **PyPI and TestPyPI:** register at <https://pypi.org/account/register/> and
   <https://test.pypi.org/account/register/> (the same username on both is simplest), and verify
   each email.
2. **2FA on both** (PyPI requires it to publish): a TOTP app or a security key. Save the recovery
   codes offline.
3. **Pending trusted publishers** for `indrajala-math-rust` on each (Your projects, Publishing, "Add
   a new pending publisher", GitHub): owner `davidbarkhuizen`, repository `indrajala-math-rust`,
   workflow `release.yml`, environment `pypi` (TestPyPI: `testpypi`).
4. **crates.io:** log in with GitHub at <https://crates.io> (it has no separate account), and
   verify an email in Account Settings (publishing requires one). Create an API token scoped to
   `publish-new` for the crate `indrajala-math`, with a short expiry, for the first publish (D9).
5. **The crate repo's environments:** `pypi` (owner as reviewer, `v*` tags only), `testpypi`
   (`testpypi-v*` only), `crates-io` (owner as reviewer, `v*` only; the token as its secret
   `CARGO_REGISTRY_TOKEN`).

Steps 1-4 are the owner's own. Claude gives step 5's settings or does it with `gh api`, except the
secret, which the owner enters (the classifier refuses secrets).

### Stage 1: abi3 or one wheel per interpreter, measured

With `abi3-py314`, one wheel per platform covers every CPython from 3.14 on. abi3 restricts pyo3 to
the stable C API, so some fast paths become slower calls.

1. On a branch, add `abi3-py314`, then build and run the tests.
2. Time abi3 against non-abi3: `focused_benchmark.py --backend rust` on the single-example dense ops
   at the smallest shapes (where the boundary is the largest share of a call), and
   `epoch_op_profile.py` for conv.
3. Adopt abi3 if nothing on a hot path regresses above noise, and use one wheel per interpreter
   otherwise. Record the decision and its numbers here.

It comes first because it touches only the boundary, which the split then rebuilds. Done when the
decision is merged and recorded.

### Stage 2: the split into `indrajala-math` and `indrajala-math-py`

**2a. Feasibility probe (one PR, or none if it fails).** Make the workspace with the dense path
only: `Array`, `linalg.rs` and `ufuncs.rs` move into `core/`, with their pyclasses over them in
`python/`. This covers most of the `unsafe` and the hottest kernels. Then check:
- the golden run, bit-identical;
- tier 2 A/B on `jebel` (the full protocol) with the dense benchmarks, under each D6 candidate;
- one pass on `litshe` with `kernel_code_diff.py`.

Adopt the candidate that loses nothing above noise, and record D6's answer and the numbers. If none
does, stop the split (D5's fallback).

**2b. The rest, one PR per module group:**
- conv, pooling and batch norm;
- attention and layer norm;
- the fused ops and recurrence;
- the RNG (`random.rs`, `generator.rs`, `cr_exp.rs`);
- the window program, factored state, tokens and mnist.

Each PR passes the golden run and a tier 1 A/B on the benchmark its group feeds (measurement.md §1's
table). The Python API and stub don't change at all: stubtest, pyright, the op suite and
indrajala-ml's tests must pass unchanged.

**2c. The Rust API:**
- the deliberate `pub` surface and the error type (D8);
- docs on every public item and a crate-level example;
- the `unsafe` audit, with `# Safety` notes and miri on the scalar paths;
- D7's contract tests and bit fixtures;
- `cargo publish --dry-run -p indrajala-math` and `cargo doc` with `-D warnings`.

The core gets its own README for crates.io.

Done when the bindings hold no arithmetic and the core builds, tests and documents without pyo3.

### Stage 3: package metadata and both READMEs

1. `[workspace.package]`: version, edition, license, repository, `rust-version = "1.98"`. The core
   adds `description`, `readme`, `keywords`, `categories` (`science`, `mathematics`) and
   `documentation` (docs.rs).
2. `rust/pyproject.toml`: `dynamic = ["version"]`, `description`, `readme`, `authors`,
   `keywords`, `[project.urls]`, and classifiers (Alpha, Rust, Python 3 only, one per supported
   version, each shipped OS, Typing :: Typed, Topic :: Scientific/Engineering :: Artificial
   Intelligence). There is no License classifier: PEP 639's SPDX field replaces it.
3. The PyPI README: what the package is, `pip install`, a five-line example, the platforms and
   Pythons, and no stability promise before 1.0. The crates.io README: the same for Rust, with a
   `Cargo.toml` line and a Rust example. The contributor sections (build, test, layout, releasing)
   stay in the repo README. Every link is absolute.
4. List the sdist's and the `.crate`'s contents (`maturin sdist`, `cargo package --list`), and
   read one wheel's `METADATA`.

### Stage 4: multi-platform build and test on every push and PR

A reusable workflow (`build-test.yml`, `on: workflow_call`), called by `ci.yml` (push and PR) and
later by `release.yml`.

1. **Lint** (one Linux job): rustfmt, clippy (workspace, all targets), ruff, pyright.
2. **Rust** (a matrix over the platforms below, without containers except musl):
   - `cargo test -p indrajala-math` (doctests and fixtures);
   - an MSRV build at `rust-version`;
   - on Linux only: `cargo doc -D warnings`, `cargo publish --dry-run`, and miri.
3. **Wheels** (`PyO3/maturin-action`, maturin pinned to `./cli`'s `maturin_version`). Each job
   uploads its wheels.

   | platform | runner | wheel tag |
   | --- | --- | --- |
   | Linux x86_64 | `ubuntu-latest` | manylinux_2_28 (container) |
   | Linux aarch64 | `ubuntu-24.04-arm` | manylinux_2_28 (container) |
   | Linux x86_64, musl | `ubuntu-latest` | musllinux_1_2 |
   | Linux aarch64, musl | `ubuntu-24.04-arm` | musllinux_1_2 |
   | macOS arm64 | `macos-latest` | macosx_11_0_arm64 |
   | macOS x86_64 | an Intel macOS runner, if GitHub still offers one | macosx_10_12_x86_64 |
   | Windows x86_64 | `windows-latest` | win_amd64 |
   | Windows arm64 | `windows-11-arm` | win_arm64 |

   There is also an sdist job. The lowest manylinux tag that numpy's current wheels also support is
   checked when the stage is built.
4. **Wheel tests** on each platform's own runner (musl in Alpine), for Python 3.14 and the newest
   CPython, or every supported interpreter if stage 1 chose per-interpreter wheels. Each job installs
   the wheel into a clean venv and runs `pytest tests/` and stubtest from outside the checkout.
5. **The sdist test** (Linux): `pip install` the sdist with rustup present, then run the tests.
6. **The package check:** `twine check --strict` on every wheel and the sdist, and a check that the
   stub, `py.typed` and LICENSE are in each wheel.

Also before merging, recorded in the PR:
- what Windows and macOS broke, one commit per fix;
- the CI wall time before and after, as the median of three runs;
- **the manylinux wheel against the local build** on `jebel`, with the stage 1 protocol.

### Stage 5: the release workflow

`release.yml`, `on: push: tags: ['v*', 'testpypi-v*']`:

1. **Verify:** the tag matches the workspace version, and the tagged commit is on `origin/main`.
2. **Build and test:** call `build-test.yml`.
3. **Publish to crates.io** (`v*` only, environment `crates-io`, D11): `cargo publish -p
   indrajala-math`, skipped if the version already exists.
4. **Publish to PyPI** (needs 2, and 3 on `v*`): `pypa/gh-action-pypi-publish` to TestPyPI or PyPI by
   tag, with `id-token: write` on this job only.
5. **Verify what users get:**
   - a pip-install matrix from the index it was published to, retrying while the index updates,
     then the tests;
   - on `v*`, a scratch cargo project depending on `indrajala-math = "=<version>"` from crates.io,
     running the crate-level example.
6. **GitHub Release** (`v*`): generated notes, with the wheels, the sdist and the `.crate`.

Also, a `Releasing` section in the crate README: bump the workspace version by PR, merge, tag
`testpypi-v<version>`, check it, tag `v<version>`, and approve `crates-io` and `pypi`.

### Stage 6: switch crates.io to trusted publishing (after stage 7's first publish)

Once `indrajala-math` exists on crates.io, configure its trusted publisher (repository, `release.yml`,
environment `crates-io`), switch the publish job to `rust-lang/crates-io-auth-action`, and the owner
deletes the token and the secret. This is a small PR, done right after 0.1.0 and listed here so the
token never outlives the first release.

### Stage 7: the first release, 0.1.0

1. Stage 0.
2. Tag `testpypi-v0.1.0`. Check the run, the TestPyPI page (README, metadata, one file per platform)
   and the verify matrix. If anything is wrong, fix it by PR, bump to 0.1.1, and rehearse again.
3. Tag `v0.1.0` (or the rehearsed version), and approve `crates-io` then `pypi`. Check the crates.io
   page, the docs.rs build, the PyPI page, both verify jobs, and the GitHub Release.
4. Stage 6.
5. Install from PyPI into a fresh venv here and run indrajala-ml's `tests/` against the PyPI wheel
   (the fused-op parity tests live only in indrajala-ml).
6. indrajala-ml: a README line on both packages, and the submodule bump. Then retire this workplan,
   with its leftovers to next-steps.md.

Done when `pip install indrajala-math-rust` and `cargo add indrajala-math` both work and their tests
pass on every shipped platform.

## After this plan

- **Free-threaded wheels** (3.14t): `gil_used = false` and extra wheels (abi3 doesn't cover them),
  after a thread-safety review of `Array`'s interior mutability.
- **ndarray interop** in the core (`From` conversions behind a feature), if a Rust user asks.
- **indrajala-ml on PyPI** (D1), after the studies-repo move.
- `cargo semver-checks` in CI once a second release exists.
- A benchmark wheel matrix on each platform's runner, only for a clear question.
- Later releases follow the Releasing section and need no plan.

## Out of scope

- Linux i686, armv7, ppc64le and s390x, and any platform without a native GitHub runner to test on.
- indrajala-ml depending on the published packages.
- Any kernel change: the stages move and time code, they don't optimise it.
- API stability promises before 1.0, on either index.
