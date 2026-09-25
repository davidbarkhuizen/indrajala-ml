# Workplan: publish indrajala-math-rust to PyPI

**Status: planned. Nothing is built yet. Stage 0 comes first, and it needs the repo owner's PyPI
and GitHub accounts.**

The goal is to publish the Rust crate (the `rust/` submodule,
[indrajala-math-rust](https://github.com/davidbarkhuizen/indrajala-math-rust)) on PyPI as
`indrajala-math-rust`, imported as `indrajala_math_rust`. Anyone should be able to
`pip install indrajala-math-rust` on Linux, macOS or Windows and get a tested prebuilt wheel. No
Rust toolchain should be needed.

Every push and PR builds and tests on every platform we ship. A `v*` tag does the same and then
publishes. A wheel is published only if it was built and tested by the same workflow run on its
own platform.

## Why

- **Anyone can use it.** Today the crate installs only from a source checkout, with rustup and
  maturin. Many more people can run `pip install`.
- **Other platforms test what has only been assumed.** Only Linux x86_64 with AVX2 has ever built
  or run the crate. On aarch64 the scalar matmul fallback is the only path, so the ~1,560 op tests
  there run the fallback for real. Today it is tested only indirectly, through the parity claims
  about the AVX2 path. Windows and macOS find anything in the build or tests that assumes Linux.
- **Each decision is measured and recorded.** The pyo3 upgrade and the abi3 choice could both
  change per-call overhead, so each one is timed against the current build before it is adopted
  (the rules in [optimizations/measurement.md](optimizations/measurement.md)).

## Where things are now

- `rust/Cargo.toml`: version 0.1.0, `license = "MIT"`, `crate-type = ["cdylib"]` (a Python module
  only, so crates.io is out of scope), and `pyo3 = "0.20"` with `extension-module` and
  `multiple-pymethods`. `[lints.rust]` allows `non_local_definitions` only for pyo3 0.20's
  expansions.
- `rust/pyproject.toml`: name, version, `license = "MIT"`, `requires-python = ">=3.9"`. It has no
  description, authors, URLs or classifiers.
- pyo3 0.20 supports CPython up to 3.12. Today the latest pyo3 is 0.29.2 (2026-08-28) and the
  latest CPython is 3.14. Python 3.9 reached end of life in October 2025.
- Of the 9 source files, 2 use `#[pyclass]` (`Array`, `ConvGeometry`). The pyo3 API appears in
  all 9, about 176 lines, most of them in `fused.rs` and `lib.rs`.
- The AVX2/FMA code is gated with `#[cfg(target_arch = "x86_64")]` and picked at runtime with
  `is_x86_feature_detected!`. A portable x86_64 wheel still uses AVX2 where the CPU has it, and
  aarch64 compiles only the scalar path.
- `indrajala_math_rust.pyi` and `py.typed` are already in the wheel. stubtest and pyright already
  check the stub in CI.
- The crate CI (`rust/.github/workflows/ci.yml`) has one Linux job: rustfmt, clippy, ruff,
  `maturin develop --release`, stubtest, pyright and pytest, on Python 3.11.
- indrajala-ml builds the crate from the submodule with `./cli build-rust` and pins maturin 1.15.0
  there. That stays as it is (see Out of scope).

## Decisions

- **PyPI only, not crates.io.** A cdylib-only crate is of no use to another Rust crate.
- **Trusted publishing** (OIDC from GitHub Actions) with `pypa/gh-action-pypi-publish`, so there
  are no API tokens. It also generates PEP 740 attestations. Each index gets its own GitHub
  environment (`testpypi`, `pypi`), and `pypi` requires a reviewer.
- **The Cargo.toml version is the only version.** `rust/pyproject.toml` declares
  `dynamic = ["version"]`, and maturin takes the version from Cargo. The release job fails unless
  the tag equals `v` plus that version and the tagged commit is on `main`.
- **Tags:** `v<version>` publishes to PyPI. `testpypi-v<version>` publishes the same version to
  TestPyPI, as a rehearsal. A version can't be uploaded twice to the same index, even after
  deletion. A failed rehearsal means a version bump before trying again.
- **Never ship an untested wheel.** If a platform can't be tested on its own runner, it is not
  shipped.
- **CI minutes don't limit this plan.** The dev matrix runs in full on every push and PR.
- **indrajala-ml keeps building from the submodule.** Parity and timing work is tied to an exact
  crate commit. A PyPI release is a snapshot of that crate, not a dependency of this repo.

## Pitfalls to design around

- **Timing baselines.** Every Rust number in docs/optimizations/ comes from a local
  `maturin build --release` on the Ryzen machine, with pyo3 0.20 and without abi3. Stages 1 and 2
  change what the build does at the Python boundary. Stage 3 changes where the published wheel is
  built (a manylinux container). Each stage measures its own change. None assumes it is free.
- **The golden run is bit-exact.** The pyo3 upgrade and abi3 don't touch the kernels, so
  `scripts/golden_training_run.py check` must pass bit-identical. A difference means the change
  reached arithmetic, and it must be explained, not accepted within a tolerance.
- **Old against new timing:** commit each build first and alternate the builds, as
  [optimizations/measurement.md](optimizations/measurement.md)'s protocols say (a stash once made
  both builds `main`).
- **Relative links in the README break on PyPI.** The README maturin uploads needs absolute
  GitHub URLs.
- **The sdist must build.** It needs `rust-toolchain.toml`, `Cargo.lock`, the stub and LICENSE.
  Without `rust-toolchain.toml`, an sdist install silently uses whatever toolchain the user has.
- **The runner labels are GitHub's to change.** Before stage 3, check that each one exists: an
  Intel macOS runner, `windows-11-arm`, `ubuntu-24.04-arm`. A platform with no native runner is
  dropped under the untested-wheel rule.
- **pyo3 renames things between versions:** the Bound API (0.21), `IntoPyObject` (0.23), the
  removal of the GIL-refs API, and `allow_threads`/`with_gil` renamed to `detach`/`attach`.
  Follow each version's migration guide in order. Don't jump straight to 0.29 and fix the
  compile errors blind.

## Stages

Each stage is one PR or more in the crate repo, followed by a submodule bump PR in indrajala-ml
with `./cli build-rust && ./cli test` green. The crate's CI and indrajala-ml's CI both pass
before merge.

### Stage 0: accounts (owner, no code)

1. On pypi.org and test.pypi.org, add a **pending trusted publisher** for the project
   `indrajala-math-rust`: owner `davidbarkhuizen`, repo `indrajala-math-rust`, workflow
   `release.yml`, environment `pypi` (TestPyPI: `testpypi`). A pending publisher lets the first
   upload create the project, so the name is claimed on the first release.
2. In the crate repo's GitHub settings, create the `pypi` and `testpypi` environments. Put a
   required reviewer on `pypi` and restrict it to `v*` tags. Restrict `testpypi` to
   `testpypi-v*` tags.
3. Enable 2FA on both PyPI accounts if it isn't on already. PyPI requires it.

Done when both pending publishers are listed and both environments exist.

### Stage 1: upgrade pyo3 from 0.20 to 0.29

The latest Python an unupgraded crate can build for is 3.12. Upgrade in steps, one commit per
step on one branch, with the crate tests green at each step:

1. 0.20 → 0.21: the Bound API. Move `&PyAny`/`&PyList`/`&PyTuple` arguments and returns to
   `Bound<'py, T>`, keeping the `gil-refs` feature only for as long as the step needs it.
2. 0.21 → 0.23: `IntoPyObject` replaces `IntoPy`/`ToPyObject`. Drop `gil-refs`.
3. 0.23 → 0.29, one minor version at a time or in larger jumps where the guide shows nothing
   relevant: the `allow_threads` → `detach` rename, `#[pyclass]` changes, and whatever else each
   guide lists for the APIs this crate uses.
4. Remove the `non_local_definitions` allow from `[lints.rust]` if clippy stays clean without it.
5. Raise `requires-python` to `>=3.10`, or to pyo3 0.29's minimum if that is higher. This matches
   indrajala-ml, and 3.9 is past end of life.

Gates:

- The crate: clippy `-D warnings`, the 1,560 tests, stubtest and pyright. The stub shouldn't
  change, and if stubtest reports a difference, the Python API changed by accident.
- indrajala-ml: `./cli test`, and `golden_training_run.py check` against a golden file recorded
  on `main` before the branch. It must be bit-identical.
- Timing, old build against new, alternated and committed first:
  `focused_benchmark.py --backend rust` over the single-example dense ops and the smallest shapes
  (the Python boundary is the largest share of the call there), and `epoch_op_profile.py` for
  conv. Record the result in `docs/optimizations/implemented.md` either way: a speedup, a null or
  a regression. A regression above noise on a hot per-call path blocks the merge until it is
  explained. Rust was adopted unconditionally, so it doesn't reverse that decision.

Done when both repos are on pyo3 0.29, the golden run is bit-identical, and the timing is recorded.

### Stage 2: abi3 or one wheel per interpreter, measured

With `abi3-py310`, one wheel per platform covers every CPython from 3.10 on, including versions
released later. Without it, each release needs a wheel per interpreter (3.10-3.14, about 5 per
platform), and a new CPython needs a new release. abi3 restricts pyo3 to the stable C API, and
some of its fast paths (such as unchecked tuple and list access) then go through slower calls.

1. On a branch, add `abi3-py310` to pyo3's features. Build it and check that everything still
   compiles. Two pyclasses and no buffer protocol are expected to be fine.
2. Time abi3 against non-abi3 with the stage 1 protocol and cases.
3. Decide from the measurement. Adopt abi3 if nothing on a hot path regresses above noise, and
   use one wheel per interpreter otherwise. Record the decision and its numbers in
   `implemented.md` or `rejected.md`.

This decision also affects indrajala-ml's own build, since `./cli build-rust` builds whatever
Cargo.toml says. So the choice holds everywhere, not only for the published wheels.

Done when the decision is merged and recorded with its numbers.

### Stage 3: multi-platform build and test on every push and PR

Restructure the crate CI into a reusable workflow (`build-test.yml`, `on: workflow_call`), called
by `ci.yml` (push and PR to `main`) and later by `release.yml` (tags).

1. **Lint** (one Linux job, as now): rustfmt, clippy, ruff, pyright.
2. **Build** (matrix, `PyO3/maturin-action` pinned to the release that ships maturin 1.15.0, or
   maturin pinned inside it; match `./cli`'s `maturin_version`). The toolchain comes from
   `rust-toolchain.toml`. Each job uploads its wheels as an artifact.

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

   Plus an **sdist** job (`maturin sdist`). manylinux_2_28 rather than 2014: if numpy's own wheels for current Pythons need
   glibc 2.28, a lower tag gains no user who could also install numpy for the tests. Check
   numpy's tags when the stage is built, and pick the lowest tag both support.
3. **Test** (matrix: every platform × Python 3.10 and 3.14, or × every supported interpreter if
   stage 2 chose per-interpreter wheels). This runs on the platform's own runner, or in an Alpine
   container for musl. Download the wheel, install it into a clean venv with `pytest numpy mypy`,
   and run `pytest tests/` and stubtest from outside the checkout. This is the step that runs the
   aarch64 scalar fallback against the numpy references.
4. **sdist test** (Linux): `pip install` the sdist with rustup present, then run the tests. This
   proves the sdist builds with the pinned toolchain.
5. **A package check**: `twine check --strict` over every wheel and the sdist. Also check that the
   stub, `py.typed` and LICENSE are in each wheel.

Also, before merging, and recorded in the PR:

- Fix whatever Windows and macOS find (paths, `/tmp` in tests, line endings, `getrusage`). Each fix
  goes in its own commit so the PR shows what broke where.
- The CI wall time before and after (median of three runs: runner timings vary 2-3x),
  and each job's time, for the README.
- **The manylinux wheel against the local build**, on the Ryzen machine: install the CI-built
  `manylinux_2_28` x86_64 wheel and time it against `./cli build-rust`'s wheel from the same
  commit, with the stage 1 protocol. If the published wheel is slower, anyone quoting timings
  needs to know. It should be the same code under the same flags, but it needs measuring.

Done when every push and PR runs the full matrix green, and the wheel comparison is recorded.

### Stage 4: package metadata and the PyPI README

1. `rust/pyproject.toml`: `dynamic = ["version"]`, `description`, `readme = "README.md"`,
   `authors`, `keywords`, `[project.urls]` (Homepage, Repository, Issues), classifiers
   (Development Status :: 3 - Alpha, Programming Language :: Rust, Python :: 3 :: Only, one per
   supported version, Operating System per shipped platform, Typing :: Typed,
   Topic :: Scientific/Engineering :: Artificial Intelligence). Leave out the License
   classifier: the SPDX `license` field replaces it under PEP 639.
2. `rust/Cargo.toml`: `description`, `repository`, `readme`, `rust-version = "1.98"`. This is
   metadata only, since nothing publishes to crates.io.
3. The README, for someone arriving from PyPI: what the package is (a numpy-like `Array` and fused
   layer ops, written as indrajala-ml's backend), `pip install indrajala-math-rust`, a
   five-line usage example, the supported platforms and Pythons, and that the API follows
   indrajala-ml's needs with no stability promise before 1.0. Keep the build, test, layout and
   release sections for contributors. Make every link absolute.
4. Run a `maturin sdist` and list its contents to check that the files above are included.
   Build one wheel locally and read its `METADATA`.

Done when `twine check --strict` passes and the README renders on TestPyPI in stage 6.

### Stage 5: the release workflow

`release.yml`, `on: push: tags: ['v*', 'testpypi-v*']`:

1. **Verify:** the tag's version (after `v` or `testpypi-v`) equals Cargo.toml's version, and the
   tagged commit is an ancestor of `origin/main`. Fail before building otherwise.
2. **Build and test:** call `build-test.yml`, the same matrix every PR runs.
3. **Publish** (needs 2): download all artifacts into `dist/` and run
   `pypa/gh-action-pypi-publish`. Use environment `testpypi` with the TestPyPI repository URL for
   `testpypi-v*`, and environment `pypi` for `v*`. `permissions: id-token: write` only on this
   job.
4. **Verify the published package** (needs 3, a matrix over every platform and both Pythons):
   `pip install indrajala-math-rust==<version>` from the index it was published to (with
   `--index-url`/`--extra-index-url` for TestPyPI, since numpy isn't there), retrying for up to
   a few minutes while the index updates, then run the tests. This checks what users get, not
   only what was built.
5. **GitHub Release** (`v*` only, needs 4): create the release with generated notes and attach
   the wheels and sdist.

Also: a `Releasing` section in the crate README: bump the version in Cargo.toml by PR, merge,
tag `testpypi-v<version>` on `main`, check it, then tag `v<version>` and approve the `pypi`
environment.

Done when the workflow is merged. It is exercised in stage 6.

### Stage 6: the first release, 0.1.0

1. Tag `testpypi-v0.1.0`. Check the whole run, the TestPyPI page (the README renders, the
   metadata and classifiers are right, one file per platform) and the verify matrix.
2. If anything is wrong, fix it by PR, bump to 0.1.1 (TestPyPI won't take 0.1.0 again), and
   rehearse again.
3. Tag `v0.1.0` (or the rehearsed version), approve `pypi`, and check the PyPI page, the verify
   matrix and the GitHub Release.
4. Install from PyPI on this machine into a fresh venv and run indrajala-ml's `tests/` against the
   PyPI wheel instead of the submodule build (`./cli test` after installing the wheel over the
   submodule's). This is the published wheel checked against the fused-op parity tests, which live
   only in indrajala-ml.
5. indrajala-ml: add a README line that the crate is on PyPI, and bump the submodule.

Done when `pip install indrajala-math-rust` works on every shipped platform and passes the tests.

## After this plan

- **Free-threaded wheels** (3.13t/3.14t): pyo3 supports them with `#[pymodule(gil_used = false)]`,
  but abi3 doesn't cover them. So they are extra wheels, and `Array`'s interior mutability needs
  review for thread safety first.
- A **benchmark wheel matrix**: time the published wheels on each platform's runner. Only for a
  clear question, since GitHub runner timings are noise-dominated.
- Later releases follow the Releasing section and need no plan.

## Out of scope

- crates.io (a cdylib is of no use there).
- Linux i686, armv7, ppc64le and s390x, and any platform without a native GitHub runner to test on.
- Making indrajala-ml depend on the PyPI package, or publishing indrajala-ml itself.
- Any kernel change. Stages 1-3 time the build changes but don't optimize.
- API stability promises before 1.0.
