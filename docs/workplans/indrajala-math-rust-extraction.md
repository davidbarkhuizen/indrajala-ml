# Workplan: extract the Rust crate into its own repo (`indrajala-math-rust`)

## Context

`rust/indrajala_ml_array/` is a self-contained PyO3/maturin extension (1,816 lines of Rust
across 8 source files) that provides a numpy-like `Array` type and a set of fused layer ops
used as the production backend for indrajala-ml's `*RustArray*` model classes. It currently
lives in-tree, built locally by `./cli build-rust` (`maturin develop --release`), and its own
pytest suite is run together with indrajala-ml's suite by `./cli test`.

The goal is to split this Rust math library into its own GitHub repo, `indrajala-math-rust`,
and have indrajala-ml consume it as an external dependency, while keeping the exact same local
dev loop (`./cli build-rust` / `./cli test`) working end to end.

Research turned up one fact that shapes the whole plan: of the 21 test files under
`rust/indrajala_ml_array/tests/`, **8 import indrajala-ml's own pure-Python reference classes**
(`AdamArrayLayer`, `L2ArrayLayer`, `DropoutArrayLayer`, `ArrayLayer`, `ReLUArrayLayer`,
`MomentumArrayLayer`, `mnist_data.load_mnist_dataset_as_array`) to parity-check the Rust
implementation against them. Those 8 files are integration tests of indrajala-ml's own model
classes, not unit tests of the math library — they belong in indrajala-ml, not in the extracted
repo. The other 13 files only need numpy and are genuinely self-contained.

Decisions:
- **Integration mechanism**: git submodule (mounted at `rust/` in indrajala-ml), not a pip/git
  dependency or a published package. Keeps `maturin develop --release` working in place with
  minimal `./cli` changes and no registry/versioning infra.
- **Parity-test split**: the 8 cross-repo files move into `indrajala-ml/tests/`; the extracted
  repo's own suite stays numpy-only.
- **Naming**: rename the crate/module `indrajala_ml_array` → `indrajala_math_rust` as part of
  the move (crate name, `#[pymodule] fn`, Python import name). The exposed `Array` class name
  is unaffected (`#[pyclass(name = "Array")]` in `array.rs` — independent of the module name).

Other relevant findings:
- No native `cargo test` suite exists — all coverage is through the pytest/PyO3 boundary.
- No `LICENSE`, no `rust-toolchain` pin, no README mention of Rust at all currently exist.
- 33 Python files in indrajala-ml (`indrajala_ml/**` + `tests/**`) import `indrajala_ml_array`
  directly and need the import renamed.
- `gh` is authenticated as `davidbarkhuizen` (repo scope); current repo is public, ssh protocol.

## Stage 1 — Create the new repo and extract history

1. `gh repo create davidbarkhuizen/indrajala-math-rust --public --description "..."` with no
   auto-init (no README/gitignore/license), so a history push lands cleanly on `main`.
2. In a scratch clone of indrajala-ml (not the working tree), run
   `git subtree split --prefix=rust/indrajala_ml_array -b extract-math-rust` to produce a branch
   containing only that subtree's history, then push it to the new repo's `main`.
3. Clone `indrajala-math-rust` locally for the next stage.

## Stage 2 — Rename and prune inside the new repo

All in a single commit on the new repo:
- `Cargo.toml`: package/lib name `indrajala_ml_array` → `indrajala_math_rust`.
- `src/lib.rs`: `#[pymodule] fn indrajala_ml_array(...)` → `fn indrajala_math_rust(...)`.
- Regenerate `Cargo.lock` (`cargo generate-lockfile`, or let the next `maturin develop` do it).
- Add a crate-root `pyproject.toml` with the maturin build backend, `name = "indrajala-math-rust"`,
  `module-name = "indrajala_math_rust"` — needed for this to build as a standalone package.
- Delete the 8 cross-repo parity test files (`test_l2_fused_layer_ops.py`,
  `test_adam_fused_layer_ops.py`, `test_dropout_fused_layer_ops.py`, `test_fused_layer_ops.py`,
  `test_relu_fused_layer_ops.py`, `test_momentum_fused_layer_ops.py`, `test_mnist_decode.py`,
  `test_numerical_parity.py` — the last one imports `indrajala_ml.mnist_data` at module level for
  its MNIST section, so the whole file moves rather than being split).
- In the remaining 13 test files, update `from indrajala_ml_array import ...` →
  `from indrajala_math_rust import ...`.
- Add a repo-local `.gitignore` (`target/`, `__pycache__/`, `*.so`, `dist/`, `build/`) — it no
  longer inherits indrajala-ml's.
- Add a minimal `README.md`: what the crate is, the PyO3/maturin toolchain, how to build
  (`maturin develop --release`) and test (`pytest tests/`), and a note that it's consumed by
  indrajala-ml as a git submodule.
- Add `.github/workflows/ci.yml`: checkout, setup-python, `pip install maturin pytest numpy`,
  `maturin develop --release`, `pytest tests/`. Optionally add a `cargo build --release` /
  `cargo check` step as a cheap compile-sanity check (no `cargo test` step needed — no native
  Rust tests exist).

## Stage 3 — Verify the new repo stands alone

Fresh clone, fresh venv, `pip install maturin pytest numpy`, `maturin develop --release`,
`pytest tests/` — must be fully green with zero reference to indrajala-ml. This is the proof
that the split boundary was drawn correctly.

## Stage 4 — Cut indrajala-ml over (one PR, atomic)

This has to land as one PR: removing the old tree without the submodule in place (or vice versa)
leaves `main` unbuildable.

- `git rm -r rust/indrajala_ml_array`, then
  `git submodule add git@github.com:davidbarkhuizen/indrajala-math-rust.git rust` (mounted
  directly at `rust/`, since the nested `indrajala_ml_array/` level no longer serves a purpose
  with only one crate).
- `./cli`: change `rust_crate_root="rust/indrajala_ml_array"` → `rust_crate_root="rust"`; add
  `git submodule update --init --recursive` to `setup()` so a fresh clone pulls submodule content.
- Rename all 33 import sites: `import indrajala_ml_array` / `from indrajala_ml_array import ...`
  → `indrajala_math_rust`. Mechanical, one pattern — representative files:
  `indrajala_ml/model/rust_array_layer.py`, `indrajala_ml/model/model_io.py`,
  `indrajala_ml/ensemble_train.py`, `tests/test_rust_array_backprop_model.py`. Do with
  `grep -rl` + `sed -i`, then re-grep to confirm zero remaining references.
- Move the 8 parity test files (content from Stage 1's pre-prune copy, with their
  `indrajala_ml_array` imports renamed to `indrajala_math_rust`) into indrajala-ml's top-level
  `tests/`. No filename collisions with existing files there (checked).
- `.github/workflows/ci.yml`: add `submodules: recursive` to the `actions/checkout@v4` step.
- `README.md`: add a short note that cloning needs `git clone --recurse-submodules` (or
  `git submodule update --init` after a plain clone), linking to indrajala-math-rust, and one
  line on how to bump the pinned submodule commit later (`git submodule update --remote rust`).

## Stage 5 — Full end-to-end verification

Fresh clone of indrajala-ml with `--recurse-submodules`, run `./cli setup && ./cli test`, confirm
all tests pass including the relocated 8 parity tests, then open the PR and confirm CI is green.

## Verification checklist

- [ ] `indrajala-math-rust` repo: fresh clone → build → `pytest tests/` green, no indrajala-ml import anywhere in it.
- [ ] `indrajala-ml` repo: fresh clone with submodules → `./cli setup` → `./cli test` green.
- [ ] `grep -r indrajala_ml_array indrajala_ml/ tests/` returns nothing.
- [ ] CI green on both repos.
