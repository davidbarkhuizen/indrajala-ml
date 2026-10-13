# Workplan: move the demos to their own repository

**Status: decisions D1-D6 settled (2026-10-13); D7 (the new repository's visibility and who
creates it) waits for the owner before stage 2. Stage 1 not started.**

The demos, the matplotlib plotting and the tkinter digit capture move to a new repository,
`indrajala-demos`, which uses indrajala-ml as a git submodule. indrajala-ml keeps the library,
the studies, the measurement tools and the tests, and loses its only dependency on matplotlib and
on `python3-tk`.

## Why

- **A smaller library to port.** The aarch64 machine (a Raspberry Pi 5 on Ubuntu 26.04;
  the aarch64 workplan) needs only the library. Every runtime dependency
  left after the split (numpy, pyarrow, jsonschema, pytest) has a wheel for Python 3.14 on Linux
  aarch64, glibc and musl alike; matplotlib has none for musl. No machine should need a GUI stack
  to run the tests.
- **The library's dependencies are the library's.** matplotlib and tkinter serve the 28 demo,
  plotting and capture modules that move, and nothing else in the package. Anyone using indrajala-ml as a
  library installs them today for nothing.
- **Demos depend on the library, never the reverse.** A separate repository makes that a fact
  the build enforces, not a convention.

## What exists

| piece | where | uses | goes |
| --- | --- | --- | --- |
| 19 demos, the capture app and the menu | `indrajala_ml/demos/` (`linear`, `backprop`, `uci_digits`, `mnist`, `capture_app.py`, `menu.py`, `registry.py`) | matplotlib, tkinter | to `indrajala-demos` |
| plotting | `indrajala_ml/graphics/` (`chart.py`, `classifier_plots.py`, `evaluation_plots.py`) | matplotlib | to `indrajala-demos` |
| digit capture | `indrajala_ml/capture/` (`capture_common.py`, `digit_capture.py`, `mnist_capture.py`) | tkinter | to `indrajala-demos` |
| per-op timing | `demos/benchmarks/demo_layer_op_timing.py` | numpy, the crate | stays: `focused_benchmark.py` imports its cases; measurement.md §5 lists it |
| batch-size scaling | `demos/benchmarks/demo_batch_size_scaling.py` | `studies.batch_size_scaling` | stays, beside the study |
| the conv comparison | `demos/conv/demo_conv_rust_vs_vectorized_digit_recognition.py` | numpy, the crate | stays: `epoch_op_profile.py`, `op_call_timing.py`, `accuracy_pass_timing.py` and `batch_size_timing.py` import its `ARCHITECTURES`, `TRAINERS`, `load_datasets`, `train_once` and op names; measurement.md §5 lists it |
| timing helpers | `demos/timing.py` (`timed_call`, `timed_train`) | the conv comparison, 4 digit demos | stays (with the conv comparison); the demos import it from its new place |
| conv depth sweep | `demos/conv/demo_conv_depth_uci_digits_comparison.py` | the Python networks | to `indrajala-demos` |
| tests | `tests/graphics/` (2), `tests/capture/` (2), `tests/demos/conv/test_demo_conv_depth_uci_digits_comparison.py`, `tests/training/test_training_pipeline.py` (plots a trained classifier) | matplotlib, tkinter | to `indrajala-demos` |
| tests | `tests/demos/benchmarks/` (2), `tests/demos/conv/test_demo_conv_rust_vs_vectorized_digit_recognition.py` | | stay, beside their modules |
| `./cli demo`, `os_packages="python3-tk"` | `cli` | | to `indrajala-demos`'s own `cli` |
| `"matplotlib"` | `pyproject.toml` dependencies | | removed |

None of the three measurement modules imports matplotlib or tkinter (checked 2026-10-13).

## Decisions

Settled for extensibility and speed (the owner's direction); every option's pros and cons are
recorded.

- **D1: the measurement pieces stay in indrajala-ml, as `indrajala_ml/measurement/` modules.**
  Settled (a).
  - (a) Move them into `indrajala_ml/measurement/` (`layer_op_timing.py`, `conv_comparison.py`,
    `timing.py`) and `batch_size_scaling`'s runner into `indrajala_ml/studies/`. Pros: the
    timing scripts and `ab.py` keep working on every machine with no demo checkout; the names
    say what they are; the package's layout matches measurement.md. Cons: every import in five
    scripts and two docs changes; `python -m` paths in measurement.md change.
  - (b) Leave them in a smaller `indrajala_ml/demos/`. Pros: no renames. Cons: a `demos`
    package that holds no demos; the next reader moves them anyway.
  - (c) Move them to `indrajala-demos` too. Pros: one cut. Cons: jebel's timing protocol would
    need the demo repository, inverting the dependency the split exists for.
- **D2: indrajala-demos uses indrajala-ml as a git submodule until 1.0.** Settled (a), the
  owner's direction (2026-10-13): no PyPI until a 1.0 release, which then follows semver.
  - (a) A submodule at `indrajala-ml/` (which brings `rust/` with it: `git submodule update
    --init --recursive`), installed editable into the demo repo's venv. Pros: the same mechanism
    as `rust/`, so one workflow; a demo runs against an exact library commit; no publishing.
    Cons: nested submodules; a library change reaches the demos only through a bump PR.
  - (b) A pip dependency on a git URL (`indrajala-ml @ git+ssh://...@<commit>`). Pros: no
    submodule. Cons: pip must build the crate itself from a nested submodule, which git-URL
    installs handle poorly; a second mechanism beside `rust/`'s.
  - (c) PyPI. Pros: what users will do. Cons: deferred to 1.0 by the owner.
- **D3: the history comes with the files.** Settled (a).
  - (a) `git filter-repo --path ...` on a fresh clone, keeping the moved paths' history (renames
    followed with `--path-rename` where a file moved within the repo before). Pros: `git log`
    and `blame` still explain every demo. Cons: a one-off tool to install; the filtered clone
    must be checked before the first push.
  - (b) Copy the files into a new repository. Pros: simplest. Cons: their history stays only in
    indrajala-ml, where the files no longer are.
- **D4: the new repository's layout mirrors the old paths' roles**: package `indrajala_demos`
  (`demos/`, `graphics/`, `capture/` under it), `tests/` beside it, its own `cli` (`setup`,
  `test`, `lint`, `demo`) and CI. Settled. Pros: imports change by one prefix
  (`indrajala_ml.demos` to `indrajala_demos.demos`, likewise `graphics` and `capture`), so the
  move is mechanical; the same lint and test conventions. Cons: none found against a flat
  layout except one more directory level.
- **D5: one cut, not a deprecation period.** Settled. indrajala-ml has no users who import its
  demos as a library (it isn't published), so nothing is kept behind as a re-export. Pros: no
  shims to remove later. Cons: an old checkout's habits (`./cli demo`) stop working at once;
  the README says where they went.
- **D6: the demo repository's CI runs its tests against the submodule's commit**, checking out
  recursively and building the crate as indrajala-ml's CI does. Settled. Pros: a library change
  that breaks a demo shows up on the bump PR. Cons: its CI builds the crate too (minutes per
  run).
- **D7: visibility and creation** (owner). The other repositories are public; the plan assumes
  `davidbarkhuizen/indrajala-demos`, public, created with `gh repo create` by the agent, unless
  the owner says otherwise.

## Stages

Each stage is one PR (stage 2: one in each repository). `./cli test` and `./cli lint` pass; the
golden run is unaffected (no training path changes) and checked anyway after stage 1, since
modules that build networks move.

### Stage 1: the measurement modules move into the package proper

1. `demos/benchmarks/demo_layer_op_timing.py` to `indrajala_ml/measurement/layer_op_timing.py`;
   `demos/conv/demo_conv_rust_vs_vectorized_digit_recognition.py` to
   `indrajala_ml/measurement/conv_comparison.py`; `demos/timing.py` to
   `indrajala_ml/measurement/timing.py`; `demos/benchmarks/demo_batch_size_scaling.py` to
   `indrajala_ml/studies/batch_size_scaling_run.py`. `git mv`, so history follows.
2. Their tests to `tests/measurement/` and `tests/studies/`; the five scripts' imports;
   measurement.md §5's commands; the registry's module paths (the menu still offers them);
   the `pyproject.toml` per-file ignore that names the conv demo.
3. An `ab.py` probe run on jebel (`focused_benchmark.py` and `epoch_op_profile.py` start and
   report) once E7 has ended, since jebel's checkout must be pulled for it.

Done when nothing under `scripts/` or `docs/measurement.md` names `indrajala_ml.demos`.

### Stage 2: indrajala-demos

1. Create the repository (D7). A fresh clone of indrajala-ml, `git filter-repo` keeping
   `indrajala_ml/demos`, `indrajala_ml/graphics`, `indrajala_ml/capture`, `tests/graphics`,
   `tests/capture`, the conv depth sweep's test and `tests/training/test_training_pipeline.py`,
   renamed into D4's layout. Check the result: the file list, `git log --follow` on two files.
2. In the demo repository: the imports, the indrajala-ml submodule, `pyproject.toml`
   (matplotlib, numpy; the library through the submodule), `cli`, CI (D6), a README. Push, CI
   green, merge.
3. In indrajala-ml: remove the moved paths, `"matplotlib"` from the dependencies,
   `python3-tk` and `demo` from `cli`, and the README's demo sections (a line pointing to
   indrajala-demos instead). Run the registry walks' tests (they walk network classes, not
   demos, but check). The handover's notes on demos move with them.

Done when indrajala-ml installs and passes `./cli test` in a venv without matplotlib, and
indrajala-demos's CI is green against indrajala-ml's main.

## After this plan

- The aarch64 workplan's stage 1 runs indrajala-ml on the Pi with no demo checkout.
- At 1.0, D2 is revisited: indrajala-demos depends on the published package.
