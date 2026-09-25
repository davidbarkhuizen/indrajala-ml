# indrajala-ml

fast neural network classifiers from first-principles with no ML framework dependencies

## The name

*Indrajala* is Sanskrit for "Indra's net". From
[Wikipedia](https://en.wikipedia.org/wiki/Indra%27s_net):

> "Indra's net" is an infinitely-large net owned by the Vedic deva Indra, which hangs over his
> palace on Mount Meru, the axis mundi of Buddhist and Hindu cosmology. In East Asian Buddhism,
> Indra's net is considered as having a multifaceted jewel at each vertex, with each jewel being
> reflected in all of the other jewels. In the Huayan school of Chinese Buddhism, which follows
> the Buddhāvataṃsaka Sūtra, the image of "Indra's net" is used to describe the interconnectedness
> or "perfect interfusion" (yuánróng, 圓融) of all phenomena in the universe.

The Rust math backend lives in a separate repo,
[indrajala-math-rust](https://github.com/davidbarkhuizen/indrajala-math-rust), mounted here as a
git submodule at `rust/`. Clone with `git clone --recurse-submodules`, or run
`git submodule update --init` after a plain clone (`./cli setup` also does this). To bump the
pinned submodule commit to the latest upstream `main`: `git submodule update --remote rust`.

## Usage

Requires Python >= 3.10, a Debian/Ubuntu host (`setup` apt-installs `python3-tk`) and
[rustup](https://rustup.rs): `rust/rust-toolchain.toml` pins the Rust toolchain, which rustup
installs on the first build (a distro `cargo` ignores the pin).

```
./cli setup          # submodule, .venv, pip deps, release build of rust/, fetch MNIST
./cli test           # pytest tests/ and rust/tests/ (or: ./cli test <path> ...)
./cli lint           # ruff check, ruff format --check, pyright
./cli demo           # interactive demo menu
./cli demo <n>       # run demo n directly
./cli build-rust     # rebuild rust/ after changing it
./cli fetch-data     # re-verify / re-fetch datasets
```

## Testing

`./cli test` runs two pytest suites in the venv:

| Suite | Tests | Covers |
| --- | --- | --- |
| `tests/` | ~3070 | this package: models, training, data loaders, Rust-vs-numpy parity |
| `rust/tests/` | ~1560 | the submodule's own `indrajala_math_rust` API, checked against numpy |

Both need the submodule checked out **and** built into `.venv`: `tests/` imports
`indrajala_math_rust` directly, and `rust/tests/` only exists once the submodule is initialised.
MNIST must also be fetched, since some tests read `data/mnist/*.bin`. `./cli setup` does all of
this; after a plain clone, or after bumping the submodule, run:

```
git submodule update --init   # populate rust/
./cli build-rust              # rebuild indrajala_math_rust into .venv
./cli fetch-data              # only needed once
./cli test
```

CI (`.github/workflows/ci.yml`) runs `./cli setup --no-os-packages --rust-wheel-dir .rust-wheel`
then `./cli test <suite>` on every push and PR, one parallel job per suite, plus a `lint` job
(`./cli lint`). It skips apt (CI's python is not apt's), and caches `.venv`, MNIST and the
crate's release wheel, keyed on the `rust/` submodule commit, so the crate only compiles when
the submodule moves.

## Linting

`./cli lint` runs `ruff check`, `ruff format --check` and `pyright`; `ruff check --fix . && ruff
format .` (in the venv) applies ruff's fixable findings. Both tools are pinned in `./cli`
(`ruff_version`, `pyright_version`), because their rule sets change between releases:

- `pyproject.toml`'s `[tool.ruff]` sets the line length (120), excludes `rust/` and holds the
  per-file ignores;
- `[tool.pyright]` type-checks `indrajala_ml/`, `tests/` and `scripts/` in standard mode. It reads
  the crate's type stub from the installed `indrajala_math_rust`, so `rust/` must be built.

A `# pyright: ignore[rule]` names its rule and says why (e.g. a numpy stub narrower than the
function). Zed runs the same pinned tools: `.zed/settings.json` points its ruff and pyright
language servers into `.venv/bin`. The crate lints its own Rust and Python tests (see
`rust/README.md`).

## Layout

| Path | Contents |
| --- | --- |
| `indrajala_ml/model/` | classifier networks and their layers |
| `indrajala_ml/demos/` | runnable demos; `registry.py` lists them in menu order |
| `indrajala_ml/train.py` | training loops (linear classifier, mini-batch backprop) and synthetic data |
| `indrajala_ml/evaluate.py`, `multiclass_evaluate.py` | disagreement rate, accuracy, confusion matrix |
| `indrajala_ml/ensemble_train.py` | one-vs-rest ensemble training over multiprocessing |
| `indrajala_ml/*_data.py` | MNIST, UCI digits and Iris loaders |
| `indrajala_ml/prepared_dataset.py` | a dataset as one backend matrix, which the array networks train from |
| `indrajala_ml/benchmark_sweep.py`, `benchmark_data.py` | multi-seed parameter sweeps over MNIST proxy tasks |
| `indrajala_ml/batch_size_scaling.py` | the batch-size scaling study (linear learning-rate scaling with warmup) |
| `indrajala_ml/graphics/chart.py` | matplotlib plotting |
| `rust/` | `indrajala_math_rust` submodule (PyO3/maturin) |
| `data/` | UCI digits and Iris (committed); MNIST (fetched into `data/mnist/`) |
| `scripts/fetch_datasets.py` | checksum-verified MNIST fetch from a pinned `indrajala-datasets-mnist` tag |
| `scripts/` (the rest) | benchmark, profiling and sweep tools (see `docs/optimizations/measurement.md`), and the refactoring golden run |
| `docs/` | optimization docs, study work plans, machine profiles |

## Models

`LinearClassifierNetwork` is Rosenblatt's perceptron / MADALINE, trained with the
minimum-disturbance rule. Backprop networks come in three implementations of the same maths:

| Implementation | Base | Class prefix |
| --- | --- | --- |
| pure Python, one object per node | `BackpropNetworkBase` | `Backprop…`, `MultiClassBackprop…` |
| numpy arrays | `ArrayNetworkBase` | `Array…`, `Vectorized…` |
| Rust arrays (production backend) | `RustArrayNetworkBase` | `RustArray…` |

The numpy and Rust networks are one implementation: `ArrayNetworkBase` calls the few array
operations that differ through a backend object (`array_backend.py`), and `RustArrayNetworkBase`
is the subclass that sets the Rust backend and layer classes. Only the layers are written once
per backend.

Each implementation has variants for the same set of features, named by a prefix on the class:
`ReLU`, `Softmax`, `CrossEntropy`, `L2`, `Momentum`, `Adam`, `Dropout`, `Ensemble`. Convolution
and max pooling (`Conv…`, `MaxPool…`) exist in all three implementations. The numpy classes are
the reference the Rust classes are tested against (`tests/test_*fused_layer_ops.py`,
`tests/test_numerical_parity.py`).

The pure-Python implementation is for correctness and parity checking only: gradient checks,
hand-computed examples, and the reference the array implementations are checked against. It is
never used for performance (speed/timing) measurement; only the numpy and Rust implementations
are timed. Accuracy comparisons of pure-Python models are fine.

## Update rules

Every update rule follows a published form, with the source's arithmetic grouping, in all three
implementations. Otherwise results aren't comparable with the literature, or between our own
backends: a different grouping changes the last bit, and training amplifies that. `g` is the
gradient summed over a batch of `B` examples, so `g / B` is the mean gradient.

| Rule | Form | Source |
| --- | --- | --- |
| SGD | `w - lr * (g / B)` | Goyal et al. 2017, eq. (2) |
| L2 weight decay (`L2…`) | `w - lr * (g / B + λ * w)`; the bias is plain SGD | Goyal et al. 2017, eq. (8) |
| Adam (`Adam…`) | Algorithm 1, on `g / B` | Kingma & Ba 2014 |
| momentum (`Momentum…`) | `u = m * u + g / B; w - lr * u` | Goyal et al. 2017, eq. (9) |

For momentum the literature has competing forms. Rumelhart et al. 1986's, Goyal et al.'s eq. (10),
folds the rate into the velocity, `v = lr * g / B + m * v; w - v`, and so needs a correction
whenever the rate changes, as in warmup. Eq. (9) needs none, and at a constant rate the two are
equivalent. `tests/test_update_rule_forms.py` checks each implementation of SGD, weight decay and
momentum against its form bit for bit. A new rule cites its source here, and where the literature
has competing forms (as for momentum), the choice is made explicitly. Consistency comes before
speed: every rule divides, `g / B`, and none multiplies by a precomputed `1 / B` or `lr / B`, which
rounds differently when `B` isn't a power of two.

## Refactoring

A structural refactoring changes structure only, never numerics. Every stage keeps every parity
test passing, and:

- **Training stays bit-identical.** Before the first stage, record a golden run on `main` with
  `python scripts/golden_training_run.py record data/refactoring/golden_run.json` (ignored by
  git: numpy's BLAS makes the file valid only on the machine that recorded it). Each stage must
  pass `... check data/refactoring/golden_run.json`, the same bits, not within a tolerance. It
  catches a 1-ULP change to the learning rate.
- **No hot-path slowdown.** A stage that touches `learn*` or `classify_rows` is timed before and
  after, numpy and Rust in separate processes (`scripts/prepared_dataset_timing.py time`), with
  both builds committed first. It must be within run-to-run noise.
- **Public names stay.** Demos, `demos/registry.py`, `ensemble_train.py` and the tests construct
  the concrete classes by name, and saved model files must still load.

## Docs

- [docs/optimizations.md](docs/optimizations.md): Rust against numpy, what has been optimized and
  rejected, the candidates left, and how to measure a change.
- [docs/conv-batch-size-scaling-workplan.md](docs/conv-batch-size-scaling-workplan.md): the
  planned study of batch-size scaling for the conv network.
