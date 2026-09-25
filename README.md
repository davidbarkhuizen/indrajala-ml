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

Requires Python >= 3.10 and a Debian/Ubuntu host (`setup` apt-installs `python3-tk` and `cargo`).

```
./cli setup          # submodule, .venv, pip deps, release build of rust/, fetch MNIST
./cli test           # pytest tests/ and rust/tests/ (or: ./cli test <path> ...)
./cli demo           # interactive demo menu
./cli demo <n>       # run demo n directly
./cli build-rust     # rebuild rust/ after changing it
./cli fetch-data     # re-verify / re-fetch datasets
```

## Testing

`./cli test` runs two pytest suites in the venv:

| Suite | Tests | Covers |
| --- | --- | --- |
| `tests/` | ~2250 | this package: models, training, data loaders, Rust-vs-numpy parity |
| `rust/tests/` | ~1200 | the submodule's own `indrajala_math_rust` API, checked against numpy |

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
then `./cli test <suite>` on every push and PR, one parallel job per suite. It skips apt (the
runner image already has cargo, and CI's python is not apt's), and caches `.venv`, MNIST and the
crate's release wheel, keyed on the `rust/` submodule commit, so the crate only compiles when
the submodule moves.

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
| `docs/` | optimization docs, refactoring and study work plans, machine profiles |

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

## Docs

- [docs/optimizations.md](docs/optimizations.md): Rust against numpy, what has been optimized and
  rejected, the candidates left, and how to measure a change.
- [docs/refactoring.md](docs/refactoring.md): structural duplication still in the code, with a
  staged plan for each item.
- [docs/conv-batch-size-scaling-workplan.md](docs/conv-batch-size-scaling-workplan.md): the
  planned study of batch-size scaling for the conv network.
