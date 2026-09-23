# indrajala-ml

fast neural network classifiers from first-principles with no ML framework dependencies

The Rust math backend lives in a separate repo,
[indrajala-math-rust](https://github.com/davidbarkhuizen/indrajala-math-rust), mounted here as a
git submodule at `rust/`. Clone with `git clone --recurse-submodules`, or run
`git submodule update --init` after a plain clone (`./cli setup` also does this). To bump the
pinned submodule commit to the latest upstream `main`: `git submodule update --remote rust`.

## Usage

Requires Python >= 3.10 and a Debian/Ubuntu host (`setup` apt-installs `python3-tk` and `cargo`).

```
./cli setup          # submodule, .venv, pip deps, release build of rust/, fetch MNIST
./cli test           # pytest tests/ and rust/tests/
./cli demo           # interactive demo menu
./cli demo <n>       # run demo n directly
./cli build-rust     # rebuild rust/ after changing it
./cli fetch-data     # re-verify / re-fetch datasets
```

## Testing

`./cli test` runs two pytest suites in the venv:

| Suite | Tests | Covers |
| --- | --- | --- |
| `tests/` | ~1500 | this package: models, training, data loaders, Rust-vs-numpy parity |
| `rust/tests/` | ~200 | the submodule's own `indrajala_math_rust` API, checked against numpy |

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

CI (`.github/workflows/ci.yml`) runs `./cli setup` then `./cli test` on every push and PR.

## Layout

| Path | Contents |
| --- | --- |
| `indrajala_ml/model/` | classifier networks and their layers |
| `indrajala_ml/demos/` | runnable demos; `registry.py` lists them in menu order |
| `indrajala_ml/train.py` | training loops (linear classifier, mini-batch backprop) and synthetic data |
| `indrajala_ml/evaluate.py`, `multiclass_evaluate.py` | disagreement rate, accuracy, confusion matrix |
| `indrajala_ml/ensemble_train.py` | one-vs-rest ensemble training over multiprocessing |
| `indrajala_ml/*_data.py` | MNIST, UCI digits and Iris loaders |
| `indrajala_ml/benchmark_sweep.py`, `benchmark_data.py` | multi-seed parameter sweeps over MNIST proxy tasks |
| `indrajala_ml/graphics/chart.py` | matplotlib plotting |
| `rust/` | `indrajala_math_rust` submodule (PyO3/maturin) |
| `data/` | UCI digits and Iris (committed); MNIST (fetched into `data/mnist/`) |
| `scripts/fetch_datasets.py` | checksum-verified MNIST fetch from a pinned `indrajala-datasets-mnist` tag |

## Models

`LinearClassifierNetwork` is Rosenblatt's perceptron / MADALINE, trained with the
minimum-disturbance rule. Backprop networks come in three implementations of the same maths:

| Implementation | Base | Class prefix |
| --- | --- | --- |
| pure Python, one object per node | `BackpropNetworkBase` | `Backprop…`, `MultiClassBackprop…` |
| numpy arrays | `ArrayNetworkBase` | `Array…`, `Vectorized…` |
| Rust arrays (production backend) | `RustArrayNetworkBase` | `RustArray…` |

Each implementation has variants for the same set of features, named by a prefix on the class:
`ReLU`, `Softmax`, `CrossEntropy`, `L2`, `Momentum`, `Adam`, `Dropout`, `Ensemble`. Convolution
(`Conv…`) exists only in the pure-Python implementation. The numpy classes are the reference
the Rust classes are tested against (`tests/test_*fused_layer_ops.py`, `tests/test_numerical_parity.py`).

The pure-Python implementation is for correctness and parity checking only: gradient checks,
hand-computed examples, and the reference the array implementations are checked against. It is
never used for performance (speed/timing) measurement; only the numpy and Rust implementations
are timed. Accuracy comparisons of pure-Python models are fine.
