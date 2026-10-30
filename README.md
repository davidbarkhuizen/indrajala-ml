# indrajala-ml

fast neural network classifiers from first-principles with no ML framework dependencies

**Paper 1** (*Learning Finite-State Transducers Through Their Approximations, with Proof*): its code
is this repository at tag [`paper1-v1.1`](https://github.com/davidbarkhuizen/indrajala-ml/tree/paper1-v1.1)
(the native crate and the manuscript at the same tag in their repositories), and its data the
datapack [doi:10.5281/zenodo.23733575](https://doi.org/10.5281/zenodo.23733575), whose README says
how to verify every number. `main` has moved on since (its `scripts/` grouped, for one).

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

Requires Python >= 3.14 and
[rustup](https://rustup.rs): `rust/rust-toolchain.toml` pins the Rust toolchain, which rustup
installs on the first build (a distro `cargo` ignores the pin).

```
./cli setup          # submodule, .venv, pip deps, release build of rust/, fetch MNIST and the text corpora
./cli test           # pytest tests/ and rust/tests/ (or: ./cli test <path> ...)
./cli lint           # ruff check, ruff format --check, pyright
./cli build-rust     # rebuild rust/ after changing it
./cli fetch-data     # re-verify / re-fetch datasets
```

The runnable demos (classifiers trained and plotted, digit recognition, drawing your own
digits) are in [indrajala-demos](https://github.com/davidbarkhuizen/indrajala-demos), which uses
this repository as a submodule; the library itself needs neither matplotlib nor tkinter.

## Testing

`./cli test` runs two pytest suites in the venv:

| Suite | Tests | Covers |
| --- | --- | --- |
| `tests/` | ~9900 | this package: models, training, data loaders, Rust-vs-numpy parity |
| `rust/tests/` | ~3000 | the submodule's own `indrajala_math_rust` API, checked against numpy |

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

CI (`.github/workflows/ci.yml`) runs `./cli setup --rust-wheel-dir .rust-wheel` then `./cli test
<suite>` on every push and PR, in parallel jobs on x86_64 and aarch64 (arm64): `rust/tests/` whole
and `tests/` in 8 shards (`--shard I/8`, `tests/conftest.py`), the `rust/tests/` job also checking
the golden run's committed Rust reference (`data/refactoring/golden_run_rust.json`), plus a `lint` job (`./cli
setup --rust-wheel-dir .rust-wheel --no-data`, then `./cli lint`). It caches `.venv`, the datasets and
the crate's release wheel, keyed on the `rust/` submodule commit, so the crate only compiles when
the submodule moves.

## Linting

`./cli lint` runs `ruff check`, `ruff format --check` and `pyright`; `ruff check --fix . && ruff
format .` (in the venv) applies ruff's fixable findings. Both tools are pinned in `./cli`
(`ruff_version`, `pyright_version`), because their rule sets change between releases:

- `pyproject.toml`'s `[tool.ruff]` sets the line length (120), excludes `rust/` and holds the
  per-file ignores;
- `[tool.pyright]` type-checks `indrajala_ml/`, `scripts/` and `tests/` in strict mode (tests may
  read private names). It reads the crate's type stub from the installed `indrajala_math_rust`, so
  `rust/` must be built.

A `# pyright: ignore[rule]` names its rule and says why (e.g. a numpy stub narrower than the
function). Zed runs the same pinned tools: `.zed/settings.json` points its ruff and pyright
language servers into `.venv/bin`. The crate lints its own Rust and Python tests (see
`rust/README.md`).

## Layout

| Path | Contents |
| --- | --- |
| `indrajala_ml/model/` | classifier networks and their layers |
| `indrajala_ml/` (top level) | `batch.py`, the batch runner's controller and workers (`scripts/runner/batch.py`); `pcg64.py`, numpy's `default_rng` in pure Python; `geometry.py`, the linear classifier's polygon targets |
| `indrajala_ml/training/` | training loops (`train.py`), synthetic training data, run checkpoints, evaluation (disagreement rate, accuracy, confusion matrix), one-vs-rest ensemble training over multiprocessing (`ensemble_train.py`) |
| `indrajala_ml/data/` | MNIST, UCI digits and Iris loaders; the text corpora's (`text_data.py`); base-3 addition (`addition_data.py`) and modular arithmetic (`modular_data.py`) for the window-recurrent studies; `prepared_dataset.py`, a dataset as one backend matrix, which the array networks train from |
| `indrajala_ml/measurement/` | the benchmark machine's profile; multi-seed parameter sweeps over MNIST proxy tasks; the per-op timing (`layer_op_timing.py`) and the conv Rust/numpy comparison (`conv_comparison.py`) the timing scripts build on, and their timing helpers (`timing.py`); the benchmark archive's records (`benchmark_archive.py`); the machine lock, one heavy job at a time (`machine_lock.py`) |
| `indrajala_ml/studies/` | the batch-size scaling study (linear learning-rate scaling with warmup); `patch_study.py`, the patch-model studies' shared protocol; `sequence_study.py`, the sequence study's; `attention_dropout_study.py`, the attention-dropout study's; `rotary_study.py` and `addition_study.py`; the window-recurrent harness (`window_recurrent_study.py`), its tasks (`window_tasks.py`, `modular_family.py`), its certificate (`window_recurrent_certificate.py`, `chow.py`), its readings (`window_diagnostics.py`, `window_drift.py`, `machine_reading.py`, `window_replay.py`), the guard and collapse trigger (`certificate_guard.py`, `collapse.py`) and the claims' statistics (`inference.py`) |
| `rust/` | `indrajala_math_rust` submodule (PyO3/maturin) |
| `data/` | UCI digits and Iris (committed); MNIST and the text corpora (fetched into `data/mnist/` and `data/<corpus>/`) |
| `scripts/data/fetch_datasets.py` | checksum-verified fetch of MNIST and the four text corpora, each from its pinned `indrajala-datasets-*` repository |
| `scripts/` (the rest) | grouped by purpose ([scripts/README.md](scripts/README.md)): `measurement/` (`ab.py`, the benchmarks, probes and the golden run; see `docs/measurement.md`), `runner/` (the batch runner), `studies/` (one script per study, `configs/`), `paper/` (paper 1's tables, datapack and Zenodo deposit) |
| `docs/` | the measurement guide, working notes, next steps, the primitives roadmap, machine profiles; `docs/workplans/` the open library and infrastructure workplans, `docs/rng-audit.md` the RNG audit. The research notes, the research protocols (experiment design, feasibility checklist), the research workplans and the dated reviews are in [indrajala-research](https://github.com/davidbarkhuizen/indrajala-research) |

## Models

`LinearClassifierNetwork` is Rosenblatt's perceptron / MADALINE, trained with the
minimum-disturbance rule. A backprop network is a list of layer specs and one update rule, built
in any of three implementations of the same maths:

```python
from indrajala_ml.model.layers.array.array_backend import RUST
from indrajala_ml.model.specs.layer_specs import Conv, Dense, Pool
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.update_rules import Momentum

network = SequentialArrayNetwork(
    input_shape=(28, 28, 1),
    layers=[Conv(5, 8), Pool(2), Dense(30), Dense(10, output=True)],
    update_rule=Momentum(0.9),
    shape="multiclass",  # or "single_output", or "sequence"
    backend=RUST,  # or NUMPY
)
```

- **Layer specs** (`layer_specs.py`) are backend-free data: `Dense(size, activation, dropout)`
  (sigmoid or ReLU, dropout on sigmoid only), `Conv(kernel_size, channel_count, stride)`,
  `Pool(pool_size, stride)`, and the output layer `Dense(size, output=True, activation, loss)`
  (sigmoid with the squared or cross-entropy loss, or softmax with cross-entropy), with
  `BatchNorm(activation)` after a linear layer (Batch normalization), `Residual(body)` for a
  residual block (Residual connections), `LayerNorm(epsilon)`, and a patch model's
  `Patches(patch_size)`, `Position()`, `Attention(heads, key_size)` and `TokenMean()` (Layer norm
  and attention), a sequence model's `Embedding(vocabulary, size)` and `Attention(causal=True)`
  (The sequence task), and `Attention(dropout=p)` and a token-wise `Dropout(p)` (Dropout among the
  tokens). Activations are fused into their layer. `validate_layer_specs` refuses a list that
  some implementation can't build.
- **Update rules** (`update_rules.py`) are data too: `SGD`, `Momentum`, `Adam` and `WeightDecay`
  (see Update rules, below).
- **The optimizer** holds all of a network's update state: `NumpyOptimizer` and `RustOptimizer`
  (`numpy_optimizer.py`, `rust_optimizer.py`), `PythonOptimizer` (`python_optimizer.py`). It keeps
  momentum's velocities and Adam's moments by layer position, with one step count `t`.
  `checkpoint()` and format 2 save
  it, and the network's generator's state beside it. Layers keep their weights and gradients, and no update formula.
- **Builders** map each spec to an implementation's layer classes: `array_layer_builder.py` for
  numpy and Rust, `python_layer_builder.py` for pure Python.

| Implementation | Base | Sequential network |
| --- | --- | --- |
| pure Python, one object per node | `BackpropNetworkBase` | `SequentialMultiClassBackpropClassifierNetwork`, `SequentialBackpropClassifierNetwork`, `SequentialSequenceBackpropNetwork` (`sequential_backprop_network.py`) |
| numpy arrays | `ArrayNetworkBase` | `SequentialArrayNetwork(..., backend=NUMPY)` |
| Rust arrays (production backend) | `RustArrayNetworkBase` | `SequentialArrayNetwork(..., backend=RUST)` |

The numpy and Rust networks are one implementation: `ArrayNetworkBase` calls the few array
operations that differ through a backend object (`array_backend.py`), and `RustArrayNetworkBase`
is the subclass that sets the Rust backend. Only the layers are written once per backend. The
numpy classes are the reference the Rust classes are tested against
(`tests/model/layers/test_*fused_layer_ops.py`, `tests/model/layers/test_numerical_parity.py`).

### Presets

A preset is a named class for one fixed combination of specs and rule, with its own constructor
arguments. It equals, by bits, the Sequential network of the same specs and rule
(`tests/array_network_contract.py`, `tests/model/networks/python/test_sequential_backprop_network.py`). The demos
([indrajala-demos](https://github.com/davidbarkhuizen/indrajala-demos)), `ensemble_train.py` and
saved files use the presets by name.

| Layers and rule | pure Python | numpy | Rust |
| --- | --- | --- | --- |
| sigmoid, one output, `SGD` | `BackpropClassifierNetwork` | `ArrayBackpropClassifierNetwork` | `RustArrayBackpropClassifierNetwork` |
| … cross-entropy loss | `BinaryCrossEntropyBackpropClassifierNetwork` | `CrossEntropyArrayBackpropClassifierNetwork` | `CrossEntropyRustArrayBackpropClassifierNetwork` |
| … fan-in-aware initialization | `FanInAwareBackpropClassifierNetwork` | | |
| … ReLU hidden layers | `ReLUBackpropClassifierNetwork` | `ReLUArrayBackpropClassifierNetwork` | `ReLURustArrayBackpropClassifierNetwork` |
| … dropout | `DropoutBackpropClassifierNetwork` | `DropoutArrayBackpropClassifierNetwork` | `DropoutRustArrayBackpropClassifierNetwork` |
| … `Momentum` | `MomentumBackpropClassifierNetwork` | `MomentumArrayBackpropClassifierNetwork` | `MomentumRustArrayBackpropClassifierNetwork` |
| … `Adam` | `AdamBackpropClassifierNetwork` | `AdamArrayBackpropClassifierNetwork` | `AdamRustArrayBackpropClassifierNetwork` |
| … `WeightDecay` | `L2RegularizedBackpropClassifierNetwork` | `L2ArrayBackpropClassifierNetwork` | `L2RustArrayBackpropClassifierNetwork` |
| one-vs-rest ensemble of one-output networks | `EnsembleBackpropClassifierNetwork` | `EnsembleArrayBackpropClassifierNetwork` | `EnsembleRustArrayBackpropClassifierNetwork` |
| sigmoid, multiclass, `SGD` | `MultiClassBackpropClassifierNetwork` | `VectorizedMultiClassBackpropClassifierNetwork` | `RustArrayMultiClassBackpropClassifierNetwork` |
| … cross-entropy loss | `CrossEntropyMultiClassBackpropClassifierNetwork` | `CrossEntropyVectorizedMultiClassBackpropClassifierNetwork` | `CrossEntropyRustArrayMultiClassBackpropClassifierNetwork` |
| … softmax output, cross-entropy loss | `SoftmaxMultiClassBackpropClassifierNetwork` | `SoftmaxVectorizedMultiClassBackpropClassifierNetwork` | `SoftmaxRustArrayMultiClassBackpropClassifierNetwork` |
| … ReLU hidden layers | `ReLUMultiClassBackpropClassifierNetwork` | `ReLUVectorizedMultiClassBackpropClassifierNetwork` | `ReLURustArrayMultiClassBackpropClassifierNetwork` |
| … dropout | `DropoutMultiClassBackpropClassifierNetwork` | `DropoutVectorizedMultiClassBackpropClassifierNetwork` | `DropoutRustArrayMultiClassBackpropClassifierNetwork` |
| … `Momentum` | `MomentumMultiClassBackpropClassifierNetwork` | `MomentumVectorizedMultiClassBackpropClassifierNetwork` | `MomentumRustArrayMultiClassBackpropClassifierNetwork` |
| … `Adam` | `AdamMultiClassBackpropClassifierNetwork` | `AdamVectorizedMultiClassBackpropClassifierNetwork` | `AdamRustArrayMultiClassBackpropClassifierNetwork` |
| … `WeightDecay` | `L2RegularizedMultiClassBackpropClassifierNetwork` | `L2VectorizedMultiClassBackpropClassifierNetwork` | `L2RustArrayMultiClassBackpropClassifierNetwork` |
| conv and pool, then sigmoid multiclass, `SGD` | `ConvMultiClassBackpropClassifierNetwork` | `ConvVectorizedMultiClassBackpropClassifierNetwork` | `ConvRustArrayMultiClassBackpropClassifierNetwork` |
| … `Momentum` | `MomentumConvMultiClassBackpropClassifierNetwork` | `MomentumConvVectorizedMultiClassBackpropClassifierNetwork` | `MomentumConvRustArrayMultiClassBackpropClassifierNetwork` |
| … `Adam` | `AdamConvMultiClassBackpropClassifierNetwork` | `AdamConvVectorizedMultiClassBackpropClassifierNetwork` | `AdamConvRustArrayMultiClassBackpropClassifierNetwork` |
| … `WeightDecay` | `L2RegularizedConvMultiClassBackpropClassifierNetwork` | `L2ConvVectorizedMultiClassBackpropClassifierNetwork` | `L2ConvRustArrayMultiClassBackpropClassifierNetwork` |
| … ReLU dense hidden layers | `ReLUConvMultiClassBackpropClassifierNetwork` | `ReLUConvVectorizedMultiClassBackpropClassifierNetwork` | `ReLUConvRustArrayMultiClassBackpropClassifierNetwork` |
| … dropout on the dense hidden layers | `DropoutConvMultiClassBackpropClassifierNetwork` | `DropoutConvVectorizedMultiClassBackpropClassifierNetwork` | `DropoutConvRustArrayMultiClassBackpropClassifierNetwork` |
| … cross-entropy loss | `CrossEntropyConvMultiClassBackpropClassifierNetwork` | `CrossEntropyConvVectorizedMultiClassBackpropClassifierNetwork` | `CrossEntropyConvRustArrayMultiClassBackpropClassifierNetwork` |
| … softmax output, cross-entropy loss | `SoftmaxConvMultiClassBackpropClassifierNetwork` | `SoftmaxConvVectorizedMultiClassBackpropClassifierNetwork` | `SoftmaxConvRustArrayMultiClassBackpropClassifierNetwork` |

Each preset changes one thing in its row's base network. The one empty cell is not a gap: the
array one-output networks already initialize fan-in-aware, so `ArrayBackpropClassifierNetwork` and
`RustArrayBackpropClassifierNetwork` are its counterparts. Combinations of two or more changes
(ReLU under `Adam`, dropout with a softmax output, ...) have no preset; the Sequential networks
build them, and [docs/next-steps.md](docs/next-steps.md), From composable layers, lists them with
those still out of reach. Batch norm, residual blocks, layer norm and patch models have no preset
either: the Sequential networks build them (Batch normalization, Residual connections, Layer norm
and attention).

The pure-Python implementation is for correctness and parity checking only: gradient checks,
hand-computed examples, and the reference the array implementations are checked against. It is
never used for performance (speed/timing) measurement; only the numpy and Rust implementations
are timed. Accuracy comparisons of pure-Python models are fine. From one seed, every pure-Python
network with an array twin starts with numpy's and Rust's weights and draws their dropout masks,
by bits (`tests/model/networks/test_seeded_init_parity.py`; [docs/rng-audit.md](docs/rng-audit.md)),
so a parity test seeds all sides alike.

## Saving and loading

Every backprop network and ensemble saves one format, format 2 (`indrajala_ml/model/persistence/format2.py`):
a JSON file with the layer specs, the update rule, the weights, the optimizer's state
(momentum's velocities, Adam's `m`, `v` and step count `t`) and the network's generator's state, so
dropout's masks resume too. A loaded network resumes training where it stopped. Training on after
`save` and `load` takes the same steps, by bits, as training on without them
(`tests/model/persistence/test_format2.py`). The trainers' epoch shuffle draws from a `random.Random` passed as
`rng=`, which the model file doesn't hold. A run file does (below).

```python
from indrajala_ml.model.persistence.load_network import load_network

network.save("model.json")
network = AdamVectorizedMultiClassBackpropClassifierNetwork.load("model.json")  # its own class
network = load_network("model.json")  # the Sequential network the file describes
```

- A preset (every named class) records its class and constructor arguments. Its `load` rebuilds
  it from them, and refuses a file whose layers, update rule, shape or input aren't its own, naming
  the difference.
- `load_network` builds the Sequential network of the file's specs, rule and implementation, never
  a class named in the file. It gives the same results, by bits, as the preset that saved the file.
- numpy files load into Rust networks and Rust files into numpy ones. Pure-Python files hold one
  weight list per node and load into pure Python only.
- An ensemble's file nests one format-2 file per sub-network, each with its own generator.
- The generator's state is numpy's `bit_generator.state` for PCG64, which all three implementations
  hold, so a numpy file's masks draw on in Rust. Its 128-bit `state` and `inc` are hex strings. A
  file saved before the field existed loads with a fresh OS-entropy generator.
- A batch-norm layer's entry records its spec. Its weights are `γ`, `β` and the running mean and
  variance, so a loaded network classifies as the saved one did and resumes training by bits; its
  linear layer's weights are `W` alone. The optimizer's state for it is per `γ` and `β`.
- A residual block's entry is `{"kind": "residual", "body": [...]}`, with its body's entries; its
  affine layer's entry alone has `"bias": true`. Weights and optimizer state are per layer of the
  flattened block (`expand_specs`): an affine layer's `W` and `b`, nothing for the fork and add.
- The patch model's entries are `"patches"`, `"position"`, `"layer_norm"`, `"attention"` and
  `"token_mean"`; a token-wise dense layer's is a `"dense"` one. Weights and optimizer state:
  `Position`'s `P`, a layer norm's `γ` and `β` (flat or over tokens), attention's `Wq, bq, Wk, bk,
  Wv, bv, Wo, bo` (heads as row blocks), nothing for `Patches` or `TokenMean`
  (`tests/model/persistence/test_attention_format2.py`). An `"attention"` entry has `"heads"` and
  `"key_size"` only when they aren't the defaults, so a one-head file is as before they existed.
- A sequence model's file has the `"sequence"` shape, an `"embedding"` entry with its table `E`
  (its weights and optimizer state per row, a token id's, in pure Python), and `"causal": true` on
  a causal attention entry, written only when true (The sequence task).
- `load` still reads each class's legacy file, written before format 2, with fresh optimizer
  state (`tests/model/persistence/test_legacy_saved_models.py`). Files saved in format 2 don't load on older versions
  of this package.

A run stopped at an epoch boundary resumes, by bits, from a run file
(`indrajala_ml/training/run_checkpoint.py`). `train_backprop_network_mini_batch` returns
`result.run_checkpoint`, taken before the pocket restores the best epoch. It holds the last
epoch's network, the pocket's network, accuracy and epoch, the per-epoch accuracies, the
convergence series, the shuffle generator's state, and the epoch and batch counters, which a
learning-rate schedule reads. `save_run` writes it as one JSON file, with both networks as format-2
network entries. The model file stays the pocketed model.

```python
from indrajala_ml.training.run_checkpoint import load_run, save_run

result = train_backprop_network_mini_batch(network, data, 32, epochs=10, rng=Random(7))
network.save("model.json")
save_run("run.json", result.run_checkpoint, network)

network = load_network("model.json")  # later, in another process
result = train_backprop_network_mini_batch(network, data, 32, epochs=20, resume_from=load_run("run.json", network))
```

`epochs` counts the whole run, so the resumed call trains epochs 10 to 19. Given the same data,
batch size, learning rate and reference classifier, it ends where an uninterrupted 20-epoch run
ends (`tests/training/test_run_checkpoint.py`). A run file holds one network's run, not an ensemble's.

## Update rules

Every update rule follows a published form, with the source's arithmetic grouping, in all three
implementations. Otherwise results aren't comparable with the literature, or between our own
backends: a different grouping changes the last bit, and training amplifies that. `g` is the
gradient summed over a batch of `B` examples, so `g / B` is the mean gradient.

| Rule | Form | Source |
| --- | --- | --- |
| `SGD` | `w - lr * (g / B)` | Goyal et al. 2017, eq. (2) |
| L2 weight decay (`WeightDecay`) | `w - lr * (g / B + λ * w)`; the bias is plain SGD | Goyal et al. 2017, eq. (8) |
| `Adam` | Algorithm 1, on `g / B` | Kingma & Ba 2014 |
| `Momentum` | `u = m * u + g / B; w - lr * u` | Goyal et al. 2017, eq. (9) |

For momentum the literature has competing forms. Rumelhart et al. 1986's, Goyal et al.'s eq. (10),
folds the rate into the velocity, `v = lr * g / B + m * v; w - v`, and so needs a correction
whenever the rate changes, as in warmup. Eq. (9) needs none, and at a constant rate the two are
equivalent. `tests/model/optimizers/test_update_rule_forms.py` checks each implementation of SGD, weight decay and
momentum against its form bit for bit. A new rule cites its source here, and where the literature
has competing forms (as for momentum), the choice is made explicitly. Consistency comes before
speed: every rule divides, `g / B`, and none multiplies by a precomputed `1 / B` or `lr / B`, which
rounds differently when `B` isn't a power of two.

## Batch normalization

numpy, pure Python and Rust build dense batch norm, `Dense(size, activation="linear"),
BatchNorm(activation)`, and conv batch norm, `Conv(kernel_size, channel_count, activation="linear"),
BatchNorm("relu")`, in the Sequential networks under every update rule, with or without ghost
groups (`BatchNorm(group_size=32)`); no preset has it. Format 2 saves it (Saving and loading). This
section fixes the forms all three implementations are held to.

A pure-Python network with batch norm trains a batch layer by layer (`layer_major.py`): forward
through each layer for the whole batch, then backward. The other layers run their per-example code
unchanged, their nodes' per-example state kept for each example in turn. Networks without batch norm
keep the example-by-example loop.

A norm layer follows a linear layer without a bias and carries the activation, as the paper places
it: "We add the BN transform immediately before the nonlinearity, by normalizing x = Wu + b. […]
Note that, since we normalize Wu+b, the bias b can be ignored since its effect will be canceled by
the subsequent mean subtraction (the role of the bias is subsumed by β in Alg. 1)" (Ioffe &
Szegedy 2015, § 3.2). A dense norm layer normalizes each feature over the batch. A conv norm layer
normalizes each channel over the batch and every position, so `m` is `B * P`.

**Training** normalizes with the batch's statistics, Algorithm 1 of Ioffe & Szegedy 2015:

| Step | Algorithm 1 |
| --- | --- |
| mini-batch mean | `μ_B ← (1/m) Σ_{i=1..m} x_i` |
| mini-batch variance | `σ²_B ← (1/m) Σ_{i=1..m} (x_i − μ_B)²` |
| normalize | `x̂_i ← (x_i − μ_B) / sqrt(σ²_B + ε)` |
| scale and shift | `y_i ← γ x̂_i + β ≡ BN_γ,β(x_i)` |

The backward pass is the paper's § 3 chain rule, "before simplification", term by term:

| Gradient | § 3 |
| --- | --- |
| `∂ℓ/∂x̂_i` | `∂ℓ/∂y_i · γ` |
| `∂ℓ/∂σ²_B` | `Σ_{i=1..m} ∂ℓ/∂x̂_i · (x_i − μ_B) · (−1/2)(σ²_B + ε)^(−3/2)` |
| `∂ℓ/∂μ_B` | `(Σ_{i=1..m} ∂ℓ/∂x̂_i · −1/sqrt(σ²_B + ε)) + ∂ℓ/∂σ²_B · (Σ_{i=1..m} −2(x_i − μ_B)) / m` |
| `∂ℓ/∂x_i` | `∂ℓ/∂x̂_i · 1/sqrt(σ²_B + ε) + ∂ℓ/∂σ²_B · 2(x_i − μ_B)/m + ∂ℓ/∂μ_B · 1/m` |
| `∂ℓ/∂γ` | `Σ_{i=1..m} ∂ℓ/∂y_i · x̂_i` |
| `∂ℓ/∂β` | `Σ_{i=1..m} ∂ℓ/∂y_i` |

**Inference** normalizes with population statistics. Algorithm 2 (step 10) averages over training
batches: `E[x] ← E_B[μ_B]`, `Var[x] ← m/(m−1) · E_B[σ²_B]`, the unbiased variance. We keep moving
averages instead, as the paper notes one can ("Using moving averages instead, we can track the
accuracy of a model as it trains", § 3.1) and as Goyal et al. 2017 do: "As in [12], we compute the
BN statistics using running average (with momentum 0.9)" (§ 5.1). Each training forward pass updates
them with the batch's statistics, at `running_rate` 0.1, their momentum 0.9, as PyTorch's default,
whose `ε` of 1e-5 we take too. Inference normalizes as training does, with the running averages in
place of `μ_B` and `σ²_B`, not with Algorithm 2 step 11's folded form
`y = γ/sqrt(Var[x]+ε) · x + (β − γE[x]/sqrt(Var[x]+ε))`, which rounds differently.

**Weight decay** doesn't apply to `γ` or `β`: "We use a weight decay λ of 0.0001 and following
[16] we do not apply weight decay on the learnable BN coefficients (namely, γ and β in [19])"
(Goyal et al. 2017, § 5.1). Under `WeightDecay` both step with plain SGD, as a bias does.

**Ghost groups** fix the statistics' sample size whatever the batch size, as Goyal et al. 2017 do
per worker: "if the per-worker sample size n is kept fixed and the total minibatch size is kn, it
can be viewed a minibatch of k samples with each sample B_j independently selected from X^n, so the
underlying loss function is unchanged"; "In this work, we use n = 32 […]. If n is adjusted, it
should be viewed as a hyper-parameter of BN, not of distributed training" (§ 2.3). This is Hoffer
et al. 2017's ghost batch norm. With a `group_size`, a training batch is split in row order into
groups of `group_size` examples, the last group the remainder. Each group is normalized, forward and
backward, with its own statistics, as a batch of its own, `m` its examples (times `P` after a conv
layer), and moves the running averages in turn, so a batch moves them once per group. The
gradients of `γ` and `β` still sum over the whole batch. A last group of one example is refused,
for the reason a batch of one is: it normalizes to 0. A network refuses such a batch before its
forward pass, and `train_backprop_network_mini_batch` refuses before training a batch size that
leaves one in its full batches or its final short batch. Without a `group_size`, or with one at
least the batch's size, the batch is one group: plain batch norm, by bits.

The exact expressions, per feature or channel, with `x_1..x_m` in row order (a conv channel's
values example by example, then position by position). Every implementation computes these, in
this grouping, left to right, with no fused multiply-add and no power function, only IEEE 754's
correctly rounded operations. The forward pass divides rather than multiplying by a precomputed
reciprocal, as the Update rules above divide by `B`; the backward pass follows § 3's factors of
`1/sqrt(σ²_B + ε)`:

```text
training forward
  mu      = sum(x_i) / m
  d_i     = x_i - mu
  ss      = sum(d_i * d_i)
  var     = ss / m
  std     = sqrt(var + eps)
  xhat_i  = d_i / std
  y_i     = gamma * xhat_i + beta                      then the activation
  running_mean = (1 - rate) * running_mean + rate * mu
  running_var  = (1 - rate) * running_var + rate * (ss / (m - 1))       unbiased

inference forward
  xhat_i  = (x_i - running_mean) / sqrt(running_var + eps)
  y_i     = gamma * xhat_i + beta                      then the activation

backward, from delta_i = dl/dy_i (the activation's derivative already applied)
  inv_std  = 1 / std
  inv_std3 = inv_std / (var + eps)                    (var + eps)^(-3/2)
  dxhat_i  = delta_i * gamma
  dvar     = sum(dxhat_i * d_i * -0.5 * inv_std3)
  dmu      = sum(dxhat_i * -inv_std) + dvar * sum(-2 * d_i) / m
  dx_i     = dxhat_i * inv_std + dvar * (2 * d_i) / m + dmu / m
  grad_gamma += sum(delta_i * xhat_i)
  grad_beta  += sum(delta_i)
```

With ghost groups the training forward and backward expressions apply to each group's `x_i`, and
`grad_gamma` and `grad_beta` to the whole batch's.

`sum` is a left fold from `0.0` in that row order, the crate's order (`sum_axis0`, and the conv
`grad_b` sum). numpy's own reductions follow it only in some layouts
(`tests/model/layers/test_summation_order.py`): `X.sum(axis=0)` does across two or more features, but sums a
single feature pairwise from 8 rows, and `D.sum(axis=(0, 2))` over a conv channel doesn't. The numpy
layers sum with `np.cumsum` along the summed axis, which does at every shape. The pure-Python layer
sums with an explicit loop: the builtin `sum` adds floats with compensated summation since Python
3.12. Given the same inputs, the pure-Python, numpy and Rust layers compute the same bits, except
for a sigmoid's `exp` (`tests/model/networks/test_batch_norm_python_network.py`,
`tests/model/networks/test_batch_norm_rust_network.py`, their conv counterparts, and
`tests/model/networks/test_batch_norm_ghost_groups.py`). Whole networks agree within their dense and conv layers'
rounding only, which differs with or without batch norm: the pure-Python parity tolerance, and
between numpy and Rust, BLAS's products against the crate's. Only `+ − × ÷` and `sqrt` appear, each
correctly rounded in IEEE 754, so every implementation that follows these forms computes the same
bits. The sigmoid's `exp` is the exception: it isn't correctly rounded, and numpy's `np.exp` picks
its implementation by CPU, so it can differ from `math.exp` and Rust's `f64::exp` in the last bit.
The tests give the numpy layer the other implementation's `exp`, and compare everything else by
bits. There's no `pow`, since `(var + eps)^(-3/2)` through a library `pow` could differ between
Python, numpy and Rust. `tests/gradient_check.py` checks each implementation's backward pass against
finite differences of the whole batch's loss.

## Residual connections

A residual block adds its input to its body's output, `out = x + F(x)`, the pre-activation form of
He et al. 2016 ("Identity Mappings in Deep Residual Networks", arXiv 1603.05027) and the
transformer's: nothing follows the add. The identity path carries the gradient past the body
unchanged. Dense blocks only, with identity shortcuts (the block's output size is its input size),
for the Sequential networks of all three implementations, under every update rule, with or
without batch norm in the body; no preset has them. Format 2 saves them (Saving and loading). This
section fixes the forms all three implementations are held to:

```python
from indrajala_ml.model.specs.layer_specs import BatchNorm, Dense, Residual

layers = [
    Dense(64, activation="relu"),
    Residual((Dense(128, activation="relu"), Dense(64, activation="linear", bias=True))),
    Residual((Dense(128, activation="linear"), BatchNorm("relu"), Dense(64, activation="linear", bias=True))),
    Dense(10, activation="softmax", output=True, loss="cross_entropy"),
]
```

A body is a valid sequence of dense hidden layers (a linear layer and a `BatchNorm` as a pair,
dropout on sigmoid only), with no conv, pool or nested block, and ends in an affine layer,
`Dense(n, activation="linear", bias=True)`, whose `n` is the block's input size. The affine layer
lets `F` be negative, so the sum isn't pushed one way. `bias=True` is accepted nowhere else.

The builders flatten each block into layers, `Fork`, the body, then `Add` (`expand_specs`). Layer
indices (snapshots, optimizer state, checkpoints) count these layers. The fork and add have no
parameters and draw nothing at initialization, so adding a block never shifts a later layer's
draws; the affine layer draws `W`, then `b`, fan-in-aware, as a dense layer does.

The exact expressions, per example (a batch row by row), with `body_first` the body's first layer
and `f'` the derivative of the activation of the layer before the fork:

```text
forward
  fork    x                                 its input, unchanged and kept
  affine  y = W h + b                       h: the layer before it, in the body
  add     out = y + x                       x: the fork's input

backward
  add     delta = downstream                from the layer after it
          its downstream: delta             the identity's
  affine  delta = downstream                the add's delta
          its downstream, gradients: a dense layer's (W^T delta; delta h^T, delta)
  fork    delta = body_first.downstream + add.delta
          its downstream: delta
  before the fork, sigmoid
          delta = (fork.delta * a) * (1 - a)
```

The add and the fork's sum are one IEEE addition each, which is commutative, so their operand order
can't move bits; no sum here has three terms. The layer before a fork multiplies in numpy's order,
`(downstream * a) * (1 - a)`, as a sigmoid layer does today; ReLU masks, and dropout scales, as
they do today. On Rust a dense layer's hidden delta is one fused call that reads the next layer's
`W` and delta; before a fork it is one fused call with the skip term,
`(body_first.delta @ body_first.W + add.delta) * f'(a)` (`layer_hidden_delta_skip` and its ReLU
and dropout forms), which computes the same bits as the crate's unfused downstream, add and mask.

`scripts/studies/residual_depth_study.py` trains plain and residual networks of 2 to 16 hidden layers on
MNIST. Without batch norm the plain network collapses to one class from 8 layers, its first
layer's gradient vanishing, while the residual one holds 96.9%; with batch norm the plain network
loses 2 points by 16 layers and the residual one 0.3 (the findings are in its docstring).

## Layer norm and attention

A patch model, a small vision transformer (Dosovitskiy et al. 2020, "An Image is Worth 16x16
Words", arXiv 2010.11929), cuts the image into patches, one token each, embeds them, adds learned
positions, passes them through pre-LN blocks, `x + F(LN(x))` (Xiong et al. 2020, arXiv
2002.04745), averages over the tokens and classifies. Multi-head self-attention (Vaswani et al. 2017, "Attention
Is All You Need", arXiv 1706.03762, §3.2.2), layer norm over tokens and over flat dense layers, for the Sequential networks of all three implementations,
under every update rule; no preset has them. Format 2 saves them (Saving and loading). This
section fixes the forms all three implementations are held to:

```python
from indrajala_ml.model.specs.layer_specs import Attention, Dense, LayerNorm, Patches, Position, Residual, TokenMean

layers = [
    Patches(7),  # (28, 28, 1) -> 16 tokens of 49
    Dense(32, activation="linear", bias=True),  # the embedding, per token
    Position(),
    Residual((LayerNorm(), Attention())),
    Residual((LayerNorm(), Dense(64, activation="relu"), Dense(32, activation="linear", bias=True))),
    TokenMean(),  # 16 tokens of 32 -> 32
    LayerNorm(),
    Dense(10, activation="softmax", output=True, loss="cross_entropy"),
]
```

A token sequence of `T` tokens of `d` features is flat and token-major, index `t * d + j`, so a
batch is an `(N, T, d)` stack and an `(N * T, d)` matrix without a copy. Between `Patches`, the
first layer, and `TokenMean` a network has a token part, in place of a conv front end: `Dense` acts
on each token with weights shared over the tokens (ReLU, or linear with `bias=True`, no dropout),
`Position` stands once, before any block, and a block's body ends in `Attention` or
`Dense(d, activation="linear", bias=True)`. Sigmoid, a dense layer's dropout and `BatchNorm` are
refused among tokens; the token-wise `Dropout` stands only where Dropout among the tokens says.
After `TokenMean` the dense part's rules hold, and in any dense network `LayerNorm` may stand
wherever a dense hidden layer may, a residual body's first layer included. A layer norm has no
activation after it.

`Attention(heads=1, key_size=None)` attends in `heads` heads of `key_size` features each, `d_k`;
`None` means `d / heads`, which `heads` must then divide (Keras's `key_dim`). Values are as wide as
keys. At the default, its parameter count, `4d² + 4d`, doesn't depend on `heads`; a set `key_size`
makes the projections `heads * key_size` wide, independently of `d`. A transformer layer is the two
blocks above; a deeper encoder writes them again.

`Patches`, `Position`, `TokenMean` and `LayerNorm` draw nothing at initialization, so adding one
never shifts a later layer's draws: `P` and `beta` start at 0, `gamma` at 1. `Attention` draws
`Wq, bq, Wk, bk, Wv, bv, Wo, bo` in that order, fan-in-aware, as a dense layer draws `W`, then `b`:
`Wq`, `Wk` and `Wv` are `(h * d_k, d)`, head `i`'s rows a block, and `Wo` is `(d, h * d_k)`, its
fan-in `h * d_k`. Under `WeightDecay` weights decay; biases, `gamma`, `beta` and `P`
don't. Each new layer is one layer in the expanded list, so layer indices count it as one.

The exact expressions, per example. `sum` is a left fold from `0.0` in index order (a batch's
gradient sums run over examples, then tokens, the rows of the `(N * T, d)` view); a product is a
matrix product (numpy's BLAS, the crate's products on Rust, as a dense layer's; in pure Python a
left fold too, where its flat dense layers use the builtin `sum`). Every
implementation computes these, in this grouping, left to right, with no fused multiply-add and no
power function:

```text
patches, size p, an (H, W, C) image stored channel-major (c * H * W + h * W + w)
  token t = u * (W / p) + v                              patch row u, column v, row-major
  feature k = c * p * p + i * p + j                      the conv kernel's (channel, row, col) order
  x_tk = image[c, u * p + i, v * p + j]
  backward: the inverse permutation of delta

token-wise dense
  a dense layer's expressions on each token, the batch as the (N * T, d) rows

position
  out_t = x_t + P_t                                      a new array; x is not written
  backward: dx_t = delta_t;  grad_P_t += sum over examples of delta_t

layer norm, each token's d features (a flat layer is one token)
  mu      = sum(x_j) / d
  c_j     = x_j - mu
  var     = sum(c_j * c_j) / d                           biased; training and inference alike
  std     = sqrt(var + eps)                              eps 1e-5
  xhat_j  = c_j / std
  y_j     = gamma_j * xhat_j + beta_j
  backward
  dxhat_j = delta_j * gamma_j
  m1      = sum(dxhat_j) / d
  m2      = sum(dxhat_j * xhat_j) / d
  dx_j    = ((dxhat_j - m1) - xhat_j * m2) / std
  grad_gamma_j += sum(delta_j * xhat_j);  grad_beta_j += sum(delta_j)    over examples and tokens

attention, X the (T, d) tokens, h heads of d_k features, s = sqrt(d_k)
  [i] is head i's block: rows i * d_k to (i + 1) * d_k - 1 of Wq, Wk, Wv and their biases,
  the same columns of Q, K, V, H and Wo; at one head with d_k = d, [0] is the whole matrix
  project
    Q = X Wq^T + bq;  K = X Wk^T + bk;  V = X Wv^T + bv   (T, h * d_k) each: the product, then the bias
  attend, each head i
    S[i] = (Q[i] K[i]^T) / s                             (T, T)
    m_t  = max_u(S[i]_tu)                                each row, max-shifted
    e_tu = exp(S[i]_tu - m_t)
    P[i]_tu = e_tu / sum_u(e_tu)
    H[i] = P[i] V[i]                                     (T, d_k)
  combine
    H = [H[0] ... H[h-1]]                                (T, h * d_k): the heads side by side
    out = H Wo^T + bo                                    one product over all h * d_k columns
  backward, in the same blocks, last first
  combine
    dH = delta Wo                                        (T, h * d_k)
    grad_Wo += delta^T H;  grad_bo += sum(delta)         over the batch's rows
  attend, each head i
    dP[i] = dH[i] V[i]^T
    dV[i] = P[i]^T dH[i]
    r_t   = sum_u(dP[i]_tu * P[i]_tu)
    dS[i]_tu = P[i]_tu * (dP[i]_tu - r_t)
    dQ[i] = (dS[i] K[i]) / s
    dK[i] = (dS[i]^T Q[i]) / s
  project
    dX = (dQ Wq + dK Wk) + dV Wv                         three products over all heads, in this order
    grad_Wq += dQ^T X;  grad_bq += sum(dQ)               and Wk, bk from dK; Wv, bv from dV

token mean
  out_j = sum_t(x_tj) / T
  backward: dx_tj = delta_j / T for every t
```

Apart from the products, only `+ − × ÷`, `sqrt`, `max` and the softmax's `exp` appear, so
given the same inputs the pure-Python, numpy and Rust layers compute the same bits outside the
products, as batch norm's do, and the tests give the numpy layer the other implementation's `exp`
(it isn't correctly rounded). numpy sums with `np.cumsum` along the summed axis
(`tests/model/layers/test_summation_order.py`), never `.sum`, whose order depends on the layout. Products
between activations are new: `Q K^T`, `P V` and their backward per example, BLAS against the
crate's products, explained as the dense layers' gap is, never accepted as a tolerance. The heads
meet only in the products that span them, `H Wo^T` and `dQ Wq` and its two siblings: each is one
product over all `h * d_k` columns, never a sum of per-head products. Exact tests need none: with
one token every `P[i] = [[1]]`, so attention is `(X Wv^T + bv) Wo^T + bo`; with `Wq = Wk = 0` and
`bq = bk = 0` every weight is exactly `1/16` at `T = 16`; with `Wo` and `bo` zero an attention
block leaves every other layer unchanged.

Two biases are inert. In each head `bk[i]` adds `q_t · bk[i]` to every score in row `t` of
`S[i]`, which the softmax ignores, so its gradient is 0 in exact arithmetic and rounding noise in
floating point, which differs between implementations; parity tests compare it apart. `bv` only
adds `Wo bv` to every output, as `bo` can, since each row of every `P[i]` sums to 1. Both stay, matching PyTorch's `nn.MultiheadAttention` and ViT's `qkv_bias`.

Every implementation's attention layer, and the crate, is the same three blocks per pass, project,
attend and combine, each tested on its own; the causal mask and dropout on `P` sit in attend (The sequence task,
Dropout among the tokens), and padding masks, grouped key/value heads and cross-attention each
have a named place among them (next-steps.md, From multi-head attention).
On Rust each layer-norm and attention pass is still one crate call, the attention ops thin
wrappers over the crate's blocks, and the token-wise dense layer is the existing dense ops on the
`(N * T, d)` view. A dense layer's hidden delta reads the next
layer's `W` in one fused call (Residual connections); right before a `LayerNorm`, or a fork whose
body starts with one, there is no `W` to read, and a sigmoid, ReLU or dropout layer takes the next
layer's downstream and a mask op instead, its own expression unchanged.

`scripts/studies/patch_attention_study.py` trains patch models with an FFN block, an attention block, or
both, against a conv network and a dense one, on MNIST under Adam (3 seeds, 5 epochs). Attention
then FFN beats FFN alone by less than a standard deviation, 95.8% against 95.4%; the attention
block alone is the weakest, 92.4%, below the dense control's 93.9%; and the conv network wins,
98.0%, as Dosovitskiy et al. 2020 predict at this data size (the findings are in its docstring).
`scripts/studies/multi_head_attention_study.py` reruns the attention-then-FFN model with 1, 2 and 4 heads,
4 heads of 32 features, and two layers, at 5 seeds. At this size more heads change nothing (96.1%,
96.2%, 96.0%), wider heads stay within noise, and a second layer helps on every seed: two layers of
4 heads reach 97.0%, a point under the conv network with 11% of its parameters.

## The sequence task

A sequence network predicts every token of a window from the tokens before it: next-character
prediction on a text corpus, a small causal transformer (Vaswani et al. 2017, §3.2.3; Radford et
al. 2018), in all three implementations and the crate, under every update rule; no preset has one.
It reads `T` token ids, gives a softmax over the vocabulary at each of the `T` positions, and is
trained on the mean of the `T` cross-entropies:

```python
from indrajala_ml.data.text_data import load_text_dataset
from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.layer_specs import Attention, Dense, Embedding, LayerNorm, Residual
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.training.sequence_evaluate import sequence_evaluate
from indrajala_ml.training.train import train_backprop_network_mini_batch

train, held_out, vocabulary = load_text_dataset()  # Tiny Shakespeare, windows of 64 + 1 characters
V = len(vocabulary)  # 65
layers = [
    Embedding(V, 64),  # 64 token ids -> 64 tokens of 64; positions inside attention (Rotary positions)
    Residual((LayerNorm(), Attention(heads=4, causal=True, positions="rotary"))),
    Residual((LayerNorm(), Dense(256, activation="relu"), Dense(64, activation="linear", bias=True))),
    LayerNorm(),
    Dense(V, activation="softmax", output=True, loss="cross_entropy"),  # a softmax per token
]
network = SequentialArrayNetwork((64,), layers, Adam(), "sequence", NUMPY)
network.randomize()
train_backprop_network_mini_batch(network, train, 32, learning_rate=0.002, epochs=10)
print(sequence_evaluate(network, held_out).bits_per_token)
```

- **The corpora** (`indrajala_ml/data/text_data.py`, `CORPORA`): Tiny Shakespeare (Karpathy 2015,
  1.1 M characters, 65 symbols), the default everywhere; Herodotus' *Histories* in Rawlinson's
  translation; Ibn Khaldun's *Muqaddimah* in Arabic; and Euclid's *Elements* in Heath's translation.
  Each lives in its own `indrajala-datasets-*` repository, cleaned there by a script that rebuilds
  it byte for byte from a pinned source, and `scripts/data/fetch_datasets.py` fetches and
  checksum-verifies it into `data/<name>/`, as it does MNIST. The *Muqaddimah*'s text is
  CC BY-NC-SA 4.0 (OpenITI's), fetched for study and never redistributed here; Euclid's encoding is
  Perseus's, CC BY-SA 4.0.
- **Examples**: the vocabulary is the corpus's characters, sorted, a token id each. The first 90%
  of the text is for training and the last 10% held out (nanoGPT's split), each cut into
  non-overlapping windows of `T + 1` characters, `T = 64`: a window's first `T` ids are its state,
  as floats, and its last `T` its labels, one class per token. The trainer is generic over the
  label, and reports accuracy per token.
- **`Embedding(vocabulary, size)`** starts the token part, as `Patches` does, over a flat input of
  `T` ids: each id's row of a learned `(vocabulary, size)` table `E`. It refuses an id that isn't
  a whole number in range. `E` is drawn as a weight matrix of fan-in `size` and doesn't decay.
- **A token-wise output**: a token part may end, without `TokenMean`, in its output layer, a
  softmax with cross-entropy applied to each token. That is the `"sequence"` network shape, beside
  `"multiclass"` and `"single_output"`, which refuse it; `classify_state` gives the per-token argmax.
- **`Attention(causal=True)`** masks each token's scores for the tokens after it, so token `t`
  attends to tokens `0` to `t` only.
- **Positions**: the example's are rotary, inside attention, the default for new sequence models
  since the rotary study (Rotary positions); a learned `Position()` after the `Embedding` is the
  alternative, and the sequence study's recorded models use one.
- **`sequence_evaluate(network, examples)`** gives the mean per-token cross-entropy in nats,
  `bits_per_token` (bits per character here) and the per-token accuracy.

The exact expressions, in the forms of Layer norm and attention:

```text
embedding, x the T ids, E (vocabulary, size)
  out_t = E[x_t]                                         a copy of the row
  backward: grad_E[x_t] += delta_t                       a scatter-add, examples then tokens in order (np.add.at's)

causal attention, attend, each head i
  S[i]_tu = -inf for u > t, after the scale              before the max shift; the diagonal is never masked
  so P[i]_tu = exp(-inf) / ... = 0 exactly, and dS[i]_tu = 0: the backward pass is unchanged

token-wise softmax output, each token's V outputs
  z_t = x_t W^T + b;  P_t = softmax(z_t)                 the softmax's sum a left fold
  loss = sum_t(-log P_t[y_t]) / T                        the mean over the tokens
  delta_t = (P_t - Y_t) / T                              Y_t one-hot
```

On Rust the mask is a `causal` keyword on the crate's two attention forward ops (the backward ops
read `P`, whose masked weights are already 0), the embedding is `embedding_forward` and
`embedding_accumulate_gradient`, and the token-wise softmax is the dense softmax op on the
`(N * T, d)` rows. Format 2 saves an `"embedding"` entry and its `E`, writes `"causal"` only when
true, and records the `"sequence"` shape (Saving and loading).

`scripts/studies/sequence_study.py` trains a 1- and 2-layer causal transformer (`d = 64`, 4 heads), an
FFN-only model and the 2-layer model without its mask against counted unigram and bigram floors,
on all four corpora, under Adam (5 seeds, 10 epochs, numpy). Held-out bits per character at 2
layers: Euclid 1.50, Herodotus 1.95, Tiny Shakespeare 2.48, the *Muqaddimah* 2.79. A per-token
model is a bigram model (FFN-only sits 0.01 bits above the counted bigram on every corpus); one
causal layer takes 0.8 to 1.5 bits under it, and a second 0.09 to 0.17 more, neither yet
converged at 10 epochs. Without the mask the model copies the next input: 0.04 to 0.06 bits per
character, held out too, on every corpus (the findings are in its docstring).

## Dropout among the tokens

A transformer drops out in two ways (Vaswani et al. 2017, §5.4; GPT-2's and nanoGPT's `attn_pdrop`,
`resid_pdrop` and `embd_pdrop`), and both are here, in all three implementations and the crate,
under every update rule, for any token part:

```python
layers = [
    Embedding(V, 64),
    Position(),
    Dropout(0.1),  # the embedding dropout
    Residual((LayerNorm(), Attention(heads=4, causal=True, dropout=0.1), Dropout(0.1))),  # attn_pdrop, resid_pdrop
    Residual((LayerNorm(), Dense(256, activation="relu"), Dense(64, activation="linear", bias=True), Dropout(0.1))),
    LayerNorm(),
    Dense(V, activation="softmax", output=True, loss="cross_entropy"),
]
```

- **`Attention(dropout=p)`** drops each attention weight in training, after the softmax and
  before the weighted sum of the values, the kept ones scaled by `1 / (1 - p)`: inverted dropout,
  as PyTorch's `scaled_dot_product_attention(dropout_p=p)`, so a row of the dropped weights no
  longer sums to 1 in training and is the plain softmax in inference.
- **`Dropout(p)`** drops each feature of each token, inverted, in training; it has no weights.
  Validation allows it in two places only: right after `Position` (the embedding dropout), and
  ending a block's body right after its `Attention` or affine `Dense` (the residual dropout,
  before the add). A dense layer's `Dense(dropout=p)` stays refused among tokens: it is fused with
  a sigmoid, and a token-wise `Dense` is ReLU or linear.
- **Training and inference.** Only `learn*` switches the layers into training for its forward
  pass; `classify_*`, `forward_rows` and `sequence_evaluate` run in inference, where neither
  mechanism draws or drops anything, so evaluation is deterministic.
- **The draws.** Masks come from the network's generator in forward order, one draw a layer a
  batch: an attention's `(N, h, T, T)` mask row-major (example, head, query, key), every entry
  drawn, the causally masked ones included; a `Dropout`'s `(N, T, d)` mask row-major (example,
  token, feature); a weight is kept when its draw `u >= p`, as a dense dropout layer's. An
  attention at `dropout=0` draws nothing and computes what it did before dropout, so a model
  without dropout keeps its bits; a `Dropout(0.0)` still draws its mask, as `Dense(dropout=0.0)`
  does. Pure Python draws in numpy's order (rng-audit.md): a network trains layer-major when a
  layer draws, example-major otherwise, and its masks match numpy's by bits.
- **Saving.** Format 2 writes an attention's `"dropout"` only when set, as `"causal"`, and a
  `{"kind": "dropout", "p": p}` entry for the token layer; the generator's saved state resumes the
  masks by bits (Saving and loading).

The exact expressions, in the forms of Layer norm and attention, `keep = 1 - p`:

```text
attention dropout, attend, each head i, in training
  M[i]_tu = 1 if u_tu >= p else 0                        u drawn as above
  P~[i] = P[i] * M[i] / keep
  H[i]  = P~[i] V[i]
  backward
  dV[i] = P~[i]^T dH[i]
  dP[i] = (dH[i] V[i]^T) * M[i] / keep
  then the softmax's backward as before, with the undropped P[i]

token dropout, in training
  M_tj = 1 if u_tj >= p else 0
  out_tj = x_tj * M_tj / keep
  backward: dx_tj = delta_tj * M_tj / keep
```

On Rust the crate's attention forward ops take `dropout`, `training` and `rng` keywords and
return the mask beside the caches (`None` when nothing was drawn), and the downstream op takes it;
the token layer is `token_dropout_forward` and `token_dropout_downstream`.

`scripts/studies/attention_dropout_study.py` trains the sequence study's 2-layer model with each
mechanism at 0.1, both at 0.1 and both at 0.2, against no dropout, on Euclid, Tiny Shakespeare and
Herodotus, with the last 10% held out and with a spread split (each of 100 blocks' last tenth),
under Adam (5 seeds, 10 epochs, one rate, numpy). The train/held-out gaps the sequence study found
are mostly the text's shift: on the spread split Euclid's falls from 0.49 to 0.04 bits per
character and Tiny Shakespeare's from 0.30 to 0.07, with the training loss unchanged. Dropout
narrows every gap by fitting the training text worse, and costs 0.01 to 0.35 bits held out
everywhere except attention dropout on Euclid's contiguous split (0.02 lower, on every seed); on
Herodotus, with no gap to close, it only costs. Attention dropout adds 10% to an epoch, the
token-wise dropout 6-7% (the findings are in its docstring).

## Rotary positions

A token part's positions are either a learned `Position`, added to each token before the first
block, or rotary positions inside attention (RoPE; Su et al. 2021, arXiv 2104.09864, the scheme of
GPT-NeoX, PaLM and LLaMA), in all three implementations and the crate, under every update rule:

```python
layers = [
    Embedding(V, 64),  # no Position
    Residual((LayerNorm(), Attention(heads=4, causal=True, positions="rotary"))),
    Residual((LayerNorm(), Dense(256, activation="relu"), Dense(64, activation="linear", bias=True))),
    LayerNorm(),
    Dense(V, activation="softmax", output=True, loss="cross_entropy"),
]
```

- **`Attention(positions="rotary", rotary_base=10000.0)`** rotates each head's queries and keys by
  their token's position, after their biases; values aren't rotated. A query at `t` and a key at
  `u` then score by their contents and `t - u` alone, so a key cached at one position stays valid as
  a window slides past it (the point of the change: generation's key/value cache). It has no
  parameters. `positions="none"`, the default, rotates nothing and keeps a model's bits.
- **Validation** refuses a token part with both a `Position` and a rotary attention (one scheme a
  model), an odd head size (features pair up), and a `rotary_base` on an unrotated attention.
  Rotary and unrotated attention may share a model; a causal and a non-causal one may both rotate.
- **The pairs** are "rotate half", feature `j` with `j + d_k / 2`, as GPT-NeoX and LLaMA, so each
  half is a contiguous slice; RoFormer's interleaved `2j, 2j + 1` is the same model up to a fixed
  permutation of `Wq`'s and `Wk`'s rows.
- **The angles** are one table per layer, `rotary_table(T, d_k, base)` in
  `indrajala_ml/model/specs/rotary_table.py`: `cos` and `sin` of `t * base^(-2j / d_k)` from
  Python's `math`, which every implementation reads, since `cos` and `sin` aren't correctly rounded
  and numpy's, Rust's and `math`'s may differ in the last bit. It derives from the spec, so it isn't
  saved; a longer table keeps a shorter one's rows.
- **Saving.** Format 2 writes `"positions"` and `"rotary_base"` only when not the default, so
  earlier files re-save unchanged.

The exact expressions, in the forms of Layer norm and attention, per head, `c = cos(t theta_j)` and
`s = sin(t theta_j)` for token `t`'s row, `j < d_k / 2`:

```text
rotary attention, the end of project, Q's and K's rows (each its own token's angles)
  q'_j          = q_j c - q_{j+d_k/2} s
  q'_{j+d_k/2}  = q_j s + q_{j+d_k/2} c                 k likewise; attend reads Q' and K'
  backward, the start of project's: the transpose, on dQ' and dK'
  dq_j          = dq'_j c + dq'_{j+d_k/2} s
  dq_{j+d_k/2}  = dq'_{j+d_k/2} c - dq'_j s             then dX and the gradients as before
```

Each entry is two products and a sum, so the three implementations rotate the same `Q` by bits. A
key bias is no longer cancelled by the softmax under rotation (it shifts each key's score by a
position-dependent amount), so `bk`'s gradient, rounding noise unrotated, is a real one here.

On Rust the crate's attention forward ops and downstream op take `cos` and `sin` keywords, the
layer's table as arrays; the downstream returns `dQ` and `dK` rotated back, so the gradient op
takes no table.

`scripts/studies/rotary_study.py` trained the sequence study's 1- and 2-layer models with rotary attention
and no `Position` against the learned-position models, seed for seed, on all four corpora (Adam, 10
epochs, 5 seeds, numpy). Rotary is lower held out on every corpus and arm, by 0.024 to 0.062 bits
per character at epoch 10 (39 of 40 seeds lower), and 0.65 to 0.89 points more accurate, with
4,096 fewer parameters; its lead is largest at epoch 1 (0.08 to 0.19 bits) and narrows without
closing. It fits faster rather than overfitting less. An epoch costs 3.5% more alone. So new
sequence models default to rotary positions, the sequence example among them (the findings are in
its docstring).

## The addition study

Which is the smallest network, built from the package's layers, that reliably adds? The task is
base-3 addition of three `n`-digit operands (two-operand addition is a zero in one slot), and a
network passes only when it meets every property of one catalogue, at 100% of its cases, on 4 of 5
seeds.

- **The catalogue** (`indrajala_ml/data/addition_data.py`) is addition's contract as code. Each
  property has a generator of cases, a verifier (Python's `+`, or a relation among the network's
  own answers) and a coverage, and a failing case shrinks to a minimal one. B1-B3 are the
  definition (adds three, adds two, a hashed tenth never trained on), A1-A3 the algebra
  (commutativity, associativity, identity), and M1-M5 the carry-lookahead algorithm's parts: the
  column sum, the carry out of a reset column, the carry over a distance `L` (reporting `L*`, the
  longest distance every case reaches), the final carry and every one of the `7^n` column-sum
  patterns. Each check is exhaustive where it can be (every triple at `n = 4`, every pattern at
  `n = 6`).
- **The candidates** (`indrajala_ml/studies/addition_study.py`) are an encoder, a network spec
  with its size knobs and a decoder: transformers with a token per column (causal, non-causal, or
  each column's digits as an unordered multiset), a string-format transformer, an FFN-only
  control, dense networks per sum digit, and two dense regressions. A new candidate, or a new
  property, is one more entry.
- **The harness** (`scripts/studies/addition_study.py`) runs a JSON config unattended: it bisects each
  shape's width, writes every run's result and trained model atomically, resumes a stopped sweep
  and builds its report at any time.

```bash
OPENBLAS_NUM_THREADS=1 python scripts/studies/addition_study.py run scripts/studies/configs/addition_sweep.json
python scripts/studies/addition_study.py report scripts/studies/configs/addition_sweep.json --md report.md
python scripts/studies/addition_study.py check data/addition/sweep/models/<run>
```

Only the column transformers passed the calibration, so the sweep (619 runs on `jebel`) searched
those three, at `n` 4 and 6, with 1-3 layers and 1-2 heads. The smallest reliable adders are 2
layers at `d` 6 at `n = 4` (831 parameters for the unordered encoding, 933 for the others), and 3
layers at `d` 6 at `n = 6` (1,197; 1,299). `d` 4 fails everywhere. One layer passes at `n = 4`
only wider and after many more epochs, and never at `n = 6` up to `d` 32. What fails is the carry
over distance (M3, M5): failing runs mostly still pass the column sum, identity and carry out of
a reset column whole.
The causal mask neither helps nor hurts from 2 layers on (the findings are in the script's
docstring; the results and threshold models are archived in `indrajala-models`). The
window-recurrent adder (`git show e45944e:docs/window-recurrent-adder-workplan.md`)
checks the same catalogue at lengths up to 64. What the adder studies have found so far, from
this study to the window-recurrent claims, is summarised in
[indrajala-research/docs/research-notes/adder-findings.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/research-notes/adder-findings.md); the window-recurrent claims themselves, each
registered before its data and decided (the snap is what extrapolates; a certified network adds
at every length; more codes buy test-passing networks, not correct ones; all at once works at
least as often as the curriculum; failures aren't unexplored states), are reported in
[indrajala-research/docs/research-notes/window-recurrent-report.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/research-notes/window-recurrent-report.md).

## Refactoring

A structural refactoring changes structure only, never numerics. Every stage keeps every parity
test passing, and:

- **Training stays bit-identical.** Before the first stage, record a golden run on `main` with
  `python scripts/measurement/golden_training_run.py record data/refactoring/golden_run.json` (ignored by
  git: numpy's BLAS makes the file valid only on the machine that recorded it). Each stage must
  pass `... check data/refactoring/golden_run.json`, the same bits, not within a tolerance. It
  catches a 1-ULP change to the learning rate. It trains the networks of all three
  implementations, pure Python included. The Rust entries are the same bits on every machine, so
  they are also committed as `data/refactoring/golden_run_rust.json`, which CI checks on every
  PR (`... check data/refactoring/golden_run_rust.json` checks it on any machine).
- **No hot-path slowdown.** A stage that touches `learn*` or `classify_rows` is timed old against
  new with `scripts/measurement/ab.py` ([docs/measurement.md](docs/measurement.md)). It must be within noise.
- **Public names stay.** The demos (indrajala-demos), `ensemble_train.py` and the tests construct
  the concrete classes by name, and saved model files must still load:
  `tests/model/persistence/test_legacy_saved_models.py` loads a committed file for every class that has `save`
  (`tests/fixtures/saved_models/`, written once by `python -m tests.saved_model_fixtures`).

## Docs

- [docs/measurement.md](docs/measurement.md): how to time a change: the machines, the tools, A/Bs
  with `scripts/measurement/ab.py`, the benchmarking tiers, and the rules for a timing claim in a PR.
- [`indrajala-benchmarks`](https://github.com/davidbarkhuizen/indrajala-benchmarks): the archive
  of golden runs, A/B runs with their reports, and the machine profiles they ran on; CI there
  re-renders every report with `ab.py` (measurement.md, §10).
- [docs/machine_profiles/i7-9700k.md](docs/machine_profiles/i7-9700k.md): the benchmark machine's
  noise and baseline (A/As of every benchmark, the per-op, conv demo and kernel tables), with the
  Ryzen laptop's numbers beside them.
- [docs/i7-9700k-rust-optimization.md](docs/i7-9700k-rust-optimization.md): where the Rust crate
  lags numpy on the benchmark machine, and what to investigate for each gap.
- [docs/workplans/infrastructure/pypi-release-workplan.md](docs/workplans/infrastructure/pypi-release-workplan.md): publishing the Rust crate to
  PyPI, with multi-platform wheels built and tested on every push, PR and release tag.
- [docs/primitives-roadmap.md](docs/primitives-roadmap.md): the proposed order for the next ML
  primitives: composable layers, batch norm, residual connections, layer norm and single-head
  attention, multi-head attention and a transformer block, and a sequence task with causal
  masking, all done.
- [docs/working-notes.md](docs/working-notes.md): the gotchas: worktrees, CI, studies, the
  machines and ssh.
- [scripts/README.md](scripts/README.md): the scripts' groups.
- [docs/next-machine-recommendation.md](docs/next-machine-recommendation.md): a buying guide for
  the next benchmark machine (2026-10-08).
- [docs/next-steps.md](docs/next-steps.md): the work left over from completed workplans, and how
  to read those workplans in git history.
- [docs/rng-audit.md](docs/rng-audit.md): the random number generators in use, how the crate's
  reproduces numpy's legacy `np.random` bit for bit, how to seed a run, and the open RNG work.

## License

MIT, see [LICENSE](LICENSE).
