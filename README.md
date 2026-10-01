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

Requires Python >= 3.14, a Debian/Ubuntu host (`setup` apt-installs `python3-tk`) and
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
| `tests/` | ~5350 | this package: models, training, data loaders, Rust-vs-numpy parity |
| `rust/tests/` | ~2000 | the submodule's own `indrajala_math_rust` API, checked against numpy |

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
| `scripts/` (the rest) | benchmark, profiling and sweep tools, `ab.py` (old-against-new timing A/Bs) and the refactoring golden run; see `docs/measurement.md` |
| `docs/` | the measurement guide, optimization docs, next steps, the PyPI release workplan, the primitives roadmap, the RNG audit, machine profiles |

## Models

`LinearClassifierNetwork` is Rosenblatt's perceptron / MADALINE, trained with the
minimum-disturbance rule. A backprop network is a list of layer specs and one update rule, built
in any of three implementations of the same maths:

```python
from indrajala_ml.model.array_backend import RUST
from indrajala_ml.model.layer_specs import Conv, Dense, Pool
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.update_rules import Momentum

network = SequentialArrayNetwork(
    input_shape=(28, 28, 1),
    layers=[Conv(5, 8), Pool(2), Dense(30), Dense(10, output=True)],
    update_rule=Momentum(0.9),
    shape="multiclass",  # or "single_output"
    backend=RUST,  # or NUMPY
)
```

- **Layer specs** (`layer_specs.py`) are backend-free data: `Dense(size, activation, dropout)`
  (sigmoid or ReLU, dropout on sigmoid only), `Conv(kernel_size, channel_count, stride)`,
  `Pool(pool_size, stride)`, and the output layer `Dense(size, output=True, activation, loss)`
  (sigmoid with the squared or cross-entropy loss, or softmax with cross-entropy). Activations are
  fused into their layer. `validate_layer_specs` refuses a list that some implementation can't
  build.
- **Update rules** (`update_rules.py`) are data too: `SGD`, `Momentum`, `Adam` and `WeightDecay`
  (see Update rules, below).
- **The optimizer** holds all of a network's update state: `NumpyOptimizer` and `RustOptimizer`
  (`optimizers.py`), `PythonOptimizer` (`python_optimizer.py`). It keeps momentum's velocities and
  Adam's moments by layer position, with one step count `t`. `checkpoint()` and format 2 save
  it. Layers keep their weights and gradients, and no update formula.
- **Builders** map each spec to an implementation's layer classes: `array_layer_builder.py` for
  numpy and Rust, `python_layer_builder.py` for pure Python.

| Implementation | Base | Sequential network |
| --- | --- | --- |
| pure Python, one object per node | `BackpropNetworkBase` | `SequentialMultiClassBackpropClassifierNetwork`, `SequentialBackpropClassifierNetwork` (`sequential_backprop_network.py`) |
| numpy arrays | `ArrayNetworkBase` | `SequentialArrayNetwork(..., backend=NUMPY)` |
| Rust arrays (production backend) | `RustArrayNetworkBase` | `SequentialArrayNetwork(..., backend=RUST)` |

The numpy and Rust networks are one implementation: `ArrayNetworkBase` calls the few array
operations that differ through a backend object (`array_backend.py`), and `RustArrayNetworkBase`
is the subclass that sets the Rust backend. Only the layers are written once per backend. The
numpy classes are the reference the Rust classes are tested against
(`tests/test_*fused_layer_ops.py`, `tests/test_numerical_parity.py`).

### Presets

A preset is a named class for one fixed combination of specs and rule, with its own constructor
arguments. It equals, by bits, the Sequential network of the same specs and rule
(`tests/array_network_contract.py`, `tests/test_sequential_backprop_network.py`). The demos,
`ensemble_train.py` and saved files use the presets by name.

| Layers and rule | pure Python | numpy | Rust |
| --- | --- | --- | --- |
| sigmoid, one output, `SGD` | `BackpropClassifierNetwork` | `ArrayBackpropClassifierNetwork` | `RustArrayBackpropClassifierNetwork` |
| … cross-entropy loss | `BinaryCrossEntropyBackpropClassifierNetwork` | `CrossEntropyArrayBackpropClassifierNetwork` | `CrossEntropyRustArrayBackpropClassifierNetwork` |
| … fan-in-aware initialization | `FanInAwareBackpropClassifierNetwork` | | |
| … ReLU hidden layers | `ReLUBackpropClassifierNetwork` | | |
| … dropout | `DropoutBackpropClassifierNetwork` | | |
| … `Momentum` | `MomentumBackpropClassifierNetwork` | | |
| … `Adam` | `AdamBackpropClassifierNetwork` | | |
| … `WeightDecay` | `L2RegularizedBackpropClassifierNetwork` | | |
| one-vs-rest ensemble of one-output networks | `EnsembleBackpropClassifierNetwork` | `EnsembleArrayBackpropClassifierNetwork` | `EnsembleRustArrayBackpropClassifierNetwork` |
| sigmoid, multiclass, `SGD` | `MultiClassBackpropClassifierNetwork` | `VectorizedMultiClassBackpropClassifierNetwork` | `RustArrayMultiClassBackpropClassifierNetwork` |
| … cross-entropy loss | | `CrossEntropyVectorizedMultiClassBackpropClassifierNetwork` | `CrossEntropyRustArrayMultiClassBackpropClassifierNetwork` |
| … softmax output, cross-entropy loss | `SoftmaxMultiClassBackpropClassifierNetwork` | `SoftmaxVectorizedMultiClassBackpropClassifierNetwork` | `SoftmaxRustArrayMultiClassBackpropClassifierNetwork` |
| … ReLU hidden layers | | `ReLUVectorizedMultiClassBackpropClassifierNetwork` | `ReLURustArrayMultiClassBackpropClassifierNetwork` |
| … dropout | | `DropoutVectorizedMultiClassBackpropClassifierNetwork` | `DropoutRustArrayMultiClassBackpropClassifierNetwork` |
| … `Momentum` | | `MomentumVectorizedMultiClassBackpropClassifierNetwork` | `MomentumRustArrayMultiClassBackpropClassifierNetwork` |
| … `Adam` | | `AdamVectorizedMultiClassBackpropClassifierNetwork` | `AdamRustArrayMultiClassBackpropClassifierNetwork` |
| … `WeightDecay` | | `L2VectorizedMultiClassBackpropClassifierNetwork` | `L2RustArrayMultiClassBackpropClassifierNetwork` |
| conv and pool, then sigmoid multiclass, `SGD` | `ConvMultiClassBackpropClassifierNetwork` | `ConvVectorizedMultiClassBackpropClassifierNetwork` | `ConvRustArrayMultiClassBackpropClassifierNetwork` |
| … `Momentum` | `MomentumConvMultiClassBackpropClassifierNetwork` | `MomentumConvVectorizedMultiClassBackpropClassifierNetwork` | `MomentumConvRustArrayMultiClassBackpropClassifierNetwork` |

An empty cell has no preset, but the Sequential network of that implementation builds the
combination, so each array preset has a pure-Python parity reference. Combinations the Sequential
networks build that no preset has, and those still out of reach, are listed in
[docs/next-steps.md](docs/next-steps.md), From composable layers. Batch norm has no preset: the
Sequential networks build it (Batch normalization).

The pure-Python implementation is for correctness and parity checking only: gradient checks,
hand-computed examples, and the reference the array implementations are checked against. It is
never used for performance (speed/timing) measurement; only the numpy and Rust implementations
are timed. Accuracy comparisons of pure-Python models are fine.

## Saving and loading

Every backprop network and ensemble saves one format, format 2 (`indrajala_ml/model/format2.py`):
a JSON file with the layer specs, the update rule, the weights and the optimizer's state
(momentum's velocities, Adam's `m`, `v` and step count `t`). A loaded network resumes training
where it stopped. Training on after `save` and `load` takes the same steps, by bits, as training
on without them (`tests/test_format2.py`). Dropout's masks and the epoch shuffle come from global
RNG state, which isn't saved.

```python
from indrajala_ml.model.load_network import load_network

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
- An ensemble's file nests one format-2 file per sub-network.
- A batch-norm layer's entry records its spec. Its weights are `γ`, `β` and the running mean and
  variance, so a loaded network classifies as the saved one did and resumes training by bits; its
  linear layer's weights are `W` alone. The optimizer's state for it is per `γ` and `β`.
- `load` still reads each class's legacy file, written before format 2, with fresh optimizer
  state (`tests/test_legacy_saved_models.py`). Files saved in format 2 don't load on older versions
  of this package.

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
equivalent. `tests/test_update_rule_forms.py` checks each implementation of SGD, weight decay and
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
(`tests/test_summation_order.py`): `X.sum(axis=0)` does across two or more features, but sums a
single feature pairwise from 8 rows, and `D.sum(axis=(0, 2))` over a conv channel doesn't. The numpy
layers sum with `np.cumsum` along the summed axis, which does at every shape. The pure-Python layer
sums with an explicit loop: the builtin `sum` adds floats with compensated summation since Python
3.12. Given the same inputs, the pure-Python, numpy and Rust layers compute the same bits, except
for a sigmoid's `exp` (`tests/test_batch_norm_python_network.py`,
`tests/test_batch_norm_rust_network.py`, their conv counterparts, and
`tests/test_batch_norm_ghost_groups.py`). Whole networks agree within their dense and conv layers'
rounding only, which differs with or without batch norm: the pure-Python parity tolerance, and
between numpy and Rust, BLAS's products against the crate's. Only `+ − × ÷` and `sqrt` appear, each
correctly rounded in IEEE 754, so every implementation that follows these forms computes the same
bits. The sigmoid's `exp` is the exception: it isn't correctly rounded, and numpy's `np.exp` picks
its implementation by CPU, so it can differ from `math.exp` and Rust's `f64::exp` in the last bit.
The tests give the numpy layer the other implementation's `exp`, and compare everything else by
bits. There's no `pow`, since `(var + eps)^(-3/2)` through a library `pow` could differ between
Python, numpy and Rust. `tests/gradient_check.py` checks each implementation's backward pass against
finite differences of the whole batch's loss.

## Refactoring

A structural refactoring changes structure only, never numerics. Every stage keeps every parity
test passing, and:

- **Training stays bit-identical.** Before the first stage, record a golden run on `main` with
  `python scripts/golden_training_run.py record data/refactoring/golden_run.json` (ignored by
  git: numpy's BLAS makes the file valid only on the machine that recorded it). Each stage must
  pass `... check data/refactoring/golden_run.json`, the same bits, not within a tolerance. It
  catches a 1-ULP change to the learning rate. It trains the networks of all three
  implementations, pure Python included.
- **No hot-path slowdown.** A stage that touches `learn*` or `classify_rows` is timed old against
  new with `scripts/ab.py` ([docs/measurement.md](docs/measurement.md)). It must be within noise.
- **Public names stay.** Demos, `demos/registry.py`, `ensemble_train.py` and the tests construct
  the concrete classes by name, and saved model files must still load:
  `tests/test_legacy_saved_models.py` loads a committed file for every class that has `save`
  (`tests/fixtures/saved_models/`, written once by `python -m tests.saved_model_fixtures`).

## Docs

- [docs/measurement.md](docs/measurement.md): how to time a change: the machine, the tools, A/Bs
  with `scripts/ab.py`, and the rules for a timing claim in a PR.
- [docs/optimizations.md](docs/optimizations.md): Rust against numpy, what has been optimized and
  rejected, the candidates left, and how to measure a change.
- [docs/pypi-release-workplan.md](docs/pypi-release-workplan.md): publishing the Rust crate to
  PyPI, with multi-platform wheels built and tested on every push, PR and release tag.
- [docs/primitives-roadmap.md](docs/primitives-roadmap.md): the proposed order for the next ML
  primitives: composable layers, batch norm, residual connections, then attention.
- [docs/next-steps.md](docs/next-steps.md): the work left over from completed workplans, and how
  to read those workplans in git history.
- [docs/rng-audit.md](docs/rng-audit.md): the random number generators in use, how the crate's
  reproduces numpy's legacy `np.random` bit for bit, how to seed a run, and the open RNG work.

## License

MIT, see [LICENSE](LICENSE).
