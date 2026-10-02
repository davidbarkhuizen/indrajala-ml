"""
The save fixtures: one small saved file per network class that has save(), which
tests/test_legacy_saved_models.py loads.

- The legacy fixtures (the composable-layers workplan, stage 0) were written by the code before
  format 2, so every legacy envelope is pinned and keeps loading after the save format changed.
- The format-2 fixtures (stage 5) are of the classes with no legacy envelope, the Sequential and
  the pure-Python single-output networks. Each is written after two training steps, so it pins a
  non-empty optimizer state too.
- The batch-norm fixtures (the batch-norm workplan, stage 5) are format-2 files of a Sequential
  network per implementation with a conv and a dense batch-norm pair, named BatchNorm<class name>.
- The residual fixtures (the residual-connections workplan, stage 5) are format-2 files of a
  Sequential network per implementation with two residual blocks, one holding a batch-norm pair,
  named Residual<class name>.
- The patch-model and layer-norm fixtures (the layer-norm and attention workplan, stage 5) are
  format-2 files of a Sequential network per implementation: a patch model with the attention and
  FFN blocks, named Attention<class name>, and a dense network with flat layer norms after a
  dropout layer and first in a residual body, named LayerNorm<class name>.
- The pure-Python multiclass presets' fixtures (the presets workplan, stage 1) are format 2 only,
  like the single-output ones: those classes never wrote a legacy envelope.
- So are the numpy and Rust one-output presets' fixtures (the presets workplan, stage 2).

Each fixture is two files in tests/fixtures/saved_models/: <name>.json, the file the class's
own save() wrote, and <name>.expected.json, what the saved network held and predicted:
its snapshot, its hyperparameters, the states it was run on, and its classify_state and predict_*
outputs there, and, for a format-2 fixture, its optimizer state. Floats are float.hex, so the test
compares bits.

The files were written once, from fixed weights (a seeded random.Random per class, independent of
either backend's RNG) and non-default hyperparameters, so a hyperparameter the loader drops shows:

    python -m tests.saved_model_fixtures

Writing refuses to overwrite a fixture: save() no longer writes the legacy envelope, so a rewritten
legacy fixture would no longer pin it, and a rewritten format-2 one would no longer pin the format
as first written. Only a new class gets a new file.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import indrajala_ml.model
from indrajala_ml.model.array_backend import NUMPY, RUST
from indrajala_ml.model.array_network_base import ArrayNetworkBase
from indrajala_ml.model.backprop_network_base import BackpropNetworkBase
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.layer_specs import (
    Attention,
    BatchNorm,
    Dense,
    InputShape,
    LayerNorm,
    LayerSpec,
    Patches,
    Position,
    Residual,
    TokenMean,
    expand_specs,
)
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.sequential_backprop_network import (
    SequentialBackpropClassifierNetwork,
    SequentialMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.update_rules import Adam, Momentum, UpdateRule, WeightDecay
from indrajala_ml.pcg64 import default_rng
from tests.helpers import bits

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "saved_models"

LAYER_SIZES = [4, 3]
DIMENSION = 5
CLASS_COUNT = 3
INPUT_BOUNDS = [(0.0, 1.0)] * DIMENSION
CONV_HEIGHT = CONV_WIDTH = 6
# a conv, an overlapping pool (a non-default stride) and a second conv: every field of both specs
CONV_SPECS: list[ConvSpec | PoolSpec] = [ConvSpec(2, 2), PoolSpec(2, stride=1), ConvSpec(2, 2)]
CONV_DENSE_LAYER_SIZES = [4]
ENSEMBLE_SIZE = 3
STATE_COUNT = 6

MOMENTUM = {"momentum": 0.85}
L2 = {"l2_lambda": 0.02}
ADAM = {"beta1": 0.8, "beta2": 0.99, "epsilon": 1e-7}
DROPOUT = {"drop_probability": 0.25}

# the Sequential networks' fixtures: a conv, an overlapping pool, a ReLU and a dropout layer and a
# softmax output under Adam, and a cross-entropy single output under momentum or weight decay
SEQUENTIAL_INPUT: InputShape = (CONV_HEIGHT, CONV_WIDTH, 1)
SEQUENTIAL_MULTICLASS: list[LayerSpec] = [
    ConvSpec(2, 2),
    PoolSpec(2, stride=1),
    Dense(4, activation="relu"),
    Dense(4, dropout=0.25),
    Dense(CLASS_COUNT, output=True, activation="softmax", loss="cross_entropy"),
]
SEQUENTIAL_SINGLE_OUTPUT: list[LayerSpec] = [Dense(4), Dense(1, output=True, loss="cross_entropy")]
# a conv and a dense batch-norm pair, an overlapping pool between them, and every BatchNorm field
# off its default in the dense pair
BATCH_NORM: list[LayerSpec] = [
    ConvSpec(2, 2, activation="linear"),
    BatchNorm("relu"),
    PoolSpec(2, stride=1),
    Dense(4, activation="linear"),
    BatchNorm("sigmoid", epsilon=1e-4, running_rate=0.2),
    Dense(CLASS_COUNT, output=True, activation="softmax", loss="cross_entropy"),
]
# two residual blocks (a ReLU body, and a batch-norm pair's), each ending in its affine layer, between
# a ReLU layer and a softmax output
RESIDUAL: list[LayerSpec] = [
    Dense(4, activation="relu"),
    Residual((Dense(3, activation="relu"), Dense(4, activation="linear", bias=True))),
    Residual((Dense(3, activation="linear"), BatchNorm("relu"), Dense(4, activation="linear", bias=True))),
    Dense(CLASS_COUNT, output=True, activation="softmax", loss="cross_entropy"),
]
# the README's patch model over the 6x6 image: 4 patches of 3x3, embedded to 4, a position, the
# attention and FFN blocks, the mean, a layer norm (epsilon off its default) and a softmax output
PATCH_MODEL: list[LayerSpec] = [
    Patches(3),
    Dense(4, activation="linear", bias=True),
    Position(),
    Residual((LayerNorm(), Attention())),
    Residual((LayerNorm(), Dense(5, activation="relu"), Dense(4, activation="linear", bias=True))),
    TokenMean(),
    LayerNorm(epsilon=1e-4),
    Dense(CLASS_COUNT, output=True, activation="softmax", loss="cross_entropy"),
]
# flat layer norms: after a dropout layer (epsilon off its default), and first in a residual body
LAYER_NORM: list[LayerSpec] = [
    Dense(4, dropout=0.25),
    LayerNorm(epsilon=1e-4),
    Residual((LayerNorm(), Dense(3, activation="relu"), Dense(4, activation="linear", bias=True))),
    Dense(CLASS_COUNT, output=True, activation="softmax", loss="cross_entropy"),
]
FORMAT_2_TRAINING_STEPS = 2


def _model_classes() -> dict[str, type[Any]]:
    # every class defined in indrajala_ml.model, by name
    classes: dict[str, type[Any]] = {}
    for module_info in pkgutil.iter_modules(indrajala_ml.model.__path__):
        module = importlib.import_module(f"indrajala_ml.model.{module_info.name}")
        for name, value in vars(module).items():
            if isinstance(value, type) and value.__module__ == module.__name__:
                classes[name] = cast("type[Any]", value)
    return classes


MODEL_CLASSES = _model_classes()


@dataclass(frozen=True)
class SavedModelFixture:
    """One saveable class: how to build it unweighted, and what its fixture records."""

    implementation: str  # "python", "numpy" or "rust"
    build: Callable[[], Any]
    predict: str  # predict_probabilities, or predict_probability for a single-output network
    hyperparameters: dict[str, float]
    # written in format 2, after FORMAT_2_TRAINING_STEPS learn_batch steps
    format2: bool = False
    # the class, when the fixture's name isn't its class's (a batch-norm fixture)
    class_name: str | None = None


def _dense(name: str, implementation: str, hyperparameters: dict[str, float] | None = None) -> SavedModelFixture:
    hyperparameters = hyperparameters or {}
    return SavedModelFixture(
        implementation,
        lambda: MODEL_CLASSES[name](LAYER_SIZES, DIMENSION, CLASS_COUNT, **hyperparameters),
        "predict_probabilities",
        hyperparameters,
    )


def _single_output(
    name: str, implementation: str, hyperparameters: dict[str, float] | None = None, format2: bool = False
) -> SavedModelFixture:
    hyperparameters = hyperparameters or {}
    return SavedModelFixture(
        implementation,
        lambda: MODEL_CLASSES[name](LAYER_SIZES, DIMENSION, **hyperparameters),
        "predict_probability",
        hyperparameters,
        format2=format2,
    )


def _conv(
    name: str, implementation: str, hyperparameters: dict[str, float] | None = None, format2: bool = False
) -> SavedModelFixture:
    hyperparameters = hyperparameters or {}
    return SavedModelFixture(
        implementation,
        lambda: MODEL_CLASSES[name](
            CONV_HEIGHT, CONV_WIDTH, CONV_SPECS, CONV_DENSE_LAYER_SIZES, CLASS_COUNT, **hyperparameters
        ),
        "predict_probabilities",
        hyperparameters,
        format2=format2,
    )


def _python_multiclass(
    name: str, hyperparameters: dict[str, float] | None = None, format2: bool = False
) -> SavedModelFixture:
    hyperparameters = hyperparameters or {}
    return SavedModelFixture(
        "python",
        lambda: MODEL_CLASSES[name](LAYER_SIZES, DIMENSION, INPUT_BOUNDS, CLASS_COUNT, **hyperparameters),
        "predict_probabilities",
        hyperparameters,
        format2=format2,
    )


def _python_single_output(name: str, hyperparameters: dict[str, float] | None = None) -> SavedModelFixture:
    hyperparameters = hyperparameters or {}
    return SavedModelFixture(
        "python",
        lambda: MODEL_CLASSES[name](LAYER_SIZES, DIMENSION, INPUT_BOUNDS, **hyperparameters),
        "predict_probability",
        hyperparameters,
        format2=True,
    )


def _sequential(
    implementation: str,
    multiclass: bool,
    rule: UpdateRule,
    class_name: str | None = None,
    architecture: tuple[InputShape, list[LayerSpec]] | None = None,
) -> SavedModelFixture:
    # architecture is a named fixture's (class_name given): its input shape and layers
    def build() -> Any:
        if architecture is not None:
            input_shape, layers = architecture
        elif multiclass:
            input_shape, layers = SEQUENTIAL_INPUT, SEQUENTIAL_MULTICLASS
        else:
            input_shape, layers = (DIMENSION,), SEQUENTIAL_SINGLE_OUTPUT
        if implementation == "python":
            cls = SequentialMultiClassBackpropClassifierNetwork if multiclass else SequentialBackpropClassifierNetwork
            return cls(input_shape, layers, rule)
        backend = NUMPY if implementation == "numpy" else RUST
        return SequentialArrayNetwork(
            input_shape, layers, rule, "multiclass" if multiclass else "single_output", backend
        )

    return SavedModelFixture(
        implementation,
        build,
        "predict_probabilities" if multiclass else "predict_probability",
        {},
        format2=True,
        class_name=class_name,
    )


def _ensemble(name: str, implementation: str, classifier_name: str) -> SavedModelFixture:
    return SavedModelFixture(
        implementation,
        lambda: MODEL_CLASSES[name](
            [MODEL_CLASSES[classifier_name](LAYER_SIZES, DIMENSION, INPUT_BOUNDS) for _ in range(ENSEMBLE_SIZE)]
        ),
        "predict_probabilities",
        {},
    )


# every class with save(), by name
FIXTURES: dict[str, SavedModelFixture] = {
    **{
        f"{prefix}{family}MultiClassBackpropClassifierNetwork": _dense(
            f"{prefix}{family}MultiClassBackpropClassifierNetwork", implementation, hyperparameters
        )
        for family, implementation in (("Vectorized", "numpy"), ("RustArray", "rust"))
        for prefix, hyperparameters in (
            ("", None),
            ("Momentum", MOMENTUM),
            ("L2", L2),
            ("Adam", ADAM),
            ("ReLU", None),
            ("Softmax", None),
            ("Dropout", DROPOUT),
            ("CrossEntropy", None),
        )
    },
    "ArrayBackpropClassifierNetwork": _single_output("ArrayBackpropClassifierNetwork", "numpy"),
    "CrossEntropyArrayBackpropClassifierNetwork": _single_output("CrossEntropyArrayBackpropClassifierNetwork", "numpy"),
    "RustArrayBackpropClassifierNetwork": _single_output("RustArrayBackpropClassifierNetwork", "rust"),
    "CrossEntropyRustArrayBackpropClassifierNetwork": _single_output(
        "CrossEntropyRustArrayBackpropClassifierNetwork", "rust"
    ),
    "ConvVectorizedMultiClassBackpropClassifierNetwork": _conv(
        "ConvVectorizedMultiClassBackpropClassifierNetwork", "numpy"
    ),
    "ConvRustArrayMultiClassBackpropClassifierNetwork": _conv(
        "ConvRustArrayMultiClassBackpropClassifierNetwork", "rust"
    ),
    "ConvMultiClassBackpropClassifierNetwork": _conv("ConvMultiClassBackpropClassifierNetwork", "python"),
    "MomentumConvVectorizedMultiClassBackpropClassifierNetwork": _conv(
        "MomentumConvVectorizedMultiClassBackpropClassifierNetwork", "numpy", MOMENTUM
    ),
    "MomentumConvRustArrayMultiClassBackpropClassifierNetwork": _conv(
        "MomentumConvRustArrayMultiClassBackpropClassifierNetwork", "rust", MOMENTUM
    ),
    "MomentumConvMultiClassBackpropClassifierNetwork": _conv(
        "MomentumConvMultiClassBackpropClassifierNetwork", "python", MOMENTUM
    ),
    "MultiClassBackpropClassifierNetwork": _python_multiclass("MultiClassBackpropClassifierNetwork"),
    "SoftmaxMultiClassBackpropClassifierNetwork": _python_multiclass("SoftmaxMultiClassBackpropClassifierNetwork"),
    "EnsembleArrayBackpropClassifierNetwork": _ensemble(
        "EnsembleArrayBackpropClassifierNetwork", "numpy", "ArrayBackpropClassifierNetwork"
    ),
    "EnsembleRustArrayBackpropClassifierNetwork": _ensemble(
        "EnsembleRustArrayBackpropClassifierNetwork", "rust", "RustArrayBackpropClassifierNetwork"
    ),
    "EnsembleBackpropClassifierNetwork": _ensemble(
        "EnsembleBackpropClassifierNetwork", "python", "BackpropClassifierNetwork"
    ),
    # format 2 only
    **{
        name: _python_single_output(name, hyperparameters)
        for name, hyperparameters in (
            ("BackpropClassifierNetwork", None),
            ("FanInAwareBackpropClassifierNetwork", None),
            ("MomentumBackpropClassifierNetwork", MOMENTUM),
            ("L2RegularizedBackpropClassifierNetwork", L2),
            ("AdamBackpropClassifierNetwork", ADAM),
            ("DropoutBackpropClassifierNetwork", DROPOUT),
            ("ReLUBackpropClassifierNetwork", None),
            ("BinaryCrossEntropyBackpropClassifierNetwork", None),
        )
    },
    # format 2 only
    **{
        f"{prefix}MultiClassBackpropClassifierNetwork": _python_multiclass(
            f"{prefix}MultiClassBackpropClassifierNetwork", hyperparameters, format2=True
        )
        for prefix, hyperparameters in (
            ("CrossEntropy", None),
            ("ReLU", None),
            ("Dropout", DROPOUT),
            ("Momentum", MOMENTUM),
            ("Adam", ADAM),
            ("L2Regularized", L2),
        )
    },
    # format 2 only
    **{
        f"{prefix}{family}BackpropClassifierNetwork": _single_output(
            f"{prefix}{family}BackpropClassifierNetwork", implementation, hyperparameters, format2=True
        )
        for family, implementation in (("Array", "numpy"), ("RustArray", "rust"))
        for prefix, hyperparameters in (
            ("ReLU", None),
            ("Dropout", DROPOUT),
            ("Momentum", MOMENTUM),
            ("Adam", ADAM),
            ("L2", L2),
        )
    },
    # format 2 only
    **{
        name: _conv(name, implementation, hyperparameters, format2=True)
        for name, implementation, hyperparameters in (
            ("AdamConvMultiClassBackpropClassifierNetwork", "python", ADAM),
            ("AdamConvVectorizedMultiClassBackpropClassifierNetwork", "numpy", ADAM),
            ("AdamConvRustArrayMultiClassBackpropClassifierNetwork", "rust", ADAM),
            ("L2RegularizedConvMultiClassBackpropClassifierNetwork", "python", L2),
            ("L2ConvVectorizedMultiClassBackpropClassifierNetwork", "numpy", L2),
            ("L2ConvRustArrayMultiClassBackpropClassifierNetwork", "rust", L2),
            ("ReLUConvMultiClassBackpropClassifierNetwork", "python", None),
            ("ReLUConvVectorizedMultiClassBackpropClassifierNetwork", "numpy", None),
            ("ReLUConvRustArrayMultiClassBackpropClassifierNetwork", "rust", None),
            ("DropoutConvMultiClassBackpropClassifierNetwork", "python", DROPOUT),
            ("DropoutConvVectorizedMultiClassBackpropClassifierNetwork", "numpy", DROPOUT),
            ("DropoutConvRustArrayMultiClassBackpropClassifierNetwork", "rust", DROPOUT),
            ("CrossEntropyConvMultiClassBackpropClassifierNetwork", "python", None),
            ("CrossEntropyConvVectorizedMultiClassBackpropClassifierNetwork", "numpy", None),
            ("CrossEntropyConvRustArrayMultiClassBackpropClassifierNetwork", "rust", None),
            ("SoftmaxConvMultiClassBackpropClassifierNetwork", "python", None),
            ("SoftmaxConvVectorizedMultiClassBackpropClassifierNetwork", "numpy", None),
            ("SoftmaxConvRustArrayMultiClassBackpropClassifierNetwork", "rust", None),
        )
    },
    "SequentialVectorizedMultiClassBackpropClassifierNetwork": _sequential("numpy", True, Adam(**ADAM)),
    "SequentialRustArrayMultiClassBackpropClassifierNetwork": _sequential("rust", True, Adam(**ADAM)),
    "SequentialMultiClassBackpropClassifierNetwork": _sequential("python", True, Adam(**ADAM)),
    "SequentialArrayBackpropClassifierNetwork": _sequential("numpy", False, Momentum(**MOMENTUM)),
    "SequentialRustArrayBackpropClassifierNetwork": _sequential("rust", False, Momentum(**MOMENTUM)),
    "SequentialBackpropClassifierNetwork": _sequential("python", False, WeightDecay(**L2)),
    **{
        f"BatchNorm{name}": _sequential(implementation, True, Adam(**ADAM), name, (SEQUENTIAL_INPUT, BATCH_NORM))
        for name, implementation in (
            ("SequentialVectorizedMultiClassBackpropClassifierNetwork", "numpy"),
            ("SequentialRustArrayMultiClassBackpropClassifierNetwork", "rust"),
            ("SequentialMultiClassBackpropClassifierNetwork", "python"),
        )
    },
    **{
        f"Residual{name}": _sequential(implementation, True, Adam(**ADAM), name, ((DIMENSION,), RESIDUAL))
        for name, implementation in (
            ("SequentialVectorizedMultiClassBackpropClassifierNetwork", "numpy"),
            ("SequentialRustArrayMultiClassBackpropClassifierNetwork", "rust"),
            ("SequentialMultiClassBackpropClassifierNetwork", "python"),
        )
    },
    **{
        f"{prefix}{name}": _sequential(implementation, True, Adam(**ADAM), name, architecture)
        for prefix, architecture in (
            ("Attention", (SEQUENTIAL_INPUT, PATCH_MODEL)),
            ("LayerNorm", ((DIMENSION,), LAYER_NORM)),
        )
        for name, implementation in (
            ("SequentialVectorizedMultiClassBackpropClassifierNetwork", "numpy"),
            ("SequentialRustArrayMultiClassBackpropClassifierNetwork", "rust"),
            ("SequentialMultiClassBackpropClassifierNetwork", "python"),
        )
    },
}


def fixture_class(name: str) -> type[Any]:
    """The class whose save() wrote fixture name, and whose load() reads it."""
    return MODEL_CLASSES[FIXTURES[name].class_name or name]


def from_bits(value: Any) -> Any:
    """bits' inverse: float.hex strings back to floats, in nested lists."""
    if isinstance(value, list):
        return [from_bits(item) for item in cast("list[Any]", value)]
    if isinstance(value, str):
        return float.fromhex(value)
    return value


def optimizer_bits(network: Any) -> list[Any]:
    """network's optimizer state (t, then [layer index, state] per stepped layer), as bits."""
    state = network.optimizer.state()
    return [state.t, [[index, bits(layer)] for index, layer in sorted(state.layers.items())]]


def outputs(network: Any, predict: str, states: list[tuple[float, ...]]) -> dict[str, Any]:
    """network's classify_state and predict_* over states, as bits."""
    return {
        "classify_state": bits([network.classify_state(state) for state in states]),
        predict: bits([getattr(network, predict)(state) for state in states]),
    }


def _random_like(rng: random.Random, value: Any) -> Any:
    if isinstance(value, list):
        return [_random_like(rng, item) for item in cast("list[Any]", value)]
    return rng.uniform(-1.0, 1.0)


def _positive_running_variances(network: Any, snapshot: list[Any]) -> list[Any]:
    # a batch-norm layer's running variance as |drawn|, so inference takes a real square root: the
    # last of [gamma, beta, running_mean, running_var] per layer on numpy and Rust, and per channel
    # in pure Python; the snapshot is per expanded layer, a residual block's body's included
    for spec, entry in zip(expand_specs(network.layer_specs), snapshot, strict=True):
        if isinstance(spec, BatchNorm):
            for values in [entry] if network.implementation != "python" else entry:
                values[3] = [abs(value) for value in values[3]] if isinstance(values[3], list) else abs(values[3])
    return snapshot


def _write(name: str, fixture: SavedModelFixture) -> None:
    model_path = FIXTURE_DIR / f"{name}.json"
    expected_path = FIXTURE_DIR / f"{name}.expected.json"
    if model_path.exists() or expected_path.exists():
        print(f"{name}: exists, not rewritten")
        return

    rng = random.Random(f"saved model fixture {name}")
    network = fixture.build()
    # the snapshot's shape as nested lists (bits), every float redrawn; every restore accepts
    # nested lists: a (weights, bias) per node or kernel in pure Python, a [W, b] per layer on
    # numpy and Rust, [] for a pool layer, one such list per ensemble member
    drawn = _random_like(rng, bits(network.snapshot()))
    network.restore(_positive_running_variances(network, drawn) if fixture.class_name is not None else drawn)
    states = [tuple(rng.uniform(0.0, 1.0) for _ in range(network_dimension(network))) for _ in range(STATE_COUNT)]
    if fixture.format2:
        # a non-empty optimizer state to pin; seeded for the dropout masks, which every network
        # draws from its own generator (the files written before the RNG generators workplan's
        # stage 4 drew pure Python's from random, seeded here then)
        _seed_generator(network, 0)
        labels = [
            rng.randrange(CLASS_COUNT) if fixture.predict == "predict_probabilities" else float(rng.randrange(2))
            for _ in states
        ]
        for _ in range(FORMAT_2_TRAINING_STEPS):
            network.learn_batch(0.1, list(zip(states, labels)))

    network.save(str(model_path))
    expected = {
        "snapshot": bits(network.snapshot()),
        "hyperparameters": fixture.hyperparameters,
        "states": bits(states),
        "outputs": outputs(network, fixture.predict, states),
    }
    if fixture.format2:
        expected["optimizer_state"] = optimizer_bits(network)
    with open(expected_path, "w") as f:
        json.dump(expected, f, indent=1)
        f.write("\n")
    print(f"{name}: written")


def network_dimension(network: Any) -> int:
    # an ensemble's input dimension is its members'
    return network.classifiers[0].dimension if hasattr(network, "classifiers") else network.dimension


def _seed_generator(network: Any, seed: int) -> None:
    # the network's own generator, which its dropout masks draw from
    if isinstance(network, ArrayNetworkBase):
        array_network = cast("ArrayNetworkBase[Any]", network)
        array_network.rng = array_network.backend.default_rng(seed)
    elif isinstance(network, BackpropNetworkBase):
        cast("BackpropNetworkBase[Any]", network).rng = default_rng(seed)


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fixture in FIXTURES.items():
        _write(name, fixture)


if __name__ == "__main__":
    main()
