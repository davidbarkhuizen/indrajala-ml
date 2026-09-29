"""
The legacy save fixtures (docs/composable-layers-workplan.md, stage 0): one small saved file per
network class that has save(), written by the code before format 2, so every legacy envelope is
pinned and keeps loading after the save format changes. tests/test_legacy_saved_models.py loads
them.

Each fixture is two files in tests/fixtures/saved_models/: <class name>.json, the file the class's
own save() wrote, and <class name>.expected.json, what the saved network held and predicted:
its snapshot, its hyperparameters, the states it was run on, and its classify_state and predict_*
outputs there. Floats are float.hex, so the test compares bits.

The files were written once, from fixed weights (a seeded random.Random per class, independent of
either backend's RNG) and non-default hyperparameters, so a hyperparameter the loader drops shows:

    python -m tests.saved_model_fixtures

Writing refuses to overwrite a fixture: after the save format changes, save() no longer writes the
legacy envelope, so a rewritten fixture would no longer pin it. Only a new class gets a new file.
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
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.max_pool_layer import PoolSpec

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


def _dense(name: str, implementation: str, hyperparameters: dict[str, float] | None = None) -> SavedModelFixture:
    hyperparameters = hyperparameters or {}
    return SavedModelFixture(
        implementation,
        lambda: MODEL_CLASSES[name](LAYER_SIZES, DIMENSION, CLASS_COUNT, **hyperparameters),
        "predict_probabilities",
        hyperparameters,
    )


def _single_output(name: str, implementation: str) -> SavedModelFixture:
    return SavedModelFixture(
        implementation, lambda: MODEL_CLASSES[name](LAYER_SIZES, DIMENSION), "predict_probability", {}
    )


def _conv(name: str, implementation: str, hyperparameters: dict[str, float] | None = None) -> SavedModelFixture:
    hyperparameters = hyperparameters or {}
    return SavedModelFixture(
        implementation,
        lambda: MODEL_CLASSES[name](
            CONV_HEIGHT, CONV_WIDTH, CONV_SPECS, CONV_DENSE_LAYER_SIZES, CLASS_COUNT, **hyperparameters
        ),
        "predict_probabilities",
        hyperparameters,
    )


def _python_multiclass(name: str) -> SavedModelFixture:
    return SavedModelFixture(
        "python",
        lambda: MODEL_CLASSES[name](LAYER_SIZES, DIMENSION, INPUT_BOUNDS, CLASS_COUNT),
        "predict_probabilities",
        {},
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
}


def bits(value: Any) -> Any:
    """value (nested lists, tuples and arrays of floats) as nested lists of float.hex strings."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [bits(item) for item in cast("list[Any] | tuple[Any, ...]", value)]
    if isinstance(value, float):
        return float.hex(value)
    return value


def from_bits(value: Any) -> Any:
    """bits' inverse: float.hex strings back to floats, in nested lists."""
    if isinstance(value, list):
        return [from_bits(item) for item in cast("list[Any]", value)]
    if isinstance(value, str):
        return float.fromhex(value)
    return value


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
    network.restore(_random_like(rng, bits(network.snapshot())))
    states = [tuple(rng.uniform(0.0, 1.0) for _ in range(network_dimension(network))) for _ in range(STATE_COUNT)]

    network.save(str(model_path))
    expected = {
        "snapshot": bits(network.snapshot()),
        "hyperparameters": fixture.hyperparameters,
        "states": bits(states),
        "outputs": outputs(network, fixture.predict, states),
    }
    with open(expected_path, "w") as f:
        json.dump(expected, f, indent=1)
        f.write("\n")
    print(f"{name}: written")


def network_dimension(network: Any) -> int:
    # an ensemble's input dimension is its members'
    return network.classifiers[0].dimension if hasattr(network, "classifiers") else network.dimension


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fixture in FIXTURES.items():
        _write(name, fixture)


if __name__ == "__main__":
    main()
