"""
Clipping by the global norm (the training-recipe workplan, D10): the batch gradient's L2 norm over
every parameter, clipped to clip_norm before the update rule reads it, in all three
implementations.
"""

import math
from random import Random
from typing import Any

import pytest

from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.networks.python.sequential_backprop_network import (
    SequentialBackpropClassifierNetwork,
)
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.layer_specs import Dense, LayerNorm, LayerSpec, Residual
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from indrajala_ml.pcg64 import default_rng
from indrajala_ml.training.train import train_backprop_network_mini_batch
from indrajala_ml.training.training_data import reachable_reference_and_training_data
from tests.helpers import bits

IMPLEMENTATIONS = ["python", "numpy", "rust"]
LAYERS: list[LayerSpec] = [
    Dense(5, activation="relu"),
    Residual((LayerNorm(), Dense(5, activation="linear", bias=True))),
    Dense(1, output=True),
]
RULES: list[UpdateRule] = [SGD(), Momentum(0.9), Adam(), WeightDecay(0.01)]


def _student(implementation: str, rule: UpdateRule) -> Any:
    if implementation == "python":
        network: Any = SequentialBackpropClassifierNetwork((2,), LAYERS, rule, square_bounds(10.0))
        network.rng = default_rng(2)
    else:
        backend = NUMPY if implementation == "numpy" else RUST
        network = SequentialArrayNetwork((2,), LAYERS, rule, shape="single_output", backend=backend)
        network.rng = backend.default_rng(2)
    network.randomize()
    return network


def _batch() -> list[tuple[tuple[float, ...], float]]:
    _reference, data = reachable_reference_and_training_data(
        1, 2, square_bounds(10.0), 8, rng=default_rng(1), data_rng=Random(1)
    )
    return data


def _flat(value: Any) -> list[float]:
    # every float of a snapshot, in order, whatever its implementation's nesting
    if isinstance(value, float | int):
        return [float(value)]
    if hasattr(value, "tolist"):
        return _flat(value.tolist())
    return [x for item in value for x in _flat(item)]


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_clipped_sgd_step_moves_the_parameters_by_the_rate_times_clip_norm(implementation: str):
    # under SGD a step is rate * g: clipped, its global norm over the rate is clip_norm
    student = _student(implementation, SGD())
    before = _flat(student.snapshot())
    student.learn_batch(0.5, _batch(), clip_norm=1e-3)
    after = _flat(student.snapshot())

    step_norm = math.sqrt(sum((a - b) ** 2 for a, b in zip(after, before))) / 0.5
    assert step_norm == pytest.approx(1e-3, rel=1e-9)


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_clip_above_the_norm_leaves_every_bit(implementation: str, rule: UpdateRule):
    unclipped, clipped = _student(implementation, rule), _student(implementation, rule)
    for _ in range(3):
        unclipped.learn_batch(0.05, _batch())
        clipped.learn_batch(0.05, _batch(), clip_norm=1e9)
    assert bits(clipped.snapshot()) == bits(unclipped.snapshot())


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
def test_the_three_implementations_clip_alike(rule: UpdateRule):
    # a clip well under the batch gradient's norm, so every step clips, through the trainer
    trained: dict[str, list[float]] = {}
    for implementation in IMPLEMENTATIONS:
        student = _student(implementation, rule)
        _reference, data = reachable_reference_and_training_data(
            1, 2, square_bounds(10.0), 24, rng=default_rng(1), data_rng=Random(1)
        )
        train_backprop_network_mini_batch(student, data, 5, learning_rate=0.05, epochs=3, rng=Random(7), clip_norm=1e-2)
        # the networks' outputs: the implementations nest their parameters differently
        trained[implementation] = [float(student.predict_probability(state)) for state, _ in data]

    assert trained["numpy"] == pytest.approx(trained["python"], rel=1e-9, abs=1e-12)
    assert trained["rust"] == pytest.approx(trained["python"], rel=1e-9, abs=1e-12)


def test_the_trainer_clips_through_the_student():
    clipped, unclipped = _student("numpy", Adam()), _student("numpy", Adam())
    _reference, data = reachable_reference_and_training_data(
        1, 2, square_bounds(10.0), 24, rng=default_rng(1), data_rng=Random(1)
    )
    train_backprop_network_mini_batch(clipped, data, 5, learning_rate=0.05, epochs=1, rng=Random(7), clip_norm=1e-3)
    train_backprop_network_mini_batch(unclipped, data, 5, learning_rate=0.05, epochs=1, rng=Random(7))
    assert bits(clipped.snapshot()) != bits(unclipped.snapshot())


@pytest.mark.parametrize("clip_norm", [0.0, -1.0, math.inf, math.nan])
def test_a_clip_norm_that_cannot_clip_is_refused(clip_norm: float):
    with pytest.raises(ValueError, match="clip_norm"):
        train_backprop_network_mini_batch(_student("numpy", Adam()), _batch(), 4, clip_norm=clip_norm)
