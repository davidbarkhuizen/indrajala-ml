"""
AdamW (the training-recipe workplan, D9): Adam(weight_decay=λ), decoupled weight decay on the
weights only, w * (1 - lr * λ) - lr * m_hat / (sqrt(v_hat) + epsilon), in all three
implementations; Adam() keeps its bits.
"""

import math
from random import Random
from typing import Any

import pytest

from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.networks.python.sequential_backprop_network import SequentialBackpropClassifierNetwork
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.persistence.format2_json import rule_from_json, rule_to_json
from indrajala_ml.model.specs.layer_specs import Dense, LayerNorm, LayerSpec, Residual
from indrajala_ml.model.specs.update_rules import Adam
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
RATE = 0.05


def _student(implementation: str, rule: Adam) -> Any:
    if implementation == "python":
        network: Any = SequentialBackpropClassifierNetwork((2,), LAYERS, rule, square_bounds(10.0))
        network.rng = default_rng(2)
    else:
        backend = NUMPY if implementation == "numpy" else RUST
        network = SequentialArrayNetwork((2,), LAYERS, rule, shape="single_output", backend=backend)
        network.rng = backend.default_rng(2)
    network.randomize()
    return network


def _data(count: int) -> list[tuple[tuple[float, ...], float]]:
    _reference, data = reachable_reference_and_training_data(
        1, 2, square_bounds(10.0), count, rng=default_rng(1), data_rng=Random(1)
    )
    return data


@pytest.mark.parametrize("implementation", ["numpy", "rust"])
def test_one_adamw_step_shrinks_the_weights_alone_by_the_rate_times_the_decay(implementation: str):
    # from the same weights and gradients, AdamW's step is Adam's less lr * λ * w on every decayed
    # parameter (the layers' W), and Adam's by bits on the rest (biases, gamma, beta)
    adam, adamw = _student(implementation, Adam()), _student(implementation, Adam(weight_decay=0.1))
    trained = [i for i, layer in enumerate(adam.layers) if hasattr(layer, "parameters")]  # not a fork or join
    before = [[parameter.tolist() for parameter in adam.layers[i].parameters()] for i in trained]
    adam.learn_batch(RATE, _data(8))
    adamw.learn_batch(RATE, _data(8))

    decayed_seen = False
    for i, start in zip(trained, before):
        layer, decayed_layer = adam.layers[i], adamw.layers[i]
        for parameter, decayed_parameter, initial, decayed in zip(
            layer.parameters(), decayed_layer.parameters(), start, layer.decayed
        ):
            stepped, decayed_step = parameter.tolist(), decayed_parameter.tolist()
            if not decayed:
                assert bits(decayed_step) == bits(stepped)
                continue
            for row, decayed_row, initial_row in zip(stepped, decayed_step, initial):
                expected = [value - RATE * 0.1 * w for value, w in zip(row, initial_row)]
                assert decayed_row == pytest.approx(expected, rel=1e-12, abs=1e-15)
            decayed_seen = True
    assert decayed_seen


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_adam_with_no_decay_is_adams_bits(implementation: str):
    adam, explicit = _student(implementation, Adam()), _student(implementation, Adam(weight_decay=0.0))
    for _ in range(3):
        adam.learn_batch(RATE, _data(8))
        explicit.learn_batch(RATE, _data(8))
    assert bits(explicit.snapshot()) == bits(adam.snapshot())


def test_the_three_implementations_decay_alike():
    data = _data(24)
    outputs: dict[str, list[float]] = {}
    for implementation in IMPLEMENTATIONS:
        student = _student(implementation, Adam(weight_decay=0.1))
        train_backprop_network_mini_batch(student, data, 5, learning_rate=RATE, epochs=3, rng=Random(7))
        # the networks' outputs: the implementations nest their parameters differently
        outputs[implementation] = [float(student.predict_probability(state)) for state, _ in data]
    undecayed = _student("numpy", Adam())
    train_backprop_network_mini_batch(undecayed, data, 5, learning_rate=RATE, epochs=3, rng=Random(7))

    assert outputs["numpy"] == pytest.approx(outputs["python"], rel=1e-9, abs=1e-12)
    assert outputs["rust"] == pytest.approx(outputs["python"], rel=1e-9, abs=1e-12)
    assert outputs["numpy"] != [float(undecayed.predict_probability(state)) for state, _ in data]


def test_a_decayed_adam_round_trips_and_an_undecayed_one_writes_as_before():
    assert rule_to_json(Adam()) == {"rule": "adam", "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-08}
    assert rule_from_json(rule_to_json(Adam())) == Adam()
    decayed = Adam(weight_decay=0.1)
    assert rule_to_json(decayed)["weight_decay"] == 0.1
    assert rule_from_json(rule_to_json(decayed)) == decayed


@pytest.mark.parametrize("weight_decay", [-0.1, math.inf, math.nan])
def test_a_weight_decay_that_cannot_train_is_refused(weight_decay: float):
    with pytest.raises(ValueError, match="weight_decay"):
        Adam(weight_decay=weight_decay)
