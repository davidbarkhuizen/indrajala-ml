"""
The gradient check (gradient_check.py) against today's layers, in all three implementations: dense
sigmoid and ReLU hidden layers, the sigmoid output under the squared and cross-entropy losses, the
softmax output, conv and pool (the batch-norm workplan, stage 0). That these pass, and that a wrong
loss fails, shows the check itself works before a batch-norm layer relies on it.

Dropout isn't checked: its masks are drawn in every training forward pass, so the perturbed
passes wouldn't drop the same nodes.
"""

import math
import random
from typing import Any

import pytest

from indrajala_ml.model.array_backend import NUMPY, RUST
from indrajala_ml.model.layer_specs import Conv, Dense, InputShape, LayerSpec, Pool
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.sequential_backprop_network import (
    SequentialBackpropClassifierNetwork,
    SequentialMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.update_rules import SGD
from tests.gradient_check import GradientMismatch, check_gradients, squared_loss

SOFTMAX = Dense(3, output=True, activation="softmax", loss="cross_entropy")

NETWORKS: dict[str, tuple[InputShape, list[LayerSpec]]] = {
    "sigmoid": ((4,), [Dense(5), Dense(3, output=True)]),
    "two_hidden": ((4,), [Dense(5), Dense(4), Dense(3, output=True)]),
    "relu_cross_entropy": ((4,), [Dense(5, activation="relu"), Dense(3, output=True, loss="cross_entropy")]),
    "softmax": ((4,), [Dense(5), SOFTMAX]),
    "single_output": ((4,), [Dense(5), Dense(1, output=True)]),
    "single_output_cross_entropy": ((4,), [Dense(5), Dense(1, output=True, loss="cross_entropy")]),
    "conv": ((5, 5, 1), [Conv(3, 2), Dense(4), Dense(3, output=True)]),
    "conv_pool": ((6, 6, 2), [Conv(3, 2), Pool(2), Dense(5, activation="relu"), SOFTMAX]),
}

IMPLEMENTATIONS = ["python", "numpy", "rust"]
BATCH_SIZE = 4


def _output_size(layers: list[LayerSpec]) -> int:
    output = layers[-1]
    assert isinstance(output, Dense)
    return output.size


def _network(implementation: str, input_shape: InputShape, layers: list[LayerSpec]) -> Any:
    single_output = _output_size(layers) == 1
    if implementation == "python":
        cls = SequentialBackpropClassifierNetwork if single_output else SequentialMultiClassBackpropClassifierNetwork
        network = cls(input_shape, layers, SGD())
        random.seed(3)
    else:
        backend = NUMPY if implementation == "numpy" else RUST
        shape = "single_output" if single_output else "multiclass"
        network = SequentialArrayNetwork(input_shape, layers, SGD(), shape=shape, backend=backend)
        network.rng = backend.default_rng(3)
    network.randomize()
    return network


def _batch(input_shape: InputShape, output_size: int) -> tuple[list[tuple[float, ...]], list[Any]]:
    rng = random.Random(1)
    states = [tuple(rng.random() for _ in range(math.prod(input_shape))) for _ in range(BATCH_SIZE)]
    labels: list[Any] = (
        [float(rng.random() < 0.5) for _ in range(BATCH_SIZE)]
        if output_size == 1
        else [rng.randrange(output_size) for _ in range(BATCH_SIZE)]
    )
    return states, labels


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("name", NETWORKS)
def test_every_gradient_matches_its_finite_difference(name: str, implementation: str):
    input_shape, layers = NETWORKS[name]
    network = _network(implementation, input_shape, layers)
    states, labels = _batch(input_shape, _output_size(layers))

    comparisons = check_gradients(network, states, labels)

    # every weight and bias of every weighted layer, pool layers having none
    assert len(comparisons) == sum(len(weights) + len(biases) for weights, biases in _weight_sets(network))


def _weight_sets(network: Any) -> list[tuple[list[float], list[float]]]:
    # per weighted layer, its weights and biases flattened
    if isinstance(network, SequentialBackpropClassifierNetwork | SequentialMultiClassBackpropClassifierNetwork):
        return [
            ([w for weights, _ in entry for w in weights], [bias for _, bias in entry])
            for entry in network.snapshot()
            if entry
        ]
    return [([w for row in entry[0].tolist() for w in row], entry[1].tolist()) for entry in network.snapshot() if entry]


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_it_fails_against_the_wrong_loss(implementation: str):
    # the softmax layer's delta is the cross-entropy loss's gradient, not the squared loss's
    input_shape, layers = NETWORKS["softmax"]
    network = _network(implementation, input_shape, layers)
    states, labels = _batch(input_shape, 3)

    with pytest.raises(GradientMismatch, match="weights mismatch"):
        check_gradients(network, states, labels, loss=squared_loss)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_it_leaves_the_network_as_it_was(implementation: str):
    input_shape, layers = NETWORKS["conv_pool"]
    network = _network(implementation, input_shape, layers)
    states, labels = _batch(input_shape, 3)
    before = _weight_sets(network)
    optimizer = network.optimizer

    check_gradients(network, states, labels)

    assert _weight_sets(network) == before
    assert network.optimizer is optimizer and optimizer.t == 0
