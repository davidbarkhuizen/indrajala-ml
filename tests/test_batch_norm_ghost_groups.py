"""
Ghost batch norm (the batch-norm workplan, stage 6, D6): BatchNorm(group_size=...) in all three
implementations. One group is plain batch norm by bits; groups of 2 in a batch of 4 by hand; each
group, the remainder one included, is a batch of its own by bits, the running averages moving once
per group in row order; the pure-Python and Rust layers are numpy's by bits; gradient checks; and
the refusals of a group_size under 2 and of a last group of one example, by the layer, the network
and train.py before training.
"""

import math
import random
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.array_backend import NUMPY, RUST
from indrajala_ml.model.base_node import AbstractNode
from indrajala_ml.model.batch_norm_array_layer import BatchNormArrayLayer, sum_rows
from indrajala_ml.model.batch_norm_layer import BatchNormLayer
from indrajala_ml.model.batch_norm_rust_array_layer import BatchNormRustArrayLayer
from indrajala_ml.model.layer_specs import BatchNorm, Conv, Dense, LayerSpec, Pool, ghost_groups
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.sequential_backprop_network import SequentialMultiClassBackpropClassifierNetwork
from indrajala_ml.model.update_rules import SGD, Adam
from indrajala_ml.train import train_backprop_network_mini_batch
from tests.gradient_check import check_gradients
from tests.test_batch_norm_python_network import math_exp  # noqa: F401  # pyright: ignore[reportUnusedImport]
from tests.test_batch_norm_rust_network import crate_exp  # noqa: F401  # pyright: ignore[reportUnusedImport]

EPSILON = 1e-5
RATE = 0.1
IMPLEMENTATIONS = ["python", "numpy", "rust"]

# (features or channels, positions, batch, group_size): groups that divide the batch, a remainder
# group, and a conv layer's channels over 4 positions
CASES = [(3, 1, 4, 2), (3, 1, 7, 4), (2, 1, 8, 3), (2, 4, 6, 4), (2, 4, 5, 3)]
# with each activation: only ReLU follows a conv layer
ACTIVATION_CASES = [
    (activation, *case) for case in CASES for activation in ("sigmoid", "relu") if case[1] == 1 or activation == "relu"
]


class _Next:
    """A next layer that sends a fixed downstream."""

    def __init__(self, downstream: Any) -> None:
        self._downstream = downstream

    def downstream_batch(self) -> Any:
        return self._downstream


def _numpy_layer(
    features: int, positions: int, group_size: int | None, activation: Any = "relu", seed: int = 0
) -> BatchNormArrayLayer:
    rng = np.random.default_rng(seed)
    layer = BatchNormArrayLayer(features * positions, activation, EPSILON, RATE, positions, group_size)
    layer.gamma, layer.beta = rng.uniform(0.5, 1.5, features), rng.uniform(-0.5, 0.5, features)
    layer.running_mean, layer.running_var = rng.uniform(-1.0, 1.0, features), rng.uniform(0.5, 2.0, features)
    layer.set_training_mode(True)
    return layer


def _inputs(features: int, positions: int, batch: int, seed: int = 1) -> tuple[Any, Any]:
    rng = np.random.default_rng(seed)
    size = features * positions
    return rng.uniform(-3.0, 3.0, (batch, size)), rng.uniform(-1.0, 1.0, (batch, size))


def _step(layer: Any, X: Any, downstream: Any) -> dict[str, Any]:
    # a numpy layer's training step: every value it computes
    A = layer.forward_batch(X)
    layer.compute_hidden_delta_batch(_Next(downstream))
    layer.accumulate_gradient_batch(X)
    dx = layer.downstream_batch()
    return {
        "a": A,
        "dx": dx,
        "grad_gamma": layer.grad_gamma,
        "grad_beta": layer.grad_beta,
        "running_mean": layer.running_mean,
        "running_var": layer.running_var,
    }


def _bits(values: dict[str, Any]) -> dict[str, list[bytes]]:
    # by bits, so -0.0 and 0.0 differ
    return {
        name: [v.tobytes() for v in np.ravel(np.asarray(value, dtype=np.float64))] for name, value in values.items()
    }


# the layers


@pytest.mark.usefixtures("math_exp")
@pytest.mark.parametrize("activation, features, positions, batch", [case[:4] for case in ACTIVATION_CASES])
@pytest.mark.parametrize("extra", [0, 1, 32])
def test_one_group_is_plain_batch_norm_by_bits(activation: str, features: int, positions: int, batch: int, extra: int):
    X, downstream = _inputs(features, positions, batch)
    plain = _step(_numpy_layer(features, positions, None, activation), X, downstream)
    grouped = _step(_numpy_layer(features, positions, batch + extra, activation), X, downstream)
    assert _bits(grouped) == _bits(plain)


def test_groups_of_2_in_a_batch_of_4_by_hand():
    layer = BatchNormArrayLayer(2, "relu", EPSILON, RATE, group_size=2)
    layer.set_training_mode(True)
    X = np.array([[1.0, -2.0], [3.0, 0.5], [2.0, 3.0], [6.0, 1.0]])
    layer.forward_batch(X)

    # feature 0: (1, 3) has mean 2 and biased variance 1, (2, 6) mean 4 and variance 4; feature 1:
    # (-2, 0.5) has mean -0.75 and variance 1.5625, (3, 1) mean 2 and variance 1
    xhat = [
        [-1 / np.sqrt(1 + EPSILON), -1.25 / np.sqrt(1.5625 + EPSILON)],
        [1 / np.sqrt(1 + EPSILON), 1.25 / np.sqrt(1.5625 + EPSILON)],
        [-2 / np.sqrt(4 + EPSILON), 1 / np.sqrt(1 + EPSILON)],
        [2 / np.sqrt(4 + EPSILON), -1 / np.sqrt(1 + EPSILON)],
    ]
    np.testing.assert_allclose(layer._xhat, xhat, rtol=1e-14)
    # the running averages move a tenth of the way to each group's mean and unbiased variance
    # (twice the biased one for 2 values), the first group first
    np.testing.assert_allclose(layer.running_mean, [0.9 * 0.1 * 2 + 0.1 * 4, 0.9 * 0.1 * -0.75 + 0.1 * 2], rtol=1e-14)
    np.testing.assert_allclose(
        layer.running_var, [0.9 * (0.9 + 0.1 * 2) + 0.1 * 8, 0.9 * (0.9 + 0.1 * 3.125) + 0.1 * 2], rtol=1e-14
    )

    # each group's gradient into the linear layer sums to 0 over the group, not only over the batch
    layer.compute_hidden_delta_batch(_Next(np.array([[0.3, -0.7], [-0.1, 0.2], [0.5, 0.4], [0.2, 0.9]])))
    dx = layer.downstream_batch()
    np.testing.assert_allclose(dx[:2].sum(axis=0), [0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(dx[2:].sum(axis=0), [0.0, 0.0], atol=1e-15)


@pytest.mark.parametrize("features, positions, batch, group_size", CASES)
def test_each_group_is_a_batch_of_its_own_by_bits(features: int, positions: int, batch: int, group_size: int):
    # the remainder group too: one plain layer run on each group in turn, carrying the running
    # averages, computes the same activations, dl/dx and running averages. The gradients of gamma
    # and beta are one fold over the whole batch
    X, downstream = _inputs(features, positions, batch)
    grouped = _numpy_layer(features, positions, group_size)
    actual = _step(grouped, X, downstream)

    plain = _numpy_layer(features, positions, None)
    parts: list[dict[str, Any]] = []
    for first, end in ghost_groups(batch, group_size):
        plain.reset_gradient_accum()
        parts.append(_step(plain, X[first:end], downstream[first:end]))
    expected = {
        "a": np.concatenate([part["a"] for part in parts]),
        "dx": np.concatenate([part["dx"] for part in parts]),
        "running_mean": plain.running_mean,
        "running_var": plain.running_var,
    }
    assert _bits({name: actual[name] for name in expected}) == _bits(expected)
    delta = grouped._rows(grouped.delta_batch)
    assert _bits({"g": actual["grad_gamma"], "b": actual["grad_beta"]}) == _bits(
        {"g": sum_rows(delta * grouped._xhat), "b": sum_rows(delta)}
    )


class _Input(AbstractNode):
    def value(self) -> float:
        return 0.0


class _InputLayer:
    def __init__(self, size: int) -> None:
        self.nodes = [_Input() for _ in range(size)]


@pytest.mark.usefixtures("math_exp")
@pytest.mark.parametrize("activation, features, positions, batch, group_size", ACTIVATION_CASES)
def test_the_python_layer_is_numpys_by_bits(
    activation: Any, features: int, positions: int, batch: int, group_size: int
):
    X, downstream = _inputs(features, positions, batch)
    array = _numpy_layer(features, positions, group_size, activation)
    python = BatchNormLayer(_InputLayer(features * positions), activation, EPSILON, RATE, positions, group_size)
    for c, node in enumerate(python.channels):
        node.gamma, node.beta = float(array.gamma[c]), float(array.beta[c])
        node.running_mean, node.running_var = float(array.running_mean[c]), float(array.running_var[c])
    python.set_training_mode(True)
    expected = _step(array, X, downstream)

    python.forward_batch(X.tolist())
    python.backward_batch(downstream.tolist())
    python.accumulate_gradients()

    def channel_major(lists: list[list[float]]) -> np.ndarray:
        # per channel, (example, position) order, as the batch's channel-major rows
        return np.array(
            [[lists[c][e * positions + p] for c in range(features) for p in range(positions)] for e in range(batch)]
        )

    actual = {
        "a": channel_major([node.activations for node in python.channels]),
        "dx": channel_major([node.dxs for node in python.channels]),
        "grad_gamma": [node.weight_gradient_accum[0] for node in python.channels],
        "grad_beta": [node.bias_gradient_accum for node in python.channels],
        "running_mean": [node.running_mean for node in python.channels],
        "running_var": [node.running_var for node in python.channels],
    }
    assert _bits(actual) == _bits(expected)


@pytest.mark.usefixtures("crate_exp")
@pytest.mark.parametrize("activation, features, positions, batch, group_size", ACTIVATION_CASES)
def test_the_rust_layer_is_numpys_by_bits(activation: Any, features: int, positions: int, batch: int, group_size: int):
    X, downstream = _inputs(features, positions, batch)
    array = _numpy_layer(features, positions, group_size, activation)
    rust = BatchNormRustArrayLayer(features * positions, activation, EPSILON, RATE, positions, group_size)
    rust.gamma, rust.beta, rust.running_mean, rust.running_var = (
        RUST.vector(values.tolist()) for values in (array.gamma, array.beta, array.running_mean, array.running_var)
    )
    rust.set_training_mode(True)
    expected = _step(array, X, downstream)

    A = rust.forward_batch(RUST.matrix(X.tolist()))
    rust.delta_batch = RUST.matrix(array.delta_batch.tolist())  # the numpy layer's dl/dy
    rust.accumulate_gradient_batch(RUST.matrix(X.tolist()))
    dx = rust.downstream_batch()
    actual = {
        "a": A.tolist(),
        "dx": dx.tolist(),
        "grad_gamma": rust.grad_gamma.tolist(),
        "grad_beta": rust.grad_beta.tolist(),
        "running_mean": rust.running_mean.tolist(),
        "running_var": rust.running_var.tolist(),
    }
    assert _bits(actual) == _bits(expected)


@pytest.mark.parametrize("layer_class", ["python", "numpy", "rust"])
def test_a_layer_refuses_a_last_group_of_one(layer_class: str):
    X, _downstream = _inputs(2, 1, 5)
    if layer_class == "python":
        layer: Any = BatchNormLayer(_InputLayer(2), "relu", EPSILON, RATE, group_size=2)
        batch: Any = X.tolist()
    elif layer_class == "numpy":
        layer, batch = BatchNormArrayLayer(2, "relu", EPSILON, RATE, group_size=2), X
    else:
        layer, batch = BatchNormRustArrayLayer(2, "relu", EPSILON, RATE, group_size=2), RUST.matrix(X.tolist())
    layer.set_training_mode(True)
    with pytest.raises(ValueError, match="a batch of 5 in groups of 2 leaves a last group of one example"):
        layer.forward_batch(batch)


# the networks

DENSE: list[LayerSpec] = [Dense(5, activation="linear"), BatchNorm(group_size=2), Dense(3, output=True)]
CONV: list[LayerSpec] = [
    Conv(3, 2, activation="linear"),
    BatchNorm("relu", group_size=2),
    Pool(2),
    Dense(3, output=True),
]
ARCHITECTURES = {"dense": ((4,), DENSE), "conv": ((6, 6, 1), CONV)}


def _network(implementation: str, input_shape: Any, layers: list[LayerSpec], rule: Any = None, seed: int = 3) -> Any:
    rule = SGD() if rule is None else rule
    if implementation == "python":
        network: Any = SequentialMultiClassBackpropClassifierNetwork(input_shape, layers, rule)
        random.seed(seed)
    else:
        backend = NUMPY if implementation == "numpy" else RUST
        network = SequentialArrayNetwork(input_shape, layers, rule, shape="multiclass", backend=backend)
        backend.seed(seed)
    network.randomize()
    return network


def _rows(input_shape: Any, count: int, seed: int = 1) -> list[tuple[tuple[float, ...], int]]:
    rng = random.Random(seed)
    size = math.prod(input_shape)
    return [(tuple(rng.random() for _ in range(size)), i % 3) for i in range(count)]


def _snapshot_bits(network: Any) -> list[bytes]:
    def leaves(tree: Any) -> list[float]:
        if isinstance(tree, list | tuple):
            return [leaf for item in tree for leaf in leaves(item)]  # pyright: ignore[reportUnknownVariableType]
        if hasattr(tree, "tolist"):
            return leaves(tree.tolist())
        return [tree]

    return [np.float64(leaf).tobytes() for leaf in leaves(network.snapshot())]


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_a_network_with_one_group_trains_as_plain_batch_norm_by_bits(implementation: str, architecture: str):
    input_shape, layers = ARCHITECTURES[architecture]
    plain_layers = [BatchNorm(spec.activation) if isinstance(spec, BatchNorm) else spec for spec in layers]
    one_group = [BatchNorm(spec.activation, group_size=6) if isinstance(spec, BatchNorm) else spec for spec in layers]
    plain = _network(implementation, input_shape, plain_layers, Adam())
    grouped = _network(implementation, input_shape, one_group, Adam())
    rows = _rows(input_shape, 18)
    for step in range(3):
        plain.learn_batch(0.1, rows[step * 6 : (step + 1) * 6])
        grouped.learn_batch(0.1, rows[step * 6 : (step + 1) * 6])
    assert _snapshot_bits(grouped) == _snapshot_bits(plain)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("batch_size", [4, 5])
def test_every_gradient_matches_its_finite_difference(implementation: str, architecture: str, batch_size: int):
    # a batch of 4 in groups of 2, and of 5 in groups of 3, the remainder 2
    input_shape, layers = ARCHITECTURES[architecture]
    if batch_size == 5:
        layers = [BatchNorm(spec.activation, group_size=3) if isinstance(spec, BatchNorm) else spec for spec in layers]
    network = _network(implementation, input_shape, layers)
    network.learn_batch(0.5, _rows(input_shape, 6, seed=2))
    rows = _rows(input_shape, batch_size)
    check_gradients(network, [state for state, _ in rows], [label for _, label in rows])


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_network_refuses_a_last_group_of_one_naming_the_layer(implementation: str):
    network = _network(implementation, (4,), DENSE)
    before = _snapshot_bits(network)
    with pytest.raises(ValueError, match=r"layer 1, BatchNorm\(.*group_size=2\): a batch of 5 in groups of 2"):
        network.learn_batch(0.1, _rows((4,), 5))
    assert _snapshot_bits(network) == before


@pytest.mark.parametrize("examples, batch_size", [(9, 4), (12, 5)])
def test_train_refuses_a_batch_size_that_leaves_a_group_of_one_before_training(examples: int, batch_size: int):
    # the full batches leave one: 4 in groups of 3 (the final batch of 1 is dropped, D4), and 5 in
    # groups of 2
    layers: list[LayerSpec] = [
        Dense(5, activation="linear"),
        BatchNorm(group_size=3 if batch_size == 4 else 2),
        Dense(3, output=True),
    ]
    network = _network("numpy", (4,), layers)
    before = _snapshot_bits(network)
    with pytest.raises(ValueError, match="leaves a last group of one example"):
        train_backprop_network_mini_batch(network, _rows((4,), examples), batch_size)
    assert _snapshot_bits(network) == before


def test_train_refuses_a_final_short_batch_that_leaves_a_group_of_one():
    # batches of 4 in groups of 2 are fine; the final batch of 11 % 4 = 3 would be 2, 1
    network = _network("numpy", (4,), DENSE)
    with pytest.raises(ValueError, match="a batch of 3 in groups of 2"):
        train_backprop_network_mini_batch(network, _rows((4,), 11), 4)


def test_train_trains_in_ghost_groups():
    # batches of 4 in groups of 2, and a final batch of 10 % 4 = 2, one group: nothing refused
    # (train's pocket may restore the untrained start, so the parameters aren't compared)
    network = _network("numpy", (4,), DENSE)
    train_backprop_network_mini_batch(network, _rows((4,), 10), 4, epochs=2)


@pytest.mark.parametrize("group_size", [0, 1, -2])
def test_a_group_size_under_2_is_refused(group_size: int):
    with pytest.raises(AssertionError, match="group_size must be 2 or more"):
        _network(
            "numpy", (4,), [Dense(5, activation="linear"), BatchNorm(group_size=group_size), Dense(3, output=True)]
        )
