"""
Dense batch norm in numpy (the batch-norm workplan, stage 1): LinearArrayLayer and
BatchNormArrayLayer against hand-computed values and a scalar transcription of the README's
expressions (Batch normalization), the layer's one-example refusal (D4), randomize, a training step
on gamma and beta, and train.py's final batch of one. The network-level tests both
backends share are tests/test_batch_norm_dense_network.py's; they, the Rust ones and the
pure-Python counterparts (tests/test_batch_norm_python_network.py) reuse this module's cases.
"""

import math
import random
from typing import Any, Literal

import numpy as np
import pytest

from indrajala_ml.model.array_backend import NUMPY
from indrajala_ml.model.array_layer import FloatArray, sigmoid
from indrajala_ml.model.batch_norm_array_layer import BatchNormArrayLayer
from indrajala_ml.model.layer_specs import BatchNorm, Dense, LayerSpec
from indrajala_ml.model.linear_array_layer import LinearArrayLayer
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from indrajala_ml.train import _chunk_into_batches, train_backprop_network_mini_batch

EPSILON = 1e-5
RATE = 0.1

# 3 examples of 2 features
X = np.array([[1.0, -2.0], [2.0, 0.5], [4.0, 3.0]])
GAMMA = np.array([1.5, 0.5])
BETA = np.array([0.25, -1.0])


def _layer(activation: Any = "sigmoid") -> BatchNormArrayLayer:
    layer = BatchNormArrayLayer(2, activation, EPSILON, RATE)
    layer.gamma, layer.beta = GAMMA.copy(), BETA.copy()
    layer.set_training_mode(True)
    return layer


def _sum(values: list[float]) -> float:
    total = 0.0
    for value in values:
        total += value
    return total


def _reference(x: list[float], gamma: float, beta: float, delta: list[float]) -> dict[str, Any]:
    """The README's expressions for one feature, in Python floats, in their grouping."""
    m = len(x)
    mu = _sum(x) / m
    d = [x_i - mu for x_i in x]
    ss = _sum([d_i * d_i for d_i in d])
    var = ss / m
    std = math.sqrt(var + EPSILON)
    xhat = [d_i / std for d_i in d]
    y = [gamma * xhat_i + beta for xhat_i in xhat]

    inv_std = 1 / std
    inv_std3 = inv_std / (var + EPSILON)
    dxhat = [delta_i * gamma for delta_i in delta]
    dvar = _sum([dxhat_i * d_i * -0.5 * inv_std3 for dxhat_i, d_i in zip(dxhat, d)])
    dmu = _sum([dxhat_i * -inv_std for dxhat_i in dxhat]) + dvar * _sum([-2 * d_i for d_i in d]) / m
    dx = [dxhat_i * inv_std + dvar * (2 * d_i) / m + dmu / m for dxhat_i, d_i in zip(dxhat, d)]
    return {
        "mu": mu,
        "var": var,
        "xhat": xhat,
        "y": y,
        "dx": dx,
        "grad_gamma": _sum([delta_i * xhat_i for delta_i, xhat_i in zip(delta, xhat)]),
        "grad_beta": _sum(delta),
        "running_mean": (1 - RATE) * 0.0 + RATE * mu,
        "running_var": (1 - RATE) * 1.0 + RATE * (ss / (m - 1)),
    }


class _Next:
    """A next layer that sends a fixed downstream."""

    def __init__(self, downstream: FloatArray) -> None:
        self._downstream = downstream

    def downstream_batch(self) -> FloatArray:
        return self._downstream


def test_the_forward_pass_normalizes_each_feature_over_the_batch_by_hand():
    layer = _layer()
    A = layer.forward_batch(X)

    # feature 0: 1, 2, 4 has mean 7/3 and biased variance 14/9; feature 1: -2, 0.5, 3 has mean
    # 1/2 and variance 25/6
    for feature, (mean, variance) in enumerate([(7 / 3, 14 / 9), (1 / 2, 25 / 6)]):
        xhat = (X[:, feature] - mean) / math.sqrt(variance + EPSILON)
        assert layer._xhat[:, feature] == pytest.approx(xhat, rel=1e-14)
        assert A[:, feature] == pytest.approx(sigmoid(GAMMA[feature] * xhat + BETA[feature]), rel=1e-14)
    # the running averages move a tenth of the way to the batch's mean and unbiased variance
    assert layer.running_mean == pytest.approx([0.1 * 7 / 3, 0.1 * 1 / 2], rel=1e-14)
    assert layer.running_var == pytest.approx([0.9 + 0.1 * 1.5 * 14 / 9, 0.9 + 0.1 * 1.5 * 25 / 6], rel=1e-14)


@pytest.mark.parametrize("activation", ["sigmoid", "relu"])
def test_every_expression_is_the_readmes_by_bits(activation: str):
    layer = _layer(activation)
    A = layer.forward_batch(X)
    downstream = np.array([[0.3, -0.7], [-0.1, 0.2], [0.5, 0.4]])
    layer.compute_hidden_delta_batch(_Next(downstream))
    layer.accumulate_gradient_batch(X)
    dx = layer.downstream_batch()

    for feature in range(2):
        delta = layer.delta_batch[:, feature].tolist()
        reference = _reference(X[:, feature].tolist(), GAMMA[feature], BETA[feature], delta)
        assert layer._xhat[:, feature].tolist() == reference["xhat"]
        Y = np.array(reference["y"])
        assert A[:, feature].tolist() == (sigmoid(Y) if activation == "sigmoid" else np.maximum(0.0, Y)).tolist()
        assert dx[:, feature].tolist() == reference["dx"]
        assert layer.grad_gamma[feature] == reference["grad_gamma"]
        assert layer.grad_beta[feature] == reference["grad_beta"]
        assert layer.running_mean[feature] == reference["running_mean"]
        assert layer.running_var[feature] == reference["running_var"]


def test_the_backward_pass_is_the_simplified_chain_rule_by_hand():
    # dl/dx = (m * dxhat - sum(dxhat) - xhat * sum(dxhat * xhat)) / (m * std), the paper's chain
    # rule simplified: a different grouping, so equal within rounding only
    layer = _layer("relu")
    layer.forward_batch(X)
    layer.compute_hidden_delta_batch(_Next(np.array([[0.3, -0.7], [-0.1, 0.2], [0.5, 0.4]])))
    dx = layer.downstream_batch()

    dxhat = layer.delta_batch * layer.gamma
    xhat, m = layer._xhat, 3
    ((_m, _d, _var, std),) = layer._stats
    simplified = (m * dxhat - dxhat.sum(axis=0) - xhat * (dxhat * xhat).sum(axis=0)) / (m * std)
    assert dx == pytest.approx(simplified, rel=1e-12, abs=1e-15)
    # so the gradient into the linear layer sums to 0 over the batch
    assert dx.sum(axis=0) == pytest.approx([0.0, 0.0], abs=1e-15)


def test_a_relu_delta_is_zero_where_the_activation_is():
    layer = _layer("relu")
    A = layer.forward_batch(X)
    layer.compute_hidden_delta_batch(_Next(np.ones((3, 2))))

    assert (A > 0).any() and (A == 0).any()
    assert layer.delta_batch.tolist() == (A > 0).astype(np.float64).tolist()


def test_inference_normalizes_with_the_running_averages():
    layer = _layer()
    layer.running_mean = np.array([0.5, -1.0])
    layer.running_var = np.array([2.0, 0.25])
    layer.set_training_mode(False)

    A = layer.forward_batch(X)
    xhat = (X - layer.running_mean) / np.sqrt(layer.running_var + EPSILON)

    assert A.tolist() == sigmoid(GAMMA * xhat + BETA).tolist()
    assert layer.forward(X[1]).tolist() == A[1].tolist()
    assert layer.running_mean.tolist() == [0.5, -1.0] and layer.running_var.tolist() == [2.0, 0.25]


def test_a_layer_refuses_one_example_in_training():
    layer = _layer()
    with pytest.raises(ValueError, match="D4"):
        layer.forward_batch(X[:1])
    with pytest.raises(ValueError, match="D4"):
        layer.forward(X[0])


def test_the_linear_layer_has_no_bias_and_passes_the_downstream_through():
    layer = LinearArrayLayer(2, 3)
    layer.W = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]])
    inputs = np.array([[1.0, 0.0, 2.0], [0.5, 1.0, -1.0]])

    assert layer.forward_batch(inputs).tolist() == [[7.0, -1.0], [-0.5, 0.0]]
    downstream = np.array([[0.1, 0.2], [0.3, -0.4]])
    layer.compute_hidden_delta_batch(_Next(downstream))
    layer.accumulate_gradient_batch(inputs)

    assert layer.delta_batch is downstream
    assert layer.grad_W.tolist() == (downstream.T @ inputs).tolist()
    assert layer.parameters() == (layer.W,) and layer.decayed == (True,)


# networks

INPUT = (4,)
SOFTMAX = Dense(3, output=True, activation="softmax", loss="cross_entropy")
Shape = Literal["multiclass", "single_output"]
NETWORKS: dict[str, tuple[list[LayerSpec], Shape]] = {
    "sigmoid": ([Dense(5, activation="linear"), BatchNorm(), Dense(3, output=True)], "multiclass"),
    "relu softmax": ([Dense(5, activation="linear"), BatchNorm("relu"), SOFTMAX], "multiclass"),
    "two pairs": (
        [Dense(5, activation="linear"), BatchNorm(), Dense(4, activation="linear"), BatchNorm("relu"), SOFTMAX],
        "multiclass",
    ),
    "after a sigmoid layer": ([Dense(6), Dense(5, activation="linear"), BatchNorm("relu"), SOFTMAX], "multiclass"),
    "single output": (
        [Dense(5, activation="linear"), BatchNorm(), Dense(1, output=True, loss="cross_entropy")],
        "single_output",
    ),
}
RULES: list[UpdateRule] = [SGD(), Momentum(0.9), Adam(), WeightDecay(0.01)]


def _network(name: str = "sigmoid", rule: UpdateRule | None = None, seed: int = 3, backend: Any = NUMPY) -> Any:
    layers, shape = NETWORKS[name]
    network = SequentialArrayNetwork(INPUT, layers, SGD() if rule is None else rule, shape=shape, backend=backend)
    network.rng = backend.default_rng(seed)
    network.randomize()
    return network


def _rows(count: int, shape: str = "multiclass", seed: int = 1) -> list[tuple[tuple[float, ...], Any]]:
    rng = random.Random(seed)
    return [
        (tuple(rng.random() for _ in range(4)), float(rng.random() < 0.5) if shape == "single_output" else i % 3)
        for i in range(count)
    ]


def test_randomize_draws_the_linear_layers_w_only_and_nothing_for_batch_norm():
    network = _network()
    linear, norm, output = network.layers
    rng = np.random.default_rng(3)
    W_linear = rng.uniform(-1 / np.sqrt(4), 1 / np.sqrt(4), (5, 4))
    W_output = rng.uniform(-1 / np.sqrt(5), 1 / np.sqrt(5), (3, 5))

    assert linear.W.tobytes() == W_linear.tobytes() and output.W.tobytes() == W_output.tobytes()
    assert norm.gamma.tolist() == [1.0] * 5 and norm.beta.tolist() == [0.0] * 5
    assert norm.running_mean.tolist() == [0.0] * 5 and norm.running_var.tolist() == [1.0] * 5


def test_a_training_step_is_the_rules_step_on_gamma_and_beta():
    network = _network(rule=Momentum(0.9))
    norm = network.layers[1]
    rows = _rows(6)
    network.learn_batch(0.5, rows)
    gamma, beta = norm.gamma.copy(), norm.beta.copy()
    velocity_gamma, velocity_beta = network.optimizer.state().layers[1]

    network.learn_batch(0.5, rows)

    # grad_gamma and grad_beta are reset by the step, so recover the second step's velocities
    new_velocity_gamma, new_velocity_beta = network.optimizer.state().layers[1]
    assert norm.gamma.tolist() == (gamma - 0.5 * new_velocity_gamma).tolist()
    assert norm.beta.tolist() == (beta - 0.5 * new_velocity_beta).tolist()
    assert new_velocity_gamma.tolist() != velocity_gamma.tolist()
    assert new_velocity_beta.tolist() != velocity_beta.tolist()


def test_train_drops_a_final_batch_of_one_for_batch_norm_only():
    assert [len(batch) for batch in _chunk_into_batches(list(range(7)), 3, drop_single=True)] == [3, 3]
    assert [len(batch) for batch in _chunk_into_batches(list(range(8)), 3, drop_single=True)] == [3, 3, 2]
    with pytest.raises(AssertionError, match="batches of 2"):
        _chunk_into_batches(list(range(7)), 1, drop_single=True)


@pytest.mark.parametrize("prepared", [False, True], ids=["tuples", "prepared"])
def test_train_trains_a_batch_norm_network_on_a_set_that_leaves_one_over(prepared: bool):
    network = _network()
    rows = _rows(7)
    sizes: list[int] = []
    learn_batch, learn_batch_rows = network.learn_batch, network.learn_batch_rows

    def record(learning_rate: float, batch: Any) -> None:
        sizes.append(len(batch))
        learn_batch(learning_rate, batch)

    def record_rows(learning_rate: float, data: Any, indices: Any) -> None:
        sizes.append(len(indices))
        learn_batch_rows(learning_rate, data, indices)

    network.learn_batch, network.learn_batch_rows = record, record_rows
    train_backprop_network_mini_batch(
        network, network.prepare_dataset(rows) if prepared else rows, 3, epochs=2, rng=random.Random(0)
    )

    assert sizes == [3, 3, 3, 3]
