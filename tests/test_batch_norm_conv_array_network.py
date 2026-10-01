"""
Conv batch norm in numpy (the batch-norm workplan, stage 4a): LinearConvArrayLayer and
BatchNormArrayLayer over a conv layer's channels, against tests/test_batch_norm_array_network.py's
scalar transcription of the README's expressions (each channel's values example by example, then
position by position), the layer's one-example refusal (D4), and randomize. The network-level
tests both backends share are tests/test_batch_norm_conv_network.py's, which reuse this module's
cases.
"""

import random
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.array_backend import NUMPY
from indrajala_ml.model.array_layer import FloatArray
from indrajala_ml.model.batch_norm_array_layer import BatchNormArrayLayer
from indrajala_ml.model.conv_array_layer import ConvArrayLayer, LinearConvArrayLayer
from indrajala_ml.model.layer_specs import BatchNorm, Conv, Dense, LayerSpec, Pool
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.update_rules import SGD, UpdateRule
from tests.test_batch_norm_array_network import EPSILON, RATE, SOFTMAX, _Next, _reference

# 3 examples of 2 channels at 4 positions, channel-major: channel c at position p is X[n, 4c + p]
X = np.array(
    [
        [1.0, -2.0, 0.5, 3.0, 0.25, 1.5, -1.0, 2.0],
        [2.0, 0.5, -0.5, 1.0, 4.0, -3.0, 0.0, 1.25],
        [4.0, 3.0, 1.5, -1.0, -2.0, 0.75, 2.5, -0.5],
    ]
)
POSITIONS = 4
GAMMA = np.array([1.5, 0.5])
BETA = np.array([0.25, -1.0])
DOWNSTREAM = np.array(
    [
        [0.3, -0.7, 0.1, 0.2, -0.4, 0.6, 0.05, -0.2],
        [-0.1, 0.2, 0.3, -0.5, 0.25, -0.15, 0.4, 0.1],
        [0.5, 0.4, -0.3, 0.1, 0.2, 0.35, -0.6, 0.45],
    ]
)


def _layer(activation: Any = "relu") -> BatchNormArrayLayer:
    layer = BatchNormArrayLayer(8, activation, EPSILON, RATE, POSITIONS)
    layer.gamma, layer.beta = GAMMA.copy(), BETA.copy()
    layer.set_training_mode(True)
    return layer


def _channel(values: FloatArray, channel: int) -> list[float]:
    """A channel's values in the README's order: example by example, then position by position."""
    return values[:, channel * POSITIONS : (channel + 1) * POSITIONS].reshape(-1).tolist()


@pytest.mark.parametrize("activation", ["sigmoid", "relu"])
def test_every_expression_is_the_readmes_over_each_channel_by_bits(activation: str):
    # sigmoid isn't accepted after a conv layer (validate_layer_specs), but the layer computes it
    layer = _layer(activation)
    A = layer.forward_batch(X)
    layer.compute_hidden_delta_batch(_Next(DOWNSTREAM))
    layer.accumulate_gradient_batch(X)
    dx = layer.downstream_batch()

    for channel in range(2):
        delta = _channel(layer.delta_batch, channel)
        reference = _reference(_channel(X, channel), GAMMA[channel], BETA[channel], delta)
        Y = np.array(reference["y"])
        expected = 1.0 / (1.0 + np.exp(-Y)) if activation == "sigmoid" else np.maximum(0.0, Y)
        assert _channel(A, channel) == expected.tolist()
        assert _channel(dx, channel) == reference["dx"]
        assert layer.grad_gamma[channel] == reference["grad_gamma"]
        assert layer.grad_beta[channel] == reference["grad_beta"]
        assert layer.running_mean[channel] == reference["running_mean"]
        assert layer.running_var[channel] == reference["running_var"]


def test_the_conv_layer_is_the_dense_layer_on_the_rows_of_positions_by_bits():
    # (N, C * P) channel-major, as (N * P, C): a dense layer of C features over N * P rows
    rows = X.reshape(3, 2, POSITIONS).transpose(0, 2, 1).reshape(3 * POSITIONS, 2)
    conv, dense = _layer(), BatchNormArrayLayer(2, "relu", EPSILON, RATE)
    dense.gamma, dense.beta = GAMMA.copy(), BETA.copy()
    dense.set_training_mode(True)

    def flat(values: FloatArray) -> FloatArray:
        return values.reshape(3, POSITIONS, 2).transpose(0, 2, 1).reshape(3, 8)

    assert conv.forward_batch(X).tobytes() == flat(dense.forward_batch(rows)).tobytes()
    conv.compute_hidden_delta_batch(_Next(DOWNSTREAM))
    dense.compute_hidden_delta_batch(_Next(DOWNSTREAM.reshape(3, 2, POSITIONS).transpose(0, 2, 1).reshape(12, 2)))
    conv.accumulate_gradient_batch(X)
    dense.accumulate_gradient_batch(rows)

    assert conv.downstream_batch().tobytes() == flat(dense.downstream_batch()).tobytes()
    for name in ("grad_gamma", "grad_beta", "running_mean", "running_var"):
        assert getattr(conv, name).tobytes() == getattr(dense, name).tobytes()


def test_the_running_variance_is_unbiased_over_every_position():
    layer = _layer()
    layer.forward_batch(X)

    for channel in range(2):
        values = np.array(_channel(X, channel))
        assert len(values) == 12  # m = B * P
        assert layer.running_var[channel] == pytest.approx(0.9 + 0.1 * values.var(ddof=1), rel=1e-14)


def test_inference_normalizes_each_channel_with_its_running_averages():
    layer = _layer()
    layer.running_mean = np.array([0.5, -1.0])
    layer.running_var = np.array([2.0, 0.25])
    layer.set_training_mode(False)

    A = layer.forward_batch(X)
    mean, var = np.repeat(layer.running_mean, POSITIONS), np.repeat(layer.running_var, POSITIONS)
    xhat = (X - mean) / np.sqrt(var + EPSILON)

    assert A.tolist() == np.maximum(0.0, np.repeat(GAMMA, POSITIONS) * xhat + np.repeat(BETA, POSITIONS)).tolist()
    assert layer.forward(X[1]).tolist() == A[1].tolist()


def test_a_layer_refuses_one_example_in_training_although_it_has_several_positions():
    # D4 is per network: a batch of one refuses whatever its m
    layer = _layer()
    with pytest.raises(ValueError, match="D4"):
        layer.forward_batch(X[:1])


def _conv_pair() -> tuple[ConvArrayLayer, LinearConvArrayLayer]:
    conv, linear = ConvArrayLayer(5, 5, 2, 3, 3, 2), LinearConvArrayLayer(5, 5, 2, 3, 3, 2)
    rng = np.random.default_rng(0)
    conv.W = rng.uniform(-1.0, 1.0, conv.W.shape)
    linear.W = conv.W.copy()
    return conv, linear


def test_the_linear_conv_layer_is_the_conv_layers_products_without_bias_or_relu():
    conv, linear = _conv_pair()
    inputs = np.random.default_rng(1).uniform(-1.0, 1.0, (3, 50))
    conv.forward_batch(inputs)

    # conv's b is 0.0, and z + 0.0 is z but for -0.0
    A = linear.forward_batch(inputs)
    assert A.tolist() == conv.Z.tolist()
    assert (A < 0.0).any()
    assert linear.forward(inputs[1]).tolist() == A[1].tolist()
    assert linear.parameters() == (linear.W,) and linear.decayed == (True,)
    assert not hasattr(linear, "b")


def test_the_linear_conv_layer_passes_the_downstream_through_to_conv_s_col2im_and_grad_w():
    conv, linear = _conv_pair()
    inputs = np.random.default_rng(1).uniform(-1.0, 1.0, (3, 50))
    downstream = np.random.default_rng(2).uniform(-1.0, 1.0, (3, linear.size))
    conv.forward_batch(inputs)
    linear.forward_batch(inputs)

    linear.compute_hidden_delta_batch(_Next(downstream))
    linear.accumulate_gradient_batch(inputs)
    conv.delta_batch = downstream
    conv.accumulate_gradient_batch(inputs)

    assert linear.delta_batch is downstream
    assert linear.downstream_batch().tobytes() == conv.downstream_batch().tobytes()
    assert linear.grad_W.tobytes() == conv.grad_W.tobytes()


# networks

INPUT = (6, 6, 1)
LINEAR_CONV = Conv(3, 2, activation="linear")
NETWORKS: dict[str, list[LayerSpec]] = {
    "conv": [LINEAR_CONV, BatchNorm("relu"), SOFTMAX],
    "conv pool": [LINEAR_CONV, BatchNorm("relu"), Pool(2), Dense(3, output=True)],
    "after a relu conv": [Conv(2, 2), Conv(2, 3, stride=2, activation="linear"), BatchNorm("relu"), SOFTMAX],
    "two conv pairs": [LINEAR_CONV, BatchNorm("relu"), Conv(2, 3, activation="linear"), BatchNorm("relu"), SOFTMAX],
    "conv and dense pairs": [LINEAR_CONV, BatchNorm("relu"), Dense(4, activation="linear"), BatchNorm(), SOFTMAX],
}


def _network(name: str = "conv pool", rule: UpdateRule | None = None, seed: int = 3, backend: Any = NUMPY) -> Any:
    network = SequentialArrayNetwork(INPUT, NETWORKS[name], SGD() if rule is None else rule, backend=backend)
    network.rng = backend.default_rng(seed)
    network.randomize()
    return network


# the conv test modules' rows
def _rows(count: int, seed: int = 1) -> list[tuple[tuple[float, ...], int]]:  # pyright: ignore[reportUnusedFunction]
    rng = random.Random(seed)
    return [(tuple(rng.random() for _ in range(36)), i % 3) for i in range(count)]


def test_randomize_draws_the_linear_conv_layers_w_only_and_nothing_for_batch_norm():
    network = _network()
    linear, norm, _pool, output = network.layers
    rng = np.random.default_rng(3)
    W_linear = rng.uniform(-1 / np.sqrt(9), 1 / np.sqrt(9), (2, 9))
    W_output = rng.uniform(-1 / np.sqrt(8), 1 / np.sqrt(8), (3, 8))

    assert linear.W.tobytes() == W_linear.tobytes() and output.W.tobytes() == W_output.tobytes()
    assert norm.gamma.tolist() == [1.0] * 2 and norm.beta.tolist() == [0.0] * 2
    assert norm.running_mean.tolist() == [0.0] * 2 and norm.running_var.tolist() == [1.0] * 2
