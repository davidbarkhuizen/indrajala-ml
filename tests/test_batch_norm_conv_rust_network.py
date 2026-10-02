"""
Conv batch norm in Rust (the batch-norm workplan, stage 4c): LinearConvRustArrayLayer and
BatchNormRustArrayLayer over a conv layer's channels against numpy's stage 4a layers, randomize,
and parity with numpy's networks. The network-level tests both backends share are
tests/test_batch_norm_conv_network.py's. Reuses tests/test_batch_norm_conv_array_network.py's cases.
"""

import random
from typing import Any

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.layers.numpy.batch_norm_array_layer import BatchNormArrayLayer
from indrajala_ml.model.layers.numpy.conv_array_layer import LinearConvArrayLayer
from indrajala_ml.model.layers.rust.batch_norm_rust_array_layer import BatchNormRustArrayLayer
from indrajala_ml.model.layers.rust.conv_rust_array_layer import ConvRustArrayLayer, LinearConvRustArrayLayer
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.layer_specs import Conv, Dense, LayerSpec, Pool
from indrajala_ml.model.specs.update_rules import Momentum, UpdateRule
from indrajala_ml.train import train_backprop_network_mini_batch
from tests.helpers import bits, max_relative_gap, to_numpy
from tests.test_batch_norm_array_network import EPSILON, RATE, RULES, SOFTMAX
from tests.test_batch_norm_conv_array_network import INPUT, NETWORKS, _network, _rows
from tests.test_batch_norm_rust_network import _rust


class _Fixed:
    """A next layer whose downstream is given, in the layer's backend: after a conv pair the next
    layer may be a pool or conv layer, with no W, so both backends read its downstream."""

    def __init__(self, downstream: Any, rust: bool) -> None:
        self.downstream = _rust(downstream) if rust else downstream

    def downstream_batch(self) -> Any:
        return self.downstream


# the layers


@pytest.mark.parametrize("batch_size, channels, positions", [(2, 1, 4), (3, 2, 9), (5, 3, 16), (4, 2, 1)])
def test_the_layer_is_numpys_by_bits(batch_size: int, channels: int, positions: int):
    # the same inputs, parameters and next layer: every value batch norm computes is the same in
    # both, since both follow the README's expressions and fold order over each channel's values
    rng = np.random.default_rng(batch_size * positions)
    size = channels * positions
    X = rng.uniform(-3.0, 3.0, (batch_size, size))
    gamma, beta = rng.uniform(0.5, 2.0, channels), rng.uniform(-1.0, 1.0, channels)
    downstream = rng.uniform(-1.0, 1.0, (batch_size, size))
    array = BatchNormArrayLayer(size, "relu", EPSILON, RATE, positions)
    rust = BatchNormRustArrayLayer(size, "relu", EPSILON, RATE, positions)
    array.gamma, array.beta = gamma.copy(), beta.copy()
    rust.gamma, rust.beta = _rust(gamma), _rust(beta)

    values: list[list[Any]] = []
    layers: list[tuple[Any, Any, bool]] = [(array, X, False), (rust, _rust(X), True)]
    for layer, inputs, is_rust in layers:
        layer.set_training_mode(True)
        activations = layer.forward_batch(inputs)
        layer.compute_hidden_delta_batch(_Fixed(downstream, is_rust))
        dx = layer.downstream_batch()
        layer.accumulate_gradient_batch(inputs)
        values.append([activations, dx, layer.grad_gamma, layer.grad_beta, layer.running_mean, layer.running_var])
    assert bits(values[1]) == bits(values[0])

    # the deltas agree in value, but not in the sign of a ReLU's zero (numpy's downstream * 0.0 is
    # -0.0 where the downstream is negative, array_relu_mask's 0.0), as for dense batch norm
    np.testing.assert_array_equal(to_numpy(rust.delta_batch), array.delta_batch)


def test_inference_is_numpys_by_bits_for_a_batch_and_one_example():
    rng = np.random.default_rng(5)
    X = rng.uniform(-3.0, 3.0, (6, 2 * 9))
    array = BatchNormArrayLayer(18, "relu", EPSILON, RATE, 9)
    rust = BatchNormRustArrayLayer(18, "relu", EPSILON, RATE, 9)
    for name in ("gamma", "beta", "running_mean", "running_var"):
        values = rng.uniform(0.5, 2.0, 2)
        setattr(array, name, values)
        setattr(rust, name, _rust(values))

    assert to_numpy(rust.forward_batch(_rust(X))).tobytes() == array.forward_batch(X).tobytes()
    for row in X:
        assert to_numpy(rust.forward(_rust(row))).tobytes() == array.forward(row).tobytes()
    assert bits(rust.running_state()) == bits(array.running_state())  # inference doesn't move them


def test_a_layer_refuses_one_example_in_training_although_it_has_several_positions():
    layer = BatchNormRustArrayLayer(8, "relu", EPSILON, RATE, 4)
    layer.set_training_mode(True)
    with pytest.raises(ValueError, match="D4"):
        layer.forward_batch(pa.Array([[1.0] * 8]))


def test_only_relu_follows_a_conv_layer():
    with pytest.raises(AssertionError, match="only ReLU"):
        BatchNormRustArrayLayer(8, "sigmoid", EPSILON, RATE, 4)


def _conv_layers() -> tuple[ConvRustArrayLayer, LinearConvRustArrayLayer, LinearConvArrayLayer]:
    conv, linear = ConvRustArrayLayer(5, 5, 2, 3, 3, 2), LinearConvRustArrayLayer(5, 5, 2, 3, 3, 2)
    array = LinearConvArrayLayer(5, 5, 2, 3, 3, 2)
    W = np.random.default_rng(0).uniform(-1.0, 1.0, (3, 18))
    conv.W, linear.W, array.W = _rust(W), _rust(W), W
    return conv, linear, array


def test_the_linear_conv_layer_is_the_conv_layers_products_without_bias_or_relu():
    conv, linear, array = _conv_layers()
    inputs = np.random.default_rng(1).uniform(-1.0, 1.0, (3, 50))
    A_conv = to_numpy(conv.forward_batch(_rust(inputs)))

    # conv's b is 0.0, so its A is the linear layer's products through the ReLU, exactly
    A = to_numpy(linear.forward_batch(_rust(inputs)))
    assert A_conv.tolist() == np.maximum(0.0, A).tolist()
    assert (A < 0.0).any()
    assert to_numpy(linear.forward(_rust(inputs[1]))).tolist() == A[1].tolist()
    assert linear.parameters() == (linear.W,) and linear.decayed == (True,)
    assert not hasattr(linear, "b")

    # numpy's products are BLAS, the crate's its own tiled kernel: the same within rounding
    np.testing.assert_allclose(A, array.forward_batch(inputs), rtol=1e-13, atol=1e-15)


def test_the_linear_conv_layer_passes_the_downstream_through_to_conv_s_col2im_and_grad_w():
    conv, linear, array = _conv_layers()
    inputs = np.random.default_rng(1).uniform(-1.0, 1.0, (3, 50))
    downstream = np.random.default_rng(2).uniform(-1.0, 1.0, (3, linear.size))
    conv.forward_batch(_rust(inputs))
    linear.forward_batch(_rust(inputs))
    array.forward_batch(inputs)

    linear.compute_hidden_delta_batch(_Fixed(downstream, rust=True))
    linear.accumulate_gradient_batch(_rust(inputs))
    conv.delta_batch = _rust(downstream)
    conv.accumulate_gradient_batch(_rust(inputs))
    array.compute_hidden_delta_batch(_Fixed(downstream, rust=False))
    array.accumulate_gradient_batch(inputs)

    assert to_numpy(linear.delta_batch).tobytes() == downstream.tobytes()
    assert to_numpy(linear.downstream_batch()).tobytes() == to_numpy(conv.downstream_batch()).tobytes()
    assert to_numpy(linear.grad_W).tobytes() == to_numpy(conv.grad_W).tobytes()
    np.testing.assert_allclose(to_numpy(linear.downstream_batch()), array.downstream_batch(), rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(to_numpy(linear.grad_W), array.grad_W, rtol=1e-13, atol=1e-15)

    linear.reset_gradient_accum()
    assert to_numpy(linear.grad_W).tolist() == np.zeros((3, 18)).tolist()
    for call in (lambda: linear.compute_hidden_delta(None), lambda: linear.downstream()):
        with pytest.raises(ValueError, match="LinearConvRustArrayLayer trains on batches only"):
            call()


# networks


@pytest.mark.parametrize("name", NETWORKS)
def test_randomize_draws_numpys_weights(name: str):
    # the crate's RNG is numpy's np.random: the same linear conv W, and batch norm draws nothing
    assert bits(_network(name, backend=RUST).snapshot()) == bits(_network(name).snapshot())


# parity with numpy


def _gap_after_training(layers: list[LayerSpec], rule: UpdateRule) -> float:
    networks: list[Any] = []
    for backend in (NUMPY, RUST):
        network = SequentialArrayNetwork(INPUT, layers, rule, backend=backend)
        network.rng = backend.default_rng(3)
        network.randomize()
        networks.append(network)
    rows = _rows(40)
    for step in range(50):
        batch = rows[(step * 5) % 40 :][:5]
        for network in networks:
            network.learn_batch(0.3, batch)
    return max_relative_gap(networks[0].snapshot(), networks[1].snapshot())


# no batch norm, the same shapes: the conv and dense layers' own gap
CONTROLS: dict[str, list[LayerSpec]] = {
    "conv": [Conv(3, 2), SOFTMAX],
    "conv pool": [Conv(3, 2), Pool(2), Dense(3, output=True)],
    "two convs": [Conv(3, 2), Conv(2, 3), SOFTMAX],
}


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", NETWORKS)
def test_training_matches_numpy_within_the_conv_and_dense_layers_rounding(name: str, rule: UpdateRule):
    # batch norm computes the same bits in both (test_the_layer_is_numpys_by_bits); the conv and
    # dense products don't (BLAS in numpy, the crate's kernels in Rust), and Adam's bias
    # corrections differ (tests/test_batch_norm_rust_network.py). 50 steps in, the networks were
    # at most 3.5e-12 apart relative when measured, and the conv networks without batch norm
    # (CONTROLS) 3.4e-13: normalizing subtracts the mean, and d = x - mu carries its inputs'
    # rounding difference as a larger relative one, as for dense batch norm (4.4e-12 vs 1.2e-12)
    assert _gap_after_training(NETWORKS[name], rule) < 1e-10


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", CONTROLS)
def test_the_conv_layers_alone_have_the_same_kind_of_gap(name: str, rule: UpdateRule):
    assert _gap_after_training(CONTROLS[name], rule) < 1e-10


def test_train_trains_a_conv_batch_norm_network_to_its_numpy_counterparts_accuracy():
    # train.py, prepared rows and a final batch of one dropped (D4), on both backends: the same
    # classifications after a few epochs
    rows = _rows(47)
    classified: list[list[int]] = []
    for backend in (NUMPY, RUST):
        network = _network("conv pool", Momentum(0.9), backend=backend)
        train_backprop_network_mini_batch(network, network.prepare_dataset(rows), 5, epochs=3, rng=random.Random(0))
        classified.append(network.classify_rows(network.prepare_dataset(rows)))
    assert classified[1] == classified[0]
