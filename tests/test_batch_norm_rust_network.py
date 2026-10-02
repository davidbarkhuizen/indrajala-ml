"""
Dense batch norm in Rust (the batch-norm workplan, stage 3): BatchNormRustArrayLayer and
LinearRustArrayLayer against numpy's layers by bits, the optimizer's pairs (a linear layer's W, and
gamma and beta) against NumpyOptimizer by bits, randomize, and parity with numpy's networks. The
network-level tests both backends share are tests/test_batch_norm_dense_network.py's. Reuses
tests/test_batch_norm_array_network.py's cases.
"""

import random
from typing import Any

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model import batch_norm_array_layer
from indrajala_ml.model.array_backend import NUMPY, RUST
from indrajala_ml.model.batch_norm_array_layer import BatchNormArrayLayer
from indrajala_ml.model.batch_norm_rust_array_layer import BatchNormRustArrayLayer
from indrajala_ml.model.linear_array_layer import LinearArrayLayer
from indrajala_ml.model.linear_rust_array_layer import LinearRustArrayLayer
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.layer_specs import Dense
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from indrajala_ml.train import train_backprop_network_mini_batch
from tests.helpers import bits, exp_by_crate, max_relative_gap, patching, sigmoid_by, to_numpy
from tests.test_batch_norm_array_network import EPSILON, INPUT, NETWORKS, RATE, RULES, _network, _rows


def _rust(values: Any) -> pa.Array:
    return pa.Array(np.asarray(values).tolist())


# the layers


# the numpy layer's sigmoid with the crate's exp
crate_exp = patching(batch_norm_array_layer, "sigmoid", sigmoid_by(exp_by_crate))


class _Next:
    """A next dense layer: the Rust layer's fused sigmoid hidden delta reads its W and delta_batch,
    and a ReLU's and numpy's its downstream, delta_batch @ W, taken here from the crate so both see
    the same values, as a Rust array for the Rust layer."""

    def __init__(self, W: Any, delta_batch: Any, rust: bool = False) -> None:
        self.W, self.delta_batch = _rust(W), _rust(delta_batch)
        self.rust = rust

    def downstream_batch(self) -> Any:
        downstream = pa.layer_downstream_batch(self.W, self.delta_batch)
        return downstream if self.rust else to_numpy(downstream)

    @staticmethod
    def fixed(downstream: pa.Array) -> Any:
        # a next layer whose downstream is given
        class Fixed:
            def downstream_batch(self) -> pa.Array:
                return downstream

        return Fixed()


@pytest.mark.usefixtures("crate_exp")
@pytest.mark.parametrize("activation", ["sigmoid", "relu"])
@pytest.mark.parametrize("batch_size", [2, 3, 8, 33])
def test_the_layer_is_numpys_by_bits(activation: Any, batch_size: int):
    # the same inputs, parameters and next layer: every value batch norm computes is the same in
    # both, since both follow the README's expressions and fold order
    rng = np.random.default_rng(batch_size)
    X = rng.uniform(-3.0, 3.0, (batch_size, 4))
    gamma, beta = rng.uniform(0.5, 2.0, 4), rng.uniform(-1.0, 1.0, 4)
    next_W, next_delta = rng.uniform(-1.0, 1.0, (3, 4)), rng.uniform(-1.0, 1.0, (batch_size, 3))
    array = BatchNormArrayLayer(4, activation, EPSILON, RATE)
    rust = BatchNormRustArrayLayer(4, activation, EPSILON, RATE)
    array.gamma, array.beta = gamma.copy(), beta.copy()
    rust.gamma, rust.beta = _rust(gamma), _rust(beta)

    values: list[list[Any]] = []
    layers: list[tuple[Any, Any, bool]] = [(array, X, False), (rust, _rust(X), True)]
    for layer, inputs, is_rust in layers:
        layer.set_training_mode(True)
        activations = layer.forward_batch(inputs)
        layer.compute_hidden_delta_batch(_Next(next_W, next_delta, is_rust))
        dx = layer.downstream_batch()
        layer.accumulate_gradient_batch(inputs)
        values.append([activations, dx, layer.grad_gamma, layer.grad_beta, layer.running_mean, layer.running_var])
    assert bits(values[1]) == bits(values[0])

    # the deltas agree in value, but not in the sign of a ReLU's zero: numpy's is the downstream
    # times 0.0, -0.0 where the downstream is negative, and the crate's is 0.0, as between
    # ReLUArrayLayer and ReLURustArrayLayer. Neither changes a sum that has a nonzero term
    np.testing.assert_array_equal(to_numpy(rust.delta_batch), array.delta_batch)


@pytest.mark.usefixtures("crate_exp")
@pytest.mark.parametrize("activation", ["sigmoid", "relu"])
def test_inference_is_numpys_by_bits_for_a_batch_and_one_example(activation: Any):
    rng = np.random.default_rng(5)
    X = rng.uniform(-3.0, 3.0, (6, 4))
    array = BatchNormArrayLayer(4, activation, EPSILON, RATE)
    rust = BatchNormRustArrayLayer(4, activation, EPSILON, RATE)
    for name in ("gamma", "beta", "running_mean", "running_var"):
        values = rng.uniform(0.5, 2.0, 4)
        setattr(array, name, values)
        setattr(rust, name, _rust(values))

    assert to_numpy(rust.forward_batch(_rust(X))).tobytes() == array.forward_batch(X).tobytes()
    for row in X:
        assert to_numpy(rust.forward(_rust(row))).tobytes() == array.forward(row).tobytes()
    assert bits(rust.running_state()) == bits(array.running_state())  # inference doesn't move them


def test_a_layer_refuses_one_example_in_training():
    layer = BatchNormRustArrayLayer(2, "sigmoid", EPSILON, RATE)
    layer.set_training_mode(True)
    for call in (
        lambda: layer.forward_batch(pa.Array([[1.0, 2.0]])),
        lambda: layer.forward(pa.Array([1.0, 2.0])),
        lambda: layer.accumulate_gradient(pa.Array([1.0, 2.0])),
    ):
        with pytest.raises(ValueError, match="BatchNormRustArrayLayer trains on batches only"):
            call()


def test_the_linear_layer_has_no_bias_and_its_delta_is_the_downstream():
    rng = np.random.default_rng(1)
    X, W = rng.uniform(-1.0, 1.0, (3, 4)), rng.uniform(-1.0, 1.0, (2, 4))
    layer = LinearRustArrayLayer(2, 4)
    layer.W = _rust(W)
    downstream = _rust(rng.uniform(-1.0, 1.0, (3, 2)))

    layer.forward_batch(_rust(X))
    layer.compute_hidden_delta_batch(_Next.fixed(downstream))
    layer.accumulate_gradient_batch(_rust(X))

    assert layer.parameters() == (layer.W,) and layer.decayed == (True,)
    assert to_numpy(layer.delta_batch).tobytes() == to_numpy(downstream).tobytes()
    expected, _grad_b = pa.layer_accumulate_gradient_batch(
        downstream, _rust(X), pa.Array.zeros((2, 4)), pa.Array.zeros(2)
    )
    assert to_numpy(layer.grad_W).tobytes() == to_numpy(expected).tobytes()
    with pytest.raises(ValueError, match="LinearRustArrayLayer trains on batches only"):
        layer.compute_hidden_delta(None)


# the optimizer: a linear layer's W alone, and gamma and beta, in the fused ops' pairs


def _optimizer_layers(rng: Any) -> tuple[list[Any], list[Any]]:
    # the same linear and batch-norm layers, with gradients, on both backends
    linear, norm = LinearArrayLayer(3, 4), BatchNormArrayLayer(3, "sigmoid", EPSILON, RATE)
    linear.W, linear.grad_W = rng.uniform(-1.0, 1.0, (3, 4)), rng.uniform(-1.0, 1.0, (3, 4))
    norm.gamma, norm.beta = rng.uniform(0.5, 2.0, 3), rng.uniform(-1.0, 1.0, 3)
    norm.grad_gamma, norm.grad_beta = rng.uniform(-1.0, 1.0, 3), rng.uniform(-1.0, 1.0, 3)
    rust_linear, rust_norm = LinearRustArrayLayer(3, 4), BatchNormRustArrayLayer(3, "sigmoid", EPSILON, RATE)
    rust_linear.W, rust_linear.grad_W = _rust(linear.W), _rust(linear.grad_W)
    rust_norm.gamma, rust_norm.beta = _rust(norm.gamma), _rust(norm.beta)
    rust_norm.grad_gamma, rust_norm.grad_beta = _rust(norm.grad_gamma), _rust(norm.grad_beta)
    return [linear, norm], [rust_linear, rust_norm]


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), WeightDecay(0.1)], ids=lambda rule: type(rule).__name__)
def test_the_optimizer_steps_a_linear_layer_and_gamma_and_beta_as_numpys_by_bits(rule: UpdateRule):
    # the fused ops compute each rule's formula in the source's grouping, as NumpyOptimizer does,
    # a parameter at a time: a missing bias is an empty array, stepped to an empty array
    rng = np.random.default_rng(2)
    numpy_optimizer, rust_optimizer = NUMPY.optimizer(rule), RUST.optimizer(rule)
    for _step in range(3):
        numpy_layers, rust_layers = _optimizer_layers(rng)
        for optimizer, layers in ((numpy_optimizer, numpy_layers), (rust_optimizer, rust_layers)):
            optimizer.begin_step()
            for index, layer in enumerate(layers):
                optimizer.apply(index, layer, 0.1, 4)
        assert bits([layer.parameters() for layer in rust_layers]) == bits(
            [layer.parameters() for layer in numpy_layers]
        )
        assert bits(list(rust_optimizer.state().layers.values())) == bits(list(numpy_optimizer.state().layers.values()))
        assert [to_numpy(layer.gradients()[0]).any() for layer in rust_layers] == [False, False]  # reset


def test_adams_step_is_numpys_within_its_bias_corrections_rounding():
    # Adam's bias corrections are 1 - beta**t: numpy's is Python's correctly rounded pow, the
    # crate's Rust's powi, which isn't, and at beta1 = 0.9 the two first differ at t = 4. So from
    # the fourth step the steps differ by an ULP or so, as they did before this optimizer took
    # pairs, for every layer, with or without batch norm
    rng = np.random.default_rng(4)
    numpy_optimizer, rust_optimizer = NUMPY.optimizer(Adam()), RUST.optimizer(Adam())
    for _step in range(5):
        numpy_layers, rust_layers = _optimizer_layers(rng)
        for optimizer, layers in ((numpy_optimizer, numpy_layers), (rust_optimizer, rust_layers)):
            optimizer.begin_step()
            for index, layer in enumerate(layers):
                optimizer.apply(index, layer, 0.1, 4)
        for numpy_layer, rust_layer in zip(numpy_layers, rust_layers):
            for expected, actual in zip(numpy_layer.parameters(), rust_layer.parameters()):
                np.testing.assert_allclose(to_numpy(actual), expected, rtol=1e-15, atol=0)
    assert [len(state) for state in rust_optimizer.state().layers.values()] == [2, 4]


# networks


@pytest.mark.parametrize("name", NETWORKS)
def test_randomize_draws_numpys_weights(name: str):
    # the crate's RNG is numpy's np.random, so the same seed draws the same linear W, and batch norm
    # draws nothing
    assert bits(_network(name, backend=RUST).snapshot()) == bits(_network(name).snapshot())


# parity with numpy


def _gap_after_training(layers: Any, shape: Any, rule: UpdateRule) -> float:
    networks: list[Any] = []
    for backend in (NUMPY, RUST):
        network = SequentialArrayNetwork(INPUT, layers, rule, shape=shape, backend=backend)
        network.rng = backend.default_rng(3)
        network.randomize()
        networks.append(network)
    rows = _rows(40, shape)
    for step in range(50):
        batch = rows[(step * 5) % 40 :][:5]
        for network in networks:
            network.learn_batch(0.3, batch)
    return max_relative_gap(networks[0].snapshot(), networks[1].snapshot())


# no batch norm, the same shapes: the dense layers' own gap
CONTROLS = {
    "sigmoid": [Dense(5), Dense(3, output=True)],
    "relu softmax": [
        Dense(6),
        Dense(5, activation="relu"),
        Dense(3, output=True, activation="softmax", loss="cross_entropy"),
    ],
}


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", NETWORKS)
def test_training_matches_numpy_within_the_dense_layers_rounding(name: str, rule: UpdateRule):
    # batch norm computes the same bits in both (test_the_layer_is_numpys_by_bits); the dense
    # products don't: numpy's are BLAS, and the crate's its own FMA chains (a linear layer's first
    # Z already differs by an ULP), and Adam's bias corrections differ as above. 50 steps in, the
    # networks were at most 4.4e-12 apart relative when measured, and the same networks without
    # batch norm (CONTROLS) 1.2e-12: the same kind of gap, from the same products
    layers, shape = NETWORKS[name]
    assert _gap_after_training(layers, shape, rule) < 1e-10


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", CONTROLS)
def test_the_dense_layers_alone_have_the_same_kind_of_gap(name: str, rule: UpdateRule):
    assert _gap_after_training(CONTROLS[name], "multiclass", rule) < 1e-10


def test_train_trains_a_batch_norm_network_to_its_numpy_counterparts_accuracy():
    # train.py, prepared rows and a final batch of one dropped (D4), on both backends: the same
    # classifications after a few epochs
    rows = _rows(47)
    classified: list[list[int]] = []
    for backend in (NUMPY, RUST):
        network = _network("relu softmax", Momentum(0.9), backend=backend)
        train_backprop_network_mini_batch(network, network.prepare_dataset(rows), 5, epochs=3, rng=random.Random(0))
        classified.append(network.classify_rows(network.prepare_dataset(rows)))
    assert classified[1] == classified[0]
