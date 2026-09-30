"""
Dense batch norm in Rust (the batch-norm workplan, stage 3): BatchNormRustArrayLayer and
LinearRustArrayLayer against numpy's layers by bits, the optimizer's pairs (a linear layer's W, and
gamma and beta) against NumpyOptimizer by bits, gradient checks under every rule, the running
averages in training and inference, snapshot and checkpoint, the one-example refusal (D4), weight
decay (D7), and parity with numpy's networks. Reuses tests/test_batch_norm_array_network.py's cases.
"""

import random
from typing import Any

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model import batch_norm_array_layer
from indrajala_ml.model.array_backend import NUMPY, RUST
from indrajala_ml.model.array_layer import FloatArray, sigmoid
from indrajala_ml.model.batch_norm_array_layer import BatchNormArrayLayer
from indrajala_ml.model.batch_norm_rust_array_layer import BatchNormRustArrayLayer
from indrajala_ml.model.layer_specs import Dense
from indrajala_ml.model.linear_array_layer import LinearArrayLayer
from indrajala_ml.model.linear_rust_array_layer import LinearRustArrayLayer
from indrajala_ml.model.optimizers import NumpyOptimizer, RustOptimizer
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from indrajala_ml.train import train_backprop_network_mini_batch
from tests.gradient_check import check_gradients
from tests.test_batch_norm_array_network import EPSILON, INPUT, NETWORKS, RATE, RULES, _rows


def _numpy(values: Any) -> Any:
    return values if isinstance(values, np.ndarray) else np.array(values.tolist())  # pyright: ignore[reportUnknownVariableType]


def _rust(values: Any) -> pa.Array:
    return pa.Array(np.asarray(values).tolist())


def _bits(values: Any) -> list[bytes]:
    # by bits, so -0.0 and 0.0 differ
    return [_numpy(array).tobytes() for entry in values for array in entry]


def _network(name: str = "sigmoid", rule: UpdateRule | None = None, seed: int = 3, backend: Any = RUST) -> Any:
    layers, shape = NETWORKS[name]
    network = SequentialArrayNetwork(INPUT, layers, SGD() if rule is None else rule, shape=shape, backend=backend)
    backend.seed(seed)
    network.randomize()
    return network


# the layers


@pytest.fixture
def crate_exp(monkeypatch: pytest.MonkeyPatch) -> None:
    """The numpy layer's sigmoid with the crate's exp. exp isn't correctly rounded: np.exp picks
    its implementation by CPU, and can differ from Rust's f64::exp in the last bit, which every
    later value then carries. Everything but exp is compared by bits."""

    def sigmoid_with_crate_exp(z: FloatArray) -> FloatArray:
        return 1.0 / (1.0 + _numpy(pa.exp(_rust(-z))))

    monkeypatch.setattr(batch_norm_array_layer, "sigmoid", sigmoid_with_crate_exp)


class _Next:
    """A next dense layer: the Rust layer's fused hidden delta reads its W and delta_batch, and numpy's
    its downstream, delta_batch @ W, taken here from the crate so both see the same values."""

    def __init__(self, W: Any, delta_batch: Any) -> None:
        self.W, self.delta_batch = _rust(W), _rust(delta_batch)

    def downstream_batch(self) -> Any:
        return _numpy(pa.layer_downstream_batch(self.W, self.delta_batch))

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
    next_layer = _Next(rng.uniform(-1.0, 1.0, (3, 4)), rng.uniform(-1.0, 1.0, (batch_size, 3)))
    array = BatchNormArrayLayer(4, activation, EPSILON, RATE)
    rust = BatchNormRustArrayLayer(4, activation, EPSILON, RATE)
    array.gamma, array.beta = gamma.copy(), beta.copy()
    rust.gamma, rust.beta = _rust(gamma), _rust(beta)

    values: list[list[Any]] = []
    layers: list[tuple[Any, Any]] = [(array, X), (rust, _rust(X))]
    for layer, inputs in layers:
        layer.set_training_mode(True)
        activations = layer.forward_batch(inputs)
        layer.compute_hidden_delta_batch(next_layer)
        dx = layer.downstream_batch()
        layer.accumulate_gradient_batch(inputs)
        values.append([activations, dx, layer.grad_gamma, layer.grad_beta, layer.running_mean, layer.running_var])
    assert _bits([values[1]]) == _bits([values[0]])

    # the deltas agree in value, but not in the sign of a ReLU's zero: numpy's is the downstream
    # times 0.0, -0.0 where the downstream is negative, and the crate's is 0.0, as between
    # ReLUArrayLayer and ReLURustArrayLayer. Neither changes a sum that has a nonzero term
    np.testing.assert_array_equal(_numpy(rust.delta_batch), array.delta_batch)


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

    assert _numpy(rust.forward_batch(_rust(X))).tobytes() == array.forward_batch(X).tobytes()
    for row in X:
        assert _numpy(rust.forward(_rust(row))).tobytes() == array.forward(row).tobytes()
    assert _bits([rust.running_state()]) == _bits([array.running_state()])  # inference doesn't move them


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
    assert _numpy(layer.delta_batch).tobytes() == _numpy(downstream).tobytes()
    expected, _grad_b = pa.layer_accumulate_gradient_batch(
        downstream, _rust(X), pa.Array.zeros((2, 4)), pa.Array.zeros(2)
    )
    assert _numpy(layer.grad_W).tobytes() == _numpy(expected).tobytes()
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
    numpy_optimizer, rust_optimizer = NumpyOptimizer(rule), RustOptimizer(rule)
    for _step in range(3):
        numpy_layers, rust_layers = _optimizer_layers(rng)
        for optimizer, layers in ((numpy_optimizer, numpy_layers), (rust_optimizer, rust_layers)):
            optimizer.begin_step()
            for index, layer in enumerate(layers):
                optimizer.apply(index, layer, 0.1, 4)
        assert _bits([layer.parameters() for layer in rust_layers]) == _bits(
            [layer.parameters() for layer in numpy_layers]
        )
        assert _bits(rust_optimizer.state().layers.values()) == _bits(numpy_optimizer.state().layers.values())
        assert [_numpy(layer.gradients()[0]).any() for layer in rust_layers] == [False, False]  # reset


def test_adams_step_is_numpys_within_its_bias_corrections_rounding():
    # Adam's bias corrections are 1 - beta**t: numpy's is Python's correctly rounded pow, the
    # crate's Rust's powi, which isn't, and at beta1 = 0.9 the two first differ at t = 4. So from
    # the fourth step the steps differ by an ULP or so, as they did before this optimizer took
    # pairs, for every layer, with or without batch norm
    rng = np.random.default_rng(4)
    numpy_optimizer, rust_optimizer = NumpyOptimizer(Adam()), RustOptimizer(Adam())
    for _step in range(5):
        numpy_layers, rust_layers = _optimizer_layers(rng)
        for optimizer, layers in ((numpy_optimizer, numpy_layers), (rust_optimizer, rust_layers)):
            optimizer.begin_step()
            for index, layer in enumerate(layers):
                optimizer.apply(index, layer, 0.1, 4)
        for numpy_layer, rust_layer in zip(numpy_layers, rust_layers):
            for expected, actual in zip(numpy_layer.parameters(), rust_layer.parameters()):
                np.testing.assert_allclose(_numpy(actual), expected, rtol=1e-15, atol=0)
    assert [len(state) for state in rust_optimizer.state().layers.values()] == [2, 4]


# networks


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", NETWORKS)
@pytest.mark.parametrize("batch_size", [2, 5])
def test_every_gradient_matches_its_finite_difference(name: str, rule: UpdateRule, batch_size: int):
    network = _network(name, rule)
    rows = _rows(batch_size, NETWORKS[name][1])
    # moved running averages and a trained step, so gamma and beta aren't at their initial values
    network.learn_batch(0.5, _rows(6, NETWORKS[name][1], seed=2))

    check_gradients(network, [state for state, _ in rows], [label for _, label in rows])


@pytest.mark.parametrize("name", NETWORKS)
def test_randomize_draws_numpys_weights(name: str):
    # the crate's RNG is numpy's np.random, so the same seed draws the same linear W, and batch norm
    # draws nothing
    assert _bits(_network(name).snapshot()) == _bits(_network(name, backend=NUMPY).snapshot())


def test_the_running_averages_move_in_training_forward_passes_only():
    network = _network()
    rows = _rows(6)
    prepared = network.prepare_dataset(rows)
    norm = network.layers[1]

    network.learn_batch(0.5, rows)
    trained = _bits([norm.running_state()])
    assert trained != _bits([(np.zeros(5), np.ones(5))])

    network.classify_rows(prepared)
    network.classify_state(rows[0][0])
    network.classify_row(prepared, 1)
    assert _bits([norm.running_state()]) == trained


def test_classifying_normalizes_with_the_running_averages():
    network = _network()
    rows = _rows(6)
    network.learn_batch(0.5, rows)
    linear, norm, output = network.layers
    Z = np.array([state for state, _ in rows]) @ _numpy(linear.W).T
    xhat = (Z - _numpy(norm.running_mean)) / np.sqrt(_numpy(norm.running_var) + EPSILON)
    hidden = sigmoid(_numpy(norm.gamma) * xhat + _numpy(norm.beta))
    expected = sigmoid(hidden @ _numpy(output.W).T + _numpy(output.b))

    assert network.classify_rows(network.prepare_dataset(rows)) == np.argmax(expected, axis=1).tolist()
    assert [network.classify_state(state) for state, _ in rows] == np.argmax(expected, axis=1).tolist()


def test_the_optimizers_state_is_per_parameter():
    network = _network(rule=Adam())
    network.learn_batch(0.1, _rows(6))
    layers = network.optimizer.state().layers

    assert [array.shape for array in layers[0]] == [(5, 4), (5, 4)]  # the linear layer's m and v
    assert [array.shape for array in layers[1]] == [(5,)] * 4  # gamma's m and v, beta's m and v
    assert [array.shape for array in layers[2]] == [(3, 5), (3, 5), (3,), (3,)]


def test_weight_decay_decays_the_linear_layers_w_and_neither_gamma_nor_beta():
    rows = _rows(6)
    sgd, decayed = _network(rule=SGD()), _network(rule=WeightDecay(0.1))

    for network in (sgd, decayed):
        network.learn_batch(0.5, rows)

    # gamma and beta step with plain SGD (D7), and nothing before them differs
    assert _bits([decayed.snapshot()[1][:2]]) == _bits([sgd.snapshot()[1][:2]])
    assert _numpy(decayed.layers[0].W).tobytes() != _numpy(sgd.layers[0].W).tobytes()


@pytest.mark.parametrize("method", ["learn", "learn_row", "learn_batch", "learn_batch_rows"])
def test_a_one_example_training_step_is_refused_naming_the_layer(method: str):
    network = _network("after a sigmoid layer")
    rows = _rows(3)
    before = _bits(network.snapshot())

    with pytest.raises(ValueError, match=r"layer 2, BatchNorm\(activation='relu'.*D4"):
        match method:
            case "learn":
                network.learn(0.5, *rows[0])
            case "learn_row":
                network.learn_row(0.5, network.prepare_dataset(rows), 0)
            case "learn_batch":
                network.learn_batch(0.5, rows[:1])
            case _:
                network.learn_batch_rows(0.5, network.prepare_dataset(rows), [2])
    assert _bits(network.snapshot()) == before


def test_snapshot_carries_the_running_averages_and_restore_returns_them():
    network = _network()
    network.learn_batch(0.5, _rows(6))
    snapshot = network.snapshot()

    assert [len(entry) for entry in snapshot] == [1, 4, 2]
    norm = network.layers[1]
    assert _bits([snapshot[1]]) == _bits([norm.parameters() + norm.running_state()])

    network.learn_batch(0.5, _rows(6, seed=5))
    network.restore([[array.tolist() for array in entry] for entry in snapshot])  # as a loaded file
    assert _bits(network.snapshot()) == _bits(snapshot)


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
def test_a_checkpoint_resumes_training_by_bits(rule: UpdateRule):
    network = _network("two pairs", rule)
    network.learn_batch(0.1, _rows(6))
    checkpoint = network.checkpoint()

    network.learn_batch(0.1, _rows(5, seed=7))
    network.learn_batch(0.1, _rows(4, seed=8))
    trained = _bits(network.snapshot())

    network.restore_checkpoint(checkpoint)
    network.learn_batch(0.1, _rows(5, seed=7))
    network.learn_batch(0.1, _rows(4, seed=8))
    assert _bits(network.snapshot()) == trained


def test_saving_batch_norm_is_refused_until_stage_5(tmp_path: Any):
    with pytest.raises(NotImplementedError, match="stage 5"):
        _network().save(str(tmp_path / "model.json"))


# parity with numpy


def _max_relative_gap(layers: Any, shape: Any, rule: UpdateRule) -> float:
    networks: list[Any] = []
    for backend in (NUMPY, RUST):
        network = SequentialArrayNetwork(INPUT, layers, rule, shape=shape, backend=backend)
        backend.seed(3)
        network.randomize()
        networks.append(network)
    rows = _rows(40, shape)
    for step in range(50):
        batch = rows[(step * 5) % 40 :][:5]
        for network in networks:
            network.learn_batch(0.3, batch)

    gap = 0.0
    for expected_entry, actual_entry in zip(networks[0].snapshot(), networks[1].snapshot()):
        for expected, actual in zip(expected_entry, actual_entry):
            gap = max(gap, float((np.abs(_numpy(actual) - expected) / np.abs(expected)).max()))
    return gap


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
    assert _max_relative_gap(layers, shape, rule) < 1e-10


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", CONTROLS)
def test_the_dense_layers_alone_have_the_same_kind_of_gap(name: str, rule: UpdateRule):
    assert _max_relative_gap(CONTROLS[name], "multiclass", rule) < 1e-10


def test_train_trains_a_batch_norm_network_to_its_numpy_counterparts_accuracy():
    # train.py, prepared rows and a final batch of one dropped (D4), on both backends: the same
    # classifications after a few epochs
    rows = _rows(47)
    classified: list[list[int]] = []
    for backend in (NUMPY, RUST):
        network = _network("relu softmax", Momentum(0.9), backend=backend)
        random.seed(0)
        train_backprop_network_mini_batch(network, network.prepare_dataset(rows), 5, epochs=3)
        classified.append(network.classify_rows(network.prepare_dataset(rows)))
    assert classified[1] == classified[0]
