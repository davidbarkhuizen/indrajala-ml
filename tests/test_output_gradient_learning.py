"""
Learning from a given gradient at the output (the NES plan's stage 1, in indrajala-research): a
linear output layer with loss="given", trained by learn_batch_output_gradient, and forward_batch.

- With the gradient set to softmax cross-entropy's, it takes learn_batch's steps on a softmax
  network bit for bit, on both backends: the same backward pass and update, only the delta given.
- Its gradients are the mean loss's, by finite differences.
- numpy and Rust agree within rtol over several Adam steps.
- It refuses target-based learning, and target-based networks refuse it.
"""

from collections.abc import Callable
from typing import Any, cast

import indrajala_math_rust as pa
import numpy as np
import numpy.typing as npt
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.networks.python.sequential_backprop_network import SequentialMultiClassBackpropClassifierNetwork
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.persistence.format2_json import layer_from_json, layer_to_json
from indrajala_ml.model.specs.layer_specs import Conv, Dense, LayerSpec, Pool
from indrajala_ml.model.specs.spec_validation import validate_layer_specs
from indrajala_ml.model.specs.update_rules import SGD, Adam

IntArray = npt.NDArray[np.int64]

INPUT_SHAPE = (6, 6, 2)
CLASSES = 5


def _specs(output: Dense) -> list[LayerSpec]:
    return [Conv(3, 4), Pool(2), Dense(16, "relu"), output]


SOFTMAX = Dense(CLASSES, "softmax", output=True, loss="cross_entropy")
GIVEN = Dense(CLASSES, "linear", output=True, loss="given")


def _seeded(output: Dense, backend: Any, rule: Any = None) -> Any:
    network = SequentialArrayNetwork(INPUT_SHAPE, _specs(output), rule or Adam(), backend=backend)
    network.rng = backend.default_rng(3)
    network.randomize()
    return network


def _batches(count: int, size: int) -> list[tuple[FloatArray, IntArray]]:
    rng = np.random.default_rng(4)
    inputs = int(np.prod(INPUT_SHAPE))
    return [(rng.uniform(-1.0, 1.0, (size, inputs)), rng.integers(0, CLASSES, size)) for _ in range(count)]


def _rows(states: FloatArray, labels: IntArray) -> list[tuple[tuple[float, ...], int]]:
    state_lists = cast("list[list[float]]", states.tolist())
    return [(tuple(state), int(label)) for state, label in zip(state_lists, labels, strict=True)]


def _weights(network: Any) -> list[Any]:
    return [[part.tolist() for part in layer] for layer in network.snapshot()]


def _numpy_cross_entropy_gradient(labels: IntArray) -> Callable[[FloatArray], FloatArray]:
    # SoftmaxArrayLayer's forward and delta, on the linear layer's output
    def gradient(Z: FloatArray) -> FloatArray:
        shifted = Z - Z.max(axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        A = exp_values / exp_values.sum(axis=1, keepdims=True)
        return A - np.eye(CLASSES)[labels]

    return gradient


def _rust_cross_entropy_gradient(labels: IntArray) -> Callable[[pa.Array], pa.Array]:
    target = pa.Array(np.eye(CLASSES)[labels].tolist())

    def gradient(Z: pa.Array) -> pa.Array:
        return pa.layer_softmax_output_delta(pa.array_softmax(Z), target)

    return gradient


@pytest.mark.parametrize("backend", [NUMPY, RUST], ids=["numpy", "rust"])
def test_the_cross_entropy_gradient_takes_learn_batchs_steps_bit_for_bit(backend: Any) -> None:
    softmax, given = _seeded(SOFTMAX, backend), _seeded(GIVEN, backend)
    assert _weights(softmax) == _weights(given)
    to_gradient = _numpy_cross_entropy_gradient if backend is NUMPY else _rust_cross_entropy_gradient
    for states, labels in _batches(4, 8):
        softmax.learn_batch(0.01, _rows(states, labels))
        given.learn_batch_output_gradient(0.01, backend.matrix(states.tolist()), to_gradient(labels))
        assert _weights(given) == _weights(softmax)


def test_the_step_follows_the_mean_losss_gradient_by_finite_differences() -> None:
    # a squared loss held by the caller, loss_i = |Y_i - T_i|^2 / 2, its gradient Y - T; one SGD
    # step at rate 1 moves each weight by the mean loss's derivative, w - g / B
    specs: list[LayerSpec] = [Dense(4, "sigmoid"), Dense(3, "linear", output=True, loss="given")]
    network = SequentialArrayNetwork((5,), specs, SGD(), backend=NUMPY)
    network.rng = NUMPY.default_rng(7)
    network.randomize()
    rng = np.random.default_rng(8)
    X, T = rng.uniform(-1.0, 1.0, (6, 5)), rng.uniform(-1.0, 1.0, (6, 3))

    def mean_loss() -> float:
        Y: FloatArray = network.forward_batch(X)
        return float(np.mean(0.5 * np.sum((Y - T) ** 2, axis=1)))

    before = [(layer.W.copy(), layer.b.copy()) for layer in network.layers]
    expected: list[tuple[FloatArray, FloatArray]] = []
    epsilon = 1e-6
    for layer in network.layers:
        derivatives: list[FloatArray] = []
        for parameter in (layer.W, layer.b):
            derivative = np.zeros_like(parameter)
            for index in np.ndindex(parameter.shape):
                saved = parameter[index]
                parameter[index] = saved + epsilon
                up = mean_loss()
                parameter[index] = saved - epsilon
                down = mean_loss()
                parameter[index] = saved
                derivative[index] = (up - down) / (2 * epsilon)
            derivatives.append(derivative)
        expected.append((derivatives[0], derivatives[1]))

    def squared_loss_gradient(Y: FloatArray) -> FloatArray:
        return Y - T

    network.learn_batch_output_gradient(1.0, X, squared_loss_gradient)
    for layer, (W0, b0), (dW, db) in zip(network.layers, before, expected, strict=True):
        np.testing.assert_allclose(W0 - layer.W, dW, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(b0 - layer.b, db, rtol=1e-6, atol=1e-9)


def test_numpy_and_rust_agree_over_several_adam_steps() -> None:
    numpy_network, rust_network = _seeded(GIVEN, NUMPY), _seeded(GIVEN, RUST)
    for states, labels in _batches(5, 16):
        numpy_network.learn_batch_output_gradient(0.01, states, _numpy_cross_entropy_gradient(labels))
        rust_network.learn_batch_output_gradient(
            0.01, pa.Array.from_rows(states.tolist()), _rust_cross_entropy_gradient(labels)
        )
    for numpy_layer, rust_layer in zip(_weights(numpy_network), _weights(rust_network), strict=True):
        for numpy_part, rust_part in zip(numpy_layer, rust_layer, strict=True):
            np.testing.assert_allclose(rust_part, numpy_part, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("backend", [NUMPY, RUST], ids=["numpy", "rust"])
def test_learn_batch_output_gradient_returns_the_steps_own_outputs(backend: Any) -> None:
    network = _seeded(GIVEN, backend)
    states, _labels = _batches(1, 8)[0]
    X = backend.matrix(states.tolist())
    before = network.forward_batch(X).tolist()
    seen: list[Any] = []

    def gradient(Y: Any) -> Any:
        seen.append(Y.tolist())
        return Y * 0.0

    returned = network.learn_batch_output_gradient(0.01, X, gradient)
    assert returned.tolist() == seen[0] == before


@pytest.mark.parametrize("backend", [NUMPY, RUST], ids=["numpy", "rust"])
def test_forward_batch_gives_forward_rows_outputs(backend: Any) -> None:
    network = _seeded(SOFTMAX, backend)
    states, labels = _batches(1, 20)[0]  # under one forward_rows chunk, so the same products
    prepared = network.prepare_dataset(_rows(states, labels))
    (by_rows,) = list(network.forward_rows(prepared))
    assert network.forward_batch(backend.matrix(states.tolist())).tolist() == by_rows.tolist()


@pytest.mark.parametrize("backend", [NUMPY, RUST], ids=["numpy", "rust"])
def test_a_given_loss_refuses_targets_and_a_target_loss_refuses_a_gradient(backend: Any) -> None:
    given, softmax = _seeded(GIVEN, backend), _seeded(SOFTMAX, backend)
    states, labels = _batches(1, 4)[0]
    rows = _rows(states, labels)
    with pytest.raises(ValueError, match="given loss"):
        given.learn_batch(0.01, rows)
    with pytest.raises(ValueError, match="given loss"):
        given.learn(0.01, rows[0][0], rows[0][1])

    def identity(Y: Any) -> Any:
        return Y

    with pytest.raises(ValueError, match="given loss"):
        softmax.learn_batch_output_gradient(0.01, backend.matrix(states.tolist()), identity)


def test_a_gradient_of_the_wrong_shape_is_refused() -> None:
    network = _seeded(GIVEN, NUMPY)
    states, _labels = _batches(1, 4)[0]

    def too_narrow(Y: FloatArray) -> FloatArray:
        return Y[:, :2]

    with pytest.raises(AssertionError, match="the output's shape"):
        network.learn_batch_output_gradient(0.01, states, too_narrow)


@pytest.mark.parametrize(
    "output",
    [
        Dense(CLASSES, "linear", output=True, loss="squared"),
        Dense(CLASSES, "sigmoid", output=True, loss="given"),
        Dense(CLASSES, "softmax", output=True, loss="given"),
    ],
    ids=["linear-squared", "sigmoid-given", "softmax-given"],
)
def test_only_a_linear_output_has_a_given_loss(output: Dense) -> None:
    with pytest.raises(AssertionError, match="given loss"):
        validate_layer_specs(_specs(output))


def test_pure_python_refuses_a_given_loss() -> None:
    with pytest.raises(AssertionError, match="array networks' only"):
        SequentialMultiClassBackpropClassifierNetwork(INPUT_SHAPE, _specs(GIVEN), Adam())


def test_a_given_loss_survives_the_format2_spec_round_trip() -> None:
    assert layer_from_json(layer_to_json(GIVEN)) == GIVEN
