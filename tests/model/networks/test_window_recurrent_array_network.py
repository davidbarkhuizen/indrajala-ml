# pyright: reportConstantRedefinition=false, reportPrivateUsage=false
"""
The numpy window-recurrent network (the window-recurrent adder workplan, stage 4): the gradient
through one, two and three windows against central differences, the codes made the identity so the
loss is differentiable; the codes' losses' terms through the network; the dropout masks a
recomputed window draws again; the plain sequence network's digits by bits with the state's
projections at zero and one window; and saving, loading and resuming by bits.
"""

from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pytest

from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.networks.numpy.window_recurrent_array_network import WindowRecurrentArrayNetwork
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    Dense,
    Dropout,
    Embedding,
    LayerNorm,
    LayerSpec,
    Position,
    Residual,
    StateInput,
)
from indrajala_ml.model.specs.update_rules import SGD, Adam
from indrajala_ml.model.specs.window_recurrence import Codes, WindowRecurrence
from indrajala_ml.pcg64 import generator_state

K = 2
ATTENTION = Residual((LayerNorm(), Attention(heads=2, causal=True)))
FFN = Residual((LayerNorm(), Dense(16, activation="relu"), Dense(8, activation="linear", bias=True)))
DIGITS = Dense(3, output=True, activation="softmax", loss="cross_entropy")
WINDOW: list[LayerSpec] = [
    Embedding(27, 8),
    Position(),
    StateInput(),
    ATTENTION,
    StateInput(),
    FFN,
    LayerNorm(),
    DIGITS,
]
RECURRENCE = WindowRecurrence(3, Codes(4))
STEP = 1e-6


def _network(window: list[LayerSpec] = WINDOW, seed: int = 7, rule: Any = None) -> WindowRecurrentArrayNetwork:
    return WindowRecurrentArrayNetwork.randomized((K,), window, RECURRENCE, rule or SGD(), seed=seed)


def _batch(windows: int, examples: int = 3, seed: int = 11) -> tuple[FloatArray, list[tuple[int, ...]]]:
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 27, size=(examples, windows * K)).astype(np.float64)
    labels = [tuple(int(d) for d in row) for row in rng.integers(0, 3, size=(examples, windows * K))]
    return X, labels


class _Trained(Protocol):
    def parameters(self) -> tuple[FloatArray, ...]: ...

    def gradients(self) -> tuple[FloatArray, ...]: ...


def _trained(network: WindowRecurrentArrayNetwork) -> list[_Trained]:
    # every layer and part with parameters, in the optimizer's order
    return [cast("_Trained", layer) for layer in [*network.layers, *network.parts] if hasattr(layer, "parameters")]


def _identity_codes(network: WindowRecurrentArrayNetwork) -> None:
    # the codes as the identity, so every state is the cell's output and the loss differentiable
    def forward(H: FloatArray) -> FloatArray:
        return H.copy()

    def backward(dQ: FloatArray, weight: float) -> FloatArray:
        return dQ

    codes: Any = network.codes
    codes.forward_batch = forward
    codes.backward_batch = backward


def _gradients(network: WindowRecurrentArrayNetwork, X: FloatArray, labels: list[tuple[int, ...]]) -> list[FloatArray]:
    network._accumulate_gradients(X, network._target_batch_array(labels))
    return [gradient.copy() for layer in _trained(network) for gradient in layer.gradients()]


def _numeric(network: WindowRecurrentArrayNetwork, X: FloatArray, labels: list[tuple[int, ...]]) -> list[FloatArray]:
    gradients: list[FloatArray] = []
    for value in (value for layer in _trained(network) for value in layer.parameters()):
        gradient = np.zeros(value.shape)
        for index in np.ndindex(value.shape):
            kept = float(value[index])
            value[index] = kept + STEP
            up = network.loss_batch(X, labels)
            value[index] = kept - STEP
            down = network.loss_batch(X, labels)
            value[index] = kept
            gradient[index] = (up - down) / (2 * STEP)
        gradients.append(gradient)
    return gradients


@pytest.mark.parametrize("windows", [1, 2, 3])
def test_the_gradient_through_the_windows_matches_central_differences(windows: int):
    network = _network()
    rng = np.random.default_rng(3)
    # every part away from its starting point, so no gradient is zero by symmetry
    network.pooling.q = rng.normal(size=network.pooling.q.shape)
    network.initial_state.c = rng.normal(size=network.initial_state.c.shape)
    _identity_codes(network)
    X, labels = _batch(windows)
    analytic = _gradients(network, X, labels)
    numeric = _numeric(network, X, labels)
    assert len(analytic) == len(numeric)
    for a, n in zip(analytic, numeric, strict=True):
        np.testing.assert_allclose(a, n, rtol=1e-5, atol=1e-7)
    # the state carries gradient back: the initial state's and the first StateInput's are not zero
    assert np.any(network.initial_state.c != 0.0)


@pytest.mark.parametrize("windows", [1, 3])
def test_the_codes_losses_count_every_snapped_state_once_at_one_over_the_windows(windows: int):
    # the codes' gradient is the codebook loss's, -2 (h - e) / W at each state's code, summed over
    # the initial state's snap and each window's; and each snap passes its gradient straight
    # through plus the commitment's, 2 * commitment * (h - e) / W
    network = _network()
    codes = network.codes
    backward = codes.backward_batch
    snaps: list[tuple[FloatArray, FloatArray, Any, FloatArray, FloatArray]] = []

    def recorded(dQ: FloatArray, weight: float) -> FloatArray:
        H, Q, indices = codes._H.copy(), codes._Q.copy(), codes.indices.copy()
        dH = backward(dQ, weight)
        assert weight == 1.0 / windows
        snaps.append((H, Q, indices, dQ.copy(), dH))
        return dH

    codes.backward_batch = recorded  # type: ignore[method-assign]
    X, labels = _batch(windows)
    network._accumulate_gradients(X, network._target_batch_array(labels))
    assert len(snaps) == windows + 1
    expected = np.zeros(codes.E.shape)
    for H, Q, indices, dQ, dH in snaps:
        np.add.at(expected, indices, -2.0 * (H - Q) / windows)
        np.testing.assert_allclose(dH, dQ + 2.0 * RECURRENCE.codes.commitment * (H - Q) / windows, rtol=1e-15)
    np.testing.assert_allclose(codes.grad_E, expected, rtol=1e-12, atol=1e-15)


def test_a_training_step_lowers_the_loss_on_its_batch():
    network = _network(rule=Adam())
    X, labels = _batch(3, examples=8)
    before = network.loss_batch(X, labels)
    for _ in range(30):
        network._learn_batch_input(0.01, X, labels)
    assert network.loss_batch(X, labels) < before


def test_a_recomputed_window_draws_its_dropout_masks_again():
    # the generator ends a step where one forward pass over the windows leaves it, and a window's
    # recompute draws the masks its forward pass drew: the gradient is the masks' network's
    window: list[LayerSpec] = [Embedding(27, 8), Position(), Dropout(0.3), StateInput(), ATTENTION, DIGITS]
    trained, forward = _network(window), _network(window)
    X, labels = _batch(3)
    trained._accumulate_gradients(X, trained._target_batch_array(labels))
    forward._set_training_mode(True)
    forward.forward_windows(X)
    assert generator_state(trained.rng) == generator_state(forward.rng)


def test_one_window_with_the_projections_at_zero_is_the_sequence_networks_digits_by_bits():
    network = _network()
    for state_input in network._state_inputs:
        state_input.P = np.zeros(state_input.P.shape)
    plain_specs = [spec for spec in WINDOW if not isinstance(spec, StateInput)]
    plain = SequentialArrayNetwork((K,), plain_specs, SGD(), shape="sequence")
    weights = [entry for layer, entry in zip(network.layers, network.snapshot()) if layer not in network._state_inputs]
    plain.restore(weights)
    X, _ = _batch(1, examples=5)
    expected = np.concatenate(list(plain.forward_rows(plain.prepare_dataset([(tuple(x), (0,) * K) for x in X]))))
    assert np.array_equal(network.forward_windows(X), expected)


def test_the_network_reads_any_window_count_and_classifies_each_column():
    network = _network()
    for windows in (1, 2, 5):
        X, _ = _batch(windows)
        assert network.forward_windows(X).shape == (3, windows * K * 3)
        digits = network.classify_state(tuple(X[0]))
        assert len(digits) == windows * K and all(d in (0, 1, 2) for d in digits)
    with pytest.raises(AssertionError, match="whole windows"):
        network.forward_windows(np.zeros((1, 3)))


def test_a_saved_network_loads_and_resumes_training_by_bits(tmp_path: Path):
    network = _network(rule=Adam())
    X, labels = _batch(2)
    network._learn_batch_input(0.01, X, labels)
    path = str(tmp_path / "window.json")
    network.save(path)
    loaded = load_network(path)
    assert isinstance(loaded, WindowRecurrentArrayNetwork)
    assert loaded.recurrence == RECURRENCE
    for a, b in zip(network.snapshot(), loaded.snapshot(), strict=True):
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    network._learn_batch_input(0.01, X, labels)
    loaded._learn_batch_input(0.01, X, labels)
    for a, b in zip(network.snapshot(), loaded.snapshot(), strict=True):
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert WindowRecurrentArrayNetwork.load(path).snapshot()[-1][0].shape == (RECURRENCE.state_size,)
