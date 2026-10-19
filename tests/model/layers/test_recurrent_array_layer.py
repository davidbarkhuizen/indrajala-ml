# pyright: reportConstantRedefinition=false
"""
A window-recurrent network's numpy layers (the window-recurrent adder workplan, stage 3): tanh
through exp (W15), the StateInput (W18), the attention pooling (W19), the GRU cell (W13) and the
codes (W14), each with a gradient check by central differences; the codes' against their
straight-through definition.
"""

import itertools
import warnings
from collections.abc import Callable

import numpy as np
import pytest

from indrajala_ml.model.layers.numpy import recurrent_array_layer
from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.layers.numpy.recurrent_array_layer import (
    AttentionPoolingArrayLayer,
    CodesArrayLayer,
    GRUArrayLayer,
    StateInputArrayLayer,
    tanh,
)

N, T, D, S = 3, 4, 5, 2
STEP = 1e-6
TOLERANCE = 1e-7


def _numeric(loss: Callable[[], float], value: FloatArray) -> FloatArray:
    # central differences of loss over each entry of value, perturbed in place
    gradient = np.zeros(value.shape)
    for index in np.ndindex(value.shape):
        kept = value[index]
        value[index] = kept + STEP
        up = loss()
        value[index] = kept - STEP
        down = loss()
        value[index] = kept
        gradient[index] = (up - down) / (2 * STEP)
    return gradient


def _close(numeric: FloatArray, analytic: FloatArray) -> None:
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=TOLERANCE)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261012)


def test_tanh_is_the_librarys_to_rounding():
    z = np.linspace(-20.0, 20.0, 4001)
    np.testing.assert_allclose(tanh(z), np.tanh(z), rtol=0.0, atol=4e-16)
    assert tanh(np.zeros(1))[0] == 0.0


def test_tanh_saturates_exactly_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert tanh(np.array([400.0, 1e300, -400.0, -1e300])).tolist() == [1.0, 1.0, -1.0, -1.0]


def test_tanh_and_the_gates_go_through_the_modules_exp(monkeypatch: pytest.MonkeyPatch):
    calls: list[int] = []

    def counted(z: FloatArray) -> FloatArray:
        calls.append(z.size)
        return np.exp(z)

    monkeypatch.setattr(recurrent_array_layer, "exp", counted)
    tanh(np.zeros(3))
    recurrent_array_layer.sigmoid(np.zeros(2))
    assert calls == [3, 2]


def test_a_state_input_adds_p_c_to_every_token(rng: np.random.Generator):
    layer = StateInputArrayLayer(T, D, S)
    layer.draw(rng)
    X, C = rng.normal(size=(N, T * D)), rng.normal(size=(N, S))
    layer.set_state(C)
    out = layer.forward_batch(X).reshape(N, T, D)
    for t in range(T):
        np.testing.assert_allclose(out[:, t, :], X.reshape(N, T, D)[:, t, :] + C @ layer.P.T, rtol=1e-15)


def test_a_state_input_with_p_at_zero_passes_its_tokens_by_bits(rng: np.random.Generator):
    layer = StateInputArrayLayer(T, D, S)
    X = rng.normal(size=(N, T * D))
    layer.set_state(rng.normal(size=(N, S)))
    assert np.array_equal(layer.forward_batch(X), X)


def test_a_state_input_refuses_a_batch_without_its_states(rng: np.random.Generator):
    layer = StateInputArrayLayer(T, D, S)
    layer.set_state(np.zeros((N + 1, S)))
    with pytest.raises(AssertionError, match="the batch's states"):
        layer.forward_batch(np.zeros((N, T * D)))


def test_a_state_inputs_gradients_match_central_differences(rng: np.random.Generator):
    layer = StateInputArrayLayer(T, D, S)
    layer.draw(rng)
    X, C, G = rng.normal(size=(N, T * D)), rng.normal(size=(N, S)), rng.normal(size=(N, T * D))

    def loss() -> float:
        layer.set_state(C)
        return float(np.sum(G * layer.forward_batch(X)))

    loss()
    layer._backward(G)
    layer.accumulate_gradient_batch(X)
    _close(_numeric(loss, layer.P), layer.grad_P)
    _close(_numeric(loss, C), layer.state_delta)
    _close(_numeric(loss, X), layer.downstream_batch())


def test_pooling_starts_as_the_mean_over_the_tokens(rng: np.random.Generator):
    layer = AttentionPoolingArrayLayer(T, D)
    X = rng.normal(size=(N, T * D))
    np.testing.assert_allclose(layer.forward_batch(X), X.reshape(N, T, D).mean(axis=1), rtol=1e-14)


def test_pooling_weights_the_tokens_by_their_softmax_against_q(rng: np.random.Generator):
    layer = AttentionPoolingArrayLayer(T, D)
    layer.q = rng.normal(size=D)
    X = rng.normal(size=(N, T * D))
    tokens = X.reshape(N, T, D)
    scores = tokens @ layer.q
    p = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(layer.forward_batch(X), np.einsum("nt,ntd->nd", p, tokens), rtol=1e-13)


def test_poolings_gradients_match_central_differences(rng: np.random.Generator):
    layer = AttentionPoolingArrayLayer(T, D)
    layer.q = rng.normal(size=D)
    X, G = rng.normal(size=(N, T * D)), rng.normal(size=(N, D))

    def loss() -> float:
        return float(np.sum(G * layer.forward_batch(X)))

    loss()
    dX = layer.backward_batch(G)
    _close(_numeric(loss, layer.q), layer.grad_q)
    _close(_numeric(loss, X), dX)


def test_the_gru_is_choss_cell(rng: np.random.Generator):
    layer = GRUArrayLayer(D, S)
    layer.draw(rng)
    u, c = rng.normal(size=(1, D)), rng.normal(size=(1, S))
    W, U, b = layer.W, layer.U, layer.b

    def gate(g: int) -> tuple[FloatArray, FloatArray, FloatArray]:
        rows = slice(g * S, (g + 1) * S)
        return W[rows], U[rows], b[rows]

    def sig(x: FloatArray) -> FloatArray:
        return 1.0 / (1.0 + np.exp(-x))

    (Wr, Ur, br), (Wz, Uz, bz), (Wh, Uh, bh) = gate(0), gate(1), gate(2)
    r = sig(Wr @ u[0] + Ur @ c[0] + br)
    z = sig(Wz @ u[0] + Uz @ c[0] + bz)
    h_tilde = np.tanh(Wh @ u[0] + Uh @ (r * c[0]) + bh)
    np.testing.assert_allclose(layer.forward_batch(u, c)[0], z * c[0] + (1.0 - z) * h_tilde, rtol=1e-13)


def test_the_grus_draw_biases_its_update_gate_toward_keeping_the_state(rng: np.random.Generator):
    layer = GRUArrayLayer(D, S, update_bias=1.0)
    layer.draw(rng)
    assert layer.b[S : 2 * S].tolist() == [1.0] * S
    limit = 1.0 / np.sqrt(D + S)
    assert np.all(np.abs(layer.W) <= limit) and np.all(np.abs(layer.U) <= limit)
    assert np.all(np.abs(layer.b[:S]) <= limit) and np.all(np.abs(layer.b[2 * S :]) <= limit)


def test_a_saturated_update_gate_keeps_the_state(rng: np.random.Generator):
    layer = GRUArrayLayer(D, S, update_bias=1000.0)
    layer.draw(rng)
    c = rng.normal(size=(N, S))
    assert np.array_equal(layer.forward_batch(rng.normal(size=(N, D)), c), c)


def test_the_grus_gradients_match_central_differences(rng: np.random.Generator):
    layer = GRUArrayLayer(D, S, update_bias=0.5)
    layer.draw(rng)
    u, c, G = rng.normal(size=(N, D)), rng.normal(size=(N, S)), rng.normal(size=(N, S))

    def loss() -> float:
        return float(np.sum(G * layer.forward_batch(u, c)))

    loss()
    du, dc = layer.backward_batch(G)
    for value, analytic in zip(layer.parameters(), layer.gradients(), strict=True):
        _close(_numeric(loss, value), analytic)
    _close(_numeric(loss, u), du)
    _close(_numeric(loss, c), dc)


def test_the_grus_gradients_accumulate_over_passes(rng: np.random.Generator):
    layer = GRUArrayLayer(D, S)
    layer.draw(rng)
    u, c, G = rng.normal(size=(N, D)), rng.normal(size=(N, S)), rng.normal(size=(N, S))
    layer.forward_batch(u, c)
    layer.backward_batch(G)
    once = [g.copy() for g in layer.gradients()]
    layer.forward_batch(u, c)
    layer.backward_batch(G)
    for twice, first in zip(layer.gradients(), once, strict=True):
        assert np.array_equal(twice, first + first)


def test_codes_snap_to_the_nearest_the_lowest_index_on_a_tie():
    layer = CodesArrayLayer(3, 2)
    layer.E = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 3.0]])
    H = np.array([[0.9, 0.1], [0.0, 0.0], [-2.0, 0.0], [0.0, 2.0]])
    assert layer.nearest(H).tolist() == [0, 0, 1, 2]
    out = layer.forward_batch(H)
    assert np.array_equal(out, layer.E[[0, 0, 1, 2]])
    out[0, 0] = 7.0
    assert layer.E[0, 0] == 1.0


def test_codes_pass_the_gradient_straight_through_plus_the_commitments(rng: np.random.Generator):
    layer = CodesArrayLayer(4, S, commitment=0.25)
    layer.draw(rng)
    H, dQ = rng.normal(size=(N, S)), rng.normal(size=(N, S))
    Q = layer.forward_batch(H)
    weight = 1.0 / 3.0
    assert np.array_equal(layer.backward_batch(dQ, weight), dQ + (2.0 * weight * 0.25) * (H - Q))


def test_the_codes_losses_match_central_differences(rng: np.random.Generator):
    # the codebook loss ||sg[h] - e||^2 moves the codes, the commitment loss
    # commitment * ||h - sg[e]||^2 moves h; the chosen codes held fixed
    layer = CodesArrayLayer(4, S, commitment=0.25)
    layer.draw(rng)
    H = rng.normal(size=(N, S))
    weight = 0.5
    layer.forward_batch(H)
    indices = layer.indices.copy()

    def codebook() -> float:
        return float(weight * np.sum((H - layer.E[indices]) ** 2))

    def commitment() -> float:
        return float(weight * 0.25 * np.sum((H - layer.E[indices]) ** 2))

    dH = layer.backward_batch(np.zeros((N, S)), weight)
    _close(_numeric(codebook, layer.E), layer.grad_E)
    _close(_numeric(commitment, H), dH)
    assert layer.loss_batch(weight) == pytest.approx(codebook() + commitment(), rel=1e-14)


def test_soft_codes_losses_and_mix_match_central_differences(rng: np.random.Generator):
    # the soft-to-hard quantiser (the commitment-control workplan, stage 3): sum(G * q) through the
    # weights, the commitment to sg[q] and the soft codebook loss with the weights and h held
    layer = CodesArrayLayer(4, S, commitment=0.25)
    layer.draw(rng)
    H, G = rng.normal(size=(N, S)), rng.normal(size=(N, S))
    weight, temperature = 0.5, 0.8
    Q = layer.forward_batch(H, temperature).copy()
    W = layer._W.copy()  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]
    H0 = H.copy()  # sg[h] in the codebook loss: _numeric moves H in place
    assert np.allclose(W.sum(axis=1), 1.0)
    assert layer.indices.tolist() == layer.nearest(H).tolist()

    def total() -> float:
        probe = CodesArrayLayer(4, S)
        probe.E = layer.E
        mixed = float(np.sum(G * probe.forward_batch(H, temperature)))
        committed = 0.25 * np.sum((H - Q) ** 2)
        codebook = np.sum(W[:, :, np.newaxis] * (H0[:, np.newaxis, :] - layer.E[np.newaxis, :, :]) ** 2)
        return mixed + float(weight * (committed + codebook))

    expected_loss = float(weight * (0.25 * np.sum((H - Q) ** 2))) + float(
        weight * np.sum(W[:, :, np.newaxis] * (H[:, np.newaxis, :] - layer.E[np.newaxis, :, :]) ** 2)
    )
    assert layer.loss_batch(weight) == pytest.approx(expected_loss, rel=1e-12)
    dH = layer.backward_batch(G, weight)
    _close(_numeric(total, layer.E), layer.grad_E)
    _close(_numeric(total, H), dH)


def test_soft_codes_cool_to_the_snap(rng: np.random.Generator):
    layer = CodesArrayLayer(4, S)
    layer.draw(rng)
    H = rng.normal(size=(N, S))
    assert np.allclose(layer.forward_batch(H, 1e-6), layer.forward_batch(H), atol=1e-12)


def test_a_target_moves_the_commitments_pull_and_leaves_the_codebooks(rng: np.random.Generator):
    # the code-consolidation workplan, D1: commitment * ||h - sg[e_target]||^2 moves h; the
    # codebook loss still moves each chosen code toward its h
    layer = CodesArrayLayer(4, S, commitment=0.25)
    layer.draw(rng)
    H = rng.normal(size=(N, S))
    weight = 0.5
    layer.forward_batch(H)
    indices = layer.indices.copy()
    layer.targets = (indices + 1) % 4
    targets = layer.targets.copy()

    def codebook() -> float:
        return float(weight * np.sum((H - layer.E[indices]) ** 2))

    def commitment() -> float:
        return float(weight * 0.25 * np.sum((H - layer.E[targets]) ** 2))

    dQ = rng.normal(size=(N, S))
    dH = layer.backward_batch(np.zeros((N, S)), weight)
    _close(_numeric(codebook, layer.E), layer.grad_E)
    _close(_numeric(commitment, H), dH)
    assert layer.loss_batch(weight) == pytest.approx(codebook() + commitment(), rel=1e-14)
    assert np.array_equal(layer.backward_batch(dQ, weight), dQ + (2.0 * weight * 0.25) * (H - layer.E[targets]))


def test_its_own_code_as_the_target_is_no_target_by_bits(rng: np.random.Generator):
    layer = CodesArrayLayer(4, S, commitment=0.25)
    layer.draw(rng)
    H, dQ = rng.normal(size=(N, S)), rng.normal(size=(N, S))
    layer.forward_batch(H)
    untargeted = (layer.backward_batch(dQ, 0.5), layer.grad_E.copy(), layer.loss_batch(0.5))
    layer.reset_gradient_accum()
    layer.targets = layer.indices.copy()
    targeted = (layer.backward_batch(dQ, 0.5), layer.grad_E.copy(), layer.loss_batch(0.5))
    assert np.array_equal(untargeted[0], targeted[0]) and np.array_equal(untargeted[1], targeted[1])
    assert untargeted[2] == targeted[2]


def test_codes_scatter_their_gradient_to_the_codes_chosen_only():
    layer = CodesArrayLayer(3, 1)
    layer.E = np.array([[0.0], [10.0], [20.0]])
    layer.forward_batch(np.array([[1.0], [2.0], [9.0]]))
    layer.backward_batch(np.zeros((3, 1)), 1.0)
    assert layer.grad_E.tolist() == [[-6.0], [2.0], [0.0]]


# the snap wall (docs/snap-wall-workplan.md)

WALLED = 6  # states, one at each pair of 4 codes' midpoint


def _walled_pair(
    rng: np.random.Generator, gate: FloatArray | None
) -> tuple[CodesArrayLayer, CodesArrayLayer, FloatArray]:
    # the same codes with the wall on and off, and states near boundaries: midpoints of code pairs, jittered
    on, off = CodesArrayLayer(4, S, wall=1.0), CodesArrayLayer(4, S)
    on.draw(rng)
    off.E = on.E.copy()
    pairs = np.array(list(itertools.combinations(range(4), 2)))  # every pair of distinct codes
    H = (on.E[pairs[:, 0]] + on.E[pairs[:, 1]]) / 2 + 0.01 * rng.normal(size=(len(pairs), S))
    on.gate = gate
    return on, off, H


@pytest.mark.parametrize("gated", [False, True])
def test_the_snap_walls_gradient_matches_central_differences(rng: np.random.Generator, gated: bool):
    gate = (np.arange(WALLED) % 2).astype(float) if gated else None
    on, off, H = _walled_pair(rng, gate)
    weight = 0.5
    on.forward_batch(H)
    off.forward_batch(H.copy())
    assert on.wall_loss_batch(weight) > 0.0
    zeros = np.zeros((WALLED, S))
    dH = on.backward_batch(zeros, weight) - off.backward_batch(zeros, weight)
    grad_E = on.grad_E - off.grad_E
    _close(_numeric(lambda: on.wall_loss_batch(weight), on.E), grad_E)
    _close(_numeric(lambda: on.wall_loss_batch(weight), H), dH)
    if gate is not None:
        assert np.all(dH[gate == 0.0] == 0.0)


def test_the_snap_wall_is_silent_past_its_ratio_and_when_off(rng: np.random.Generator):
    layer = CodesArrayLayer(2, 1, wall=1.0, wall_ratio=1.5)
    layer.E = np.array([[0.0], [10.0]])
    # distances 1 and 9 (ratio 9, past 1.5) and 4.5 and 5.5 (ratio 1.22, inside it)
    layer.forward_batch(np.array([[1.0], [4.5]]))
    assert layer.wall_loss_batch(1.0) == pytest.approx((np.log(1.5**2) - np.log(5.5**2 / 4.5**2)) ** 2)
    dH = layer.backward_batch(np.zeros((2, 1)), 1.0) - 2.0 * 0.25 * (np.array([[1.0], [4.5]]) - layer._Q)
    assert dH[0, 0] == 0.0 and dH[1, 0] > 0.0  # descent moves the near state back to its code at 0
    off = CodesArrayLayer(2, 1)
    off.E = layer.E.copy()
    off.forward_batch(np.array([[4.5]]))
    assert off.wall_loss_batch(1.0) == 0.0
