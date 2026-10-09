"""
Patch models in numpy (the layer-norm and attention workplan, stage 2; the multi-head attention
workplan, stage 3; README, Layer norm and attention): what randomize draws, Patches, Position and
TokenMean against the README's indices and sums, attention's softmax and its backward per head by
bits against a scalar transcription, its packed head blocks, the exact tests (one token, uniform
attention, identical heads, a silent head, an identity attention block, the blocks called in turn),
and a short MNIST run in which the README's model learns. The gradient check, the README model's wiring and learn against a batch of
one are tests/model/networks/test_attention_network.py's, on every implementation.
"""

import math
import random
from typing import Any

import numpy as np
import pytest

from indrajala_ml.data.mnist_data import load_mnist_dataset
from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.layers.numpy import attention_array_layer
from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.layers.numpy.attention_array_layer import AttentionArrayLayer
from indrajala_ml.model.layers.numpy.token_array_layer import (
    PatchesArrayLayer,
    PositionArrayLayer,
    TokenMeanArrayLayer,
)
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    Dense,
    LayerNorm,
    LayerSpec,
    Patches,
    Position,
    Residual,
    TokenMean,
)
from indrajala_ml.model.specs.spec_shapes import InputShape
from indrajala_ml.model.specs.update_rules import SGD, Adam, UpdateRule
from tests.gradient_check import analytic_gradients
from tests.helpers import bits, exp_by_math, patching, split
from tests.model.specs.test_layer_specs import ATTENTION_BLOCK, EMBED, FFN_BLOCK, PATCHES, SOFTMAX

# test_layer_specs' patch models read a (4, 4, 1) image: Patches(2) gives 4 tokens of 4
IMAGE: InputShape = (4, 4, 1)


def network(specs: list[LayerSpec], rule: UpdateRule | None = None, seed: int = 3, image: InputShape = IMAGE) -> Any:
    built = SequentialArrayNetwork(image, specs, SGD() if rule is None else rule, backend=NUMPY)
    built.rng = NUMPY.default_rng(seed)
    built.randomize()
    return built


def rows(count: int, seed: int = 1, size: int = 16, classes: int = 3) -> list[tuple[tuple[float, ...], int]]:
    rng = random.Random(seed)
    return [(tuple(rng.uniform(0.0, 1.0) for _ in range(size)), i % classes) for i in range(count)]


# the softmax's exp as math.exp, so a scalar transcription computes the same bits
math_exp = patching(attention_array_layer, "exp", exp_by_math)


def test_randomize_draws_attentions_projections_in_order_and_nothing_for_the_parameter_free_layers():
    built = network([PATCHES, EMBED, Position(), ATTENTION_BLOCK, TokenMean(), LayerNorm(), SOFTMAX])
    rng = np.random.default_rng(3)
    expected: list[Any] = []
    # the embedding, attention's Wq, bq, Wk, bk, Wv, bv, Wo, bo, then the output layer
    for rows_, fan_in in [(6, 4), (6, 6), (6, 6), (6, 6), (6, 6), (3, 6)]:
        limit = 1.0 / np.sqrt(fan_in)
        expected += [rng.uniform(-limit, limit, (rows_, fan_in)), rng.uniform(-limit, limit, rows_)]
    snapshot = built.snapshot()
    assert bits([array for i in (1, 5, 9) for array in snapshot[i]]) == bits(expected)
    # positions, gamma and beta start at 0, 1 and 0
    assert bits(snapshot[2]) == bits([np.zeros((4, 6))])
    assert bits([*snapshot[4], *snapshot[8]]) == bits([np.ones(6), np.zeros(6)] * 2)


def test_patches_follow_the_readmes_indices_and_send_the_inverse_permutation_back():
    height, width, channels, p = 6, 4, 3, 2
    layer = PatchesArrayLayer(height, width, channels, p)
    X = np.arange(2 * height * width * channels, dtype=np.float64).reshape(2, -1)
    tokens = layer.forward_batch(X)

    features = p * p * channels
    for n in range(2):
        for u in range(height // p):
            for v in range(width // p):
                for c in range(channels):
                    for i in range(p):
                        for j in range(p):
                            t, k = u * (width // p) + v, c * p * p + i * p + j
                            image = c * height * width + (u * p + i) * width + (v * p + j)
                            assert tokens[n, t * features + k] == X[n, image]

    layer.delta_batch = tokens
    assert bits(layer.downstream_batch()) == bits(X)


def test_position_adds_its_table_into_a_new_array_and_sums_its_gradient_over_the_examples():
    layer = PositionArrayLayer(2, 3)
    layer.P = np.arange(6, dtype=np.float64).reshape(2, 3) / 8
    X = np.ones((3, 6))
    out = layer.forward_batch(X)
    assert out is not X and bits(X) == bits(np.ones((3, 6)))
    assert bits(out) == bits(1.0 + np.tile(layer.P.reshape(6), (3, 1)))

    delta = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.7, 0.0, -1.0, 2.5, 0.1, 0.3]])
    layer.delta_batch = delta
    layer.accumulate_gradient_batch(X)
    expected = [((0.0 + delta[0, k]) + delta[1, k]) + delta[2, k] for k in range(6)]
    assert bits(layer.grad_P) == bits(np.array(expected).reshape(2, 3))
    assert layer.downstream_batch() is delta


def test_the_token_mean_is_a_left_fold_over_the_tokens_divided_by_their_count():
    layer = TokenMeanArrayLayer(3, 2)
    X = np.array([[0.1, 1e16, 0.2, 1.0, 0.3, -1e16]])
    expected = [((0.0 + X[0, j]) + X[0, 2 + j]) + X[0, 4 + j] for j in range(2)]
    assert bits(layer.forward_batch(X)) == bits(np.array([expected]) / 3)

    layer.delta_batch = np.array([[3.0, -1.5]])
    assert bits(layer.downstream_batch()) == bits(np.array([[1.0, -0.5] * 3]))


def _softmax_rows(S: list[list[float]]) -> list[list[float]]:
    # the README's softmax, max-shifted, a left fold for the sum
    P: list[list[float]] = []
    for row in S:
        m = max(row)
        e = [math.exp(s - m) for s in row]
        total = 0.0
        for value in e:
            total += value
        P.append([value / total for value in e])
    return P


# (heads, key_size) over 6 features: one head as step 4's, heads splitting the width, and key sizes
# that make h * d_k differ from d both ways (the defaults hide d against h * d_k mix-ups)
HEADS = [(1, None), (2, None), (3, None), (2, 4), (4, 1)]
HEAD_IDS = ["one head", "two heads of 3", "three heads of 2", "two heads of 4", "four heads of 1"]


def _attention_block(heads: int, key_size: int | None) -> Residual:
    return Residual((LayerNorm(), Attention(heads=heads, key_size=key_size)))


def _block(heads: FloatArray, i: int, key_size: int) -> FloatArray:
    # head i's columns of (..., h * d_k) rows
    return heads[..., i * key_size : (i + 1) * key_size]


@pytest.mark.usefixtures("math_exp")
@pytest.mark.parametrize(("heads", "key_size"), HEADS, ids=HEAD_IDS)
def test_attentions_softmax_and_its_backward_follow_the_readme_per_head_by_bits(heads: int, key_size: int | None):
    built = network([PATCHES, EMBED, _attention_block(heads, key_size), TokenMean(), SOFTMAX])
    batch = rows(3)
    built.learn_batch(0.5, batch)
    attention = built.layers[4]
    assert isinstance(attention, AttentionArrayLayer)
    d_k = attention.key_size
    assert d_k == (6 // heads if key_size is None else key_size)
    width = heads * d_k

    Wo = attention.Wo.copy()  # the optimizer steps it in place after the backward pass
    built.learn_batch(0.5, batch)  # the forward and backward passes to compare against
    # the caches: Q, K, V as (N, h, T, d_k) views, P (N, h, T, T), H and dQ, dK, dV as (N * T, h * d_k)
    Q, K, V, P = attention._Q, attention._K, attention._V, attention._P
    assert (Q.shape, P.shape, attention._H.shape, attention._dQ.shape) == (
        (3, heads, 4, d_k),
        (3, heads, 4, 4),
        (12, width),
        (12, width),
    )
    dH = (attention.delta_batch.reshape(-1, 6) @ Wo).reshape(3, 4, width)
    for i in range(heads):
        Qi, Ki, Vi, Pi = Q[:, i], K[:, i], V[:, i], P[:, i]
        S = (Qi @ Ki.transpose(0, 2, 1)) / math.sqrt(d_k)
        for n in range(3):
            assert bits(Pi[n]) == bits(np.array(_softmax_rows(S[n].tolist())))
        H = attention._H.reshape(3, 4, width)
        assert bits(_block(H, i, d_k)) == bits(Pi @ Vi)

        # dS from dP: r_t = sum_u(dP_tu * P_tu), a left fold, then P_tu * (dP_tu - r_t)
        dHi = _block(dH, i, d_k)
        dP = dHi @ Vi.transpose(0, 2, 1)
        dS = np.empty_like(dP)
        for n in range(3):
            for t in range(4):
                r = 0.0
                for u in range(4):
                    r += dP[n, t, u] * Pi[n, t, u]
                dS[n, t] = [Pi[n, t, u] * (dP[n, t, u] - r) for u in range(4)]
        dQ, dK, dV = (
            attention._dQ.reshape(3, 4, width),
            attention._dK.reshape(3, 4, width),
            attention._dV.reshape(3, 4, width),
        )
        assert bits(_block(dQ, i, d_k)) == bits((dS @ Ki) / math.sqrt(d_k))
        assert bits(_block(dK, i, d_k)) == bits((dS.transpose(0, 2, 1) @ Qi) / math.sqrt(d_k))
        assert bits(_block(dV, i, d_k)) == bits(Pi.transpose(0, 2, 1) @ dHi)


def _parameters(layer: AttentionArrayLayer, seed: int) -> list[FloatArray]:
    rng = np.random.default_rng(seed)
    return [rng.uniform(-0.5, 0.5, parameter.shape) for parameter in layer.parameters()]


@pytest.mark.parametrize(("heads", "key_size"), [(1, None), (2, None), (3, 4)], ids=["one head", "two", "three of 4"])
def test_the_projections_are_packed_head_blocks_and_wos_fan_in_is_their_width(heads: int, key_size: int | None):
    layer = AttentionArrayLayer(4, 6, heads, key_size)
    width = heads * layer.key_size
    assert layer.projection_shapes == ((width, 6), (width, 6), (width, 6), (6, width))
    assert [p.shape for p in layer.parameters()] == [(width, 6), (width,)] * 3 + [(6, width), (6,)]
    assert [g.shape for g in layer.gradients()] == [p.shape for p in layer.parameters()]
    assert layer.scale == math.sqrt(layer.key_size)


def test_randomize_draws_wo_at_fan_in_heads_times_key_size():
    built = network([PATCHES, EMBED, _attention_block(3, 4), TokenMean(), SOFTMAX])
    rng = np.random.default_rng(3)
    expected: list[Any] = []
    # the embedding, attention's three (12, 6) projections and Wo (6, 12), then the output layer
    for rows_, fan_in in [(6, 4), (12, 6), (12, 6), (12, 6), (6, 12), (3, 6)]:
        limit = 1.0 / np.sqrt(fan_in)
        expected += [rng.uniform(-limit, limit, (rows_, fan_in)), rng.uniform(-limit, limit, rows_)]
    snapshot = built.snapshot()
    assert bits([array for i in (1, 4, 7) for array in snapshot[i]]) == bits(expected)


@pytest.mark.parametrize(("heads", "key_size"), HEADS, ids=HEAD_IDS)
def test_one_token_attends_only_to_itself_so_attention_is_two_affine_maps_by_bits(heads: int, key_size: int | None):
    layer = AttentionArrayLayer(1, 6, heads, key_size)
    layer.set_parameters(_parameters(layer, 1))
    *_, Wv, bv, Wo, bo = layer.parameters()
    X = np.random.default_rng(2).uniform(-1.0, 1.0, (4, 6))

    out = layer.forward_batch(X)
    assert bits(layer._P) == bits(np.ones((4, heads, 1, 1)))
    assert bits(out) == bits((X @ Wv.T + bv) @ Wo.T + bo)


@pytest.mark.parametrize(
    ("heads", "key_size"), [(1, None), (2, None), (4, None), (4, 3)], ids=["1", "2", "4", "4 of 3"]
)
def test_zero_queries_and_keys_weigh_every_token_exactly_one_sixteenth(heads: int, key_size: int | None):
    # T = 16: every score is 0, every weight 1/16, and each head's H is the token mean of its V
    layer = AttentionArrayLayer(16, 8, heads, key_size)
    parameters = _parameters(layer, 3)
    for i in range(4):  # Wq, bq, Wk, bk
        parameters[i] = np.zeros_like(parameters[i])
    layer.set_parameters(parameters)
    X = np.random.default_rng(4).uniform(-1.0, 1.0, (3, 16 * 8))

    layer.forward_batch(X)
    assert bits(layer._P) == bits(np.full((3, heads, 16, 16), 1 / 16))
    width = layer.width
    V = layer._V.transpose(0, 2, 1, 3).reshape(3, 16 * width)
    mean = TokenMeanArrayLayer(16, width).forward_batch(V)
    assert bits(layer._H.reshape(3, 16, width)) == bits(np.broadcast_to(mean[:, np.newaxis, :], (3, 16, width)))


@pytest.mark.parametrize(("heads", "key_size"), [(2, None), (3, None), (2, 4)], ids=["2", "3", "2 of 4"])
def test_identical_heads_weigh_and_mix_alike_by_bits(heads: int, key_size: int | None):
    layer = AttentionArrayLayer(4, 6, heads, key_size)
    parameters = _parameters(layer, 5)
    d_k = layer.key_size
    for i in range(6):  # each head's blocks of Wq, bq, Wk, bk, Wv, bv: head 0's
        parameters[i] = np.concatenate([parameters[i][:d_k]] * heads)
    layer.set_parameters(parameters)
    X = np.random.default_rng(6).uniform(-1.0, 1.0, (3, 4 * 6))

    layer.forward_batch(X)
    H = layer._H.reshape(3, 4, layer.width)
    for i in range(1, heads):
        assert bits(layer._P[:, i]) == bits(layer._P[:, 0])
        assert bits(_block(H, i, d_k)) == bits(_block(H, 0, d_k))


@pytest.mark.parametrize(("heads", "key_size"), [(2, None), (3, None), (2, 4)], ids=["2", "3", "2 of 4"])
def test_a_silent_heads_projection_gradients_are_exactly_zero(heads: int, key_size: int | None):
    # with head i's columns of Wo zero, dH[i] is exactly zero, and so is all that flows from it
    layer = AttentionArrayLayer(4, 6, heads, key_size)
    parameters = _parameters(layer, 7)
    d_k, silent = layer.key_size, heads - 1
    parameters[6][:, silent * d_k : (silent + 1) * d_k] = 0.0
    layer.set_parameters(parameters)
    rng = np.random.default_rng(8)
    X = rng.uniform(-1.0, 1.0, (3, 4 * 6))
    layer.forward_batch(X)
    layer._backward(rng.uniform(-0.5, 0.5, (3, 4 * 6)))
    layer.accumulate_gradient_batch(X)

    gradients = layer.gradients()
    for weight, bias in ((0, 1), (2, 3), (4, 5)):  # Wq, bq; Wk, bk; Wv, bv
        assert not np.any(gradients[weight][silent * d_k : (silent + 1) * d_k])
        assert not np.any(gradients[bias][silent * d_k : (silent + 1) * d_k])
        assert np.any(gradients[weight][: silent * d_k])  # the others' blocks are not


@pytest.mark.parametrize(("heads", "key_size"), HEADS, ids=HEAD_IDS)
def test_the_blocks_called_in_turn_are_the_layers_passes_by_bits(heads: int, key_size: int | None):
    layer = AttentionArrayLayer(4, 6, heads, key_size)
    layer.set_parameters(_parameters(layer, 9))
    rng = np.random.default_rng(10)
    X, delta = rng.uniform(-1.0, 1.0, (3, 4 * 6)), rng.uniform(-0.5, 0.5, (3, 4 * 6))

    Q, K, V = layer._project(X.reshape(-1, 6))
    P, H = layer._attend(layer._heads(Q), layer._heads(K), layer._heads(V))
    out = layer._combine(layer._side_by_side(H))
    assert bits(layer.forward_batch(X)) == bits(out.reshape(3, -1))

    dQ, dK, dV = layer._attend_backward(layer._heads(layer._combine_backward(delta.reshape(-1, 6))))
    dX = layer._project_backward(layer._side_by_side(dQ), layer._side_by_side(dK), layer._side_by_side(dV))
    layer._backward(delta)
    assert bits(layer.downstream_batch()) == bits(dX.reshape(3, -1))
    assert bits(layer._P) == bits(P)


def test_an_identity_attention_block_changes_no_output_and_no_other_layers_gradient_by_bits():
    # with Wo and bo zero the block adds 0 forward, and backward its body's downstream is 0
    plain_specs: list[LayerSpec] = [PATCHES, EMBED, Position(), FFN_BLOCK, TokenMean(), SOFTMAX]
    plain = network(plain_specs, seed=5)
    plain.learn_batch(0.5, rows(4, seed=2))  # positions, gamma and beta off their initial values
    block_specs: list[LayerSpec] = [PATCHES, EMBED, Position(), ATTENTION_BLOCK, FFN_BLOCK, TokenMean(), SOFTMAX]
    blocked = network(block_specs, seed=7)

    snapshot = plain.snapshot()
    attention = list(blocked.snapshot()[5])
    attention[6], attention[7] = np.zeros_like(attention[6]), np.zeros_like(attention[7])
    blocked.restore([*snapshot[:3], (), blocked.snapshot()[4], tuple(attention), (), *snapshot[3:]])

    batch = rows(5)
    states, labels = split(batch)
    for state in states:
        assert bits(blocked._forward(state)) == bits(plain._forward(state))

    outside = [0, 1, 2, *range(7, 14)]
    plain_gradients = analytic_gradients(plain, states, labels)
    blocked_gradients = analytic_gradients(blocked, states, labels)
    assert [np.asarray(g).tobytes() for i in outside for g in blocked_gradients[i]] == [
        np.asarray(g).tobytes() for layer in plain_gradients for g in layer
    ]


def test_on_mnist_the_readmes_patch_model_learns():
    data = load_mnist_dataset("data/mnist/mnist-train.bin", limit=640)
    train, held = data[:512], data[512:]
    readme: list[LayerSpec] = [
        Patches(7),
        Dense(32, activation="linear", bias=True),
        Position(),
        Residual((LayerNorm(), Attention())),
        Residual((LayerNorm(), Dense(64, activation="relu"), Dense(32, activation="linear", bias=True))),
        TokenMean(),
        LayerNorm(),
        Dense(10, activation="softmax", output=True, loss="cross_entropy"),
    ]
    built = network(readme, Adam(), image=(28, 28, 1))

    def held_loss() -> float:
        outputs = NUMPY.matrix([state for state, _ in held])
        for layer in built.layers:
            outputs = layer.forward_batch(outputs)
        return -float(np.mean([math.log(outputs[i, label]) for i, (_, label) in enumerate(held)]))

    before = held_loss()
    for _epoch in range(3):
        for start in range(0, len(train), 32):
            built.learn_batch(1e-3, train[start : start + 32])
    after = held_loss()
    # at initialization about ln 10 = 2.30
    assert before > 2.0 and after < before - 0.5
