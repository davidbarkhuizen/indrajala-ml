"""
Rotary positions in numpy (the rotary-positions workplan, stage 3): the table (D4: math's cos and sin
of t * theta_j, whose rows don't change as it grows), the rotation at the end of project, after the
biases, on Q and K and not V (D3: feature j with j + d_k / 2), and its inverse on dQ and dK before
project's products, both by bits against Python's floats; the rotation keeps each pair's norm; a
score depends on its tokens' distance only; position 0 is unrotated, so one token is the unrotated
layer by bits; the gradient check, learn against a batch of one, save and load by bits, and a short
run in which a causal model without a Position learns a text.
"""

import math
import random
from typing import Any

import numpy as np
import pytest

from indrajala_ml.data.text_data import Vocabulary, windows
from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.layers.numpy.attention_array_layer import AttentionArrayLayer
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.layer_specs import Attention, Dense, Embedding, LayerNorm, LayerSpec, Residual
from indrajala_ml.model.specs.rotary_table import rotary_table
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.training.sequence_evaluate import sequence_evaluate
from indrajala_ml.training.train import train_backprop_network_mini_batch
from tests.gradient_check import check_gradients
from tests.helpers import bits
from tests.model.networks.test_sequence_array_network import network, rows
from tests.model.specs.test_layer_specs import ROTARY

# a batch of 3 examples of 5 tokens of 8 features
N, T, D = 3, 5, 8
HEADS = [(1, None), (2, None), (2, 2), (3, 4)]
HEAD_IDS = ["one head of 8", "two of 4", "two of 2", "three of 4"]


def _attention(heads: int, key_size: int | None, positions: str = "rotary", tokens: int = T) -> AttentionArrayLayer:
    layer = AttentionArrayLayer(tokens, D, heads, key_size, positions=positions, rotary_base=100.0)  # pyright: ignore[reportArgumentType]
    rng = np.random.default_rng(0)
    layer.set_parameters([rng.uniform(-1.0, 1.0, parameter.shape) for parameter in layer.parameters()])
    return layer


def _X(tokens: int = T, seed: int = 1) -> Any:
    return np.random.default_rng(seed).uniform(-1.0, 1.0, (N, tokens * D))


def _rotated(rows: Any, layer: AttentionArrayLayer, inverse: bool = False) -> list[list[float]]:
    # (N * T, h * d_k) rows rotated in Python's floats, one entry at a time: the workplan's form
    cos, sin = rotary_table(layer.tokens, layer.key_size, layer.rotary_base)
    d_k, half = layer.key_size, layer.key_size // 2
    out: list[list[float]] = []
    for r, row in enumerate(rows.tolist()):
        t = r % layer.tokens
        new = list(row)
        for i in range(layer.heads):
            for j in range(half):
                a, b = row[i * d_k + j], row[i * d_k + j + half]
                c, s = cos[t][j], sin[t][j]
                if inverse:
                    new[i * d_k + j], new[i * d_k + j + half] = a * c + b * s, b * c - a * s
                else:
                    new[i * d_k + j], new[i * d_k + j + half] = a * c - b * s, a * s + b * c
        out.append(new)
    return out


# the table


def test_the_table_is_maths_cos_and_sin_of_t_times_theta_j():
    cos, sin = rotary_table(3, 4, 100.0)
    thetas = [100.0 ** (-2 * j / 4) for j in range(2)]
    assert cos == [[math.cos(t * theta) for theta in thetas] for t in range(3)]
    assert sin == [[math.sin(t * theta) for theta in thetas] for t in range(3)]
    assert cos[0] == [1.0, 1.0] and sin[0] == [0.0, 0.0]


def test_a_longer_table_keeps_a_shorter_ones_rows():
    # generation extends it past a window's tokens
    short, long = rotary_table(5, 6, 10000.0), rotary_table(64, 6, 10000.0)
    assert long[0][:5] == short[0] and long[1][:5] == short[1]


@pytest.mark.parametrize(("positions", "key_size", "base"), [(3, 3, 10.0), (0, 4, 10.0), (3, 4, 0.0)])
def test_the_table_refuses_an_odd_head_no_positions_and_a_base_of_zero(positions: int, key_size: int, base: float):
    with pytest.raises(AssertionError):
        rotary_table(positions, key_size, base)


# the rotation


@pytest.mark.parametrize(("heads", "key_size"), HEADS, ids=HEAD_IDS)
def test_project_rotates_q_and_k_after_their_biases_and_leaves_v_by_bits(heads: int, key_size: int | None):
    layer = _attention(heads, key_size)
    plain = _attention(heads, key_size, positions="none")
    R = _X().reshape(-1, D)
    Q, K, V = layer._project(R)  # pyright: ignore[reportPrivateUsage]
    Q0, K0, V0 = plain._project(R)  # pyright: ignore[reportPrivateUsage]
    assert bits(Q) == bits(_rotated(Q0, layer))
    assert bits(K) == bits(_rotated(K0, layer))
    assert bits(V) == bits(V0)


@pytest.mark.parametrize(("heads", "key_size"), HEADS, ids=HEAD_IDS)
def test_the_backward_pass_rotates_dq_and_dk_back_before_projects_products_by_bits(heads: int, key_size: int | None):
    layer = _attention(heads, key_size)
    X, delta = _X(), _X(seed=2)
    layer.forward_batch(X)
    layer._backward(delta)  # pyright: ignore[reportPrivateUsage]
    # attend's gradients of the rotated Q and K, then the inverse rotation, then the products
    dQ, dK, dV = layer._attend_backward(layer._heads(layer._combine_backward(delta.reshape(-1, D))))  # pyright: ignore[reportPrivateUsage]
    dQ, dK, dV = (layer._side_by_side(g) for g in (dQ, dK, dV))  # pyright: ignore[reportPrivateUsage]
    assert bits(layer._dQ) == bits(_rotated(dQ, layer, inverse=True))  # pyright: ignore[reportPrivateUsage]
    assert bits(layer._dK) == bits(_rotated(dK, layer, inverse=True))  # pyright: ignore[reportPrivateUsage]
    assert bits(layer._dV) == bits(dV)  # pyright: ignore[reportPrivateUsage]
    dX = layer._project_backward(layer._dQ, layer._dK, layer._dV)  # pyright: ignore[reportPrivateUsage]
    assert bits(layer.downstream_batch()) == bits(dX.reshape(N, -1))


def test_the_rotation_keeps_each_pairs_norm():
    layer = _attention(2, None)
    plain = _attention(2, None, positions="none")
    R = _X().reshape(-1, D)
    Q, Q0 = layer._project(R)[0], plain._project(R)[0]  # pyright: ignore[reportPrivateUsage]
    pairs = [(Q[:, i * 4 + j], Q[:, i * 4 + j + 2]) for i in range(2) for j in range(2)]
    pairs0 = [(Q0[:, i * 4 + j], Q0[:, i * 4 + j + 2]) for i in range(2) for j in range(2)]
    for (a, b), (a0, b0) in zip(pairs, pairs0, strict=True):
        np.testing.assert_allclose(np.hypot(a, b), np.hypot(a0, b0), rtol=1e-15, atol=1e-15)
    assert not np.array_equal(Q, Q0)


@pytest.mark.parametrize(("heads", "key_size"), HEADS, ids=HEAD_IDS)
def test_a_score_depends_on_its_tokens_distance_only(heads: int, key_size: int | None):
    # every token the same: unrotated every score of a row is the same; rotated, the score of
    # tokens t and u is a function of t - u, the same along each diagonal, within rounding
    tokens = 7
    layer = _attention(heads, key_size, tokens=tokens)
    row = np.random.default_rng(3).uniform(-1.0, 1.0, D)
    layer.forward_batch(np.tile(row, (1, tokens)))
    for i in range(heads):
        S = layer._Q[0, i] @ layer._K[0, i].T  # pyright: ignore[reportPrivateUsage]
        for offset in range(-tokens + 1, tokens):
            diagonal = np.diagonal(S, offset)
            np.testing.assert_allclose(diagonal, np.full_like(diagonal, diagonal[0]), rtol=1e-12, atol=1e-12)
        assert not np.allclose(S[0], S[0, 0])  # the distance matters


@pytest.mark.parametrize("causal", [False, True], ids=["unmasked", "causal"])
@pytest.mark.parametrize(("heads", "key_size"), HEADS, ids=HEAD_IDS)
def test_one_token_at_position_0_is_the_unrotated_layer_by_bits(heads: int, key_size: int | None, causal: bool):
    rotary = _attention(heads, key_size, tokens=1)
    plain = _attention(heads, key_size, positions="none", tokens=1)
    rotary.causal = plain.causal = causal
    X, delta = _X(tokens=1), _X(tokens=1, seed=2)
    for layer in (rotary, plain):
        layer.forward_batch(X)
        layer._backward(delta)  # pyright: ignore[reportPrivateUsage]
        layer.accumulate_gradient_batch(X)
    assert bits(rotary.A) == bits(plain.A)
    assert bits(rotary.downstream_batch()) == bits(plain.downstream_batch())
    assert bits(rotary.gradients()) == bits(plain.gradients())


def test_position_0s_queries_and_keys_are_unrotated_by_bits():
    layer, plain = _attention(2, None), _attention(2, None, positions="none")
    X = _X()
    layer.forward_batch(X)
    plain.forward_batch(X)
    assert bits(layer._Q[:, :, 0]) == bits(plain._Q[:, :, 0])  # pyright: ignore[reportPrivateUsage]
    assert bits(layer._K[:, :, 0]) == bits(plain._K[:, :, 0])  # pyright: ignore[reportPrivateUsage]
    assert bits(layer._V) == bits(plain._V)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("positions", "key_size"), [("alibi", None), ("rotary", 3)], ids=["an unknown scheme", "an odd head"]
)
def test_an_attention_refuses_an_unknown_scheme_and_an_odd_rotary_head(positions: str, key_size: int | None):
    with pytest.raises(AssertionError):
        AttentionArrayLayer(T, D, 2, key_size, positions=positions)  # pyright: ignore[reportArgumentType]


# networks

SEQUENCE_ROTARY = {name: specs for name, specs in ROTARY.items()}


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("name", SEQUENCE_ROTARY)
def test_every_gradient_matches_its_finite_difference(name: str, batch_size: int):
    specs = SEQUENCE_ROTARY[name]
    built = network(specs)
    built.learn_batch(0.5, rows(specs, 4, seed=2))
    states, labels = zip(*rows(specs, batch_size))
    check_gradients(built, states, labels, rng=lambda: NUMPY.default_rng(11))  # "with dropout" holds its masks


def test_a_causal_model_with_rotary_attention_and_no_position_learns_a_text():
    # test_sequence_array_network's text and model, its Position out and its attention rotary
    text = "".join(random.Random(4).choice(["abc", "abd", "bad", "cab"]) for _ in range(1200))
    vocabulary = Vocabulary.of(text)
    examples = windows(vocabulary.encode(text), 8)
    size = len(vocabulary)
    specs: list[LayerSpec] = [
        Embedding(size, 16),
        Residual((LayerNorm(), Attention(heads=2, causal=True, positions="rotary"))),
        Residual((LayerNorm(), Dense(32, activation="relu"), Dense(16, activation="linear", bias=True))),
        LayerNorm(),
        Dense(size, output=True, activation="softmax", loss="cross_entropy"),
    ]
    built = SequentialArrayNetwork((8,), specs, Adam(), shape="sequence")
    built.rng = NUMPY.default_rng(5)
    built.randomize()
    before = sequence_evaluate(built, examples)
    train_backprop_network_mini_batch(built, examples, 16, 0.01, epochs=8, rng=random.Random(5))
    after = sequence_evaluate(built, examples)
    assert before.bits_per_token > 1.5 and after.bits_per_token < 0.9 * before.bits_per_token
    assert after.accuracy > 0.6
