"""
Patch models in numpy (the layer-norm and attention workplan, stage 2; README, Layer norm and
attention): the gradient check on every accepted patch model, the layers the builder wires and
what randomize draws, Patches, Position and TokenMean against the README's indices and sums,
attention's softmax and its backward by bits against a scalar transcription, the exact tests (one
token, uniform attention, an identity attention block), learn against a batch of one, and a short
MNIST run in which the README's model learns.
"""

import math
import random
from typing import Any

import numpy as np
import pytest

from indrajala_ml.mnist_data import load_mnist_dataset
from indrajala_ml.model import attention_array_layer
from indrajala_ml.model.array_backend import NUMPY
from indrajala_ml.model.array_layer import FloatArray
from indrajala_ml.model.attention_array_layer import AttentionArrayLayer
from indrajala_ml.model.layer_norm_array_layer import LayerNormArrayLayer
from indrajala_ml.model.layer_specs import (
    Attention,
    Dense,
    InputShape,
    LayerNorm,
    LayerSpec,
    Patches,
    Position,
    Residual,
    TokenMean,
    batch_norm_index,
)
from indrajala_ml.model.residual_array_layer import AddArrayLayer, ForkArrayLayer
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.token_array_layer import (
    PatchesArrayLayer,
    PositionArrayLayer,
    TokenDenseArrayLayer,
    TokenMeanArrayLayer,
)
from indrajala_ml.model.update_rules import SGD, Adam, Momentum, UpdateRule
from tests.gradient_check import analytic_gradients, check_gradients
from tests.helpers import bits, exp_by_math, patching, split
from tests.test_layer_specs import ATTENTION_BLOCK, EMBED, FFN_BLOCK, PATCHES, SOFTMAX, TOKENS

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


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("name", TOKENS)
def test_every_gradient_matches_its_finite_difference(name: str, batch_size: int):
    if batch_size == 1 and batch_norm_index(TOKENS[name]) is not None:
        pytest.skip("batch norm trains on batches only (the batch-norm workplan, D4)")
    built = network(TOKENS[name])
    # a trained step first, so gamma, beta and the positions aren't at their initial values
    built.learn_batch(0.5, rows(4, seed=2))

    check_gradients(built, *split(rows(batch_size)))


def test_the_readme_model_builds_its_layers_wired_together():
    built = network(TOKENS["the README's model"])
    kinds: list[type] = [type(layer) for layer in built.layers]
    assert kinds == [
        PatchesArrayLayer,
        TokenDenseArrayLayer,
        PositionArrayLayer,
        ForkArrayLayer,
        LayerNormArrayLayer,
        AttentionArrayLayer,
        AddArrayLayer,
        ForkArrayLayer,
        LayerNormArrayLayer,
        TokenDenseArrayLayer,
        TokenDenseArrayLayer,
        AddArrayLayer,
        TokenMeanArrayLayer,
        LayerNormArrayLayer,
        type(built.layers[-1]),
    ]
    patches, embed, position, fork, norm, attention, add = built.layers[:7]
    assert fork.body_first is norm and fork.add is add
    assert (patches.size, embed.W.shape, position.P.shape) == (16, (6, 4), (4, 6))
    assert (norm.tokens, norm.features, attention.tokens, attention.features) == (4, 6, 4, 6)
    final = built.layers[13]
    assert (final.tokens, final.features) == (1, 6)
    # per layer: patches 0, W and b, P, fork 0, gamma and beta, attention's 8, add 0, ...
    assert [len(entry) for entry in built.snapshot()] == [0, 2, 1, 0, 2, 8, 0, 0, 2, 2, 2, 0, 0, 2, 2]


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


@pytest.mark.usefixtures("math_exp")
def test_attentions_softmax_and_its_backward_follow_the_readme_by_bits():
    built = network([PATCHES, EMBED, ATTENTION_BLOCK, TokenMean(), SOFTMAX])
    batch = rows(3)
    built.learn_batch(0.5, batch)
    attention = built.layers[4]
    assert isinstance(attention, AttentionArrayLayer)

    Wo = attention.Wo.copy()  # the optimizer steps it in place after the backward pass
    built.learn_batch(0.5, batch)  # the forward and backward passes to compare against
    Q, K, P = attention._Q, attention._K, attention._P
    S = (Q @ K.transpose(0, 2, 1)) / math.sqrt(6)
    for n in range(3):
        assert bits(P[n]) == bits(np.array(_softmax_rows(S[n].tolist())))

    # dS from dP: r_i = sum_j(dP_ij * P_ij), a left fold, then P_ij * (dP_ij - r_i)
    dH = (attention.delta_batch.reshape(-1, 6) @ Wo).reshape(3, 4, 6)
    dP = dH @ attention._V.transpose(0, 2, 1)
    dS = np.empty_like(dP)
    for n in range(3):
        for i in range(4):
            r = 0.0
            for j in range(4):
                r += dP[n, i, j] * P[n, i, j]
            dS[n, i] = [P[n, i, j] * (dP[n, i, j] - r) for j in range(4)]
    assert bits(attention._dQ) == bits(((dS @ K) / math.sqrt(6)).reshape(-1, 6))
    assert bits(attention._dK) == bits(((dS.transpose(0, 2, 1) @ Q) / math.sqrt(6)).reshape(-1, 6))


def _parameters(layer: AttentionArrayLayer, seed: int) -> list[FloatArray]:
    rng = np.random.default_rng(seed)
    return [rng.uniform(-0.5, 0.5, parameter.shape) for parameter in layer.parameters()]


def test_one_token_attends_only_to_itself_so_attention_is_two_affine_maps_by_bits():
    layer = AttentionArrayLayer(1, 5)
    layer.set_parameters(_parameters(layer, 1))
    *_, Wv, bv, Wo, bo = layer.parameters()
    X = np.random.default_rng(2).uniform(-1.0, 1.0, (4, 5))

    out = layer.forward_batch(X)
    assert bits(layer._P) == bits(np.ones((4, 1, 1)))
    assert bits(out) == bits((X @ Wv.T + bv) @ Wo.T + bo)


def test_zero_queries_and_keys_weigh_every_token_exactly_one_sixteenth():
    # T = 16: every score is 0, every weight 1/16, and H is the token mean of V, the same fold
    layer = AttentionArrayLayer(16, 8)
    parameters = _parameters(layer, 3)
    for i in range(4):  # Wq, bq, Wk, bk
        parameters[i] = np.zeros_like(parameters[i])
    layer.set_parameters(parameters)
    X = np.random.default_rng(4).uniform(-1.0, 1.0, (3, 16 * 8))

    layer.forward_batch(X)
    assert bits(layer._P) == bits(np.full((3, 16, 16), 1 / 16))
    mean = TokenMeanArrayLayer(16, 8).forward_batch(layer._V.reshape(3, -1))
    assert bits(layer._H) == bits(np.broadcast_to(mean[:, np.newaxis, :], (3, 16, 8)))


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


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), Adam()], ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", ["the README's model", "token-wise layers after a block"])
def test_learn_and_a_learn_batch_of_one_example_agree(name: str, rule: UpdateRule):
    single, batched = network(TOKENS[name], rule), network(TOKENS[name], rule)
    for state, label in rows(6, seed=4):
        single.learn(0.5, state, label)
        batched.learn_batch(0.5, [(state, label)])

    # the token layers' single-example passes are their batch passes on a batch of one, but a flat
    # dense layer's W @ x and X @ W.T are different BLAS calls, which can differ in the last bit
    # (test_residual_array_network)
    for one, other in zip(single.snapshot(), batched.snapshot(), strict=True):
        for a, b in zip(one, other, strict=True):
            assert a == pytest.approx(b, rel=1e-12, abs=1e-15)


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
