"""
Patch models in pure Python (the layer-norm and attention workplan, stage 3; README, Layer norm and
attention): the gradient check on every accepted patch model, the layers the builder wires and what
randomize draws, Patches, Position and TokenMean against the README's indices and sums and against
numpy by bits, the exact tests (one token, uniform attention, an identity attention block), learn
against a batch of one, the layer-major path's lanes (attention's own caches among them) against
the example-major loop, and parity with numpy after 50 steps. The cases are
tests/test_attention_array_network.py's.
"""

import math
import random
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.array_backend import NUMPY
from indrajala_ml.model.attention_layer import AttentionLayer
from indrajala_ml.model.batch_norm_layer import BatchNormLayer, fold
from indrajala_ml.model.layer_norm_layer import LayerNormLayer
from indrajala_ml.model.layer_specs import InputShape, LayerNorm, LayerSpec, Position, TokenMean, batch_norm_index
from indrajala_ml.model.linear_layer import LinearLayer
from indrajala_ml.model.residual_layer import AddLayer, ForkLayer
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.sequential_backprop_network import SequentialMultiClassBackpropClassifierNetwork
from indrajala_ml.model.state_layer import StateLayer
from indrajala_ml.model.token_array_layer import PatchesArrayLayer, PositionArrayLayer, TokenMeanArrayLayer
from indrajala_ml.model.token_layer import PatchesLayer, PositionLayer, TokenDenseLayer, TokenMeanLayer
from indrajala_ml.model.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from indrajala_ml.pcg64 import default_rng
from tests.gradient_check import analytic_gradients, check_gradients
from tests.test_attention_array_network import IMAGE, rows
from tests.test_batch_norm_python_network import _as_array_snapshot, _bits
from tests.test_layer_specs import ATTENTION_BLOCK, EMBED, FFN_BLOCK, PATCHES, SOFTMAX, TOKENS


def network(specs: list[LayerSpec], rule: UpdateRule | None = None, seed: int = 3, image: InputShape = IMAGE) -> Any:
    built = SequentialMultiClassBackpropClassifierNetwork(image, specs, SGD() if rule is None else rule)
    built.rng = default_rng(seed)
    built.randomize()
    return built


def _split(batch: list[tuple[tuple[float, ...], int]]) -> tuple[list[tuple[float, ...]], list[int]]:
    return [state for state, _ in batch], [label for _, label in batch]


def as_array_snapshot(python: Any) -> list[tuple[Any, ...]]:
    """
    The pure-Python snapshot as numpy's: a layer norm's per-feature ([gamma], beta) as (gamma,
    beta), attention's rows as (Wq, bq, Wk, bk, Wv, bv, Wo, bo), the position table's rows as P, and
    every other layer as _as_array_snapshot's.
    """
    snapshot: list[tuple[Any, ...]] = list(_as_array_snapshot(python))
    for i, (layer, entry) in enumerate(zip(python.trainable_layers, python.snapshot(), strict=True)):
        if isinstance(layer, LayerNormLayer):
            snapshot[i] = ([gamma for (gamma,), _ in entry], [beta for _, beta in entry])
        elif isinstance(layer, AttentionLayer):
            d = layer.features
            projections = [entry[p * d : (p + 1) * d] for p in range(4)]
            snapshot[i] = tuple(
                values for rows_ in projections for values in ([w for w, _ in rows_], [b for _, b in rows_])
            )
    return snapshot


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("name", TOKENS)
def test_every_gradient_matches_its_finite_difference(name: str, batch_size: int):
    if batch_size == 1 and batch_norm_index(TOKENS[name]) is not None:
        pytest.skip("batch norm trains on batches only (the batch-norm workplan, D4)")
    built = network(TOKENS[name])
    # a trained step first, so gamma, beta and the positions aren't at their initial values
    built.learn_batch(0.5, rows(4, seed=2))

    check_gradients(built, *_split(rows(batch_size)))


def test_the_readme_model_builds_its_layers_wired_together():
    built = network(TOKENS["the README's model"])
    kinds: list[type] = [type(layer) for layer in built.trainable_layers]
    assert kinds == [
        PatchesLayer,
        TokenDenseLayer,
        PositionLayer,
        ForkLayer,
        LayerNormLayer,
        AttentionLayer,
        AddLayer,
        ForkLayer,
        LayerNormLayer,
        TokenDenseLayer,
        TokenDenseLayer,
        AddLayer,
        TokenMeanLayer,
        LayerNormLayer,
        type(built.output_layer),
    ]
    patches, embed, position, fork, norm, attention, add = built.trainable_layers[:7]
    assert fork.body_first is norm and fork.add is add and attention.input_layer is norm
    assert (patches.size, len(embed.units), embed.input_size, len(position.rows)) == (16, 6, 4, 4)
    assert (norm.tokens, norm.features, attention.tokens, attention.features) == (4, 6, 4, 6)
    final = built.trainable_layers[13]
    assert (final.tokens, final.features) == (1, 6)
    # per layer, its weight sets: embedding units, position rows, features, attention's 4 * 6 rows, ...
    assert [len(entry) for entry in built.snapshot()] == [0, 6, 4, 0, 6, 24, 0, 0, 6, 8, 6, 0, 0, 6, 3]


def test_randomize_draws_weights_then_bias_per_unit_and_row_and_nothing_for_the_parameter_free_layers():
    built = network([PATCHES, EMBED, Position(), ATTENTION_BLOCK, TokenMean(), LayerNorm(), SOFTMAX])
    rng = default_rng(3)
    expected: list[Any] = []
    # the embedding's units, attention's Wq, Wk, Wv and Wo rows, then the output layer's nodes
    for size, fan_in in [(6, 4), (24, 6), (3, 6)]:
        limit = 1 / math.sqrt(fan_in)
        expected.append(
            [([rng.uniform(-limit, limit) for _ in range(fan_in)], rng.uniform(-limit, limit)) for _ in range(size)]
        )
    snapshot = built.snapshot()
    assert _bits([snapshot[i] for i in (1, 5, 9)]) == _bits(expected)
    # positions, gamma and beta start at 0, 1 and 0
    assert _bits(snapshot[2]) == _bits([([0.0] * 6,)] * 4)
    assert _bits([snapshot[4], snapshot[8]]) == _bits([[([1.0], 0.0)] * 6] * 2)


def _input(values: list[float]) -> StateLayer:
    layer = StateLayer(len(values), [(-1e20, 1e20)] * len(values))
    layer.update_state(tuple(values))
    return layer


class _Downstream:
    """A next layer whose downstream_sum is a given row."""

    def __init__(self, row: list[float]) -> None:
        self.row = row

    def downstream_sum(self, own_index: int) -> float:
        return self.row[own_index]


def downstream(row: list[float]) -> Any:
    """A next layer for a layer's compute_hidden_deltas: its downstream_sum is row."""
    return _Downstream(row)


def test_patches_follow_the_readmes_indices_and_send_the_inverse_permutation_back():
    height, width, channels, p = 6, 4, 3, 2
    X = [float(i) for i in range(height * width * channels)]
    layer = PatchesLayer(_input(X), height, width, channels, p)
    layer.forward()
    tokens = [node.value() for node in layer.nodes]

    array = PatchesArrayLayer(height, width, channels, p)
    assert _bits(tokens) == _bits(array.forward_batch(np.array([X]))[0].tolist())

    layer.compute_hidden_deltas(downstream(tokens))
    assert _bits([layer.downstream_sum(i) for i in range(len(X))]) == _bits(X)


def test_position_adds_its_table_and_sums_its_gradient_over_the_examples_as_numpy():
    P = [[i / 8 for i in range(3)], [i / 8 for i in range(3, 6)]]
    delta = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.7, 0.0, -1.0, 2.5, 0.1, 0.3]]
    inputs = _input([0.0] * 6)
    layer = PositionLayer(inputs, 2, 3)
    layer.restore_state([(row,) for row in P])
    array = PositionArrayLayer(2, 3)
    array.P = np.array(P)
    X = [[1.0] * 6, [0.5] * 6, [-2.0] * 6]

    for x, row in zip(X, delta, strict=True):
        inputs.update_state(tuple(x))
        layer.forward()
        assert _bits([node.value() for node in layer.nodes]) == _bits((np.array(x) + array.P.reshape(6)).tolist())
        layer.compute_hidden_deltas(downstream(row))
        assert [layer.downstream_sum(i) for i in range(6)] == row
        layer.accumulate_gradients()

    array.delta_batch = np.array(delta)
    array.accumulate_gradient_batch(np.array(X))
    assert _bits([row.weight_gradient_accum for row in layer.rows]) == _bits(array.grad_P.tolist())


def test_the_token_mean_is_a_left_fold_over_the_tokens_divided_by_their_count_as_numpy():
    X = [0.1, 1e16, 0.2, 1.0, 0.3, -1e16]
    layer = TokenMeanLayer(_input(X), 3, 2)
    layer.forward()
    expected = [((0.0 + X[j]) + X[2 + j]) + X[4 + j] for j in range(2)]
    assert _bits([node.value() for node in layer.nodes]) == _bits([value / 3 for value in expected])
    assert _bits([node.value() for node in layer.nodes]) == _bits(
        TokenMeanArrayLayer(3, 2).forward_batch(np.array([X]))[0].tolist()
    )

    layer.compute_hidden_deltas(downstream([3.0, -1.5]))
    assert [layer.downstream_sum(i) for i in range(6)] == [1.0, -0.5] * 3


def _attention(tokens: int, features: int, X: list[float], seed: int) -> AttentionLayer:
    layer = AttentionLayer(_input(X), tokens, features)
    rng = random.Random(seed)
    layer.restore_state(
        [([rng.uniform(-0.5, 0.5) for _ in range(features)], rng.uniform(-0.5, 0.5)) for _ in range(4 * features)]
    )
    return layer


def test_one_token_attends_only_to_itself_so_attention_is_two_affine_maps_by_bits():
    rng = random.Random(2)
    X = [rng.uniform(-1.0, 1.0) for _ in range(5)]
    layer = _attention(1, 5, X, seed=1)
    layer.forward()

    def affine(x: list[float], rows_: list[tuple[list[float], float]]) -> list[float]:
        return [fold([x_j * w_j for x_j, w_j in zip(x, weights, strict=True)]) + b for weights, b in rows_]

    snapshot = layer.snapshot_state()
    assert layer._P == [[1.0]]
    expected = affine(affine(X, snapshot[10:15]), snapshot[15:20])
    assert _bits([node.value() for node in layer.nodes]) == _bits(expected)


def test_zero_queries_and_keys_weigh_every_token_exactly_one_sixteenth():
    # T = 16: every score is 0, every weight 1/16, and H is the token mean of V, the same fold
    rng = random.Random(4)
    layer = _attention(16, 8, [rng.uniform(-1.0, 1.0) for _ in range(16 * 8)], seed=3)
    snapshot = layer.snapshot_state()
    layer.restore_state([([0.0] * 8, 0.0)] * 16 + snapshot[16:])  # Wq, bq, Wk, bk
    layer.forward()

    assert layer._P == [[1 / 16] * 16] * 16
    mean = TokenMeanLayer(_input([v for row in layer._V for v in row]), 16, 8)
    mean.forward()
    assert _bits(layer._H) == _bits([[node.value() for node in mean.nodes]] * 16)


def test_an_identity_attention_block_changes_no_output_and_no_other_layers_gradient_by_bits():
    # with Wo and bo zero the block adds 0 forward, and backward its body's downstream is 0
    plain_specs: list[LayerSpec] = [PATCHES, EMBED, Position(), FFN_BLOCK, TokenMean(), SOFTMAX]
    plain = network(plain_specs, seed=5)
    plain.learn_batch(0.5, rows(4, seed=2))  # positions, gamma and beta off their initial values
    block_specs: list[LayerSpec] = [PATCHES, EMBED, Position(), ATTENTION_BLOCK, FFN_BLOCK, TokenMean(), SOFTMAX]
    blocked = network(block_specs, seed=7)

    snapshot = plain.snapshot()
    attention = blocked.snapshot()[5]
    attention[18:] = [([0.0] * 6, 0.0)] * 6  # Wo's rows and bo
    blocked.restore([*snapshot[:3], [], blocked.snapshot()[4], attention, [], *snapshot[3:]])

    batch = rows(5)
    states, labels = _split(batch)
    for state in states:
        assert _bits(blocked._forward(state)) == _bits(plain._forward(state))

    outside = [0, 1, 2, *range(7, 14)]
    blocked_gradients = analytic_gradients(blocked, states, labels)
    assert _bits([blocked_gradients[i] for i in outside]) == _bits(analytic_gradients(plain, states, labels))


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), Adam()], ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", ["the README's model", "token-wise layers after a block"])
def test_learn_and_a_learn_batch_of_one_example_agree_by_bits(name: str, rule: UpdateRule):
    single, batched = network(TOKENS[name], rule), network(TOKENS[name], rule)
    for state, label in rows(6, seed=4):
        single.learn(0.5, state, label)
        batched.learn_batch(0.5, [(state, label)])

    assert _bits(single.snapshot()) == _bits(batched.snapshot())


@pytest.mark.parametrize("rule", [Momentum(0.9), Adam()], ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", ["the README's model", "two layers' blocks", "token-wise layers after a block"])
def test_the_layer_major_path_is_the_example_major_loop_by_bits(name: str, rule: UpdateRule):
    # the lanes of every token layer, attention's own caches (example_fields) among them
    example_major, layer_major = network(TOKENS[name], rule), network(TOKENS[name], rule)
    for size in (3, 5, 4):
        batch = rows(size, seed=size)
        example_major.learn_batch(0.3, batch)
        layer_major._learn_batch_layer_major(0.3, batch)

    assert _bits(layer_major.snapshot()) == _bits(example_major.snapshot())
    assert _bits(list(layer_major.optimizer.state().layers.values())) == _bits(
        list(example_major.optimizer.state().layers.values())
    )


# the rules whose step is linear in the gradient (tests/test_residual_rust_network.py). Adam's,
# lr * g / (|g| + epsilon), is steepest for |g| at or under epsilon, where it turns a gradient's
# rounding into up to lr / epsilon of its step, and attention puts gradients there: bk's is rounding
# noise (D6: a key bias shifts each row of scores by a constant, which the softmax cancels), and
# Wq's, bq's and Wk's were down to 8e-6. So under Adam the trajectory itself is sensitive: 50 steps
# in, measured, pure Python against numpy 1e-8 to 5e-6 relative on the models with attention, and
# numpy against numpy with the first trained weight nudged by one ulp 2e-8 to 3e-5. With bk's
# gradient zeroed on both sides, the two models it alone explains drop from 1.7e-8 and 9.4e-9 to
# 2.4e-12. Under Adam the gradients are compared step by step instead (below)
LINEAR_RULES = [SGD(), Momentum(0.9), WeightDecay(0.01)]


@pytest.mark.parametrize("rule", LINEAR_RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", TOKENS)
def test_training_matches_numpy_within_the_dense_layers_rounding(name: str, rule: UpdateRule):
    # the token layers' and layer norm's sums are folds in both, but the products aren't: pure
    # Python's folds against numpy's BLAS, as for the dense layers, so the networks agree within
    # the tolerance every pure-Python parity test allows. Measured: 3.5e-12 relative at most
    python = network(TOKENS[name], rule)
    array = SequentialArrayNetwork(IMAGE, TOKENS[name], rule, backend=NUMPY)
    array.restore(as_array_snapshot(python))
    data = rows(40)

    for step in range(50):
        batch = data[(step * 5) % 40 :][:5]
        python.learn_batch(0.3, batch)
        array.learn_batch(0.3, batch)

    for expected, actual in zip(as_array_snapshot(python), array.snapshot(), strict=True):
        for values, array_values in zip(expected, actual, strict=True):
            np.testing.assert_allclose(array_values, values, rtol=1e-9, atol=1e-9)


def _as_array_gradients(python: Any, gradients: list[Any]) -> list[list[Any]]:
    # analytic_gradients' per-weight-set lists as numpy's per parameter, as as_array_snapshot
    arrays: list[list[Any]] = []
    for layer, entry in zip(python.trainable_layers, gradients, strict=True):
        if isinstance(layer, AttentionLayer):
            d = layer.features
            projections = [entry[p * d : (p + 1) * d] for p in range(4)]
            arrays.append(
                [values for rows_ in projections for values in ([w for w, _ in rows_], [b for _, b in rows_])]
            )
        elif isinstance(layer, LayerNormLayer | BatchNormLayer):
            arrays.append([[gamma for (gamma,), _ in entry], [beta for _, beta in entry]])
        else:
            arrays.append([list(values) for values in zip(*entry)])
    return arrays


def _scales(python: Any, gradients: list[list[Any]]) -> list[float]:
    # each layer's largest gradient, but a linear layer's is its batch norm's: batch norm's dl/dx
    # is a difference of terms that cancel (to 1e-7 of them, measured here, once Adam's steps kill
    # the ReLU layer before), so its rounding is the terms', as the batch-norm parity tests found
    largest = [max((float(np.abs(np.array(values)).max()) for values in layer), default=0.0) for layer in gradients]
    return [
        largest[i + 1] if isinstance(layer, LinearLayer) else largest[i]
        for i, layer in enumerate(python.trainable_layers)
    ]


def assert_every_step_has_numpys_gradients(python: Any, array: Any, data: list[tuple[tuple[float, ...], int]]) -> None:
    """
    50 of python's training steps, each first checked against array's gradients from the same
    weights and the same draws (dropout's masks): within 1e-10 of each layer's scale (_scales).
    Attention's bk, rounding noise (D6), is compared apart: within 1e-13 of attention's largest
    gradient on both sides.
    """
    for step in range(50):
        batch = data[(step * 5) % 40 :][:5]
        array.restore(as_array_snapshot(python))
        states, labels = _split(batch)
        python.rng, array.rng = default_rng(step), NUMPY.default_rng(step)
        expected = _as_array_gradients(python, analytic_gradients(python, states, labels))
        actual = analytic_gradients(array, states, labels)
        layers = zip(python.trainable_layers, expected, actual, _scales(python, expected), strict=True)
        for layer, python_layer, array_layer, scale in layers:
            for i, (values, array_values) in enumerate(zip(python_layer, array_layer, strict=True)):
                if isinstance(layer, AttentionLayer) and i == 3:
                    assert max(np.abs(values).max(), np.abs(array_values).max()) <= 1e-13 * scale
                    continue
                np.testing.assert_allclose(array_values, values, rtol=0.0, atol=1e-10 * scale)
        python.rng = default_rng(step)
        python.learn_batch(0.3, batch)


@pytest.mark.parametrize("name", TOKENS)
def test_every_step_under_adam_has_numpys_gradients(name: str):
    # measured within 2.6e-13 of each layer's scale, and bk's within 4e-16 of attention's
    python = network(TOKENS[name], Adam())
    array = SequentialArrayNetwork(IMAGE, TOKENS[name], SGD(), backend=NUMPY)
    assert_every_step_has_numpys_gradients(python, array, rows(40))
