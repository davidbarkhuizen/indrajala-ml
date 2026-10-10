"""
Patch models in pure Python (the layer-norm and attention workplan, stage 3; the multi-head
attention workplan, stage 4; README, Layer norm and attention): what randomize draws, Patches,
Position and TokenMean against the README's indices and sums and against numpy by bits, attention's
packed head blocks, the exact tests (one token, uniform attention, identical heads, a silent head,
an identity attention block, the blocks called in turn), the layer-major path's lanes (attention's
own caches among them) against the example-major loop, and parity with numpy after 50 steps (step
by step under Adam), at one head and several. The cases are tests/model/networks/test_attention_array_network.py's;
the gradient check, the README model's wiring and learn against a batch of one are
tests/model/networks/test_attention_network.py's.
"""

import math
import random
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.layers.numpy.token_array_layer import PatchesArrayLayer, PositionArrayLayer, TokenMeanArrayLayer
from indrajala_ml.model.layers.python.attention_layer import AttentionLayer
from indrajala_ml.model.layers.python.batch_norm_layer import BatchNormLayer, fold
from indrajala_ml.model.layers.python.layer_norm_layer import LayerNormLayer
from indrajala_ml.model.layers.python.linear_layer import LinearLayer
from indrajala_ml.model.layers.python.state_layer import StateLayer
from indrajala_ml.model.layers.python.token_layer import PatchesLayer, PositionLayer, TokenMeanLayer
from indrajala_ml.model.networks.python.sequential_backprop_network import SequentialMultiClassBackpropClassifierNetwork
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.layer_specs import LayerNorm, LayerSpec, Position, TokenMean
from indrajala_ml.model.specs.spec_shapes import InputShape
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from indrajala_ml.pcg64 import default_rng
from tests.gradient_check import analytic_gradients
from tests.helpers import assert_snapshots_close, batches, bits, learn_in_step, split
from tests.model.networks.test_attention_array_network import IMAGE, rows
from tests.model.networks.test_attention_network import MULTI_HEAD
from tests.model.specs.test_layer_specs import ATTENTION_BLOCK, EMBED, FFN_BLOCK, PATCHES, SOFTMAX, TOKENS
from tests.python_array_snapshot import as_array_snapshot, projections, seeded_like


def network(specs: list[LayerSpec], rule: UpdateRule | None = None, seed: int = 3, image: InputShape = IMAGE) -> Any:
    built = SequentialMultiClassBackpropClassifierNetwork(image, specs, SGD() if rule is None else rule)
    built.rng = default_rng(seed)
    built.randomize()
    return built


def test_randomize_draws_numpys_weights_and_nothing_for_the_parameter_free_layers():
    specs: list[LayerSpec] = [PATCHES, EMBED, Position(), ATTENTION_BLOCK, TokenMean(), LayerNorm(), SOFTMAX]
    built = network(specs)
    # numpy's weights from the same seed: the embedding's, attention's Wq, Wk, Wv and Wo, each W
    # then b, and the output layer's
    seeded_like(built, SequentialArrayNetwork(IMAGE, specs, SGD(), backend=NUMPY), 3)
    snapshot = built.snapshot()
    # positions, gamma and beta start at 0, 1 and 0
    assert bits(snapshot[2]) == bits([([0.0] * 6,)] * 4)
    assert bits([snapshot[4], snapshot[8]]) == bits([[([1.0], 0.0)] * 6] * 2)


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
    assert bits(tokens) == bits(array.forward_batch(np.array([X]))[0].tolist())

    layer.compute_hidden_deltas(downstream(tokens))
    assert bits([layer.downstream_sum(i) for i in range(len(X))]) == bits(X)


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
        assert bits([node.value() for node in layer.nodes]) == bits((np.array(x) + array.P.reshape(6)).tolist())
        layer.compute_hidden_deltas(downstream(row))
        assert [layer.downstream_sum(i) for i in range(6)] == row
        layer.accumulate_gradients()

    array.delta_batch = np.array(delta)
    array.accumulate_gradient_batch(np.array(X))
    assert bits([row.weight_gradient_accum for row in layer.rows]) == bits(array.grad_P.tolist())


def test_the_token_mean_is_a_left_fold_over_the_tokens_divided_by_their_count_as_numpy():
    X = [0.1, 1e16, 0.2, 1.0, 0.3, -1e16]
    layer = TokenMeanLayer(_input(X), 3, 2)
    layer.forward()
    expected = [((0.0 + X[j]) + X[2 + j]) + X[4 + j] for j in range(2)]
    assert bits([node.value() for node in layer.nodes]) == bits([value / 3 for value in expected])
    assert bits([node.value() for node in layer.nodes]) == bits(
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
    assert bits([node.value() for node in layer.nodes]) == bits(expected)


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
    assert bits(layer._H) == bits([[node.value() for node in mean.nodes]] * 16)


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
    states, labels = split(batch)
    for state in states:
        assert bits(blocked._forward(state)) == bits(plain._forward(state))

    outside = [0, 1, 2, *range(7, 14)]
    blocked_gradients = analytic_gradients(blocked, states, labels)
    assert bits([blocked_gradients[i] for i in outside]) == bits(analytic_gradients(plain, states, labels))


@pytest.mark.parametrize("rule", [Momentum(0.9), Adam()], ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", ["the README's model", "two layers' blocks", "token-wise layers after a block"])
def test_the_layer_major_path_is_the_example_major_loop_by_bits(name: str, rule: UpdateRule):
    # the lanes of every token layer, attention's own caches (example_fields) among them
    example_major, layer_major = network(TOKENS[name], rule), network(TOKENS[name], rule)
    for size in (3, 5, 4):
        batch = rows(size, seed=size)
        example_major.learn_batch(0.3, batch)
        layer_major._learn_batch_layer_major(0.3, batch)

    assert bits(layer_major.snapshot()) == bits(example_major.snapshot())
    assert bits(list(layer_major.optimizer.state().layers.values())) == bits(
        list(example_major.optimizer.state().layers.values())
    )


# the rules whose step is linear in the gradient (tests/model/networks/test_residual_rust_network.py). Adam's,
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
    array = seeded_like(python, SequentialArrayNetwork(IMAGE, TOKENS[name], rule, backend=NUMPY), 3)
    data = rows(40)

    learn_in_step(0.3, data, (python, array))

    assert_snapshots_close(as_array_snapshot(python), array.snapshot())


def _as_array_gradients(python: Any, gradients: list[Any]) -> list[list[Any]]:
    # analytic_gradients' per-weight-set lists as numpy's per parameter, as as_array_snapshot
    arrays: list[list[Any]] = []
    for layer, entry in zip(python.trainable_layers, gradients, strict=True):
        if isinstance(layer, AttentionLayer):
            arrays.append(
                [
                    values
                    for rows_ in projections(layer, entry)
                    for values in ([w for w, _ in rows_], [b for _, b in rows_])
                ]
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
    Attention's bk, rounding noise (D6), is compared apart: within 1e-11 of attention's largest
    gradient on both sides. The noise follows the terms that cancel, not the gradients, so once
    Adam's steps shrink attention's gradients it reaches 2.8e-12 of them (a dense part after the
    mean, over seeds 0 to 11), and under 6e-14 in every other model.
    """
    for step, batch in enumerate(batches(data)):
        array.restore(as_array_snapshot(python))
        states, labels = split(batch)
        python.rng, array.rng = default_rng(step), NUMPY.default_rng(step)
        expected = _as_array_gradients(python, analytic_gradients(python, states, labels))
        actual = analytic_gradients(array, states, labels)
        layers = zip(python.trainable_layers, expected, actual, _scales(python, expected), strict=True)
        for layer, python_layer, array_layer, scale in layers:
            for i, (values, array_values) in enumerate(zip(python_layer, array_layer, strict=True)):
                if isinstance(layer, AttentionLayer) and i == 3:
                    assert max(np.abs(values).max(), np.abs(array_values).max()) <= 1e-11 * scale
                    continue
                np.testing.assert_allclose(array_values, values, rtol=0.0, atol=1e-10 * scale)
        python.rng = default_rng(step)
        python.learn_batch(0.3, batch)


@pytest.mark.parametrize("name", TOKENS)
def test_every_step_under_adam_has_numpys_gradients(name: str):
    # measured within 2.6e-13 of each layer's scale (seed 3, before pure Python drew numpy's order)
    python = network(TOKENS[name], Adam())
    array = SequentialArrayNetwork(IMAGE, TOKENS[name], SGD(), backend=NUMPY)
    assert_every_step_has_numpys_gradients(python, array, rows(40))


# multi-head (the multi-head attention workplan, stage 4): the shared models over 4 tokens of 6
@pytest.mark.parametrize("rule", LINEAR_RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", MULTI_HEAD)
def test_multi_head_training_matches_numpy_within_the_dense_layers_rounding(name: str, rule: UpdateRule):
    # at 0.1, not the one-head test's 0.3: two multi-head layers under Momentum diverge at 0.3 (numpy
    # overflows), which would compare two blow-ups rather than two trainings
    python = network(MULTI_HEAD[name], rule)
    array = seeded_like(python, SequentialArrayNetwork(IMAGE, MULTI_HEAD[name], rule, backend=NUMPY), 3)
    data = rows(40)

    learn_in_step(0.1, data, (python, array))

    assert_snapshots_close(as_array_snapshot(python), array.snapshot())


@pytest.mark.parametrize("name", MULTI_HEAD)
def test_every_multi_head_step_under_adam_has_numpys_gradients(name: str):
    python = network(MULTI_HEAD[name], Adam())
    array = SequentialArrayNetwork(IMAGE, MULTI_HEAD[name], SGD(), backend=NUMPY)
    assert_every_step_has_numpys_gradients(python, array, rows(40))


@pytest.mark.parametrize("rule", [Momentum(0.9), Adam()], ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", ["two heads", "four heads of 2, wider than d", "two layers"])
def test_the_multi_head_layer_major_path_is_the_example_major_loop_by_bits(name: str, rule: UpdateRule):
    example_major, layer_major = network(MULTI_HEAD[name], rule), network(MULTI_HEAD[name], rule)
    for size in (3, 5, 4):
        batch = rows(size, seed=size)
        example_major.learn_batch(0.3, batch)
        layer_major._learn_batch_layer_major(0.3, batch)

    assert bits(layer_major.snapshot()) == bits(example_major.snapshot())


def _heads_attention(tokens: int, X: list[float], heads: int, key_size: int | None, seed: int) -> AttentionLayer:
    # 6 features, every weight set drawn from seed
    layer = AttentionLayer(_input(X), tokens, 6, heads, key_size)
    rng = random.Random(seed)
    layer.restore_state(
        [([rng.uniform(-0.5, 0.5) for _ in row.weights], rng.uniform(-0.5, 0.5)) for row in layer.weight_sets()]
    )
    return layer


HEADS = [(1, None), (2, None), (3, None), (2, 4), (4, 1)]
HEAD_IDS = ["one head", "two heads of 3", "three heads of 2", "two heads of 4", "four heads of 1"]


@pytest.mark.parametrize(("heads", "key_size"), HEADS, ids=HEAD_IDS)
def test_the_weight_sets_are_packed_head_blocks_and_wos_rows_are_their_width(heads: int, key_size: int | None):
    layer = AttentionLayer(_input([0.0] * 24), 4, 6, heads, key_size)
    w = heads * layer.key_size
    assert (layer.width, layer.scale) == (w, math.sqrt(layer.key_size))
    assert [len(row.weights) for row in layer.weight_sets()] == [6] * (3 * w) + [w] * 6


@pytest.mark.parametrize(("heads", "key_size"), HEADS, ids=HEAD_IDS)
def test_one_token_attends_only_to_itself_in_every_head_by_bits(heads: int, key_size: int | None):
    rng = random.Random(2)
    X = [rng.uniform(-1.0, 1.0) for _ in range(6)]
    layer = _heads_attention(1, X, heads, key_size, seed=1)
    layer.forward()

    def affine(x: list[float], rows_: list[tuple[list[float], float]]) -> list[float]:
        return [fold([x_j * w_j for x_j, w_j in zip(x, weights, strict=True)]) + b for weights, b in rows_]

    w = layer.width
    snapshot = layer.snapshot_state()
    assert layer._P == [[1.0] * heads]
    expected = affine(affine(X, snapshot[2 * w : 3 * w]), snapshot[3 * w :])
    assert bits([node.value() for node in layer.nodes]) == bits(expected)


@pytest.mark.parametrize(("heads", "key_size"), [(1, None), (2, None), (3, 4)], ids=["1", "2", "3 of 4"])
def test_zero_queries_and_keys_weigh_every_token_exactly_one_sixteenth_in_every_head(heads: int, key_size: int | None):
    rng = random.Random(4)
    layer = _heads_attention(16, [rng.uniform(-1.0, 1.0) for _ in range(16 * 6)], heads, key_size, seed=3)
    w = layer.width
    snapshot = layer.snapshot_state()
    layer.restore_state([([0.0] * 6, 0.0)] * (2 * w) + snapshot[2 * w :])  # Wq, bq, Wk, bk
    layer.forward()

    assert layer._P == [[1 / 16] * (16 * heads)] * 16
    mean = TokenMeanLayer(_input([v for row in layer._V for v in row]), 16, w)
    mean.forward()
    assert bits(layer._H) == bits([[node.value() for node in mean.nodes]] * 16)


@pytest.mark.parametrize(("heads", "key_size"), [(2, None), (3, None), (2, 4)], ids=["2", "3", "2 of 4"])
def test_identical_heads_weigh_and_mix_alike_by_bits(heads: int, key_size: int | None):
    rng = random.Random(6)
    layer = _heads_attention(4, [rng.uniform(-1.0, 1.0) for _ in range(24)], heads, key_size, seed=5)
    d_k, w = layer.key_size, layer.width
    snapshot = layer.snapshot_state()
    # each of Wq, Wk, Wv: head 0's rows (with their biases) for every head
    same = [row for p in range(3) for _ in range(heads) for row in snapshot[p * w : p * w + d_k]]
    layer.restore_state(same + snapshot[3 * w :])
    layer.forward()

    for i in range(1, heads):
        assert bits([row[i * 4 : (i + 1) * 4] for row in layer._P]) == bits([row[:4] for row in layer._P])
        assert bits([row[i * d_k : (i + 1) * d_k] for row in layer._H]) == bits([row[:d_k] for row in layer._H])


@pytest.mark.parametrize(("heads", "key_size"), [(2, None), (3, None), (2, 4)], ids=["2", "3", "2 of 4"])
def test_a_silent_heads_projection_gradients_are_exactly_zero(heads: int, key_size: int | None):
    rng = random.Random(8)
    layer = _heads_attention(4, [rng.uniform(-1.0, 1.0) for _ in range(24)], heads, key_size, seed=7)
    d_k, w, silent = layer.key_size, layer.width, heads - 1
    snapshot = layer.snapshot_state()
    outputs = [
        ([0.0 if silent * d_k <= j < (silent + 1) * d_k else v for j, v in enumerate(weights)], b)
        for weights, b in snapshot[3 * w :]
    ]
    layer.restore_state(snapshot[: 3 * w] + outputs)
    layer.forward()
    layer.compute_hidden_deltas(downstream([rng.uniform(-0.5, 0.5) for _ in range(24)]))
    layer.accumulate_gradients()

    for p in range(3):  # Wq, Wk, Wv: the silent head's rows and biases
        rows_ = layer.weight_sets()[p * w : (p + 1) * w]
        assert all(
            g == 0.0 for row in rows_[silent * d_k :] for g in [*row.weight_gradient_accum, row.bias_gradient_accum]
        )
        assert any(g != 0.0 for row in rows_[: silent * d_k] for g in row.weight_gradient_accum)


@pytest.mark.parametrize(("heads", "key_size"), HEADS, ids=HEAD_IDS)
def test_the_blocks_called_in_turn_are_the_layers_passes_by_bits(heads: int, key_size: int | None):
    rng = random.Random(10)
    layer = _heads_attention(4, [rng.uniform(-1.0, 1.0) for _ in range(24)], heads, key_size, seed=9)
    delta = [rng.uniform(-0.5, 0.5) for _ in range(24)]

    Q, K, V = layer._project(layer._inputs())
    P, H = layer._attend(Q, K, V)
    out = layer._combine(H)
    layer.forward()
    assert bits([node.value() for node in layer.nodes]) == bits([v for row in out for v in row])

    layer.compute_hidden_deltas(downstream(delta))
    dQ, dK, dV = layer._attend_backward(layer._combine_backward(layer._deltas()))
    assert bits(layer._dX) == bits(layer._project_backward(dQ, dK, dV))
    assert bits(layer._P) == bits(P)
