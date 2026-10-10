"""
Sequence models in pure Python (the sequence task workplan, stage 6, D10): randomize's draws against
numpy's by bits; Embedding's rows and scatter-add gradient against numpy's by bits, and its
refusals; the token-wise softmax against numpy's by bits where no product rounds, and its delta
(P - Y) / T; the causal mask (masked weights exactly 0, a later token can't change an earlier
output, one token as the unmasked layer); the sequence shape's class, targets and refusals, and
save and load; learn against a batch of one; the layer-major path; and parity with numpy, after 50
steps under the rules whose step is linear in the gradient and per step under Adam, at a small T
and vocabulary. The cases are tests/model/specs/test_layer_specs.py's SEQUENCE; numpy's exps are
math.exp, as pure Python's.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.layers.numpy import attention_array_layer, token_array_layer
from indrajala_ml.model.layers.numpy.token_array_layer import EmbeddingArrayLayer, TokenSoftmaxArrayLayer
from indrajala_ml.model.layers.python.attention_layer import AttentionLayer
from indrajala_ml.model.layers.python.state_layer import StateLayer
from indrajala_ml.model.layers.python.token_layer import EmbeddingLayer, TokenSoftmaxLayer
from indrajala_ml.model.networks.python.sequential_backprop_network import (
    SequentialBackpropClassifierNetwork,
    SequentialMultiClassBackpropClassifierNetwork,
    SequentialSequenceBackpropNetwork,
)
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.persistence.format2 import network_to_json
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.layer_specs import Dense, LayerSpec, token_wise_output
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule
from indrajala_ml.pcg64 import default_rng
from tests.helpers import (
    assert_learn_and_a_batch_of_one_agree,
    assert_snapshots_close,
    bits,
    exp_by_math,
    learn_in_step,
    numpy_draws,
    patching,
)
from tests.model.networks.test_attention_python_network import (
    LINEAR_RULES,
    as_array_snapshot,
    assert_every_step_has_numpys_gradients,
    downstream,
)
from tests.model.networks.test_sequence_array_network import (
    CAUSAL,
    LEAK,
    TOKEN_WISE,
    TOKENS,
    VOCABULARY,
    _input_shape,  # pyright: ignore[reportPrivateUsage]
    rows,
)
from tests.model.specs.test_layer_specs import SEQUENCE


def network(specs: list[LayerSpec], rule: UpdateRule | None = None, seed: int = 3) -> Any:
    cls = (
        SequentialSequenceBackpropNetwork if token_wise_output(specs) else SequentialMultiClassBackpropClassifierNetwork
    )
    built = cls(_input_shape(specs), specs, SGD() if rule is None else rule)
    built.rng = default_rng(seed)
    built.randomize()
    return built


def numpy_network(specs: list[LayerSpec], rule: UpdateRule | None = None, seed: int = 3) -> Any:
    shape = "sequence" if token_wise_output(specs) else "multiclass"
    built = SequentialArrayNetwork(_input_shape(specs), specs, SGD() if rule is None else rule, shape=shape)
    built.rng = NUMPY.default_rng(seed)
    built.randomize()
    return built


def _input(values: list[float]) -> StateLayer:
    layer = StateLayer(len(values), [(-1e20, 1e20)] * len(values))
    layer.update_state(tuple(values))
    return layer


# numpy's softmax exps, attention's and the token-wise output's, as pure Python's (math.exp)
attention_exp = patching(attention_array_layer, "exp", exp_by_math)
token_exp = patching(token_array_layer, "exp", exp_by_math)


def test_randomize_draws_numpys_embedding_table_then_output_weights_then_biases():
    built = network(SEQUENCE["the embedding, then the output"])
    assert bits(built.snapshot()) == bits(numpy_draws(3, [(VOCABULARY, 6, False), (VOCABULARY, 6, True)]))


def test_a_sequence_network_in_pure_python_is_its_class():
    assert isinstance(network(CAUSAL), SequentialSequenceBackpropNetwork)


# Embedding


def _embeddings(ids: list[float]) -> tuple[EmbeddingLayer, EmbeddingArrayLayer]:
    table = np.random.default_rng(0).uniform(-1.0, 1.0, (4, 2))
    python, array = EmbeddingLayer(_input(ids), 3, 4, 2), EmbeddingArrayLayer(3, 4, 2)
    python.restore_state([(row,) for row in table.tolist()])
    array.E = table
    return python, array


def test_the_embedding_gives_numpys_rows_and_scatter_adds_numpys_gradient_by_bits():
    X = [[3.0, 0.0, 3.0], [3.0, 2.0, 0.0]]
    # large and small values in one row of E, so a different order of the adds would show
    delta = [[0.1, 1e16, 0.2, 1.0, 0.3, -1e16], [0.7, 1.0, 0.5, -2.0, 0.25, 3.0]]
    python, array = _embeddings(X[0])
    assert isinstance(python.input_layer, StateLayer)
    for x, row in zip(X, delta, strict=True):
        python.input_layer.update_state(tuple(x))
        python.forward()
        assert bits([node.value() for node in python.nodes]) == bits(array.forward_batch(np.array([x]))[0].tolist())
        python.compute_hidden_deltas(downstream(row))
        python.accumulate_gradients()
    array._backward(np.array(delta))  # pyright: ignore[reportPrivateUsage]
    array.accumulate_gradient_batch(np.array(X))
    assert bits([row.weight_gradient_accum for row in python.rows]) == bits(array.grad_E.tolist())
    with pytest.raises(NotImplementedError, match="the first layer"):
        python.downstream_sum(0)


@pytest.mark.parametrize(
    "ids", [[0.5, 1.0, 2.0], [0.0, 4.0, 1.0], [-1.0, 0.0, 1.0]], ids=["a fraction", "past the vocabulary", "negative"]
)
def test_the_embedding_refuses_what_isnt_a_token_id(ids: list[float]):
    python, _ = _embeddings(ids)
    with pytest.raises(AssertionError, match="token ids"):
        python.forward()


def test_the_embedding_draws_each_row_as_a_linear_layers_node_and_is_not_decayed():
    python, _ = _embeddings([0.0, 1.0, 2.0])
    python.randomize_fan_in_aware(default_rng(5))
    expected = NUMPY.random_weights(NUMPY.default_rng(5), 4, 2)
    assert bits([list(row.weights) for row in python.rows]) == bits(expected.tolist())
    assert all(not row.weights_decayed and not row.has_bias for row in python.weight_sets())


# the token-wise output


@pytest.mark.usefixtures("token_exp")
def test_the_token_wise_softmax_is_numpys_by_bits_where_no_product_rounds():
    # W of 0 and 1 and b of halves: each z is a sum of at most two of X's values, exact either way,
    # so only exp could differ, and it is math.exp on both sides
    W, b = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0.0, 0.5, -0.5]
    X = [[0.25, -0.375, 1.0, 2.0], [-1.5, 0.75, 0.0, 0.125]]
    Y = [[0.0, 1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]]
    inputs = _input(X[0])
    python, array = TokenSoftmaxLayer(inputs, 3, 2), TokenSoftmaxArrayLayer(3, 2, 2)
    python.restore_state(list(zip(W, b, strict=True)))
    array.W, array.b = np.array(W), np.array(b)
    expected = array.forward_batch(np.array(X))
    array.compute_output_delta_batch(np.array(Y))
    for x, y, a, delta in zip(X, Y, expected.tolist(), array.delta_batch.tolist(), strict=True):
        inputs.update_state(tuple(x))
        python.forward()
        assert bits([node.value() for node in python.nodes]) == bits(a)
        for node, target in zip(python.nodes, y, strict=True):
            node.compute_output_delta(target)
        assert bits([node.delta for node in python.nodes]) == bits(delta)


def test_the_token_wise_softmax_is_only_an_output_layer():
    python = TokenSoftmaxLayer(_input([0.0] * 4), 3, 2)
    with pytest.raises(NotImplementedError, match="the output layer"):
        python.compute_hidden_deltas(downstream([0.0] * 6))


# the mask


def _attention(causal: bool, X: list[float], tokens: int = 4, features: int = 6, heads: int = 2) -> AttentionLayer:
    layer = AttentionLayer(_input(X), tokens, features, heads, causal=causal)
    rng = default_rng(0)
    layer.restore_state(
        [([rng.uniform(-1.0, 1.0) for _ in row.weights], rng.uniform(-1.0, 1.0)) for row in layer.weight_sets()]
    )
    return layer


def _uniform(count: int, seed: int) -> list[float]:
    return np.random.default_rng(seed).uniform(-1.0, 1.0, count).tolist()


def test_a_causal_layers_masked_weights_are_exactly_zero():
    layer = _attention(True, _uniform(24, 1))
    layer.forward()
    # packed (T, h * T): head i in columns i * T..
    for head in range(2):
        for i, row in enumerate(layer._P):  # pyright: ignore[reportPrivateUsage]
            weights = row[head * 4 : (head + 1) * 4]
            assert all(w == 0.0 for w in weights[i + 1 :]) and all(w > 0.0 for w in weights[: i + 1])
        assert layer._P[0][head * 4] == 1.0  # pyright: ignore[reportPrivateUsage]


def test_in_a_causal_layer_the_last_token_reaches_only_its_own_output():
    # its key and value meet only its own query, so with its delta zero nothing reaches its input
    layer = _attention(True, _uniform(24, 1))
    layer.forward()
    delta = [*_uniform(18, 2), *[0.0] * 6]
    layer.compute_hidden_deltas(downstream(delta))
    dX = [layer.downstream_sum(i) for i in range(24)]
    assert all(d == 0.0 for d in dX[18:]) and all(d != 0.0 for d in dX[:18])


def test_a_causal_layer_over_one_token_is_the_unmasked_layer_by_bits():
    X, delta = _uniform(6, 1), _uniform(6, 2)
    masked, unmasked = _attention(True, X, tokens=1), _attention(False, X, tokens=1)
    for layer in (masked, unmasked):
        layer.forward()
        layer.compute_hidden_deltas(downstream(delta))
    assert bits([node.value() for node in masked.nodes]) == bits([node.value() for node in unmasked.nodes])
    assert bits([masked.downstream_sum(i) for i in range(6)]) == bits([unmasked.downstream_sum(i) for i in range(6)])


@pytest.mark.parametrize("position", range(TOKENS))
def test_a_later_token_cant_change_an_earlier_output_in_a_causal_model(position: int):
    built = network(CAUSAL)
    state = (1.0, 4.0, 0.0, 6.0, 2.0)
    changed = tuple(float((v + 3) % VOCABULARY) if t == position else v for t, v in enumerate(state))
    before, after = built.predict_probabilities(state), built.predict_probabilities(changed)
    assert bits(before[:position]) == bits(after[:position])
    assert all(bits(b) != bits(a) for b, a in zip(before[position:], after[position:]))


def test_without_the_mask_a_later_token_changes_every_output():
    built = network(LEAK)
    before = built.predict_probabilities((1.0, 4.0, 0.0, 6.0, 2.0))
    after = built.predict_probabilities((1.0, 4.0, 0.0, 6.0, 3.0))
    assert all(bits(b) != bits(a) for b, a in zip(before, after))


# the network


def test_a_sequence_networks_class_is_the_argmax_per_token_and_its_targets_one_hot_per_token():
    built = network(CAUSAL)
    state = (1.0, 4.0, 0.0, 6.0, 2.0)
    probabilities = built.predict_probabilities(state)
    assert len(probabilities) == TOKENS and all(len(token) == VOCABULARY for token in probabilities)
    assert built.classify_state(state) == tuple(int(np.argmax(token)) for token in probabilities)
    array = numpy_network(CAUSAL)
    array.restore(as_array_snapshot(built))
    assert built.classify_state(state) == array.classify_state(state)

    label = (2, 0, 6, 6, 1)
    built._output_deltas(label)  # pyright: ignore[reportPrivateUsage]
    flat = [p for token in probabilities for p in token]
    target = [1.0 if k == label[t] else 0.0 for t in range(TOKENS) for k in range(VOCABULARY)]
    assert bits([node.delta for node in built.output_layer.nodes]) == bits(
        [(p - y) / TOKENS for p, y in zip(flat, target, strict=True)]
    )
    with pytest.raises(AssertionError, match="one class per token"):
        built._output_deltas(label[:-1])  # pyright: ignore[reportPrivateUsage]


def test_the_multiclass_and_single_output_networks_refuse_a_token_wise_output():
    for cls in (SequentialMultiClassBackpropClassifierNetwork, SequentialBackpropClassifierNetwork):
        with pytest.raises(AssertionError, match="shape='sequence'"):
            cls(_input_shape(CAUSAL), CAUSAL, SGD())


def test_the_sequence_network_refuses_one_without_a_token_wise_output():
    specs = SEQUENCE["an embedding, then the mean"]
    with pytest.raises(AssertionError, match="applied to each token"):
        SequentialSequenceBackpropNetwork(_input_shape(specs), specs, SGD())


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), Adam()], ids=["sgd", "momentum", "adam"])
@pytest.mark.parametrize("name", SEQUENCE)
def test_learn_and_a_batch_of_one_agree(name: str, rule: UpdateRule):
    specs = SEQUENCE[name]
    assert_learn_and_a_batch_of_one_agree("python", network(specs, rule), network(specs, rule), rows(specs, 4))


@pytest.mark.parametrize("rule", [Momentum(0.9), Adam()], ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", TOKEN_WISE)
def test_the_layer_major_path_is_the_example_major_loop_by_bits(name: str, rule: UpdateRule):
    specs = TOKEN_WISE[name]
    example_major, layer_major = network(specs, rule), network(specs, rule)
    for size in (3, 5, 4):
        batch = rows(specs, size, seed=size)
        example_major.learn_batch(0.3, batch)
        layer_major._learn_batch_layer_major(0.3, batch)  # pyright: ignore[reportPrivateUsage]
    assert bits(layer_major.snapshot()) == bits(example_major.snapshot())


@pytest.mark.parametrize("name", TOKEN_WISE)
def test_a_saved_sequence_network_loads_and_trains_on_by_bits(name: str, tmp_path: Path):
    specs = TOKEN_WISE[name]
    built = network(specs, Adam())
    data = rows(specs, 6)
    built.learn_batch(0.1, data[:3])
    path = str(tmp_path / "sequence.json")
    built.save(path)

    state = network_to_json(built)
    assert state["shape"] == "sequence" and state["implementation"] == "python"
    loaded = load_network(path)
    assert isinstance(loaded, SequentialSequenceBackpropNetwork)
    assert isinstance(SequentialSequenceBackpropNetwork.load(path), SequentialSequenceBackpropNetwork)
    built.learn_batch(0.1, data[3:])
    loaded.learn_batch(0.1, data[3:])
    assert bits(loaded.snapshot()) == bits(built.snapshot())


# parity with numpy


@pytest.mark.usefixtures("attention_exp", "token_exp")
@pytest.mark.parametrize("rule", LINEAR_RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", SEQUENCE)
def test_training_matches_numpy_within_the_dense_layers_rounding(name: str, rule: UpdateRule):
    # the sums are folds in both, the products not: pure Python's folds against numpy's BLAS, so the
    # networks agree within the tolerance every pure-Python parity test allows
    specs = SEQUENCE[name]
    python, array = network(specs, rule), numpy_network(specs, rule)
    array.restore(as_array_snapshot(python))

    learn_in_step(0.1, rows(specs, 40), (python, array))

    assert_snapshots_close(as_array_snapshot(python), array.snapshot())


@pytest.mark.usefixtures("attention_exp", "token_exp")
@pytest.mark.parametrize("name", SEQUENCE)
def test_every_step_under_adam_has_numpys_gradients(name: str):
    specs = SEQUENCE[name]
    assert_every_step_has_numpys_gradients(network(specs, Adam()), numpy_network(specs), rows(specs, 40))


def test_the_output_layer_is_a_dense_one_applied_to_each_token():
    built = network(CAUSAL)
    output = CAUSAL[-1]
    assert isinstance(output, Dense) and isinstance(built.output_layer, TokenSoftmaxLayer)
    assert len(built.output_layer.nodes) == TOKENS * output.size
