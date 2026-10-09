"""
Sequence models in numpy (the sequence task workplan, stage 3): the causal mask in attend (a later
token can't change an earlier output; masked weights exactly zero; an unmasked layer as before),
Embedding (its rows, its scatter-add gradient, its refusals), the token-wise softmax output and its
delta, the sequence shape (targets, classification, save and load, its refusals), the gradient
check on every accepted sequence model, learn against a batch of one, the trainer's per-token
accuracy, the evaluation (D8), and a short run in which a causal model learns a text.
"""

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from indrajala_ml.data.text_data import Vocabulary, windows
from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.layers.numpy.attention_array_layer import AttentionArrayLayer
from indrajala_ml.model.layers.numpy.token_array_layer import EmbeddingArrayLayer, TokenSoftmaxArrayLayer
from indrajala_ml.model.networks.sequential_array_network import (
    SequentialArrayNetwork,
    SequentialSequenceArrayNetwork,
)
from indrajala_ml.model.persistence.format2 import network_from_json, network_to_json
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.persistence.model_io import save_json
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    Dense,
    Embedding,
    LayerNorm,
    LayerSpec,
    Position,
    Residual,
    token_wise_output,
)
from indrajala_ml.model.specs.spec_shapes import InputShape
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule
from indrajala_ml.training.sequence_evaluate import sequence_evaluate
from indrajala_ml.training.train import train_backprop_network_mini_batch
from tests.gradient_check import check_gradients
from tests.helpers import assert_learn_and_a_batch_of_one_agree, bits
from tests.model.specs.test_layer_specs import FFN_BLOCK, SEQUENCE, TOKEN_OUTPUT

# test_layer_specs' sequence models read 5 ids of a vocabulary of 7, or a (4, 4, 1) image whose
# Patches(2) gives 4 tokens, each with 3 classes
TOKENS, VOCABULARY = 5, 7
IDS: InputShape = (TOKENS,)
IMAGE: InputShape = (4, 4, 1)

TOKEN_WISE = {name: specs for name, specs in SEQUENCE.items() if token_wise_output(specs)}
CAUSAL = SEQUENCE["a causal transformer"]
LEAK = SEQUENCE["the leak arm, unmasked"]


def _input_shape(specs: list[LayerSpec]) -> InputShape:
    return IDS if isinstance(specs[0], Embedding) else IMAGE


def network(specs: list[LayerSpec], rule: UpdateRule | None = None, seed: int = 3) -> Any:
    shape = "sequence" if token_wise_output(specs) else "multiclass"
    built = SequentialArrayNetwork(_input_shape(specs), specs, SGD() if rule is None else rule, shape=shape)
    built.rng = NUMPY.default_rng(seed)
    built.randomize()
    return built


def rows(specs: list[LayerSpec], count: int, seed: int = 1) -> list[tuple[tuple[float, ...], Any]]:
    """count examples for specs: ids or an image, and a class per token (or one class, for a mean)."""
    rng = random.Random(seed)
    output = specs[-1]
    assert isinstance(output, Dense)
    examples: list[tuple[tuple[float, ...], Any]] = []
    for _ in range(count):
        if isinstance(specs[0], Embedding):
            state = tuple(float(rng.randrange(VOCABULARY)) for _ in range(TOKENS))
            tokens = TOKENS
        else:
            state = tuple(rng.uniform(0.0, 1.0) for _ in range(16))
            tokens = 4
        if token_wise_output(specs):
            label: Any = tuple(rng.randrange(output.size) for _ in range(tokens))
        else:
            label = rng.randrange(output.size)
        examples.append((state, label))
    return examples


def _split(batch: list[tuple[tuple[float, ...], Any]]) -> tuple[list[tuple[float, ...]], list[Any]]:
    return [state for state, _ in batch], [label for _, label in batch]


# the mask


def _attention(causal: bool, tokens: int = 4, features: int = 6, heads: int = 2, seed: int = 0) -> AttentionArrayLayer:
    layer = AttentionArrayLayer(tokens, features, heads, causal=causal)
    rng = np.random.default_rng(seed)
    layer.set_parameters([rng.uniform(-1.0, 1.0, parameter.shape) for parameter in layer.parameters()])
    return layer


def test_a_causal_layers_masked_weights_are_exactly_zero_and_each_row_sums_to_one():
    layer = _attention(True)
    layer.forward_batch(np.random.default_rng(1).uniform(-1.0, 1.0, (3, 24)))
    P = layer._P  # pyright: ignore[reportPrivateUsage]
    future = np.triu(np.ones((4, 4), dtype=bool), k=1)
    assert (P[..., future] == 0.0).all() and (P[..., ~future] > 0.0).all()
    np.testing.assert_allclose(P.sum(axis=-1), 1.0, rtol=1e-15)
    # the first token attends to itself alone
    assert (P[..., 0, 0] == 1.0).all()


def test_in_a_causal_layer_the_last_token_reaches_only_its_own_output():
    # its key and value meet only its own query, so with its delta zero nothing reaches its input
    layer = _attention(True)
    layer.forward_batch(np.random.default_rng(1).uniform(-1.0, 1.0, (3, 24)))
    delta = np.random.default_rng(2).uniform(-1.0, 1.0, (3, 24))
    delta[:, 18:] = 0.0
    layer._backward(delta)  # pyright: ignore[reportPrivateUsage]
    assert (layer.downstream_batch()[:, 18:] == 0.0).all()
    assert (layer.downstream_batch()[:, :18] != 0.0).all()


def test_a_causal_layer_over_one_token_is_the_unmasked_layer_by_bits():
    X = np.random.default_rng(1).uniform(-1.0, 1.0, (3, 6))
    masked, unmasked = _attention(True, tokens=1), _attention(False, tokens=1)
    assert bits(masked.forward_batch(X)) == bits(unmasked.forward_batch(X))
    delta = np.random.default_rng(2).uniform(-1.0, 1.0, (3, 6))
    masked._backward(delta)  # pyright: ignore[reportPrivateUsage]
    unmasked._backward(delta)  # pyright: ignore[reportPrivateUsage]
    assert bits(masked.downstream_batch()) == bits(unmasked.downstream_batch())


def _outputs(built: Any, state: tuple[float, ...]) -> list[list[float]]:
    return built.predict_probabilities(state)


@pytest.mark.parametrize("position", range(TOKENS))
def test_a_later_token_cant_change_an_earlier_output_in_a_causal_model(position: int):
    built = network(CAUSAL)
    state = (1.0, 4.0, 0.0, 6.0, 2.0)
    changed = tuple(float((v + 3) % VOCABULARY) if t == position else v for t, v in enumerate(state))
    before, after = _outputs(built, state), _outputs(built, changed)
    assert bits(before[:position]) == bits(after[:position])
    assert all(bits(b) != bits(a) for b, a in zip(before[position:], after[position:]))


def test_without_the_mask_a_later_token_changes_every_output():
    built = network(LEAK)
    before = _outputs(built, (1.0, 4.0, 0.0, 6.0, 2.0))
    after = _outputs(built, (1.0, 4.0, 0.0, 6.0, 3.0))
    assert all(bits(b) != bits(a) for b, a in zip(before, after))


# Embedding


def test_the_embedding_gives_each_ids_row_of_its_table():
    layer = EmbeddingArrayLayer(3, 4, 2)
    layer.E = np.arange(8, dtype=np.float64).reshape(4, 2)
    out = layer.forward_batch(np.array([[3.0, 0.0, 3.0], [1.0, 2.0, 0.0]]))
    assert bits(out) == bits(np.array([[6.0, 7.0, 0.0, 1.0, 6.0, 7.0], [2.0, 3.0, 4.0, 5.0, 0.0, 1.0]]))


def test_the_embeddings_gradient_is_a_scatter_add_in_row_order():
    layer = EmbeddingArrayLayer(3, 4, 2)
    X = np.array([[3.0, 0.0, 3.0], [3.0, 2.0, 0.0]])
    layer.forward_batch(X)
    delta = np.array([[0.1, 1e16, 0.2, 1.0, 0.3, -1e16], [0.7, 1.0, 0.5, -2.0, 0.25, 3.0]])
    layer._backward(delta)  # pyright: ignore[reportPrivateUsage]
    layer.accumulate_gradient_batch(X)
    rows_ = delta.reshape(6, 2)
    ids = [3, 0, 3, 3, 2, 0]
    expected = np.zeros((4, 2))
    for row, i in zip(rows_, ids, strict=True):
        for j in range(2):
            expected[i, j] = expected[i, j] + row[j]
    assert bits(layer.grad_E) == bits(expected)
    with pytest.raises(NotImplementedError, match="the first layer"):
        layer.downstream_batch()


@pytest.mark.parametrize(
    "ids", [[0.5, 1.0, 2.0], [0.0, 4.0, 1.0], [-1.0, 0.0, 1.0]], ids=["a fraction", "past the vocabulary", "negative"]
)
def test_the_embedding_refuses_what_isnt_a_token_id(ids: list[float]):
    with pytest.raises(AssertionError, match="token ids"):
        EmbeddingArrayLayer(3, 4, 2).forward_batch(np.array([ids]))


def test_randomize_draws_the_embedding_as_a_weight_matrix_of_its_vocabulary_rows():
    built = network(SEQUENCE["the embedding, then the output"])
    rng = np.random.default_rng(3)
    expected: list[Any] = [rng.uniform(-1 / np.sqrt(6), 1 / np.sqrt(6), (7, 6))]
    expected += [rng.uniform(-1 / np.sqrt(6), 1 / np.sqrt(6), shape) for shape in ((7, 6), (7,))]
    assert bits([array for entry in built.snapshot() for array in entry]) == bits(expected)


# the token-wise output


def test_the_token_wise_softmax_normalizes_each_token_and_its_delta_is_the_mean_losss():
    layer = TokenSoftmaxArrayLayer(3, 2, 2)
    layer.W = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    layer.b = np.array([0.0, 0.5, -0.5])
    X = np.array([[0.2, -0.3, 1.0, 2.0]])
    A = layer.forward_batch(X)
    for t in range(2):
        z = layer.W @ X[0, 2 * t : 2 * t + 2] + layer.b
        e = np.exp(z - z.max())
        np.testing.assert_allclose(A[0, 3 * t : 3 * t + 3], e / e.sum(), rtol=1e-15)
    Y = np.array([[0.0, 1.0, 0.0, 1.0, 0.0, 0.0]])
    layer.compute_output_delta_batch(Y)
    assert bits(layer.delta_batch) == bits((A - Y) / 2)


# the network


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("name", SEQUENCE)
def test_every_gradient_matches_its_finite_difference(name: str, batch_size: int):
    specs = SEQUENCE[name]
    built = network(specs)
    # a trained step first, so the positions, gamma and beta aren't at their initial values
    built.learn_batch(0.5, rows(specs, 4, seed=2))
    check_gradients(built, *_split(rows(specs, batch_size)))


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), Adam()], ids=["sgd", "momentum", "adam"])
@pytest.mark.parametrize("name", SEQUENCE)
def test_learn_and_a_batch_of_one_agree(name: str, rule: UpdateRule):
    specs = SEQUENCE[name]
    assert_learn_and_a_batch_of_one_agree("numpy", network(specs, rule), network(specs, rule), rows(specs, 4))


def test_a_sequence_networks_targets_are_one_hot_per_token_and_its_class_the_argmax_per_token():
    built = network(CAUSAL)
    target = built._target_batch_array([(0, 6, 1, 1, 2)])  # pyright: ignore[reportPrivateUsage]
    expected = np.zeros((1, TOKENS * VOCABULARY))
    for t, label in enumerate((0, 6, 1, 1, 2)):
        expected[0, t * VOCABULARY + label] = 1.0
    assert bits(target) == bits(expected)
    state = (1.0, 4.0, 0.0, 6.0, 2.0)
    probabilities = built.predict_probabilities(state)
    assert built.classify_state(state) == tuple(int(np.argmax(token)) for token in probabilities)
    with pytest.raises(AssertionError, match="one class per token"):
        built.learn(0.5, state, (0, 1))


def test_the_numpy_and_rust_backends_take_the_same_argmax_per_token():
    matrix = [[0.1, 0.5, 0.5, 0.9, 0.0, 0.2], [0.3, 0.2, 0.1, 0.0, 0.0, 0.0]]
    expected = [(1, 0), (0, 0)]
    assert NUMPY.argmax_token_rows(np.array(matrix), 3) == expected
    assert RUST.argmax_token_rows(RUST.matrix(matrix), 3) == expected


def test_the_multiclass_and_single_output_shapes_refuse_a_token_wise_output():
    for shape in ("multiclass", "single_output"):
        with pytest.raises(AssertionError, match="shape='sequence'"):
            SequentialArrayNetwork(IDS, CAUSAL, SGD(), shape=shape)


def test_the_sequence_shape_refuses_a_network_without_a_token_wise_output():
    with pytest.raises(AssertionError, match="applied to each token"):
        SequentialArrayNetwork(IDS, SEQUENCE["an embedding, then the mean"], SGD(), shape="sequence")


@pytest.mark.parametrize("name", TOKEN_WISE)
def test_a_saved_sequence_network_loads_and_trains_on_by_bits(name: str, tmp_path: Path):
    specs = TOKEN_WISE[name]
    built = network(specs, Adam())
    data = rows(specs, 6)
    built.learn_batch(0.1, data[:3])
    path = str(tmp_path / "sequence.json")
    built.save(path)

    state = network_to_json(built)
    assert state["shape"] == "sequence" and "preset" not in state
    loaded = load_network(path)
    assert isinstance(loaded, SequentialSequenceArrayNetwork)
    assert isinstance(SequentialSequenceArrayNetwork.load(path), SequentialSequenceArrayNetwork)
    built.learn_batch(0.1, data[3:])
    loaded.learn_batch(0.1, data[3:])
    assert bits(loaded.snapshot()) == bits(built.snapshot())


def test_a_pure_python_sequence_file_is_refused_until_stage_6(tmp_path: Path):
    state = network_to_json(network(CAUSAL))
    state["implementation"] = "python"
    network_from_json(state)  # the file itself is well formed
    path = str(tmp_path / "python.json")
    save_json(path, state)
    with pytest.raises(NotImplementedError, match="stage 6"):
        load_network(path)


# training and evaluation


def test_the_trainers_accuracy_counts_tokens():
    built = network(SEQUENCE["the embedding, then the output"])
    data = rows(SEQUENCE["the embedding, then the output"], 6)
    result = train_backprop_network_mini_batch(built, data, 3, 0.0, epochs=1, rng=random.Random(1))
    predictions = [built.classify_state(state) for state, _ in data]
    right = sum(p == c for predicted, (_, labels) in zip(predictions, data) for p, c in zip(predicted, labels))
    assert result.diagnostic.best_training_accuracy == right / (6 * TOKENS)


def test_the_evaluation_is_the_mean_token_cross_entropy_and_the_token_accuracy():
    built = network(CAUSAL)
    data = rows(CAUSAL, 40)
    evaluation = sequence_evaluate(built, data)
    losses = [
        -math.log(built.predict_probabilities(state)[t][label])
        for state, labels in data
        for t, label in enumerate(labels)
    ]
    right = [
        predicted == label
        for state, labels in data
        for predicted, label in zip(built.classify_state(state), labels, strict=True)
    ]
    assert evaluation.tokens == 40 * TOKENS
    assert evaluation.cross_entropy == pytest.approx(math.fsum(losses) / len(losses), rel=1e-12)
    assert evaluation.bits_per_token == pytest.approx(evaluation.cross_entropy / math.log(2.0), rel=1e-15)
    assert evaluation.accuracy == sum(right) / len(right)


def test_a_causal_model_learns_a_text():
    # words of three letters in a random order: within a word, the letters before predict the next
    text = "".join(random.Random(4).choice(["abc", "abd", "bad", "cab"]) for _ in range(1200))
    vocabulary = Vocabulary.of(text)
    examples = windows(vocabulary.encode(text), 8)
    size = len(vocabulary)
    specs: list[LayerSpec] = [
        Embedding(size, 16),
        Position(),
        Residual((LayerNorm(), Attention(heads=2, causal=True))),
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


def test_an_ffn_only_model_is_blind_to_the_other_tokens():
    # the study's FFN-only arm (D9): a token's output reads its own id and position only
    specs: list[LayerSpec] = [Embedding(VOCABULARY, 6), Position(), FFN_BLOCK, TOKEN_OUTPUT]
    built = network(specs)
    before = _outputs(built, (1.0, 4.0, 0.0, 6.0, 2.0))
    after = _outputs(built, (1.0, 4.0, 3.0, 6.0, 2.0))
    assert bits(before[:2] + before[3:]) == bits(after[:2] + after[3:])
