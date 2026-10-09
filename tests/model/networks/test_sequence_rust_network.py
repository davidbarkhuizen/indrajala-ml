"""
Sequence models on Rust (the sequence task workplan, stage 5): randomize's draws against numpy's by
bits; Embedding's rows and scatter-add gradient against numpy's by bits, and its refusals; the
token-wise softmax against numpy's by bits where no product rounds, and its delta (P - Y) / T; the
causal mask (masked weights exactly 0, a later token can't change an earlier output, one token as
the unmasked layer); the sequence shape's class and save and load; learn against a batch of one; and
parity with numpy, after 50 steps under the rules whose step is linear in the gradient and per step
under Adam. The cases are tests/model/specs/test_layer_specs.py's SEQUENCE; the gradient check is
tests/model/networks/test_sequence_array_network.py's, on numpy.
"""

from pathlib import Path
from typing import Any

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.layers.numpy import attention_array_layer, token_array_layer
from indrajala_ml.model.layers.numpy.token_array_layer import EmbeddingArrayLayer, TokenSoftmaxArrayLayer
from indrajala_ml.model.layers.rust.attention_rust_array_layer import AttentionRustArrayLayer
from indrajala_ml.model.layers.rust.token_rust_array_layer import EmbeddingRustArrayLayer, TokenSoftmaxRustArrayLayer
from indrajala_ml.model.networks.sequential_array_network import (
    SequentialArrayNetwork,
    SequentialSequenceRustArrayNetwork,
)
from indrajala_ml.model.persistence.format2 import network_to_json
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.layer_specs import LayerSpec, token_wise_output
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule
from tests.helpers import assert_learn_and_a_batch_of_one_agree, bits, exp_by_crate, patching, to_numpy
from tests.model.networks.test_attention_rust_network import (
    LINEAR_RULES,
    _assert_every_adam_step_has_numpys_gradients,  # pyright: ignore[reportPrivateUsage]
    _assert_training_matches_numpy,  # pyright: ignore[reportPrivateUsage]
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


def network(specs: list[LayerSpec], rule: UpdateRule | None = None, backend: Any = RUST, seed: int = 3) -> Any:
    shape = "sequence" if token_wise_output(specs) else "multiclass"
    built = SequentialArrayNetwork(
        _input_shape(specs), specs, SGD() if rule is None else rule, shape=shape, backend=backend
    )
    built.rng = backend.default_rng(seed)
    built.randomize()
    return built


def _array(values: Any) -> pa.Array:
    return pa.Array(np.asarray(values, dtype=np.float64).tolist())


# numpy's softmax exps, attention's and the token-wise output's, as the crate's (Rust's f64::exp)
attention_exp = patching(attention_array_layer, "exp", exp_by_crate)
token_exp = patching(token_array_layer, "exp", exp_by_crate)


@pytest.mark.parametrize("name", SEQUENCE)
def test_randomize_draws_numpys_parameters_by_bits(name: str):
    rust, numpy = network(SEQUENCE[name]), network(SEQUENCE[name], backend=NUMPY)
    assert bits([to_numpy(a) for entry in rust.snapshot() for a in entry]) == bits(
        [a for entry in numpy.snapshot() for a in entry]
    )


def test_a_sequence_network_on_rust_is_its_class():
    assert isinstance(network(CAUSAL), SequentialSequenceRustArrayNetwork)


# Embedding


def _embeddings() -> tuple[EmbeddingRustArrayLayer, EmbeddingArrayLayer]:
    table = np.random.default_rng(0).uniform(-1.0, 1.0, (4, 2))
    rust, numpy = EmbeddingRustArrayLayer(3, 4, 2), EmbeddingArrayLayer(3, 4, 2)
    rust.E, numpy.E = _array(table), table
    return rust, numpy


def test_the_embedding_gives_numpys_rows_and_scatter_adds_numpys_gradient_by_bits():
    rust, numpy = _embeddings()
    X = np.array([[3.0, 0.0, 3.0], [3.0, 2.0, 0.0]])
    assert bits(to_numpy(rust.forward_batch(_array(X)))) == bits(numpy.forward_batch(X))
    # large and small values in one row of E, so a different order of the adds would show
    delta = np.array([[0.1, 1e16, 0.2, 1.0, 0.3, -1e16], [0.7, 1.0, 0.5, -2.0, 0.25, 3.0]])
    rust.delta_batch = _array(delta)
    numpy._backward(delta)  # pyright: ignore[reportPrivateUsage]
    rust.accumulate_gradient_batch(_array(X))
    numpy.accumulate_gradient_batch(X)
    assert bits(to_numpy(rust.grad_E)) == bits(numpy.grad_E)
    with pytest.raises(NotImplementedError, match="the first layer"):
        rust.downstream_batch()


def test_the_embeddings_single_example_is_its_batch_of_one():
    rust, _ = _embeddings()
    x = [3.0, 0.0, 1.0]
    assert bits(to_numpy(rust.forward(_array(x)))) == bits(to_numpy(rust.forward_batch(_array([x])))[0])


@pytest.mark.parametrize(
    "ids", [[0.5, 1.0, 2.0], [0.0, 4.0, 1.0], [-1.0, 0.0, 1.0]], ids=["a fraction", "past the vocabulary", "negative"]
)
def test_the_embedding_refuses_what_isnt_a_token_id(ids: list[float]):
    rust, _ = _embeddings()
    with pytest.raises(ValueError, match="token ids"):
        rust.forward_batch(_array([ids]))


# the token-wise output


@pytest.mark.usefixtures("token_exp")
def test_the_token_wise_softmax_is_numpys_by_bits_where_no_product_rounds():
    # W of 0 and 1 and b of halves: each Z is a sum of at most two of X's values, exact either way,
    # so only exp could differ, and it is the crate's on both sides
    rust, numpy = TokenSoftmaxRustArrayLayer(3, 2, 2), TokenSoftmaxArrayLayer(3, 2, 2)
    W, b = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([0.0, 0.5, -0.5])
    rust.W, rust.b, numpy.W, numpy.b = _array(W), _array(b), W, b
    X = np.array([[0.25, -0.375, 1.0, 2.0], [-1.5, 0.75, 0.0, 0.125]])
    assert bits(to_numpy(rust.forward_batch(_array(X)))) == bits(numpy.forward_batch(X))

    Y = np.array([[0.0, 1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
    rust.compute_output_delta_batch(_array(Y))
    numpy.compute_output_delta_batch(Y)
    assert bits(to_numpy(rust.delta_batch)) == bits(numpy.delta_batch)

    rust.forward(_array(X[0]))
    rust.compute_output_delta(_array(Y[0]))
    assert bits(to_numpy(rust.delta)) == bits(numpy.delta_batch[0])


# the mask


def _attention(causal: bool, tokens: int = 4, features: int = 6, heads: int = 2) -> AttentionRustArrayLayer:
    layer = AttentionRustArrayLayer(tokens, features, heads, causal=causal)
    rng = np.random.default_rng(0)
    layer.set_parameters([_array(rng.uniform(-1.0, 1.0, to_numpy(p).shape)) for p in layer.parameters()])
    return layer


def test_a_causal_layers_masked_weights_are_exactly_zero():
    layer = _attention(True)
    layer.forward_batch(_array(np.random.default_rng(1).uniform(-1.0, 1.0, (3, 24))))
    # packed (N * T, h * T): head i in columns i * T..
    P = to_numpy(layer._P).reshape(3, 4, 2, 4)  # pyright: ignore[reportPrivateUsage]
    future = np.triu(np.ones((4, 4), dtype=bool), k=1)
    for head in range(2):
        weights = P[:, :, head, :]
        assert (weights[:, future] == 0.0).all() and (weights[:, ~future] > 0.0).all()
        assert (weights[:, 0, 0] == 1.0).all()


def test_a_causal_layer_over_one_token_is_the_unmasked_layer_by_bits():
    X = _array(np.random.default_rng(1).uniform(-1.0, 1.0, (3, 6)))
    masked, unmasked = _attention(True, tokens=1), _attention(False, tokens=1)
    assert bits(to_numpy(masked.forward_batch(X))) == bits(to_numpy(unmasked.forward_batch(X)))


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


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), Adam()], ids=["sgd", "momentum", "adam"])
@pytest.mark.parametrize("name", SEQUENCE)
def test_learn_and_a_batch_of_one_agree(name: str, rule: UpdateRule):
    specs = SEQUENCE[name]
    assert_learn_and_a_batch_of_one_agree("rust", network(specs, rule), network(specs, rule), rows(specs, 4))


@pytest.mark.parametrize("name", TOKEN_WISE)
def test_a_saved_sequence_network_loads_and_trains_on_by_bits(name: str, tmp_path: Path):
    specs = TOKEN_WISE[name]
    built = network(specs, Adam())
    data = rows(specs, 6)
    built.learn_batch(0.1, data[:3])
    path = str(tmp_path / "sequence.json")
    built.save(path)

    state = network_to_json(built)
    assert state["shape"] == "sequence" and state["implementation"] == "rust"
    loaded = load_network(path)
    assert isinstance(loaded, SequentialSequenceRustArrayNetwork)
    built.learn_batch(0.1, data[3:])
    loaded.learn_batch(0.1, data[3:])
    assert bits([to_numpy(a) for entry in loaded.snapshot() for a in entry]) == bits(
        [to_numpy(a) for entry in built.snapshot() for a in entry]
    )


# parity with numpy


@pytest.mark.usefixtures("attention_exp", "token_exp")
@pytest.mark.parametrize("rule", LINEAR_RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", SEQUENCE)
def test_training_matches_numpy_within_the_dense_layers_rounding(name: str, rule: UpdateRule):
    _assert_training_matches_numpy(SEQUENCE[name], rule, 0.1, network, rows)


@pytest.mark.usefixtures("attention_exp", "token_exp")
@pytest.mark.parametrize("name", SEQUENCE)
def test_every_step_under_adam_has_numpys_gradients(name: str):
    _assert_every_adam_step_has_numpys_gradients(SEQUENCE[name], 0.1, network, rows)
