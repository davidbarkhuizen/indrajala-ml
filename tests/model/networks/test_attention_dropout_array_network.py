"""
Dropout among the tokens in numpy (the attention-dropout workplan, stage 3): the mask on attention's
weights P (D3, D4: one (N, h, T, T) draw, row-major, every entry drawn; P~ = P * M / keep; the
backward pass by hand), the token-wise Dropout (D2), nothing drawn in inference or at dropout 0, the
network's draw order (its layers' masks in forward order from its one generator), the gradient check
with the masks held, learn against a batch of one, save and load by bits, and a short run in which
a model with GPT's three dropouts still learns a text.
"""

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from indrajala_ml.data.text_data import Vocabulary, windows
from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.layers.numpy.attention_array_layer import AttentionArrayLayer
from indrajala_ml.model.layers.numpy.token_array_layer import TokenDropoutArrayLayer
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
)
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule
from indrajala_ml.training.sequence_evaluate import sequence_evaluate
from indrajala_ml.training.train import train_backprop_network_mini_batch
from tests.gradient_check import check_gradients
from tests.helpers import assert_learn_and_a_batch_of_one_agree, bits
from tests.model.networks.test_sequence_array_network import network, rows
from tests.model.specs.test_layer_specs import DROPOUT

GPT = DROPOUT["GPT's three dropouts"]
# a batch of 3 examples of 4 tokens of 6 features, in 2 heads of 3
N, T, D, H = 3, 4, 6, 2


def _attention(dropout: float, causal: bool = False, seed: int = 0) -> AttentionArrayLayer:
    layer = AttentionArrayLayer(T, D, H, causal=causal, dropout=dropout)
    rng = np.random.default_rng(seed)
    layer.set_parameters([rng.uniform(-1.0, 1.0, parameter.shape) for parameter in layer.parameters()])
    return layer


def _X(seed: int = 1) -> Any:
    return np.random.default_rng(seed).uniform(-1.0, 1.0, (N, T * D))


def _passes(layer: Any, X: Any, delta: Any) -> list[Any]:
    out = layer.forward_batch(X)
    layer._backward(delta)  # pyright: ignore[reportPrivateUsage]
    return [out, layer.downstream_batch()]


def _state(rng: np.random.Generator) -> Any:
    return rng.bit_generator.state


# attention's mask


@pytest.mark.parametrize("training", [False, True], ids=["inference", "training at dropout 0"])
def test_without_drawing_an_attention_with_dropout_is_the_one_without_by_bits(training: bool):
    # in inference a dropping layer draws nothing; at dropout 0 none does, training or not
    dropping = _attention(0.0 if training else 0.3)
    plain = _attention(0.0)
    dropping.set_rng(np.random.default_rng(7))
    dropping.set_training_mode(training)
    before = _state(dropping.rng)
    delta = _X(2)
    assert bits(_passes(dropping, _X(), delta)) == bits(_passes(plain, _X(), delta))
    assert _state(dropping.rng) == before


@pytest.mark.parametrize("causal", [False, True], ids=["unmasked", "causal"])
def test_the_mask_is_one_draw_per_weight_row_major_the_causally_masked_ones_included(causal: bool):
    layer = _attention(0.3, causal)
    layer.set_rng(np.random.default_rng(5))
    layer.set_training_mode(True)
    layer.forward_batch(_X())
    expected = np.random.default_rng(5).random((N, H, T, T)) >= 0.3
    assert bits(layer._M) == bits(expected.astype(np.float64))  # pyright: ignore[reportPrivateUsage]
    # exactly that many draws: the generator is where one (N, h, T, T) draw leaves it
    after = np.random.default_rng(5)
    after.random((N, H, T, T))
    assert _state(layer.rng) == _state(after)


def test_attend_drops_the_weights_and_scales_the_kept_ones_and_the_undropped_p_is_kept():
    layer, plain = _attention(0.3), _attention(0.0)
    layer.set_rng(np.random.default_rng(5))
    layer.set_training_mode(True)
    layer.forward_batch(_X())
    plain.forward_batch(_X())
    P, M = plain._P, layer._M  # pyright: ignore[reportPrivateUsage]
    assert M is not None and 0.0 < M.mean() < 1.0
    # the softmax's P as without dropout, then P~ = P * M / keep and H = P~ V
    assert bits(layer._P) == bits(P)  # pyright: ignore[reportPrivateUsage]
    dropped = P * M / 0.7
    assert bits(layer._H) == bits(plain._side_by_side(dropped @ plain._V))  # pyright: ignore[reportPrivateUsage]
    # a kept row no longer sums to 1, and a dropped weight is exactly 0
    assert (dropped[M == 0.0] == 0.0).all()


def test_the_backward_pass_takes_p_tilde_for_dv_and_the_mask_for_dp_by_bits():
    layer = _attention(0.3, causal=True)
    layer.set_rng(np.random.default_rng(5))
    layer.set_training_mode(True)
    layer.forward_batch(_X())
    layer.set_training_mode(False)  # as the network does before its backward pass
    delta = _X(2)
    layer._backward(delta)  # pyright: ignore[reportPrivateUsage]
    Q, K, V, P, M = layer._Q, layer._K, layer._V, layer._P, layer._M  # pyright: ignore[reportPrivateUsage]
    assert M is not None
    dH = layer._heads(layer._rows(delta, D) @ layer.Wo)  # pyright: ignore[reportPrivateUsage]
    dV = (P * M / 0.7).swapaxes(-1, -2) @ dH
    dP = (dH @ V.swapaxes(-1, -2)) * M / 0.7
    r = np.cumsum(dP * P, axis=-1)[..., -1:]
    dS = P * (dP - r)
    assert bits(layer._dV) == bits(layer._side_by_side(dV))  # pyright: ignore[reportPrivateUsage]
    assert bits(layer._dQ) == bits(layer._side_by_side((dS @ K) / layer.scale))  # pyright: ignore[reportPrivateUsage]
    dK = (dS.swapaxes(-1, -2) @ Q) / layer.scale
    assert bits(layer._dK) == bits(layer._side_by_side(dK))  # pyright: ignore[reportPrivateUsage]


def test_an_attention_refuses_a_dropout_outside_zero_to_one():
    for dropout in (-0.1, 1.0):
        with pytest.raises(AssertionError, match="dropout is in"):
            AttentionArrayLayer(T, D, H, dropout=dropout)


# the token-wise dropout


def test_the_token_dropout_masks_each_feature_and_scales_the_kept_ones_by_bits():
    layer = TokenDropoutArrayLayer(T, D, 0.25)
    layer.set_rng(np.random.default_rng(5))
    layer.set_training_mode(True)
    X, delta = _X(), _X(2)
    out, downstream = _passes(layer, X, delta)
    M = (np.random.default_rng(5).random((N, T * D)) >= 0.25).astype(np.float64)
    assert 0.0 < M.mean() < 1.0
    assert bits(out) == bits(X * M / 0.75)
    assert bits(downstream) == bits(delta * M / 0.75)


def test_in_inference_the_token_dropout_passes_its_input_on_and_draws_nothing():
    layer = TokenDropoutArrayLayer(T, D, 0.25)
    layer.set_rng(np.random.default_rng(5))
    before = _state(layer.rng)
    X, delta = _X(), _X(2)
    assert bits(_passes(layer, X, delta)) == bits([X, delta])
    assert _state(layer.rng) == before


def test_the_token_dropout_refuses_a_probability_outside_zero_to_one():
    for p in (-0.1, 1.0):
        with pytest.raises(AssertionError, match="drop_probability"):
            TokenDropoutArrayLayer(T, D, p)


# the network


def test_the_networks_layers_draw_their_masks_in_forward_order_from_its_one_generator():
    built = network(GPT)
    built.rng = NUMPY.default_rng(9)
    data = rows(GPT, 3)
    built.learn_batch(0.5, data)
    # the embedding's dropout, the attention's weights, the attention block's dropout, the FFN's
    masks = [
        layer._mask_batch if isinstance(layer, TokenDropoutArrayLayer) else layer._M  # pyright: ignore[reportPrivateUsage]
        for layer in built.layers
        if isinstance(layer, TokenDropoutArrayLayer | AttentionArrayLayer)
    ]
    rng = np.random.default_rng(9)
    tokens = (3, 5 * 6)
    expected = [
        rng.random(shape) >= p for shape, p in [(tokens, 0.1), ((3, 2, 5, 5), 0.1), (tokens, 0.1), (tokens, 0.1)]
    ]
    assert bits(masks) == bits([mask.astype(np.float64) for mask in expected])
    assert _state(built.rng) == _state(rng)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("name", DROPOUT)
def test_every_gradient_matches_its_finite_difference_with_the_masks_held(name: str, batch_size: int):
    specs = DROPOUT[name]
    built = network(specs)
    built.learn_batch(0.5, rows(specs, 4, seed=2))
    states, labels = zip(*rows(specs, batch_size))
    check_gradients(built, states, labels, rng=lambda: NUMPY.default_rng(11))


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), Adam()], ids=["sgd", "momentum", "adam"])
@pytest.mark.parametrize("name", DROPOUT)
def test_learn_and_a_batch_of_one_agree(name: str, rule: UpdateRule):
    # a single example's masks are a batch of one's, from the same generator
    specs = DROPOUT[name]
    assert_learn_and_a_batch_of_one_agree("numpy", network(specs, rule), network(specs, rule), rows(specs, 4))


def _without_dropout(specs: list[LayerSpec]) -> list[LayerSpec]:
    # the same layers, every dropout 0: the same snapshot layout
    def plain(spec: LayerSpec) -> LayerSpec:
        if isinstance(spec, Residual):
            return Residual(tuple(plain(layer) for layer in spec.body))
        if isinstance(spec, Attention):
            return Attention(spec.heads, spec.key_size, spec.causal)
        return Dropout(0.0) if isinstance(spec, Dropout) else spec

    return [plain(spec) for spec in specs]


def test_inference_draws_nothing_and_is_the_network_without_dropout_by_bits():
    built, plain = network(GPT), network(_without_dropout(GPT))
    built.learn_batch(0.5, rows(GPT, 4, seed=2))
    plain.restore(built.snapshot())
    before = _state(built.rng)
    prepared = built.prepare_dataset(rows(GPT, 5))
    assert bits(list(built.forward_rows(prepared))) == bits(list(plain.forward_rows(prepared)))
    assert _state(built.rng) == before


@pytest.mark.parametrize("name", DROPOUT)
def test_a_saved_dropout_network_loads_and_trains_on_by_bits(name: str, tmp_path: Path):
    # the generator's state is saved, so the masks resume by bits (D7)
    specs = DROPOUT[name]
    built = network(specs, Adam())
    data = rows(specs, 6)
    built.learn_batch(0.1, data[:3])
    path = str(tmp_path / "dropout.json")
    built.save(path)
    loaded = load_network(path)
    assert loaded.layer_specs == specs
    built.learn_batch(0.1, data[3:])
    loaded.learn_batch(0.1, data[3:])
    assert bits(loaded.snapshot()) == bits(built.snapshot())


def test_a_causal_model_with_gpts_three_dropouts_learns_a_text():
    # test_sequence_array_network's text and model, every dropout 0.1
    text = "".join(random.Random(4).choice(["abc", "abd", "bad", "cab"]) for _ in range(1200))
    vocabulary = Vocabulary.of(text)
    examples = windows(vocabulary.encode(text), 8)
    size = len(vocabulary)
    specs: list[LayerSpec] = [
        Embedding(size, 16),
        Position(),
        Dropout(0.1),
        Residual((LayerNorm(), Attention(heads=2, causal=True, dropout=0.1), Dropout(0.1))),
        Residual((LayerNorm(), Dense(32, activation="relu"), Dense(16, activation="linear", bias=True), Dropout(0.1))),
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
