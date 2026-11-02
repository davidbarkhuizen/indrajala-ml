"""
Grouped-query attention in numpy (the grouped-query attention workplan, stage 2; D2, D3, D9): the
gradient check on the shared tables' models with their heads grouped, a grouped layer against the
ungrouped layer whose key/value projections repeat each group's, the backward's fold in head order,
the dropout's draws (per query head, as before), the decoder's cache at g * d_k, decoding against
the full pass and past the window, and save and load. g = heads is today's attention, which the
golden run and every earlier test hold by bits.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.forward_context import ForwardContext
from indrajala_ml.model.layers.numpy.attention_array_layer import AttentionArrayLayer, fold
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.layer_specs import Attention, Embedding, LayerNorm, LayerSpec, Residual
from indrajala_ml.model.specs.update_rules import Adam
from tests.generation.test_numpy_decoding import (
    ATOL,
    WINDOW,
    _decoded,
    _full,
    _ids,
    _network,
    _sliding_reference,
    _specs,
)
from tests.gradient_check import check_gradients
from tests.helpers import bits, split
from tests.model.networks import test_attention_array_network as patches
from tests.model.networks import test_sequence_array_network as sequences
from tests.model.networks.test_attention_network import MULTI_HEAD
from tests.model.specs.test_layer_specs import SEQUENCE, TOKEN_OUTPUT


def grouped(specs: list[LayerSpec], key_value_heads: int) -> list[LayerSpec]:
    """specs with every Attention's heads sharing key_value_heads key/value heads, None where that's its heads."""

    def group(spec: LayerSpec) -> LayerSpec:
        if isinstance(spec, Residual):
            return replace(spec, body=tuple(group(layer) for layer in spec.body))
        if isinstance(spec, Attention):
            return replace(spec, key_value_heads=None if key_value_heads == spec.heads else key_value_heads)
        return spec

    return [group(spec) for spec in specs]


def _has_attention(specs: list[LayerSpec]) -> bool:
    return any(isinstance(layer, Attention) for spec in specs for layer in getattr(spec, "body", (spec,)))


# the multi-head patch models, multi-query and (four heads) two groups; the sequence models with
# attention, multi-query (their two heads sharing one)
GROUPED_PATCH = {
    **{f"{name}, multi-query": grouped(specs, 1) for name, specs in MULTI_HEAD.items()},
    "four heads of 2 sharing two": grouped(MULTI_HEAD["four heads of 2, wider than d"], 2),
}
GROUPED_SEQUENCE = {
    f"{name}, multi-query": grouped(specs, 1) for name, specs in SEQUENCE.items() if _has_attention(specs)
}


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("name", GROUPED_PATCH)
def test_every_grouped_patch_models_gradient_matches_its_finite_difference(name: str, batch_size: int):
    built = patches.network(GROUPED_PATCH[name])
    built.learn_batch(0.5, patches.rows(4, seed=2))
    check_gradients(built, *split(patches.rows(batch_size)))


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("name", GROUPED_SEQUENCE)
def test_every_grouped_sequence_models_gradient_matches_its_finite_difference(name: str, batch_size: int):
    specs = GROUPED_SEQUENCE[name]
    built = sequences.network(specs)
    built.learn_batch(0.5, sequences.rows(specs, 4, seed=2))
    check_gradients(built, *split(sequences.rows(specs, batch_size)))


@pytest.mark.parametrize("key_value_heads", [1, 2])
def test_a_grouped_rotary_sequence_models_gradient_matches_its_finite_difference(key_value_heads: int):
    # four causal heads of 2 over 6 features, rotary over the grouped keys
    attention = Attention(heads=4, key_size=2, causal=True, positions="rotary", key_value_heads=key_value_heads)
    specs: list[LayerSpec] = [Embedding(7, 6), Residual((LayerNorm(), attention)), TOKEN_OUTPUT]
    built = sequences.network(specs)
    built.learn_batch(0.5, sequences.rows(specs, 4, seed=2))
    check_gradients(built, *split(sequences.rows(specs, 3)))


# the layer against the ungrouped layer

T, D, HEADS, KEY = 6, 8, 4, 2


def _layers(key_value_heads: int, causal: bool = True) -> tuple[AttentionArrayLayer, AttentionArrayLayer]:
    # a grouped layer, randomized, and the ungrouped layer with the same parameters, Wk's, bk's, Wv's
    # and bv's blocks each repeated for its group's heads
    rng = np.random.default_rng(4)
    kwargs: dict[str, Any] = {"causal": causal, "positions": "rotary"}
    grouped_layer = AttentionArrayLayer(T, D, HEADS, KEY, key_value_heads=key_value_heads, **kwargs)
    grouped_layer.set_parameters([rng.normal(size=p.shape) for p in grouped_layer.parameters()])
    full = AttentionArrayLayer(T, D, HEADS, KEY, **kwargs)
    repeats = HEADS // key_value_heads

    def repeated(p: np.ndarray) -> np.ndarray:
        blocks = p.reshape(key_value_heads, KEY, *p.shape[1:])
        return np.repeat(blocks, repeats, axis=0).reshape(HEADS * KEY, *p.shape[1:])

    Wq, bq, Wk, bk, Wv, bv, Wo, bo = grouped_layer.parameters()
    full.set_parameters([Wq, bq, repeated(Wk), repeated(bk), repeated(Wv), repeated(bv), Wo, bo])
    return grouped_layer, full


def _group_sums(per_head: np.ndarray, key_value_heads: int) -> np.ndarray:
    # an ungrouped layer's (h * d_k, ...) gradient summed over each group's head blocks
    blocks = per_head.reshape(key_value_heads, HEADS // key_value_heads, KEY, *per_head.shape[1:])
    return blocks.sum(axis=1).reshape(key_value_heads * KEY, *per_head.shape[1:])


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("key_value_heads", [1, 2])
def test_a_grouped_layer_is_the_ungrouped_layer_with_each_groups_projections_repeated(
    key_value_heads: int, causal: bool
):
    grouped_layer, full = _layers(key_value_heads, causal)
    X = np.random.default_rng(5).normal(size=(3, T * D))
    downstream = np.random.default_rng(6).normal(size=(3, T * D))
    outputs: list[np.ndarray] = []
    for layer in (grouped_layer, full):
        layer.reset_gradient_accum()
        outputs.append(layer.forward_batch(X))
        layer._backward(downstream)  # pyright: ignore[reportPrivateUsage]
        layer.accumulate_gradient_batch(X)
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=0, atol=1e-13)
    np.testing.assert_allclose(grouped_layer.downstream_batch(), full.downstream_batch(), rtol=0, atol=1e-12)
    g, f = grouped_layer.gradients(), full.gradients()
    for i in (0, 1, 6, 7):  # Wq, bq, Wo, bo: the query heads' and the output's, unshared
        np.testing.assert_allclose(g[i], f[i], rtol=0, atol=1e-12)
    for i in (2, 3, 4, 5):  # Wk, bk, Wv, bv: each group's the sum of its heads'
        np.testing.assert_allclose(g[i], _group_sums(f[i], key_value_heads), rtol=0, atol=1e-12)


def test_the_grouped_parameters_are_wk_and_wv_at_g_key_value_heads():
    layer = AttentionArrayLayer(T, D, HEADS, KEY, key_value_heads=2)
    assert (layer.width, layer.kv_width) == (8, 4)
    assert [p.shape for p in layer.parameters()] == [(8, D), (8,), (4, D), (4,), (4, D), (4,), (D, 8), (D,)]


def test_the_backward_folds_each_group_left_to_right_from_its_first_head():
    # D3: ((a + b) + c) + d, not pairwise and not from zero, so the crate and pure Python add the same
    a, b, c, d = 1e16, 1.0, -1e16, 1.0
    per_head = np.array([a, b, c, d]).reshape(1, 4, 1, 1)
    assert fold(per_head, 1).item() == ((a + b) + c) + d == 1.0
    assert (a + b) + (c + d) == 0.0  # pairwise differs
    # two groups of two: ((a + b), (c + d))
    assert fold(per_head, 2).ravel().tolist() == [a + b, c + d]
    # from the first head, not from zero: a group of -0.0s folds to -0.0
    zeros = np.full((1, 4, 1, 1), -0.0)
    assert bits(fold(zeros, 1)) == bits(zeros[:, :1])


@pytest.mark.parametrize("key_value_heads", [1, 2])
def test_the_dropout_draws_per_query_head_as_an_ungrouped_layer(key_value_heads: int):
    grouped_layer = AttentionArrayLayer(T, D, HEADS, KEY, causal=True, dropout=0.2, key_value_heads=key_value_heads)
    full = AttentionArrayLayer(T, D, HEADS, KEY, causal=True, dropout=0.2)
    X = np.random.default_rng(5).normal(size=(3, T * D))
    for layer in (grouped_layer, full):
        layer.set_rng(np.random.default_rng(7))
        layer.set_training_mode(True)
        layer.forward_batch(X)
    assert grouped_layer.rng.bit_generator.state == full.rng.bit_generator.state
    mask = grouped_layer.dropout_mask()
    assert mask is not None and mask.shape == (3, HEADS, T, T)
    assert bits(mask) == bits(full.dropout_mask())


# decoding


@pytest.mark.parametrize("key_value_heads", [1, 2])
def test_the_decoders_cache_holds_the_key_value_heads(key_value_heads: int):
    layer = AttentionArrayLayer(T, D, HEADS, KEY, causal=True, key_value_heads=key_value_heads)
    cache = layer.start_decoding()
    assert cache.width == key_value_heads * KEY
    X = np.random.default_rng(5).normal(size=(2 * 3, D))
    layer.prefill(X, cache, ForwardContext(examples=2))
    assert cache.K.shape == cache.V.shape == (2, 3, key_value_heads * KEY)


@pytest.mark.parametrize("positions", ["rotary", "none"])
@pytest.mark.parametrize("key_value_heads", [1, 2])
@pytest.mark.parametrize("prompt", [1, 3, WINDOW])
def test_grouped_steps_within_the_window_are_the_full_pass(positions: str, key_value_heads: int, prompt: int):
    network = _network(_specs(positions, heads=4, key_value_heads=key_value_heads))
    ids = _ids(WINDOW)
    np.testing.assert_allclose(_decoded(network, ids, prompt)[1:], _full(network, ids)[prompt:], rtol=0, atol=ATOL)


@pytest.mark.parametrize("positions", ["rotary", "none"])
@pytest.mark.parametrize("key_value_heads", [1, 2])
def test_past_the_window_a_grouped_window_slides(positions: str, key_value_heads: int):
    specs = _specs(positions, heads=4, key_value_heads=key_value_heads)
    network = _network(specs)
    ids = _ids(3 * WINDOW)
    reference = _sliding_reference(specs, network, ids)
    np.testing.assert_allclose(_decoded(network, ids, 3), reference[2:], rtol=0, atol=ATOL)


# save and load


def test_a_saved_grouped_network_loads_and_trains_on_by_bits(tmp_path: Path):
    specs = GROUPED_SEQUENCE["a causal transformer, multi-query"]
    built = sequences.network(specs, Adam())
    data = sequences.rows(specs, 6)
    built.learn_batch(0.1, data[:3])
    path = str(tmp_path / "grouped.json")
    built.save(path)
    loaded = load_network(path)
    attention: Any = next(layer for layer in loaded.layers if isinstance(layer, AttentionArrayLayer))
    assert (attention.heads, attention.key_value_heads) == (2, 1)
    built.learn_batch(0.1, data[3:])
    loaded.learn_batch(0.1, data[3:])
    assert bits(loaded.snapshot()) == bits(built.snapshot())
