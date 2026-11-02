"""
Grouped-query attention in pure Python (the grouped-query attention workplan, stage 5; D2, D3):
randomize draws numpy's weights, Wk and Wv at g * d_k; the gradient check; parity with numpy after
50 steps under the rules whose step is linear in the gradient, and per step under Adam, on
test_grouped_attention_array_network's grouped patch models; the layer-major path the
example-major loop by bits; and decoding, within the window the full pass by bits and past it
numpy's decode.
"""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.layers.python.attention_layer import AttentionLayer
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.layer_specs import Attention, LayerSpec, Residual
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule
from tests.generation import test_python_decoding as decoding
from tests.gradient_check import check_gradients
from tests.helpers import assert_snapshots_close, bits, learn_in_step, split
from tests.model.networks.test_attention_array_network import IMAGE, rows
from tests.model.networks.test_attention_python_network import (
    LINEAR_RULES,
    assert_every_step_has_numpys_gradients,
    network,
)
from tests.model.networks.test_grouped_attention_array_network import GROUPED_PATCH
from tests.python_array_snapshot import as_array_snapshot, seeded_like


def _attention(built: Any) -> AttentionLayer:
    return next(layer for layer in built.trainable_layers if isinstance(layer, AttentionLayer))


@pytest.mark.parametrize("name", GROUPED_PATCH)
def test_randomize_draws_numpys_grouped_weights(name: str):
    built = network(GROUPED_PATCH[name])
    seeded_like(built, SequentialArrayNetwork(IMAGE, GROUPED_PATCH[name], SGD(), backend=NUMPY), 3)
    attention = _attention(built)
    assert len(attention.keys) == len(attention.values) == attention.kv_width
    assert len(attention.queries) == attention.width


@pytest.mark.parametrize("name", ["two heads, multi-query", "four heads of 2 sharing two", "two layers, multi-query"])
def test_every_grouped_gradient_matches_its_finite_difference(name: str):
    built = network(GROUPED_PATCH[name])
    built.learn_batch(0.5, rows(4, seed=2))
    check_gradients(built, *split(rows(3)))


@pytest.mark.parametrize("rule", LINEAR_RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", GROUPED_PATCH)
def test_grouped_training_matches_numpy_within_the_dense_layers_rounding(name: str, rule: UpdateRule):
    python = network(GROUPED_PATCH[name], rule)
    array = seeded_like(python, SequentialArrayNetwork(IMAGE, GROUPED_PATCH[name], rule, backend=NUMPY), 3)
    learn_in_step(0.1, rows(40), (python, array))
    assert_snapshots_close(as_array_snapshot(python), array.snapshot())


@pytest.mark.parametrize("name", GROUPED_PATCH)
def test_every_grouped_step_under_adam_has_numpys_gradients(name: str):
    python = network(GROUPED_PATCH[name], Adam())
    array = SequentialArrayNetwork(IMAGE, GROUPED_PATCH[name], SGD(), backend=NUMPY)
    assert_every_step_has_numpys_gradients(python, array, rows(40))


@pytest.mark.parametrize("rule", [Momentum(0.9), Adam()], ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", ["four heads of 2 sharing two", "two layers, multi-query"])
def test_the_grouped_layer_major_path_is_the_example_major_loop_by_bits(name: str, rule: UpdateRule):
    example_major, layer_major = network(GROUPED_PATCH[name], rule), network(GROUPED_PATCH[name], rule)
    for size in (3, 5, 4):
        batch = rows(size, seed=size)
        example_major.learn_batch(0.3, batch)
        layer_major._learn_batch_layer_major(0.3, batch, None)  # pyright: ignore[reportPrivateUsage]
    assert bits(layer_major.snapshot()) == bits(example_major.snapshot())


# decoding: the decode tests' model at 4 heads of 2 sharing key_value_heads


def _grouped_specs(positions: str, key_value_heads: int, dropout: float = 0.0) -> list[LayerSpec]:
    def group(spec: LayerSpec) -> LayerSpec:
        if isinstance(spec, Residual):
            return replace(spec, body=tuple(group(layer) for layer in spec.body))
        if isinstance(spec, Attention):
            return replace(spec, heads=4, key_value_heads=key_value_heads)
        return spec

    return [group(spec) for spec in decoding._specs(positions, dropout=dropout)]  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("positions", ["rotary", "none"])
@pytest.mark.parametrize("key_value_heads", [1, 2])
@pytest.mark.parametrize("prompt", [1, decoding.WINDOW])
def test_grouped_steps_within_the_window_are_the_full_pass_by_bits(positions: str, key_value_heads: int, prompt: int):
    built = decoding._network(_grouped_specs(positions, key_value_heads))  # pyright: ignore[reportPrivateUsage]
    ids = decoding._ids(decoding.WINDOW)  # pyright: ignore[reportPrivateUsage]
    decoded = decoding._decoded(built, ids, prompt)  # pyright: ignore[reportPrivateUsage]
    assert decoded[1:] == decoding._full(built, ids)[prompt:]  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("positions", ["rotary", "none"])
@pytest.mark.parametrize("key_value_heads", [1, 2])
def test_past_the_window_a_grouped_decode_slides_as_numpys(positions: str, key_value_heads: int):
    specs = _grouped_specs(positions, key_value_heads)
    ids = decoding._ids(3 * decoding.WINDOW + 1)  # pyright: ignore[reportPrivateUsage]
    python, twin = decoding._network(specs), decoding._numpy_twin(specs)  # pyright: ignore[reportPrivateUsage]
    np.testing.assert_allclose(
        decoding._decoded(python, ids, 2),  # pyright: ignore[reportPrivateUsage]
        decoding._decoded(twin, ids, 2),  # pyright: ignore[reportPrivateUsage]
        rtol=0,
        atol=decoding.ATOL,
    )
