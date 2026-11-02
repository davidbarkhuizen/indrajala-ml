"""
Grouped-query attention on Rust (the grouped-query attention workplan, stage 4; D4): randomize's
draws numpy's by bits, Wk and Wv at g * d_k; parity with numpy, after 50 steps under the rules
whose step is linear in the gradient and per step under Adam, on test_grouped_attention_array_network's
grouped patch and sequence models; the decoder's cache at g * d_k, a decode within the window the
full pass by bits and past it numpy's decode; and save and load. The crate's ops against their
tiled composition are the crate's own tests (tests/test_attention_grouped.py there).
"""

from pathlib import Path
from typing import Any

import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.layers.numpy import attention_array_layer, token_array_layer
from indrajala_ml.model.layers.rust.attention_rust_array_layer import AttentionRustArrayLayer
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.update_rules import Adam, UpdateRule
from tests.generation import test_rust_decoding as decoding
from tests.helpers import bits, exp_by_crate, patching, to_numpy
from tests.model.networks import test_sequence_rust_network as sequences
from tests.model.networks.test_attention_rust_network import (
    LINEAR_RULES,
    _assert_every_adam_step_has_numpys_gradients,  # pyright: ignore[reportPrivateUsage]
    _assert_training_matches_numpy,  # pyright: ignore[reportPrivateUsage]
    network,
)
from tests.model.networks.test_grouped_attention_array_network import GROUPED_PATCH, GROUPED_SEQUENCE
from tests.model.networks.test_sequence_array_network import rows

# numpy's softmax exps, attention's and the token-wise output's, as the crate's
attention_exp = patching(attention_array_layer, "exp", exp_by_crate)
token_exp = patching(token_array_layer, "exp", exp_by_crate)

# a few of the sequence models, to keep the parity runs short: the plain causal transformer, the
# SwiGLU one and the leak arm (unmasked)
SEQUENCE_PARITY = {
    name: GROUPED_SEQUENCE[name]
    for name in (
        "a causal transformer, multi-query",
        "a causal transformer, SwiGLU, multi-query",
        "the leak arm, unmasked, multi-query",
    )
}


@pytest.mark.parametrize("name", GROUPED_PATCH)
def test_randomize_draws_numpys_grouped_parameters_by_bits(name: str):
    rust, numpy = network(GROUPED_PATCH[name]), network(GROUPED_PATCH[name], backend=NUMPY)
    assert bits([to_numpy(a) for entry in rust.snapshot() for a in entry]) == bits(
        [a for entry in numpy.snapshot() for a in entry]
    )
    attention = next(layer for layer in rust.layers if isinstance(layer, AttentionRustArrayLayer))
    assert attention.Wk.shape == (attention.kv_width, attention.features)


@pytest.mark.usefixtures("attention_exp")
@pytest.mark.parametrize("rule", LINEAR_RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", GROUPED_PATCH)
def test_grouped_training_matches_numpy_within_the_dense_layers_rounding(name: str, rule: UpdateRule):
    _assert_training_matches_numpy(GROUPED_PATCH[name], rule, 0.1)


@pytest.mark.usefixtures("attention_exp")
@pytest.mark.parametrize("name", GROUPED_PATCH)
def test_every_grouped_step_under_adam_has_numpys_gradients(name: str):
    _assert_every_adam_step_has_numpys_gradients(GROUPED_PATCH[name], 0.1)


@pytest.mark.usefixtures("attention_exp", "token_exp")
@pytest.mark.parametrize("rule", LINEAR_RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", SEQUENCE_PARITY)
def test_grouped_sequence_training_matches_numpy_within_the_dense_layers_rounding(name: str, rule: UpdateRule):
    _assert_training_matches_numpy(SEQUENCE_PARITY[name], rule, 0.1, sequences.network, rows)


@pytest.mark.usefixtures("attention_exp", "token_exp")
@pytest.mark.parametrize("name", SEQUENCE_PARITY)
def test_every_grouped_sequence_step_under_adam_has_numpys_gradients(name: str):
    _assert_every_adam_step_has_numpys_gradients(SEQUENCE_PARITY[name], 0.1, sequences.network, rows)


# decoding


def _grouped_specs(positions: str, key_value_heads: int) -> Any:
    specs = decoding._specs(positions)  # pyright: ignore[reportPrivateUsage]
    # the decode tests' model at 4 heads of 4 sharing key_value_heads
    from dataclasses import replace

    from indrajala_ml.model.specs.layer_specs import Attention, Residual

    def group(spec: Any) -> Any:
        if isinstance(spec, Residual):
            return replace(spec, body=tuple(group(layer) for layer in spec.body))
        if isinstance(spec, Attention):
            return replace(spec, heads=4, key_value_heads=key_value_heads)
        return spec

    return [group(spec) for spec in specs]


@pytest.mark.parametrize("key_value_heads", [1, 2])
def test_the_decoders_cache_holds_the_key_value_heads(key_value_heads: int):
    network = decoding._network(_grouped_specs("rotary", key_value_heads))  # pyright: ignore[reportPrivateUsage]
    attention = next(layer for layer in network.layers if isinstance(layer, AttentionRustArrayLayer))
    assert attention.start_decoding().width == key_value_heads * attention.key_size


@pytest.mark.parametrize("positions", ["rotary", "none"])
@pytest.mark.parametrize("key_value_heads", [1, 2])
@pytest.mark.parametrize("prompt", [1, 3, decoding.WINDOW])
def test_grouped_steps_within_the_window_are_the_full_pass_by_bits(positions: str, key_value_heads: int, prompt: int):
    network = decoding._network(_grouped_specs(positions, key_value_heads))  # pyright: ignore[reportPrivateUsage]
    ids = decoding._ids(decoding.WINDOW)  # pyright: ignore[reportPrivateUsage]
    decoded = decoding._decoded(network, ids, prompt)  # pyright: ignore[reportPrivateUsage]
    assert decoded[1:] == decoding._full(network, ids)[prompt:]  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("positions", ["rotary", "none"])
@pytest.mark.parametrize("key_value_heads", [1, 2])
def test_past_the_window_a_grouped_decode_slides_as_numpys(positions: str, key_value_heads: int):
    import numpy as np

    specs = _grouped_specs(positions, key_value_heads)
    rust = decoding._network(specs)  # pyright: ignore[reportPrivateUsage]
    twin = decoding._numpy_twin(specs, rust)  # pyright: ignore[reportPrivateUsage]
    ids = decoding._ids(3 * decoding.WINDOW)  # pyright: ignore[reportPrivateUsage]
    np.testing.assert_allclose(
        decoding._decoded(rust, ids, 3),  # pyright: ignore[reportPrivateUsage]
        decoding._decoded(twin, ids, 3),  # pyright: ignore[reportPrivateUsage]
        rtol=0,
        atol=decoding.ATOL,
    )


# save and load


def test_a_saved_grouped_network_loads_and_trains_on_by_bits(tmp_path: Path):
    specs = GROUPED_SEQUENCE["a causal transformer, multi-query"]
    built = sequences.network(specs, Adam())
    data = rows(specs, 6)
    built.learn_batch(0.1, data[:3])
    path = str(tmp_path / "grouped.json")
    built.save(path)
    loaded = load_network(path)
    attention = next(layer for layer in loaded.layers if isinstance(layer, AttentionRustArrayLayer))
    assert (attention.heads, attention.key_value_heads) == (2, 1)
    built.learn_batch(0.1, data[3:])
    loaded.learn_batch(0.1, data[3:])
    assert bits([to_numpy(a) for entry in loaded.snapshot() for a in entry]) == bits(
        [to_numpy(a) for entry in built.snapshot() for a in entry]
    )
