"""
numpy's incremental decoding (the generation workplan, D2, D3, D7, stage 3): the Decoder against the
network's own full pass, within rounding (a one-row product and a whole-window product may take
different BLAS kernels), and past the window against a sliding-window reference: a longer network
with the same parameters whose attentions mask a band of T keys.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.layers.numpy.attention_array_layer import AttentionArrayLayer
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
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
from indrajala_ml.model.specs.update_rules import Adam

VOCABULARY = 11
FEATURES = 16
WINDOW = 8
# products of one row against products of a window: BLAS rounding only
ATOL = 1e-13


def _specs(positions: str, causal: bool = True, dropout: float = 0.0) -> list[LayerSpec]:
    attention = Attention(
        heads=2, causal=causal, dropout=dropout, positions="rotary" if positions == "rotary" else "none"
    )
    # a token-wise dropout ends the FFN's body
    body = (LayerNorm(), Dense(32, activation="relu"), Dense(FEATURES, activation="linear", bias=True))
    ffn = Residual((*body, Dropout(dropout)) if dropout else body)
    blocks = [Residual((LayerNorm(), attention)), ffn] * 2
    return [
        Embedding(VOCABULARY, FEATURES),
        *([Position()] if positions == "learned" else []),
        *blocks,
        LayerNorm(),
        Dense(VOCABULARY, output=True, activation="softmax", loss="cross_entropy"),
    ]


def _network(specs: list[LayerSpec], window: int = WINDOW, seed: int = 1) -> Any:
    network = SequentialArrayNetwork((window,), specs, Adam(), "sequence", NUMPY)
    network.rng = np.random.default_rng(seed)
    network.randomize()
    return network


def _ids(n: int, seed: int = 2) -> list[int]:
    return [int(i) for i in np.random.default_rng(seed).integers(0, VOCABULARY, n)]


def _full(network: Any, ids: list[int]) -> np.ndarray:
    return np.array(network.predict_probabilities(tuple(float(i) for i in ids)))


def _decoded(network: Any, ids: list[int], prompt: int = 1) -> np.ndarray:
    # each position's probabilities, the first prompt tokens prefilled, the rest stepped
    decoder = network.decoder()
    rows = [decoder.prefill(ids[:prompt])]
    rows += [decoder.step(token) for token in ids[prompt:]]
    return np.array(rows)


@pytest.mark.parametrize("positions", ["rotary", "learned", "none"])
def test_each_prompt_length_prefills_the_full_pass(positions: str) -> None:
    network = _network(_specs(positions))
    ids = _ids(WINDOW)
    full = _full(network, ids)
    decoder = network.decoder()
    for length in range(1, WINDOW + 1):
        np.testing.assert_allclose(decoder.prefill(ids[:length]), full[length - 1], rtol=0, atol=ATOL)


@pytest.mark.parametrize("positions", ["rotary", "learned", "none"])
@pytest.mark.parametrize("prompt", [1, 3, WINDOW])
def test_steps_within_the_window_are_the_full_pass(positions: str, prompt: int) -> None:
    network = _network(_specs(positions))
    ids = _ids(WINDOW)
    np.testing.assert_allclose(_decoded(network, ids, prompt)[1:], _full(network, ids)[prompt:], rtol=0, atol=ATOL)


def _sliding_reference(specs: list[LayerSpec], network: Any, ids: list[int]) -> np.ndarray:
    # the same parameters over all of ids at once, each attention masking key j for query i unless
    # i - T < j <= i: a band of T keys, the window sliding
    n = len(ids)
    long = _network(specs, window=n)
    long.restore(network.snapshot())
    band = np.triu(np.ones((n, n), dtype=np.bool_), k=1) | np.tril(np.ones((n, n), dtype=np.bool_), k=-WINDOW)
    for layer in long.layers:
        if isinstance(layer, AttentionArrayLayer):
            layer._future = band  # pyright: ignore[reportPrivateUsage]
    return _full(long, ids)


@pytest.mark.parametrize("positions", ["rotary", "none"])
@pytest.mark.parametrize("prompt", [1, WINDOW, 2 * WINDOW + 3])
def test_past_the_window_the_window_slides(positions: str, prompt: int) -> None:
    specs = _specs(positions)
    network = _network(specs)
    ids = _ids(3 * WINDOW)
    reference = _sliding_reference(specs, network, ids)
    np.testing.assert_allclose(_decoded(network, ids, prompt), reference[prompt - 1 :], rtol=0, atol=ATOL)


def test_the_sliding_reference_differs_from_a_full_pass_over_the_last_window() -> None:
    # two layers carry context from before the window: the reference isn't trivially the last window's pass
    specs = _specs("rotary")
    network = _network(specs)
    ids = _ids(3 * WINDOW)
    reference = _sliding_reference(specs, network, ids)
    assert not np.allclose(reference[-1], _full(network, ids[-WINDOW:])[-1], rtol=0, atol=1e-6)


def test_a_learned_position_past_the_window_is_a_full_pass_over_the_last_window_said_once() -> None:
    network = _network(_specs("learned"))
    ids = _ids(2 * WINDOW + 2)
    decoder = network.decoder()
    decoder.prefill(ids[:WINDOW])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for t in range(WINDOW, len(ids)):
            p = decoder.step(ids[t])
            assert p == _full(network, ids[t - WINDOW + 1 : t + 1]).tolist()[-1]
    assert len(caught) == 1
    assert "full pass" in str(caught[0].message)


def test_a_long_prompt_is_the_window_prefilled_then_stepped() -> None:
    network = _network(_specs("rotary"))
    ids = _ids(2 * WINDOW + 1)
    decoder = network.decoder()
    whole = decoder.prefill(ids)
    stepped = decoder.prefill(ids[:WINDOW])
    for token in ids[WINDOW:]:
        stepped = decoder.step(token)
    assert whole == stepped
    assert decoder.position == len(ids)


def test_a_new_prefill_starts_a_new_decode() -> None:
    network = _network(_specs("rotary"))
    ids, other = _ids(WINDOW), _ids(WINDOW, seed=9)
    decoder = network.decoder()
    first = decoder.prefill(ids[:3])
    decoder.prefill(other)
    assert decoder.prefill(ids[:3]) == first


def test_dropout_decodes_as_in_inference() -> None:
    network = _network(_specs("rotary", dropout=0.3))
    ids = _ids(WINDOW)
    np.testing.assert_allclose(_decoded(network, ids, 2)[1:], _full(network, ids)[2:], rtol=0, atol=ATOL)


def test_decoding_past_the_window_leaves_the_full_pass_unchanged() -> None:
    network = _network(_specs("rotary"))
    ids = _ids(WINDOW)
    before = _full(network, ids)
    _decoded(network, _ids(4 * WINDOW, seed=5))
    assert np.array_equal(_full(network, ids), before)


def test_a_non_causal_attention_is_refused_at_decoder() -> None:
    network = _network(_specs("rotary", causal=False))
    with pytest.raises(ValueError, match="non-causal"):
        network.decoder()
