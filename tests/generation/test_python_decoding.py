"""
Pure Python's incremental decoding (the generation workplan, D2, D3, D7, stage 6): its products are
left folds a row at a time, so within the window a decode is the network's own full pass by bits;
past the window it is numpy's decode (checked against a sliding-window reference there), from the
same seed, within the products' rounding.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.networks.python.sequential_backprop_network import SequentialSequenceBackpropNetwork
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
from indrajala_ml.pcg64 import default_rng

VOCABULARY = 7
FEATURES = 8
WINDOW = 5
# pure Python's left folds against numpy's BLAS products
ATOL = 1e-13


def _specs(positions: str, causal: bool = True, dropout: float = 0.0) -> list[LayerSpec]:
    attention = Attention(
        heads=2, causal=causal, dropout=dropout, positions="rotary" if positions == "rotary" else "none"
    )
    body = (LayerNorm(), Dense(16, activation="relu"), Dense(FEATURES, activation="linear", bias=True))
    ffn = Residual((*body, Dropout(dropout)) if dropout else body)
    return [
        Embedding(VOCABULARY, FEATURES),
        *([Position()] if positions == "learned" else []),
        *[Residual((LayerNorm(), attention)), ffn] * 2,
        LayerNorm(),
        Dense(VOCABULARY, output=True, activation="softmax", loss="cross_entropy"),
    ]


def _network(specs: list[LayerSpec], seed: int = 1) -> Any:
    network = SequentialSequenceBackpropNetwork((WINDOW,), specs, Adam())
    network.rng = default_rng(seed)
    network.randomize()
    return network


def _numpy_twin(specs: list[LayerSpec], seed: int = 1) -> Any:
    # pure Python draws numpy's stream in numpy's order: the same weights by bits from one seed
    network = SequentialArrayNetwork((WINDOW,), specs, Adam(), "sequence", NUMPY)
    network.rng = np.random.default_rng(seed)
    network.randomize()
    return network


def _ids(n: int, seed: int = 2) -> list[int]:
    return [int(i) for i in np.random.default_rng(seed).integers(0, VOCABULARY, n)]


def _full(network: Any, ids: list[int]) -> list[list[float]]:
    return network.predict_probabilities(tuple(float(i) for i in ids))


def _decoded(network: Any, ids: list[int], prompt: int = 1) -> list[list[float]]:
    decoder = network.decoder()
    return [decoder.prefill(ids[:prompt]), *(decoder.step(token) for token in ids[prompt:])]


@pytest.mark.parametrize("positions", ["rotary", "learned", "none"])
def test_each_prompt_length_prefills_the_full_pass_by_bits(positions: str) -> None:
    network = _network(_specs(positions))
    ids = _ids(WINDOW)
    full = _full(network, ids)
    decoder = network.decoder()
    for length in range(1, WINDOW + 1):
        assert decoder.prefill(ids[:length]) == full[length - 1], length


@pytest.mark.parametrize("positions", ["rotary", "learned", "none"])
@pytest.mark.parametrize("prompt", [1, 3])
def test_steps_within_the_window_are_the_full_pass_by_bits(positions: str, prompt: int) -> None:
    network = _network(_specs(positions))
    ids = _ids(WINDOW)
    assert _decoded(network, ids, prompt)[1:] == _full(network, ids)[prompt:]


@pytest.mark.parametrize("positions", ["rotary", "none"])
def test_past_the_window_it_slides_as_numpys_decode(positions: str) -> None:
    specs = _specs(positions)
    ids = _ids(3 * WINDOW + 1)
    python, twin = _network(specs), _numpy_twin(specs)
    np.testing.assert_allclose(_decoded(python, ids, 2), _decoded(twin, ids, 2), rtol=0, atol=ATOL)


def test_a_learned_position_past_the_window_is_a_full_pass_over_the_last_window_said_once() -> None:
    network = _network(_specs("learned"))
    ids = _ids(2 * WINDOW)
    decoder = network.decoder()
    decoder.prefill(ids[:WINDOW])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for t in range(WINDOW, len(ids)):
            assert decoder.step(ids[t]) == _full(network, ids[t - WINDOW + 1 : t + 1])[-1]
    assert len(caught) == 1


def test_dropout_decodes_as_in_inference() -> None:
    network = _network(_specs("rotary", dropout=0.3))
    ids = _ids(WINDOW)
    assert _decoded(network, ids, 2)[1:] == _full(network, ids)[2:]


def test_decoding_past_the_window_leaves_the_full_pass_unchanged() -> None:
    network = _network(_specs("rotary"))
    ids = _ids(WINDOW)
    before = _full(network, ids)
    _decoded(network, _ids(4 * WINDOW, seed=5))
    assert _full(network, ids) == before


def test_a_non_causal_attention_is_refused_at_decoder() -> None:
    network = _network(_specs("rotary", causal=False))
    with pytest.raises(ValueError, match="non-causal"):
        network.decoder()
