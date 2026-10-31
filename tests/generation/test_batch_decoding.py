"""
Batched, padded decoding (the segments workplan, D8, stage 8), in each implementation: a batch's
rows, left-padded to the longest prompt, decode as each row alone; a separator, in a prompt or
stepped, starts a paragraph that attends to itself only, with positions from its first token, as
the network's own full pass masks it (D4, D5); and a decode whose current paragraph lies in the
window is the full pass over the last window for ever, a learned Position's included. Pure Python
and the crate compute each output as one fixed chain, the masked zeros' sums exact, so they agree by
bits, but a rotary model's rows at other columns (its angles differ, and its scores depend on t - u
within rounding only); numpy's products of different shapes may take different BLAS kernels, so
within rounding.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.networks.python.sequential_backprop_network import SequentialSequenceBackpropNetwork
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    Dense,
    Embedding,
    LayerNorm,
    LayerSpec,
    Position,
    Residual,
)
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.pcg64 import default_rng

# symbols 0 to 6, the separator 7 and the pad 8
VOCABULARY = 9
SEPARATOR, PAD = VOCABULARY - 2, VOCABULARY - 1
FEATURES = 8
WINDOW = 6
ATOL = 1e-13
IMPLEMENTATIONS = ["numpy", "rust", "python"]
POSITIONS = ["rotary", "learned", "none"]


def _specs(positions: str, segments: bool = True) -> list[LayerSpec]:
    attention = Attention(heads=2, causal=True, positions="rotary" if positions == "rotary" else "none")
    body = (LayerNorm(), Dense(16, activation="relu"), Dense(FEATURES, activation="linear", bias=True))
    return [
        Embedding(VOCABULARY, FEATURES, segments=segments),
        *([Position()] if positions == "learned" else []),
        *[Residual((LayerNorm(), attention)), Residual(body)] * 2,
        LayerNorm(),
        Dense(VOCABULARY - 1 if segments else VOCABULARY, output=True, activation="softmax", loss="cross_entropy"),
    ]


def _network(implementation: str, positions: str, segments: bool = True) -> Any:
    specs = _specs(positions, segments)
    if implementation == "python":
        network: Any = SequentialSequenceBackpropNetwork((WINDOW,), specs, Adam())
        network.rng = default_rng(1)
    else:
        backend = NUMPY if implementation == "numpy" else RUST
        network = SequentialArrayNetwork((WINDOW,), specs, Adam(), "sequence", backend)
        network.rng = backend.default_rng(1)
    network.randomize()
    return network


def _assert_equal(implementation: str, got: Any, expected: Any, positions: str = "none") -> None:
    # by bits but numpy's, and a rotary model's at other columns: its angles differ, and its scores
    # depend on t - u only within rounding
    if implementation == "numpy" or positions == "rotary":
        np.testing.assert_allclose(np.array(got), np.array(expected), rtol=0, atol=ATOL)
    else:
        assert got == expected


def _symbols(n: int, seed: int = 2) -> list[int]:
    return [int(i) for i in np.random.default_rng(seed).integers(0, SEPARATOR, n)]


def _paragraphs(n: int, every: int, seed: int = 2) -> list[int]:
    # n ids, every every-th a separator
    return [SEPARATOR if (t + 1) % every == 0 else i for t, i in enumerate(_symbols(n, seed))]


def _last_window(network: Any, ids: list[int]) -> list[float]:
    # the full pass over the last window of ids, left-padded when shorter: its last token's probabilities
    window = ids[-WINDOW:]
    padded = [PAD] * (WINDOW - len(window)) + window
    return network.predict_probabilities(tuple(float(i) for i in padded))[-1]


@pytest.mark.parametrize("positions", POSITIONS)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_padded_batch_decodes_each_row_as_alone(implementation: str, positions: str) -> None:
    network = _network(implementation, positions)
    prompts = [_symbols(1, seed=3), [*_symbols(2, seed=4), SEPARATOR, *_symbols(2, seed=5)], _symbols(4, seed=6)]
    # within the window for a learned Position; past it otherwise
    steps = WINDOW - 5 if positions == "learned" else 2 * WINDOW
    tokens = [_paragraphs(steps, 4, seed=7 + n) for n in range(len(prompts))]
    batch = network.decoder()
    rows = [batch.prefill_batch(prompts)]
    rows += [batch.step_batch([row[s] for row in tokens]) for s in range(steps)]
    for n, prompt in enumerate(prompts):
        alone = network.decoder()
        expected = [alone.prefill(prompt)] + [alone.step(token) for token in tokens[n]]
        _assert_equal(implementation, [r[n] for r in rows], expected, positions)


@pytest.mark.parametrize("positions", POSITIONS)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_prompt_is_the_padded_full_pass(implementation: str, positions: str) -> None:
    # a short prompt with a separator: the full pass over it left-padded to the window, which the
    # network's own segment context masks
    network = _network(implementation, positions)
    prompt = [*_symbols(2), SEPARATOR, *_symbols(1, seed=3)]
    _assert_equal(implementation, network.decoder().prefill(prompt), _last_window(network, prompt), positions)


@pytest.mark.parametrize("positions", POSITIONS)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_paragraphs_inside_the_window_decode_as_the_last_window_for_ever(implementation: str, positions: str) -> None:
    # every paragraph shorter than the window: each step's query reads its own paragraph only, all
    # in the window, so it is the last window's full pass, past T too, and no learned Position runs
    # out of rows
    network = _network(implementation, positions)
    ids = _paragraphs(4 * WINDOW, 4)
    decoder = network.decoder()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = [decoder.prefill(ids[:2])] + [decoder.step(token) for token in ids[2:]]
    expected = [_last_window(network, ids[: t + 1]) for t in range(1, len(ids))]
    _assert_equal(implementation, got, expected, positions)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_learned_position_past_the_window_is_a_full_pass_of_every_row_said_once(implementation: str) -> None:
    network = _network(implementation, "learned")
    # row 0's paragraph outgrows the window, row 1's never does
    rows = [_symbols(2 * WINDOW + 2), _paragraphs(2 * WINDOW + 2, 3, seed=5)]
    decoder = network.decoder()
    decoder.prefill_batch([row[:2] for row in rows])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for t in range(2, len(rows[0])):
            got = decoder.step_batch([row[t] for row in rows])
            expected = [_last_window(network, row[: t + 1]) for row in rows]
            _assert_equal(implementation, got, expected)
    assert len(caught) == 1
    assert "full pass" in str(caught[0].message)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_model_without_segments_decodes_prompts_of_one_length(implementation: str) -> None:
    network = _network(implementation, "rotary", segments=False)
    prompts = [_symbols(3, seed=3), _symbols(3, seed=4)]
    batch = network.decoder()
    got = [batch.prefill_batch(prompts), batch.step_batch([1, 2])]
    for n, prompt in enumerate(prompts):
        alone = network.decoder()
        _assert_equal(implementation, [r[n] for r in got], [alone.prefill(prompt), alone.step([1, 2][n])])
    with pytest.raises(ValueError, match="one length"):
        network.decoder().prefill_batch([_symbols(3), _symbols(2)])


@pytest.mark.parametrize("positions", POSITIONS)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_prompt_longer_than_the_window_pads_a_short_row_past_it(implementation: str, positions: str) -> None:
    # the short row's first window is all pads: prefilled as pads, then stepped through pads to its prompt
    network = _network(implementation, positions)
    prompts = [_symbols(2, seed=3), _paragraphs(2 * WINDOW + 1, 4, seed=4)]
    batch = network.decoder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a learned Position's full pass past the window
        got = [batch.prefill_batch(prompts), batch.step_batch([1, 2])]
        for n, prompt in enumerate(prompts):
            alone = network.decoder()
            expected = [alone.prefill(prompt), alone.step([1, 2][n])]
            _assert_equal(implementation, [r[n] for r in got], expected, positions)
