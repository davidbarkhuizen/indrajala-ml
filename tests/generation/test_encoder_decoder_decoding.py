"""
Decoding an encoder-decoder (the cross-attention workplan, stage 8b, D7, D15, D16): each
implementation's decoder(sources) encodes the sources once, and each CrossAttention keeps the
memory's keys and values from prefill on. A decode from one separator, then a target token a step,
is the network's full pass over the source and that target, token by token: by bits on Rust and in
pure Python (every sum a fixed chain whatever the rows around it), within rounding on numpy (BLAS).
The target's segments are D13's, a separator opening its pair. Sources of several lengths decode
in one batch as each alone. translate stops a row at its next separator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from indrajala_ml.data.text_data import pad_id, separator_id
from indrajala_ml.generation.sampling import Sampler
from indrajala_ml.generation.translate import translate
from indrajala_ml.model.networks.numpy.encoder_decoder_array_network import EncoderDecoderArrayNetwork
from indrajala_ml.model.networks.python.encoder_decoder_backprop_network import EncoderDecoderBackpropNetwork
from indrajala_ml.model.networks.rust.encoder_decoder_rust_array_network import EncoderDecoderRustArrayNetwork
from indrajala_ml.model.specs.update_rules import SGD
from tests.helpers import bits
from tests.model.networks import test_encoder_decoder_numpy as numpy_tests

CLASSES = {
    "python": EncoderDecoderBackpropNetwork,
    "numpy": EncoderDecoderArrayNetwork,
    "rust": EncoderDecoderRustArrayNetwork,
}
WINDOW = numpy_tests.WINDOW
SEPARATOR, PAD = separator_id(numpy_tests.TARGET), pad_id(numpy_tests.TARGET)
SOURCE_SEPARATOR, SOURCE_PAD = separator_id(numpy_tests.SOURCE), pad_id(numpy_tests.SOURCE)
# a source pair (its Greek and its separator) and a target filling the window from its separator
SOURCES = [[0, 1, 2, SOURCE_SEPARATOR], [3, 3, 0, 1, 2, 1, SOURCE_SEPARATOR]]
TARGETS = [[SEPARATOR, 0, 1, 2, 3, 0, 1, 2, 3, 0], [SEPARATOR, 3, 2, 1, 0, 3, 2, 1, 0, 3]]


def network(implementation: str, seed: int = 4) -> Any:
    return CLASSES[implementation].randomized(numpy_tests.spec(), WINDOW, WINDOW, SGD(), seed=seed)


def full(net: Any, source: list[int], target: list[int]) -> list[list[float]]:
    # the full pass's probabilities at each target token
    state = tuple(float(i) for i in [SOURCE_PAD] * (WINDOW - len(source)) + source + target)
    return [list(map(float, row)) for row in net.predict_probabilities(state)]


def decoded(net: Any, sources: list[list[int]], targets: list[list[int]]) -> list[list[list[float]]]:
    # each row's probabilities after its separator, then after each later target token
    decoder = net.decoder(sources)
    steps = [decoder.prefill_batch([[t[0]] for t in targets])]
    for column in range(1, WINDOW):
        steps.append(decoder.step_batch([t[column] for t in targets]))
    return [[list(map(float, steps[c][r])) for c in range(WINDOW)] for r in range(len(targets))]


@pytest.mark.parametrize("implementation", ["python", "rust"])
def test_a_decode_is_the_full_pass_token_by_token_by_bits(implementation: str):
    net = network(implementation)
    for source, target in zip(SOURCES, TARGETS, strict=True):
        assert bits(decoded(net, [source], [target])[0]) == bits(full(net, source, target))


def test_a_numpy_decode_is_the_full_pass_within_rounding():
    net = network("numpy")
    for source, target in zip(SOURCES, TARGETS, strict=True):
        np.testing.assert_allclose(decoded(net, [source], [target])[0], full(net, source, target), rtol=0, atol=1e-12)


@pytest.mark.parametrize("implementation", ["python", "rust"])
def test_sources_of_several_lengths_decode_in_one_batch_as_each_alone_by_bits(implementation: str):
    net = network(implementation)
    batch = decoded(net, SOURCES, TARGETS)
    alone = [decoded(net, [source], [target])[0] for source, target in zip(SOURCES, TARGETS, strict=True)]
    assert bits(batch) == bits(alone)


def test_the_three_implementations_decode_alike_within_rounding():
    rows = {name: decoded(network(name), SOURCES, TARGETS) for name in CLASSES}
    np.testing.assert_allclose(rows["rust"], rows["numpy"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(rows["python"], rows["numpy"], rtol=0, atol=1e-12)


def test_a_decode_takes_as_many_rows_as_sources():
    decoder = network("numpy").decoder(SOURCES)
    with pytest.raises(AssertionError, match="decodes 2 rows"):
        decoder.prefill_batch([[SEPARATOR]])


class _Scripted:
    """A decoder whose rows' next tokens are scripted: probability 1 on each."""

    def __init__(self, scripts: list[list[int]], classes: int) -> None:
        self.scripts, self.classes, self.column = scripts, classes, 0

    def _rows(self) -> list[list[float]]:
        rows: list[list[float]] = []
        for script in self.scripts:
            p = [0.0] * self.classes
            p[script[min(self.column, len(script) - 1)]] = 1.0
            rows.append(p)
        return rows

    def prefill_batch(self, prompts: list[list[int]]) -> list[list[float]]:
        self.column = 0
        return self._rows()

    def step_batch(self, tokens: list[int]) -> list[list[float]]:
        self.column += 1
        return self._rows()


class _Uniforms:
    def random(self) -> float:
        return 0.5


def test_translate_stops_each_row_at_its_separator_or_the_cap():
    sep = 4
    scripted = _Scripted([[0, 1, sep, 2], [3, 3, 3, 3, 3, 3]], classes=5)
    out = translate(scripted, 2, sep, 4, Sampler(temperature=0.0), _Uniforms())
    assert out == [[0, 1], [3, 3, 3, 3]]
