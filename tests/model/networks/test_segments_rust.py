"""
Segments on Rust (the segments workplan, stage 6): the crate's attention masked by the segment ids,
positions per segment and the counted loss through embedding_forward gathers, the pad's row; parity
with numpy, as every step's.
"""

import random
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.training.sequence_evaluate import sequence_evaluate
from tests.model.networks.test_segments_numpy import PAD, SEP, SYMBOLS, T, _specs


def _pair(positions: str) -> tuple[Any, Any]:
    numpy_network = SequentialArrayNetwork((T,), _specs(positions), Adam(), "sequence", NUMPY)
    numpy_network.rng = NUMPY.default_rng(0)
    numpy_network.randomize()
    rust_network = SequentialArrayNetwork((T,), _specs(positions), Adam(), "sequence", RUST)
    rust_network.restore(numpy_network.snapshot())
    return numpy_network, rust_network


def _rows(seed: int) -> list[tuple[tuple[float, ...], tuple[int, ...]]]:
    rng = random.Random(seed)
    text = [i for _ in range(30) for i in (*[rng.randrange(SYMBOLS) for _ in range(rng.randrange(2, 6))], SEP)]
    rows = [(tuple(float(i) for i in text[s : s + T]), tuple(text[s + 1 : s + T + 1])) for s in range(0, 90, T + 1)]
    # a left-padded row too
    padded = (float(PAD), float(PAD), float(PAD), 1.0, 2.0, 3.0, float(SEP), 4.0)
    return [*rows, (padded, (0, 0, 1, 2, 3, SEP, 4, 0))]


def _flat(network: Any) -> list[float]:
    return [float(v) for entry in network.snapshot() for array in entry for v in np.ravel(array.tolist())]


@pytest.mark.parametrize("positions", ["learned", "rotary"])
def test_rust_trains_on_segments_as_numpy(positions: str):
    numpy_network, rust_network = _pair(positions)
    rows = _rows(1)
    for _ in range(3):
        numpy_network.learn_batch(0.01, rows)
        rust_network.learn_batch(0.01, rows)
    numpy_network.learn(0.01, *rows[0])
    rust_network.learn(0.01, *rows[0])
    np.testing.assert_allclose(_flat(rust_network), _flat(numpy_network), rtol=1e-9, atol=1e-12)
    assert sequence_evaluate(rust_network, rows).cross_entropy == pytest.approx(
        sequence_evaluate(numpy_network, rows).cross_entropy, rel=1e-9
    )


def test_the_pads_row_stays_zero_on_rust():
    _numpy_network, rust_network = _pair("rotary")
    for _ in range(3):
        rust_network.learn_batch(0.05, _rows(2))
    assert rust_network.layers[0].E.tolist()[PAD] == [0.0] * 4
