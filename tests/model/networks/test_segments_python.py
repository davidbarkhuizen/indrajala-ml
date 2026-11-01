"""
Segments in pure Python (the segments workplan, stage 7): the same contexts as numpy's, as lists of
rows, read one example at a time; parity with numpy, both seeded alike (tests/python_array_snapshot.py),
per example and layer-major (dropout).
"""

from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.layers.numpy.segments import segment_context
from indrajala_ml.model.layers.python.segments import segment_rows
from indrajala_ml.model.networks.python.sequential_backprop_network import SequentialSequenceBackpropNetwork
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.layer_specs import Dropout, LayerSpec
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.pcg64 import default_rng
from tests.model.networks.test_segments_numpy import PAD, SEP, T, _specs
from tests.model.networks.test_segments_rust import _rows
from tests.python_array_snapshot import as_array_snapshot, seeded_like


def test_the_python_context_is_numpys_as_lists():
    states = [state for state, _ in _rows(3)]
    rows, arrays = segment_rows(states, SEP, PAD), segment_context(np.array(states), SEP, PAD)
    assert rows.segments == arrays.segments.tolist() and rows.positions == arrays.positions.tolist()
    assert rows.counted == arrays.counted.tolist() and (rows.count, rows.examples) == (arrays.count, arrays.examples)
    assert segment_rows([[0.0, 1.0]], SEP, PAD).segments is None


def _pair(specs: list[LayerSpec]) -> tuple[Any, Any]:
    python = SequentialSequenceBackpropNetwork((T,), specs, Adam())
    python.rng = default_rng(5)
    python.randomize()
    array = seeded_like(python, SequentialArrayNetwork((T,), specs, Adam(), "sequence", NUMPY), 5)
    return python, array


def _flat(snapshot: list[tuple[Any, ...]]) -> list[float]:
    return [float(v) for entry in snapshot for array in entry for v in np.ravel(np.asarray(array, dtype=np.float64))]


@pytest.mark.parametrize(
    "specs",
    [_specs("learned"), _specs("rotary"), [*_specs("learned")[:2], Dropout(0.1), *_specs("learned")[2:]]],
    ids=["learned", "rotary", "dropout (layer-major)"],
)
def test_python_trains_on_segments_as_numpy(specs: list[LayerSpec]):
    python, array = _pair(specs)
    rows = _rows(4)
    for _ in range(2):
        python.learn_batch(0.01, rows)
        array.learn_batch(0.01, rows)
    np.testing.assert_allclose(_flat(as_array_snapshot(python)), _flat(array.snapshot()), rtol=1e-9, atol=1e-12)
    state = rows[-1][0]
    python_probabilities = np.asarray(python.predict_probabilities(state), dtype=np.float64).ravel()
    array_probabilities = np.asarray(array.predict_probabilities(state), dtype=np.float64).ravel()
    np.testing.assert_allclose(python_probabilities, array_probabilities, rtol=1e-9)


def test_the_pads_row_stays_zero_in_pure_python():
    python, _array = _pair(_specs("rotary"))
    for _ in range(2):
        python.learn_batch(0.05, _rows(2))
    assert list(np.asarray(as_array_snapshot(python)[0][0])[PAD]) == [0.0] * 4
