"""The array backends' shared operations where Rust's and numpy's are written apart."""

import math

import indrajala_math_rust as pa
import numpy as np

from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST


def _argmax_by_max_and_index(rows: list[list[float]], classes: int) -> list[tuple[int, ...]]:
    """RUST.argmax_token_rows as first written (before the window-program workplan's stage 1)."""
    return [
        tuple(token.index(max(token)) for token in (row[i : i + classes] for i in range(0, len(row), classes)))
        for row in rows
    ]


def test_token_rows_argmax_is_max_and_index_ties_to_the_first():
    rng = np.random.default_rng(0)
    for classes in (2, 3, 8, 12):
        rows = rng.integers(-2, 3, size=(7, 5 * classes)).astype(float).tolist()  # many ties
        expected = _argmax_by_max_and_index(rows, classes)
        assert RUST.argmax_token_rows(pa.Array(rows), classes) == expected
        assert NUMPY.argmax_token_rows(np.array(rows), classes) == expected


def test_token_rows_argmax_with_a_nan_is_max_and_index():
    rows = [[math.nan, 1.0, 2.0, 0.5, math.nan, 0.1], [3.0, 1.0, 2.0, 0.0, 0.0, 0.0]]
    assert RUST.argmax_token_rows(pa.Array(rows), 3) == _argmax_by_max_and_index(rows, 3) == [(0, 0), (0, 0)]


def test_token_rows_argmax_of_no_rows_is_empty():
    assert RUST.argmax_token_rows(pa.Array.zeros((0, 6)), 3) == []
