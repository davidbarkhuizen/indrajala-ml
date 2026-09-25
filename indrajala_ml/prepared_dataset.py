from __future__ import annotations

from collections.abc import Sequence

import indrajala_math_rust as pa
import numpy as np

from indrajala_ml.mnist_data import RECORD_SIZE, _read_binary_records, load_mnist_dataset_as_array, load_mnist_labels

BACKENDS = ("numpy", "rust")

# classify_rows forwards this many rows at once: the accuracy-pass timing
# (docs/optimizations/rejected.md) measured 32 fastest in Rust (512 was slower) and within 3% of
# 512's saving in numpy
CLASSIFY_CHUNK_ROWS = 32


class PreparedDataset:
    """
    A training or evaluation set held as one backend matrix (one row per example) plus its
    labels, so the array networks can read rows and batches by index instead of converting a
    tuple on every call (docs/optimizations/implemented.md). backend is "numpy" (an
    np.ndarray, float64) or "rust" (a pa.Array); a network only accepts its own backend's.

    Built once per run - by the trainers from a (state, label) tuple list, or by a caller from
    an array loader (prepared_mnist) - and never modified: the numpy networks read rows as views.
    """

    def __init__(self, states, labels: Sequence, backend: str) -> None:
        assert backend in BACKENDS, f"backend must be one of {BACKENDS}; got {backend!r}"
        assert len(labels) >= 1, "a prepared dataset must not be empty"
        assert states.shape[0] == len(labels), f"{states.shape[0]} rows but {len(labels)} labels"
        self.states = states
        self.labels = list(labels)
        self.backend = backend

    def __len__(self) -> int:
        return len(self.labels)

    @classmethod
    def from_rows(cls, rows: Sequence[tuple[tuple[float, ...], object]], backend: str) -> PreparedDataset:
        assert len(rows) >= 1, "a prepared dataset must not be empty"
        states = [state for state, _label in rows]
        matrix = np.array(states, dtype=np.float64) if backend == "numpy" else pa.Array.from_rows(states)
        return cls(matrix, [label for _state, label in rows], backend)


def prepared_mnist(path: str, backend: str, limit: int | None = None) -> PreparedDataset:
    """
    An MNIST binary file (mnist_data's format) straight into a PreparedDataset, without the
    per-pixel Python floats load_mnist_dataset builds: the same values, in milliseconds.
    """

    if backend == "numpy":
        states = load_mnist_dataset_as_array(path, limit)
    else:
        states = pa.decode_mnist_pixels(_read_binary_records(path, limit), RECORD_SIZE)
    labels = load_mnist_labels(path)[: states.shape[0]]
    return PreparedDataset(states, labels, backend)
