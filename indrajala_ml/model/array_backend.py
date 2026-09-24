from __future__ import annotations

from typing import Sequence

import indrajala_math_rust as pa
import numpy as np

from indrajala_ml.model.array_layer import fan_in_aware_random_layer
from indrajala_ml.model.rust_array_layer import fan_in_aware_random_rust_layer


class NumpyBackend:
    """
    The array operations ArrayNetworkBase needs from numpy. The two backends differ only in these
    (docs/refactoring.md, item 1); everything else the network does goes through its layers,
    whose methods have the same names on both backends.
    """

    name = "numpy"
    random_layer = staticmethod(fan_in_aware_random_layer)

    @staticmethod
    def vector(state: Sequence[float]) -> np.ndarray:
        return np.array(state, dtype=np.float64)

    @staticmethod
    def matrix(states: Sequence[Sequence[float]]) -> np.ndarray:
        return np.array(states, dtype=np.float64)

    @staticmethod
    def row(states: np.ndarray, index: int) -> np.ndarray:
        # a row of a C-contiguous matrix is a view; no layer writes into its input
        return states[index]

    @staticmethod
    def rows(states: np.ndarray, indices: Sequence[int]) -> np.ndarray:
        return states[list(indices)]

    @staticmethod
    def row_range(states: np.ndarray, start: int, stop: int) -> np.ndarray:
        return states[start:stop]

    @staticmethod
    def owned(values) -> np.ndarray:
        # a copy the caller can't alias, from an array or nested lists (a loaded file)
        return np.array(values, dtype=np.float64).copy()


class RustBackend:
    """NumpyBackend's operations on indrajala_math_rust arrays."""

    name = "rust"
    random_layer = staticmethod(fan_in_aware_random_rust_layer)

    @staticmethod
    def vector(state: Sequence[float]) -> "pa.Array":
        return pa.Array(list(state))

    @staticmethod
    def matrix(states: Sequence[Sequence[float]]) -> "pa.Array":
        return pa.Array([list(state) for state in states])

    @staticmethod
    def row(states: "pa.Array", index: int) -> "pa.Array":
        return states.row(index)

    @staticmethod
    def rows(states: "pa.Array", indices: Sequence[int]) -> "pa.Array":
        return states.take_rows(list(indices))

    @staticmethod
    def row_range(states: "pa.Array", start: int, stop: int) -> "pa.Array":
        return states.take_rows(list(range(start, stop)))

    @staticmethod
    def owned(values) -> "pa.Array":
        # nested lists let a snapshot cross a multiprocessing.Pool worker boundary as plain,
        # picklable lists (ensemble_train._picklable_snapshot)
        return values.copy() if isinstance(values, pa.Array) else pa.Array(values)


NUMPY = NumpyBackend()
RUST = RustBackend()
