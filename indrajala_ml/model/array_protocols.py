"""
The structural interfaces of the array-backed networks, generic in the backend's array type A:
FloatArray (numpy) or indrajala_math_rust.Array (Rust). Every layer and backend class satisfies
them for its own A without inheriting from them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

from typing_extensions import Self


class BackendArray(Protocol):
    """What the networks do to a backend's array directly: numpy's ndarray and indrajala_math_rust.Array."""

    def copy(self) -> Self: ...

    def tolist(self) -> Any: ...

    def __setitem__(self, index: Any, value: float, /) -> None: ...


# the backend's array type: FloatArray (numpy) or indrajala_math_rust.Array (Rust)
A = TypeVar("A", bound=BackendArray)


class ArrayNetworkLayer(Protocol[A]):
    """What ArrayNetworkBase drives layer by layer: dense, conv and max-pool layers alike."""

    def forward(self, x: A, /) -> A: ...

    def forward_batch(self, X: A, /) -> A: ...

    def compute_output_delta(self, reference: A, /) -> None: ...

    def compute_output_delta_batch(self, reference_batch: A, /) -> None: ...

    # the next layer of the same backend: a dense layer reads its weights and delta, a conv or
    # pool layer its downstream()
    def compute_hidden_delta(self, next_layer: Any, /) -> None: ...

    def compute_hidden_delta_batch(self, next_layer: Any, /) -> None: ...

    def downstream(self) -> A: ...

    def downstream_batch(self) -> A: ...

    def accumulate_gradient(self, input_activation: A, /) -> None: ...

    def accumulate_gradient_batch(self, input_activation_batch: A, /) -> None: ...

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None: ...

    def sgd_step(self, input_activation: A, learning_rate: float, /) -> None: ...


@runtime_checkable
class WeightedArrayLayer(ArrayNetworkLayer[A], Protocol[A]):
    """A layer with weights: a dense layer (all of a dense network's) or a conv layer."""

    size: int
    W: A
    b: A


class ArrayBackend(Protocol[A]):
    """The array operations ArrayNetworkBase needs from a backend (array_backend.py)."""

    @property
    def name(self) -> str: ...

    def random_layer(self, size: int, previous_size: int) -> tuple[A, A]: ...

    def vector(self, state: Sequence[float]) -> A: ...

    def matrix(self, states: Sequence[Sequence[float]]) -> A: ...

    def row(self, states: A, index: int) -> A: ...

    def rows(self, states: A, indices: Sequence[int]) -> A: ...

    def row_range(self, states: A, start: int, stop: int) -> A: ...

    # an array the caller can't alias, from an array or nested lists (a loaded file)
    def owned(self, values: Any) -> A: ...

    def zeros(self, shape: int | tuple[int, int]) -> A: ...

    def argmax(self, vector: A) -> int: ...

    def argmax_rows(self, matrix: A) -> list[int]: ...
