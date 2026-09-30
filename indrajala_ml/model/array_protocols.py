"""
The structural interfaces of the array-backed networks, generic in the backend's array type A:
FloatArray (numpy) or indrajala_math_rust.Array (Rust). Every layer and backend class satisfies
them for its own A without inheriting from them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, Self, runtime_checkable

from indrajala_ml.model.checkpoint import OptimizerState
from indrajala_ml.model.update_rules import UpdateRule


class BackendArray(Protocol):
    """What the networks do to a backend's array directly: numpy's ndarray and indrajala_math_rust.Array."""

    def copy(self) -> Self: ...

    def tolist(self) -> Any: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    def __setitem__(self, index: Any, value: float, /) -> None: ...


class ArrayNetworkLayer[A: BackendArray](Protocol):
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


@runtime_checkable
class WeightedArrayLayer[A: BackendArray](ArrayNetworkLayer[A], Protocol):
    """
    A layer with weights: a dense layer (all of a dense network's) or a conv layer. The network's
    optimizer (optimizers.py) steps W and b from the accumulated gradients, then resets them.
    """

    size: int
    W: A
    b: A
    grad_W: A
    grad_b: A

    def reset_gradient_accum(self) -> None: ...


@runtime_checkable
class TrainingModeLayer(Protocol):
    """A layer that behaves differently in training (dropout): learn* switches it on and off."""

    def set_training_mode(self, training: bool, /) -> None: ...


class ArrayOptimizer[A: BackendArray](Protocol):
    """What ArrayNetworkBase drives to update its layers (optimizers.py)."""

    rule: UpdateRule

    def begin_step(self) -> None: ...

    # copies of t and the rule's state, and their inverse (checkpoint.py)
    def state(self) -> OptimizerState[list[A]]: ...

    def load_state(self, state: OptimizerState[Any], /) -> None: ...

    def apply(self, index: int, layer: ArrayNetworkLayer[A], learning_rate: float, batch_size: int) -> None: ...

    def step_single(
        self, index: int, layer: ArrayNetworkLayer[A], input_activation: A, learning_rate: float
    ) -> None: ...


class ArrayBackend[A: BackendArray](Protocol):
    """The array operations ArrayNetworkBase needs from a backend (array_backend.py)."""

    @property
    def name(self) -> str: ...

    def seed(self, seed: int | None = None) -> None: ...

    # a new optimizer applying rule on this backend's arrays
    def optimizer(self, rule: UpdateRule) -> ArrayOptimizer[A]: ...

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
