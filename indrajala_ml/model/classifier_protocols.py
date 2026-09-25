"""
The structural interfaces the training and evaluation functions take, rather than one concrete
class: every network and target in the package satisfies them without inheriting from them.

L is the label type: float (0.0/1.0) for the single-output networks and targets, int (a class
index) for the multiclass ones. A function generic in L checks that a student and its training
data agree on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from indrajala_ml.prepared_dataset import PreparedDataset

L = TypeVar("L")
L_co = TypeVar("L_co", covariant=True)

State = tuple[float, ...]
Example = tuple[State, L]


class StateClassifier(Protocol[L_co]):
    def classify_state(self, state: State) -> L_co: ...


class TargetClassifier(StateClassifier[L_co], Protocol[L_co]):
    """A reference to sample and score against: a LinearClassifierNetwork, or a targets.py target."""

    @property
    def input_bounds(self) -> list[tuple[float, float]]: ...


class TrainableClassifier(StateClassifier[L], Protocol[L]):
    """A student train_linear_classifier_network can train: learn one example, and pocket snapshots."""

    def learn(self, learning_rate: float, state: State, category: L) -> None: ...

    # each network's own snapshot type, only ever passed back to the same network's restore
    def snapshot(self) -> Any: ...

    def restore(self, snapshot: Any) -> None: ...


class BatchTrainableClassifier(TrainableClassifier[L], Protocol[L]):
    """A student train_backprop_network_mini_batch can train: a gradient-based network."""

    def learn_batch(self, learning_rate: float, batch: Sequence[Example[L]]) -> None: ...


@runtime_checkable
class PreparedTrainableClassifier(BatchTrainableClassifier[L], Protocol[L]):
    """An array network, which trains from rows of one backend matrix (a PreparedDataset)."""

    def prepare_dataset(self, rows: Sequence[Example[L]]) -> PreparedDataset: ...

    def learn_row(self, learning_rate: float, prepared: PreparedDataset, index: int) -> None: ...

    def learn_batch_rows(self, learning_rate: float, prepared: PreparedDataset, indices: Sequence[int]) -> None: ...

    def classify_rows(self, prepared: PreparedDataset) -> list[L]: ...


class BinaryClassifier(TrainableClassifier[float], Protocol):
    """A single-output network: one of an ensemble's per-class sub-networks."""

    def predict_probability(self, state: State) -> float: ...


BinaryClassifierT_co = TypeVar("BinaryClassifierT_co", bound=BinaryClassifier, covariant=True)


class BinaryClassifierClass(Protocol[BinaryClassifierT_co]):
    """
    A BinaryClassifier class as the ensemble trainers take it: BackpropClassifierNetwork's
    constructor and randomized() signature.
    """

    def __call__(
        self, layer_sizes: list[int], dimension: int, input_bounds: list[tuple[float, float]]
    ) -> BinaryClassifierT_co: ...

    def randomized(
        self, layer_sizes: list[int], dimension: int, input_bounds: list[tuple[float, float]]
    ) -> BinaryClassifierT_co: ...
