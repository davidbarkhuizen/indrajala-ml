from __future__ import annotations

from typing import Sequence

import indrajala_ml_array as pa

from indrajala_ml.model.bounds import validate_class_count
from indrajala_ml.model.model_io import load_array_model_json, save_array_model_json
from indrajala_ml.model.rust_array_network_base import RustArrayNetworkBase


class RustArrayMultiClassBackpropClassifierNetwork(RustArrayNetworkBase):
    """
    The Rust-array-core-backed sibling of VectorizedMultiClassBackpropClassifierNetwork. Mirrors
    that class's external contract exactly
    (learn, learn_batch, classify_state, predict_probabilities, randomize/randomized,
    snapshot/restore, save/load) so indrajala_ml/train.py's duck-typed
    train_linear_classifier_network/train_backprop_network_mini_batch work unchanged - only the
    array backend (`indrajala_ml_array.Array` via `RustArrayLayer`, not numpy via `ArrayLayer`)
    differs. The multiclass shape over RustArrayNetworkBase, the same relationship
    VectorizedMultiClassBackpropClassifierNetwork has to ArrayNetworkBase - every array-based
    multiclass sibling's Rust-matmul-backed counterpart subclasses this directly.

    This class is the intended production backend unconditionally - not contingent on beating
    VectorizedMultiClassBackpropClassifierNetwork's numpy benchmark, which stays on permanently
    as the comparison point, not a bar this class had to clear first.
    """

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int) -> None:
        validate_class_count(class_count)
        self.class_count = class_count
        super().__init__(layer_sizes, dimension, class_count)

    def predict_probabilities(self, state: tuple[float, ...]) -> list[float]:
        return self._forward(state).tolist()

    def classify_state(self, state: tuple[float, ...]) -> int:
        return pa.argmax(self._forward(state))

    def _target_array(self, category: int) -> "pa.Array":
        target = pa.Array.zeros(self.class_count)
        target[category] = 1.0
        return target

    def _target_batch_array(self, batch: Sequence[tuple[tuple[float, ...], int]], batch_size: int) -> "pa.Array":
        target_batch = pa.Array.zeros((batch_size, self.class_count))
        for row, (_state, category) in enumerate(batch):
            target_batch[row, category] = 1.0
        return target_batch

    @classmethod
    def randomized(
        cls,
        layer_sizes: list[int],
        dimension: int,
        class_count: int,
    ) -> "RustArrayMultiClassBackpropClassifierNetwork":
        network = cls(layer_sizes, dimension, class_count)
        network.randomize()
        return network

    def _extra_state(self) -> dict:
        return {}

    @classmethod
    def _extra_init_kwargs(cls, state: dict) -> dict:
        return {}

    def save(self, path: str) -> None:
        # save_array_model_json (model_io.py) - the shared envelope every array-backed multiclass
        # sibling uses, not save_model_json: no input_bounds/StateLayer notion here.
        save_array_model_json(
            path,
            layer_sizes=self.layer_sizes,
            dimension=self.dimension,
            class_count=self.class_count,
            snapshot=self.snapshot(),
            extra=self._extra_state(),
        )

    @classmethod
    def load(cls, path: str) -> "RustArrayMultiClassBackpropClassifierNetwork":
        state = load_array_model_json(path)
        network = cls(
            state["layer_sizes"],
            state["dimension"],
            state["class_count"],
            **cls._extra_init_kwargs(state),
        )
        network.restore([(pa.Array(W), pa.Array(b)) for W, b in state["snapshot"]])
        return network
