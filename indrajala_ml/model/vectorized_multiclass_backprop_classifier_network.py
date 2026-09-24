from __future__ import annotations

from typing import Sequence

import numpy as np

from indrajala_ml.model.array_network_base import ArrayNetworkBase
from indrajala_ml.model.bounds import validate_class_count
from indrajala_ml.model.model_io import load_array_model_json, save_array_model_json


class VectorizedMultiClassBackpropClassifierNetwork(ArrayNetworkBase):
    """
    A numpy-array-backed sibling of MultiClassBackpropClassifierNetwork: array-based vectorization
    replaces "one Python object, one method call, per node" with "one array, one matrix operation,
    for the whole layer", so there's no per-node compute_hidden_delta(next_layer_nodes, own_index)
    to reuse and no StateLayer/BackpropLayer involved at all.

    The multiclass shape over ArrayNetworkBase - argmax-based classify_state/predict_probabilities,
    class_count validation, one-hot target encoding, and the class_count-carrying save/load
    envelope - the array-level analogue of MultiClassBackpropClassifierNetwork's own relationship
    to BackpropNetworkBase. Every array-based multiclass sibling (momentum, L2, Adam, ReLU,
    softmax, dropout, cross-entropy) subclasses this directly, the same way their per-node
    counterparts subclass MultiClassBackpropClassifierNetwork.

    Parity-checked against the pure-Python reference implementation; numpy is a vectorization
    backend here, not a permanent dependency commitment (RustArrayNetworkBase's Rust-backed
    siblings are the production path).
    """

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int) -> None:
        validate_class_count(class_count)
        self.class_count = class_count
        super().__init__(layer_sizes, dimension, class_count)

    def predict_probabilities(self, state: tuple[float, ...]) -> list[float]:
        return self._forward(state).tolist()

    def classify_state(self, state: tuple[float, ...]) -> int:
        return self._classify_output(self._forward(state))

    def _classify_output(self, output: np.ndarray) -> int:
        return int(np.argmax(output))

    def _classify_output_batch(self, output_batch: np.ndarray) -> list[int]:
        return np.argmax(output_batch, axis=1).tolist()

    def _target_array(self, category: int) -> np.ndarray:
        target = np.zeros(self.class_count)
        target[category] = 1.0
        return target

    def _target_batch_array(self, categories: Sequence[int]) -> np.ndarray:
        target_batch = np.zeros((len(categories), self.class_count))
        for row, category in enumerate(categories):
            target_batch[row, category] = 1.0
        return target_batch

    @classmethod
    def randomized(
        cls,
        layer_sizes: list[int],
        dimension: int,
        class_count: int,
    ) -> "VectorizedMultiClassBackpropClassifierNetwork":
        network = cls(layer_sizes, dimension, class_count)
        network.randomize()
        return network

    def _extra_state(self) -> dict:
        # override point for a sibling with its own hyperparameter to round-trip through
        # save/load (e.g. {"l2_lambda": self.l2_lambda}) - empty for this plain class and every
        # hyperparameter-free sibling (ReLU, softmax, cross-entropy)
        return {}

    @classmethod
    def _extra_init_kwargs(cls, state: dict) -> dict:
        # the inverse of _extra_state: reconstructs a sibling's extra constructor kwargs from a
        # loaded state dict - empty for this plain class and every hyperparameter-free sibling
        return {}

    def save(self, path: str) -> None:
        # not save_model_json (model_io.py) - that envelope hardcodes input_bounds, which this
        # class has no notion of (no StateLayer). save_array_model_json is the shared envelope
        # every array-backed multiclass sibling uses instead.
        save_array_model_json(
            path,
            layer_sizes=self.layer_sizes,
            dimension=self.dimension,
            class_count=self.class_count,
            snapshot=self.snapshot(),
            extra=self._extra_state(),
        )

    @classmethod
    def load(cls, path: str) -> "VectorizedMultiClassBackpropClassifierNetwork":
        state = load_array_model_json(path)
        network = cls(
            state["layer_sizes"],
            state["dimension"],
            state["class_count"],
            **cls._extra_init_kwargs(state),
        )
        network.restore([(np.array(W), np.array(b)) for W, b in state["snapshot"]])
        return network
