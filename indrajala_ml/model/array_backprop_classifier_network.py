from __future__ import annotations

from typing import Sequence

import numpy as np

from indrajala_ml.model.array_network_base import ArrayNetworkBase
from indrajala_ml.model.model_io import load_single_output_array_model_json, save_single_output_array_model_json


class ArrayBackpropClassifierNetwork(ArrayNetworkBase):
    """
    The single-output numpy-array-backed sibling of FanInAwareBackpropClassifierNetwork. This
    class exists specifically to host
    EnsembleArrayBackpropClassifierNetwork's sub-networks, one independent binary classifier per
    class, not a jointly-trained multiclass network.

    The single-output shape over ArrayNetworkBase - 0.5-threshold classify_state/
    predict_probability, a scalar target, and the class_count-free save/load envelope - the
    array-level analogue of ArrayNetworkBase's relationship to
    VectorizedMultiClassBackpropClassifierNetwork's own multiclass shape.
    CrossEntropyArrayBackpropClassifierNetwork subclasses this directly.

    randomize() (inherited from ArrayNetworkBase) implements only the fan-in-aware scheme (limit
    = 1/sqrt(fan_in)), skipping BackpropClassifierNetwork.randomize()'s own
    per-dimension-bounds-width scaling entirely: that scheme is "tuned for 1-2D geometric
    problems" (FanInAwareBackpropClassifierNetwork's own docstring), while this class exists for
    the ensemble's high-dimensional (784-pixel real-MNIST) use case, where fan-in-aware init is
    the only scheme ever measured to work.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        # input_bounds is accepted and discarded - this class has no StateLayer/input_bounds
        # notion, but ensemble_train.py's classifier_cls contract always calls
        # classifier_cls(layer_sizes, dimension, input_bounds) /
        # classifier_cls.randomized(layer_sizes, dimension, input_bounds); accepting it here is
        # a duck-typing relaxation, rather than changing that shared contract.
        super().__init__(layer_sizes, dimension, 1)

    def predict_probability(self, state: tuple[float, ...]) -> float:
        return float(self._forward(state)[0])

    def classify_state(self, state: tuple[float, ...]) -> float:
        return self._classify_output(self._forward(state))

    def _classify_output(self, output: np.ndarray) -> float:
        return 1.0 if float(output[0]) > 0.5 else 0.0

    def _target_array(self, category: float) -> np.ndarray:
        return np.array([category], dtype=np.float64)

    def _target_batch_array(self, categories: Sequence[float]) -> np.ndarray:
        return np.array([[category] for category in categories], dtype=np.float64)

    @classmethod
    def randomized(
        cls,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]] | None = None,
    ) -> "ArrayBackpropClassifierNetwork":
        network = cls(layer_sizes, dimension, input_bounds)
        network.randomize()
        return network

    def _extra_state(self) -> dict:
        # override point for a sibling with its own hyperparameter to round-trip through
        # save/load - empty for this plain class and for CrossEntropyArrayBackpropClassifierNetwork
        return {}

    @classmethod
    def _extra_init_kwargs(cls, state: dict) -> dict:
        return {}

    def save(self, path: str) -> None:
        # save_single_output_array_model_json (model_io.py), not save_array_model_json - this
        # class has no class_count notion at all, unlike every multiclass array-backed sibling.
        save_single_output_array_model_json(
            path,
            layer_sizes=self.layer_sizes,
            dimension=self.dimension,
            snapshot=self.snapshot(),
            extra=self._extra_state(),
        )

    @classmethod
    def load(cls, path: str) -> "ArrayBackpropClassifierNetwork":
        state = load_single_output_array_model_json(path)
        network = cls(state["layer_sizes"], state["dimension"], **cls._extra_init_kwargs(state))
        network.restore([(np.array(W), np.array(b)) for W, b in state["snapshot"]])
        return network
