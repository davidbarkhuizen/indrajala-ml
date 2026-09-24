from __future__ import annotations

from typing import Sequence

from indrajala_ml.model.bounds import validate_class_count
from indrajala_ml.model.model_io import (
    load_array_model_json,
    load_single_output_array_model_json,
    save_array_model_json,
    save_single_output_array_model_json,
)


class ArrayMultiClassShape:
    """
    The multiclass shape over ArrayNetworkBase, for either backend: argmax-based
    classify_state/predict_probabilities, class_count validation, one-hot target encoding, and
    the class_count-carrying save/load envelope - the array-level analogue of
    MultiClassBackpropClassifierNetwork's own relationship to BackpropNetworkBase.

    A mixin, listed before the backend's base (ArrayNetworkBase or RustArrayNetworkBase), which
    supplies self.backend: VectorizedMultiClassBackpropClassifierNetwork and
    RustArrayMultiClassBackpropClassifierNetwork are this shape on numpy and on Rust.
    """

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int) -> None:
        validate_class_count(class_count)
        self.class_count = class_count
        super().__init__(layer_sizes, dimension, class_count)

    def predict_probabilities(self, state: tuple[float, ...]) -> list[float]:
        return self._forward(state).tolist()

    def classify_state(self, state: tuple[float, ...]) -> int:
        return self._classify_output(self._forward(state))

    def _classify_output(self, output) -> int:
        return self.backend.argmax(output)

    def _classify_output_batch(self, output_batch) -> list[int]:
        return self.backend.argmax_rows(output_batch)

    def _target_array(self, category: int):
        target = self.backend.zeros(self.class_count)
        target[category] = 1.0
        return target

    def _target_batch_array(self, categories: Sequence[int]):
        target_batch = self.backend.zeros((len(categories), self.class_count))
        for row, category in enumerate(categories):
            target_batch[row, category] = 1.0
        return target_batch

    @classmethod
    def randomized(cls, layer_sizes: list[int], dimension: int, class_count: int):
        network = cls(layer_sizes, dimension, class_count)
        network.randomize()
        return network

    def _extra_state(self) -> dict:
        # override point for a sibling with its own hyperparameter to round-trip through
        # save/load (e.g. {"l2_lambda": self.l2_lambda}) - empty for the plain classes and every
        # hyperparameter-free sibling (ReLU, softmax, cross-entropy)
        return {}

    @classmethod
    def _extra_init_kwargs(cls, state: dict) -> dict:
        # the inverse of _extra_state: reconstructs a sibling's extra constructor kwargs from a
        # loaded state dict - empty for the plain classes and every hyperparameter-free sibling
        return {}

    def save(self, path: str) -> None:
        # not save_model_json (model_io.py) - that envelope hardcodes input_bounds, which this
        # shape has no notion of (no StateLayer). save_array_model_json is the shared envelope
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
    def load(cls, path: str):
        state = load_array_model_json(path)
        network = cls(
            state["layer_sizes"],
            state["dimension"],
            state["class_count"],
            **cls._extra_init_kwargs(state),
        )
        # restore converts the file's nested lists through the backend
        network.restore(state["snapshot"])
        return network


class ArraySingleOutputShape:
    """
    The single-output shape over ArrayNetworkBase, for either backend: 0.5-threshold
    classify_state/predict_probability, a scalar target, and the class_count-free save/load
    envelope. It exists to host the ensembles' sub-networks, one independent binary classifier
    per class, not a jointly-trained multiclass network.

    A mixin, listed before the backend's base, like ArrayMultiClassShape:
    ArrayBackpropClassifierNetwork and RustArrayBackpropClassifierNetwork are this shape on numpy
    and on Rust.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        # input_bounds is accepted and discarded - this shape has no StateLayer/input_bounds
        # notion, but ensemble_train.py's classifier_cls contract always calls
        # classifier_cls(layer_sizes, dimension, input_bounds) /
        # classifier_cls.randomized(layer_sizes, dimension, input_bounds); accepting it here is
        # a duck-typing relaxation, rather than changing that shared contract.
        super().__init__(layer_sizes, dimension, 1)

    def predict_probability(self, state: tuple[float, ...]) -> float:
        return self._forward(state).tolist()[0]

    def classify_state(self, state: tuple[float, ...]) -> float:
        return self._classify_output(self._forward(state))

    def _classify_output(self, output) -> float:
        return 1.0 if output.tolist()[0] > 0.5 else 0.0

    def _classify_output_batch(self, output_batch) -> list[float]:
        return [1.0 if row[0] > 0.5 else 0.0 for row in output_batch.tolist()]

    def _target_array(self, category: float):
        return self.backend.vector([category])

    def _target_batch_array(self, categories: Sequence[float]):
        return self.backend.matrix([[category] for category in categories])

    @classmethod
    def randomized(
        cls,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]] | None = None,
    ):
        network = cls(layer_sizes, dimension, input_bounds)
        network.randomize()
        return network

    def _extra_state(self) -> dict:
        # override point for a sibling with its own hyperparameter to round-trip through
        # save/load - empty for the plain classes and the cross-entropy siblings
        return {}

    @classmethod
    def _extra_init_kwargs(cls, state: dict) -> dict:
        return {}

    def save(self, path: str) -> None:
        # save_single_output_array_model_json (model_io.py), not save_array_model_json - this
        # shape has no class_count notion at all, unlike every multiclass array-backed sibling.
        save_single_output_array_model_json(
            path,
            layer_sizes=self.layer_sizes,
            dimension=self.dimension,
            snapshot=self.snapshot(),
            extra=self._extra_state(),
        )

    @classmethod
    def load(cls, path: str):
        state = load_single_output_array_model_json(path)
        network = cls(state["layer_sizes"], state["dimension"], **cls._extra_init_kwargs(state))
        network.restore(state["snapshot"])
        return network
