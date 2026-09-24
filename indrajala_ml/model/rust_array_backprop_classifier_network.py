from __future__ import annotations

from typing import Sequence

import indrajala_math_rust as pa

from indrajala_ml.model.model_io import load_single_output_array_model_json, save_single_output_array_model_json
from indrajala_ml.model.rust_array_network_base import RustArrayNetworkBase


class RustArrayBackpropClassifierNetwork(RustArrayNetworkBase):
    """
    The Rust-array-core-backed sibling of ArrayBackpropClassifierNetwork. Built unconditionally
    alongside the numpy sibling, not gated behind a wall-clock comparison - this codebase treats
    the Rust backend as the intended production path unconditionally, not contingent on beating
    the numpy benchmark first. The single-output shape over RustArrayNetworkBase, the same
    relationship ArrayBackpropClassifierNetwork has to ArrayNetworkBase.
    CrossEntropyRustArrayBackpropClassifierNetwork subclasses this directly.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        # input_bounds is accepted and discarded - see ArrayBackpropClassifierNetwork's own
        # docstring for why (duck-type compatibility with ensemble_train.py's classifier_cls
        # contract).
        super().__init__(layer_sizes, dimension, 1)

    def predict_probability(self, state: tuple[float, ...]) -> float:
        return self._forward(state).tolist()[0]

    def classify_state(self, state: tuple[float, ...]) -> float:
        return self._classify_output(self._forward(state))

    def _classify_output(self, output: "pa.Array") -> float:
        return 1.0 if output.tolist()[0] > 0.5 else 0.0

    def _classify_output_batch(self, output_batch: "pa.Array") -> list[float]:
        return [1.0 if row[0] > 0.5 else 0.0 for row in output_batch.tolist()]

    def _target_array(self, category: float) -> "pa.Array":
        return pa.Array([category])

    def _target_batch_array(self, categories: Sequence[float]) -> "pa.Array":
        return pa.Array([[category] for category in categories])

    @classmethod
    def randomized(
        cls,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]] | None = None,
    ) -> "RustArrayBackpropClassifierNetwork":
        network = cls(layer_sizes, dimension, input_bounds)
        network.randomize()
        return network

    def _extra_state(self) -> dict:
        return {}

    @classmethod
    def _extra_init_kwargs(cls, state: dict) -> dict:
        return {}

    def save(self, path: str) -> None:
        save_single_output_array_model_json(
            path,
            layer_sizes=self.layer_sizes,
            dimension=self.dimension,
            snapshot=self.snapshot(),
            extra=self._extra_state(),
        )

    @classmethod
    def load(cls, path: str) -> "RustArrayBackpropClassifierNetwork":
        state = load_single_output_array_model_json(path)
        network = cls(state["layer_sizes"], state["dimension"], **cls._extra_init_kwargs(state))
        network.restore([(pa.Array(W), pa.Array(b)) for W, b in state["snapshot"]])
        return network
