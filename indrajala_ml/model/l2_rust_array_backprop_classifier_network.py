from __future__ import annotations

from indrajala_ml.model.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork
from indrajala_ml.model.update_rules import WeightDecay


class L2RustArrayBackpropClassifierNetwork(RustArrayBackpropClassifierNetwork):
    """
    L2ArrayBackpropClassifierNetwork on the Rust backend.
    """

    hyperparameters = ("l2_lambda",)

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]] | None = None,
        *,
        l2_lambda: float,
    ) -> None:
        self.l2_lambda = l2_lambda
        super().__init__(layer_sizes, dimension, input_bounds)

    def _update_rule(self) -> WeightDecay:
        return WeightDecay(self.l2_lambda)
