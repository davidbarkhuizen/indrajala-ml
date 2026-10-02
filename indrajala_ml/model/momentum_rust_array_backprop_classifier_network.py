from __future__ import annotations

from indrajala_ml.model.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork
from indrajala_ml.model.update_rules import Momentum


class MomentumRustArrayBackpropClassifierNetwork(RustArrayBackpropClassifierNetwork):
    """
    MomentumArrayBackpropClassifierNetwork on the Rust backend.
    """

    hyperparameters = ("momentum",)

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]] | None = None,
        *,
        momentum: float,
    ) -> None:
        self.momentum = momentum
        super().__init__(layer_sizes, dimension, input_bounds)

    def _update_rule(self) -> Momentum:
        return Momentum(self.momentum)
