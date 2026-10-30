from __future__ import annotations

from indrajala_ml.model.networks.rust.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork
from indrajala_ml.model.specs.spec_hooks import RulePreset
from indrajala_ml.model.specs.update_rules import Momentum


class MomentumRustArrayBackpropClassifierNetwork(RulePreset, RustArrayBackpropClassifierNetwork):
    """
    MomentumArrayBackpropClassifierNetwork on the Rust backend.
    """

    rule = Momentum

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
