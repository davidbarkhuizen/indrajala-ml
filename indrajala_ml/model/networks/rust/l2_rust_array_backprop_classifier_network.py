from __future__ import annotations

from indrajala_ml.model.networks.rust.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork
from indrajala_ml.model.specs.spec_hooks import RulePreset
from indrajala_ml.model.specs.update_rules import WeightDecay


class L2RustArrayBackpropClassifierNetwork(RulePreset, RustArrayBackpropClassifierNetwork):
    """
    L2ArrayBackpropClassifierNetwork on the Rust backend.
    """

    rule = WeightDecay

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
