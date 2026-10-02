from __future__ import annotations

from indrajala_ml.model.networks.numpy.array_backprop_classifier_network import ArrayBackpropClassifierNetwork
from indrajala_ml.model.specs.update_rules import WeightDecay


class L2ArrayBackpropClassifierNetwork(ArrayBackpropClassifierNetwork):
    """
    The L2 (weight decay) sibling of ArrayBackpropClassifierNetwork: its optimizer applies the
    WeightDecay rule, with l2_lambda required (keyword-only, after the ignored input_bounds), as in
    L2RegularizedBackpropClassifierNetwork. L2 keeps no per-parameter state, so
    snapshot()/restore() of W/b is complete.
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
