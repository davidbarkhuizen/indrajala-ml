from __future__ import annotations

from indrajala_ml.model.array_backprop_classifier_network import ArrayBackpropClassifierNetwork
from indrajala_ml.model.update_rules import Momentum


class MomentumArrayBackpropClassifierNetwork(ArrayBackpropClassifierNetwork):
    """
    The momentum sibling of ArrayBackpropClassifierNetwork: its optimizer applies the Momentum
    rule, with momentum required (keyword-only, after the ignored input_bounds), as in
    MomentumBackpropClassifierNetwork. snapshot()/restore() cover only W/b, not the optimizer's
    velocities.
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
