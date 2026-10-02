from __future__ import annotations

from typing import ClassVar

from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.specs.update_rules import Momentum


class MomentumMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    The momentum sibling of MultiClassBackpropClassifierNetwork: its optimizer applies the
    Momentum rule, with momentum required, as in MomentumBackpropClassifierNetwork. The
    pure-Python sibling of MomentumVectorizedMultiClassBackpropClassifierNetwork and
    MomentumRustArrayMultiClassBackpropClassifierNetwork.
    """

    hyperparameters: ClassVar[tuple[str, ...]] = ("momentum",)

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        class_count: int,
        momentum: float,
    ) -> None:
        self.momentum = momentum
        super().__init__(layer_sizes, dimension, input_bounds, class_count)

    def _update_rule(self) -> Momentum:
        return Momentum(self.momentum)
