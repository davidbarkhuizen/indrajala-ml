from __future__ import annotations

from typing import ClassVar

from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.specs.update_rules import WeightDecay


class L2RegularizedMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    The L2 (weight decay) sibling of MultiClassBackpropClassifierNetwork: its optimizer applies the
    WeightDecay rule, with l2_lambda required, as in L2RegularizedBackpropClassifierNetwork. The
    pure-Python sibling of L2VectorizedMultiClassBackpropClassifierNetwork and
    L2RustArrayMultiClassBackpropClassifierNetwork.
    """

    hyperparameters: ClassVar[tuple[str, ...]] = ("l2_lambda",)

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        class_count: int,
        l2_lambda: float,
    ) -> None:
        self.l2_lambda = l2_lambda
        super().__init__(layer_sizes, dimension, input_bounds, class_count)

    def _update_rule(self) -> WeightDecay:
        return WeightDecay(self.l2_lambda)
