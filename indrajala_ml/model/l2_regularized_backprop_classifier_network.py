from __future__ import annotations

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.update_rules import WeightDecay


class L2RegularizedBackpropClassifierNetwork(BackpropClassifierNetwork):
    """
    An L2 (weight decay) sibling of BackpropClassifierNetwork: its optimizer applies the
    WeightDecay rule, adding l2_lambda * weight to every weight's gradient; biases aren't
    regularized. l2_lambda is required.

    Measured on a small fixed proxy dataset a network can overfit (not the XOR target the other
    siblings used): 0.0001 and 0.001 made training and held-out accuracy slightly worse than none;
    0.01 closed the train/test gap by lowering training accuracy to test accuracy (93.75% both);
    0.1 and above collapsed the network to a constant prediction (hidden weights decayed to about
    0.003). No coefficient beat the unregularized held-out accuracy, plausibly because these small
    networks don't overfit enough for a weight penalty to help.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        l2_lambda: float,
    ) -> None:
        self.l2_lambda = l2_lambda
        super().__init__(layer_sizes, dimension, input_bounds)

    def _update_rule(self) -> WeightDecay:
        return WeightDecay(self.l2_lambda)
