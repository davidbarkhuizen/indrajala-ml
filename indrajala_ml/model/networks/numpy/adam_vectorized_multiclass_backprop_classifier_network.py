from __future__ import annotations

from indrajala_ml.model.networks.numpy.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.update_rules import DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON, Adam


class AdamVectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    The Adam sibling of VectorizedMultiClassBackpropClassifierNetwork: its optimizer applies the
    Adam rule, with beta1/beta2/epsilon defaulting to Kingma & Ba's published values, as in
    AdamBackpropClassifierNetwork.

    snapshot()/restore() cover only W/b, not Adam's m/v/t, as every network's do; resuming training
    with m/v/t intact would need an extended envelope.
    """

    hyperparameters = ("beta1", "beta2", "epsilon")

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        class_count: int,
        beta1: float = DEFAULT_BETA1,
        beta2: float = DEFAULT_BETA2,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        super().__init__(layer_sizes, dimension, class_count)

    def _update_rule(self) -> Adam:
        return Adam(self.beta1, self.beta2, self.epsilon)
