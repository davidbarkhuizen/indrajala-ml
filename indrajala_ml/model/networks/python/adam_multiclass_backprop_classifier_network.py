from __future__ import annotations

from indrajala_ml.model.networks.python.multiclass_backprop_classifier_network import (
    MultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.spec_hooks import RulePreset
from indrajala_ml.model.specs.update_rules import DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON, Adam


class AdamMultiClassBackpropClassifierNetwork(RulePreset, MultiClassBackpropClassifierNetwork):
    """
    The Adam sibling of MultiClassBackpropClassifierNetwork: its optimizer applies the Adam rule,
    with beta1/beta2/epsilon defaulting to Kingma & Ba's published values, as in
    AdamBackpropClassifierNetwork. The pure-Python sibling of
    AdamVectorizedMultiClassBackpropClassifierNetwork and
    AdamRustArrayMultiClassBackpropClassifierNetwork.
    """

    rule = Adam

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        class_count: int,
        beta1: float = DEFAULT_BETA1,
        beta2: float = DEFAULT_BETA2,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        super().__init__(layer_sizes, dimension, input_bounds, class_count)
