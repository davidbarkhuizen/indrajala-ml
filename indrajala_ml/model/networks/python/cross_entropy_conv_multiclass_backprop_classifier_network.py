from __future__ import annotations

from indrajala_ml.model.networks.python.conv_multiclass_backprop_classifier_network import (
    ConvMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.layer_specs import Dense


class CrossEntropyConvMultiClassBackpropClassifierNetwork(ConvMultiClassBackpropClassifierNetwork):
    """
    ConvMultiClassBackpropClassifierNetwork with a CrossEntropyOutputLayer output: an independent
    cross-entropy delta at each of the class_count sigmoid outputs, as in
    CrossEntropyMultiClassBackpropClassifierNetwork. predict_probabilities doesn't sum to 1.0;
    classify_state is the argmax. The pure-Python sibling of
    CrossEntropyConvVectorizedMultiClassBackpropClassifierNetwork and
    CrossEntropyConvRustArrayMultiClassBackpropClassifierNetwork.
    """

    def _output_spec(self, size: int) -> Dense:
        return Dense(size, output=True, loss="cross_entropy")
