from __future__ import annotations

from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.specs.layer_specs import Dense


class CrossEntropyMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    MultiClassBackpropClassifierNetwork with a CrossEntropyOutputLayer output: an independent
    cross-entropy delta at each of the class_count sigmoid outputs, one-vs-rest rather than
    SoftmaxMultiClassBackpropClassifierNetwork's joint normalization. predict_probabilities doesn't
    sum to 1.0; classify_state is the argmax. The pure-Python sibling of
    CrossEntropyVectorizedMultiClassBackpropClassifierNetwork and
    CrossEntropyRustArrayMultiClassBackpropClassifierNetwork.
    """

    def _output_spec(self, size: int) -> Dense:
        return Dense(size, output=True, loss="cross_entropy")
