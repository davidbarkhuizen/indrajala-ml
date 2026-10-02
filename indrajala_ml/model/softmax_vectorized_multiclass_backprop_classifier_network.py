from __future__ import annotations

from indrajala_ml.model.specs.layer_specs import Dense
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class SoftmaxVectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    VectorizedMultiClassBackpropClassifierNetwork with a SoftmaxArrayLayer output. The layer's
    forward normalizes jointly, so classify_state and predict_probabilities need no override.
    """

    def _output_spec(self, size: int) -> Dense:
        return Dense(size, output=True, activation="softmax", loss="cross_entropy")
