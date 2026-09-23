from __future__ import annotations

from indrajala_ml.model.cross_entropy_array_layer import CrossEntropyArrayLayer
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class CrossEntropyVectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    A cross-entropy-output-layer sibling of VectorizedMultiClassBackpropClassifierNetwork. Unlike
    softmax's jointly-normalized
    output (one shared normalization across the whole output vector), this class's output layer
    applies an independent per-node cross-entropy delta at each of class_count outputs - a
    one-vs-rest-with-cross-entropy-loss variant, not a jointly-trained multiclass distribution:
    predict_probabilities does not sum to 1.0, and classify_state still picks the argmax, the
    same convention VectorizedMultiClassBackpropClassifierNetwork's own plain-sigmoid output
    already uses (both inherited unchanged here).

    Hidden layers stay plain ArrayLayer (sigmoid); only the output layer is a
    CrossEntropyArrayLayer - the array-level analogue of
    BinaryCrossEntropyBackpropClassifierNetwork's own output_layer_cls-only override, applied
    class_count-wide instead of single-output. No hyperparameter and no extra constructor
    parameter, matching softmax's own posture.
    """

    output_layer_cls = CrossEntropyArrayLayer
