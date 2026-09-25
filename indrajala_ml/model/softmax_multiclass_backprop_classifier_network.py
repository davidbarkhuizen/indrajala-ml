from __future__ import annotations

from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.softmax_output_layer import SoftmaxOutputLayer


class SoftmaxMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    MultiClassBackpropClassifierNetwork with a softmax output and cross-entropy loss, the standard
    multiclass treatment (e.g. Nielsen's "Neural Networks and Deep Learning"), instead of
    independent sigmoids with quadratic loss:

    - predict_probabilities() sums to 1.0, a distribution over mutually exclusive classes.
    - The delta (activation - target) stays proportional to the error; quadratic loss's gradient
      vanishes on a confidently wrong, saturated sigmoid.

    Only output_layer_cls differs: softmax couples the nodes in the forward pass only, and its
    cross-entropy delta is per node (SoftmaxOutputNode.compute_output_delta). The one-vs-rest
    network stays in use by demos and saved models.
    """

    output_layer_cls = SoftmaxOutputLayer
