from __future__ import annotations

import numpy as np

from indrajala_ml.model.array_layer import ArrayLayer


class CrossEntropyArrayLayer(ArrayLayer):
    """
    The array-based counterpart to binary_cross_entropy_backprop_classifier_network.CrossEntropyOutputNode:
    binary cross-entropy loss's delta simplifies to activation - target, with no extra
    sigmoid-derivative (a*(1-a)) factor - the same simplification SoftmaxArrayLayer's own delta
    uses for the multi-class case (see that class's own compute_output_delta).

    forward/forward_batch/compute_hidden_delta/compute_hidden_delta_batch/
    apply_accumulated_gradient are all inherited unchanged from ArrayLayer - unlike
    SoftmaxArrayLayer, a single sigmoid-activated output needs nothing from any sibling node, the
    same point CrossEntropyOutputNode's own docstring makes ("a single output node's activation
    needs nothing from any sibling - forward() is inherited completely unchanged").

    No size restriction (unlike SoftmaxArrayLayer's own size >= 2): cross-entropy's delta is
    independent per node, so this works equally as a single-node output (the literal array
    counterpart of CrossEntropyOutputLayer) or as a class_count-wide output layer (an independent
    per-node cross-entropy delta at each output - a one-vs-rest-with-cross-entropy-loss variant,
    distinct from softmax's jointly-normalized one).
    """

    def compute_output_delta(self, reference: np.ndarray) -> None:
        self.delta = self.a - reference

    def compute_output_delta_batch(self, reference_batch: np.ndarray) -> None:
        self.delta_batch = self.A - reference_batch
