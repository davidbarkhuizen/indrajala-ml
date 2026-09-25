from __future__ import annotations

from indrajala_ml.model.array_layer import ArrayLayer


class L2ArrayLayer(ArrayLayer):
    """
    The array-based counterpart to l2_regularization_layer.make_l2_node_cls: the same L2 (weight
    decay) update rule - w -= learning_rate*(accum/batch_size + l2_lambda*w), bias left
    unregularized - but as a whole-array numpy op over the layer's (size, input_size) weight
    matrix, instead of a per-weight Python loop.

    Unlike MomentumArrayLayer/AdamArrayLayer, needs no persistent per-parameter state between
    steps at all: l2_lambda is a fixed scalar and the penalty term is a pure function of the
    *current* weight, not any running history.

    l2_lambda is a required constructor argument, no default, mirroring make_l2_node_cls's own
    posture - this codebase's own measurements never found a value worth recommending as a
    default.
    """

    hyperparameters = ("l2_lambda",)

    def __init__(self, size: int, input_size: int, l2_lambda: float) -> None:
        super().__init__(size, input_size)
        self._l2_lambda = l2_lambda

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        self.W -= learning_rate * (self._grad_W / batch_size + self._l2_lambda * self.W)
        self.b -= learning_rate * self._grad_b / batch_size  # bias unregularized, matching
        # make_l2_node_cls's own comment
        self._reset_gradient_accum()
