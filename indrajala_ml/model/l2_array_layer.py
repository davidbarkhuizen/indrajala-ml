from __future__ import annotations

from indrajala_ml.model.array_layer import ArrayLayer


class L2ArrayLayer(ArrayLayer):
    """
    L2 weight decay over arrays, as make_l2_node_cls: w -= learning_rate * (accum/batch_size +
    l2_lambda * w), bias unregularized. No state between steps; l2_lambda is required.
    """

    hyperparameters = ("l2_lambda",)

    def __init__(self, size: int, input_size: int, l2_lambda: float) -> None:
        super().__init__(size, input_size)
        self._l2_lambda = l2_lambda

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        self.W -= learning_rate * (self._grad_W / batch_size + self._l2_lambda * self.W)
        self.b -= learning_rate * (self._grad_b / batch_size)  # bias unregularized
        self._reset_gradient_accum()
