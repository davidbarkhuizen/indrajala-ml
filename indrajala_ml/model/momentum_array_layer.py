from __future__ import annotations

import numpy as np

from indrajala_ml.model.array_layer import ArrayLayer


class MomentumArrayLayer(ArrayLayer):
    """
    Momentum as Goyal et al. 2017's eq. (9) over arrays, as make_momentum_node_cls: a velocity for
    W and for b, u = m * u + g / B, and the step w - lr * u. momentum is required.
    """

    hyperparameters = ("momentum",)

    def __init__(self, size: int, input_size: int, momentum: float) -> None:
        super().__init__(size, input_size)
        self._momentum = momentum
        self._velocity_W = np.zeros((size, input_size))
        self._velocity_b = np.zeros(size)

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        self._velocity_W = self._momentum * self._velocity_W + self._grad_W / batch_size
        self._velocity_b = self._momentum * self._velocity_b + self._grad_b / batch_size
        self.W -= learning_rate * self._velocity_W
        self.b -= learning_rate * self._velocity_b
        self._reset_gradient_accum()
