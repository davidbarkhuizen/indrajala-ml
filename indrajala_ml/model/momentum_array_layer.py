from __future__ import annotations

import numpy as np

from indrajala_ml.model.array_layer import ArrayLayer


class MomentumArrayLayer(ArrayLayer):
    """
    Momentum (Rumelhart, Hinton & Williams, 1986) over arrays, as make_momentum_node_cls:
    Δw(n) = η·δ·a + α·Δw(n-1), keeping the previous delta of W and b. momentum is required.
    """

    hyperparameters = ("momentum",)

    def __init__(self, size: int, input_size: int, momentum: float) -> None:
        super().__init__(size, input_size)
        self._momentum = momentum
        self._prev_delta_W = np.zeros((size, input_size))
        self._prev_delta_b = np.zeros(size)

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        delta_W = learning_rate * self._grad_W / batch_size + self._momentum * self._prev_delta_W
        delta_b = learning_rate * self._grad_b / batch_size + self._momentum * self._prev_delta_b
        self.W -= delta_W
        self.b -= delta_b
        self._prev_delta_W, self._prev_delta_b = delta_W, delta_b
        self._reset_gradient_accum()
