from __future__ import annotations

import numpy as np

from indrajala_ml.model.array_layer import ArrayLayer


class MomentumArrayLayer(ArrayLayer):
    """
    The array-based counterpart to momentum_layer.make_momentum_node_cls: the same momentum term
    from Rumelhart, Hinton & Williams (1986)'s own generalized delta rule -
    Δw(n) = η·δ·a + α·Δw(n-1) - but as whole-array numpy ops over the layer's (size, input_size)
    weight matrix and size-length bias vector, instead of a per-weight Python loop.

    One previous-delta array per parameter tensor, no bias correction, no second moment -
    simpler than AdamArrayLayer's own state, the same shape used here.

    momentum is a required constructor argument, no default - the same posture
    make_momentum_node_cls itself takes (this codebase's own measurements never found a value
    worth recommending).
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
