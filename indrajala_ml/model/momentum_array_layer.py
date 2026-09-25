# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, which strict mode takes for constants)
from __future__ import annotations

import numpy as np

from indrajala_ml.model.array_layer import ArrayLayer, FloatArray


def momentum_update(
    W: FloatArray,
    b: FloatArray,
    grad_W: FloatArray,
    grad_b: FloatArray,
    velocity_W: FloatArray,
    velocity_b: FloatArray,
    momentum: float,
    learning_rate: float,
    batch_size: int,
) -> tuple[FloatArray, FloatArray]:
    """
    Goyal et al. 2017's eq. (9) for one layer's (W, b), whatever their shape: u = m * u + g / B,
    then w - lr * u. Steps W and b in place and returns the new velocities. The numpy counterpart
    of the fused layer_momentum_apply_accumulated_gradient, shared by the dense and conv layers.
    """
    velocity_W = momentum * velocity_W + grad_W / batch_size
    velocity_b = momentum * velocity_b + grad_b / batch_size
    W -= learning_rate * velocity_W
    b -= learning_rate * velocity_b
    return velocity_W, velocity_b


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
        self._velocity_W, self._velocity_b = momentum_update(
            self.W,
            self.b,
            self._grad_W,
            self._grad_b,
            self._velocity_W,
            self._velocity_b,
            self._momentum,
            learning_rate,
            batch_size,
        )
        self._reset_gradient_accum()
