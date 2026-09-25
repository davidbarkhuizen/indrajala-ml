from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.array_layer import unfused_sgd_step
from indrajala_ml.model.rust_array_layer import RustArrayLayer


class MomentumRustArrayLayer(RustArrayLayer):
    """
    MomentumArrayLayer on the Rust backend: the same update and velocity state, applied by one
    fused call (layer_momentum_apply_accumulated_gradient). momentum is required.
    """

    hyperparameters = ("momentum",)

    def __init__(self, size: int, input_size: int, momentum: float) -> None:
        super().__init__(size, input_size)
        self._momentum = momentum
        self._velocity_W = pa.Array.zeros((size, input_size))
        self._velocity_b = pa.Array.zeros(size)

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        (
            self.W,
            self.b,
            self._velocity_W,
            self._velocity_b,
        ) = pa.layer_momentum_apply_accumulated_gradient(
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

    def sgd_step(self, input_activation: pa.Array, learning_rate: float) -> None:
        # not plain SGD, so not RustArrayLayer's fused step
        unfused_sgd_step(self, input_activation, learning_rate)
