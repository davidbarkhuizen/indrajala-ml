from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.array_layer import unfused_sgd_step
from indrajala_ml.model.rust_array_layer import RustArrayLayer


class L2RustArrayLayer(RustArrayLayer):
    """
    L2ArrayLayer on the Rust backend, applied by one fused call
    (layer_l2_apply_accumulated_gradient). l2_lambda is required.
    """

    hyperparameters = ("l2_lambda",)

    def __init__(self, size: int, input_size: int, l2_lambda: float) -> None:
        super().__init__(size, input_size)
        self._l2_lambda = l2_lambda

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        self.W, self.b = pa.layer_l2_apply_accumulated_gradient(
            self.W,
            self.b,
            self._grad_W,
            self._grad_b,
            self._l2_lambda,
            learning_rate,
            batch_size,
        )
        self._reset_gradient_accum()

    def sgd_step(self, input_activation: "pa.Array", learning_rate: float) -> None:
        # not plain SGD, so not RustArrayLayer's fused step
        unfused_sgd_step(self, input_activation, learning_rate)
