from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.array_layer import unfused_sgd_step
from indrajala_ml.model.rust_array_layer import RustArrayLayer


class AdamRustArrayLayer(RustArrayLayer):
    """
    The Rust-matmul-backed counterpart to AdamArrayLayer. Same Adam (Kingma & Ba, 2014) update rule, same m/v/t state,
    but `apply_accumulated_gradient` is a single fused Rust call
    (`layer_adam_apply_accumulated_gradient`, `fused.rs`) instead of a numpy expression -
    mirroring how `RustArrayLayer` itself relates to `ArrayLayer`.

    beta1/beta2/epsilon are required here (no defaults), the same posture AdamArrayLayer already
    has - the safe Kingma & Ba defaults live one level up, on
    AdamRustArrayMultiClassBackpropClassifierNetwork.
    """

    def __init__(self, size: int, input_size: int, beta1: float, beta2: float, epsilon: float) -> None:
        super().__init__(size, input_size)
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._m_W = pa.Array.zeros((size, input_size))
        self._v_W = pa.Array.zeros((size, input_size))
        self._m_b = pa.Array.zeros(size)
        self._v_b = pa.Array.zeros(size)
        self._t = 0

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        self._t += 1
        (
            self.W,
            self.b,
            self._m_W,
            self._v_W,
            self._m_b,
            self._v_b,
        ) = pa.layer_adam_apply_accumulated_gradient(
            self.W,
            self.b,
            self._grad_W,
            self._grad_b,
            self._m_W,
            self._v_W,
            self._m_b,
            self._v_b,
            self._t,
            self._beta1,
            self._beta2,
            self._epsilon,
            learning_rate,
            batch_size,
        )
        self._reset_gradient_accum()

    def sgd_step(self, input_activation: "pa.Array", learning_rate: float) -> None:
        # not plain SGD, so not RustArrayLayer's fused step
        unfused_sgd_step(self, input_activation, learning_rate)
