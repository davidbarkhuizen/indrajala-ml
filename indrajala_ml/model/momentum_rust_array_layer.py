from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.rust_array_layer import RustArrayLayer, unfused_sgd_step


class MomentumRustArrayLayer(RustArrayLayer):
    """
    The Rust-matmul-backed counterpart to MomentumArrayLayer. Same momentum update rule, same
    previous-delta state, but `apply_accumulated_gradient` is a single fused Rust call
    (`layer_momentum_apply_accumulated_gradient`, `fused.rs`) instead of a numpy expression -
    mirroring how `RustArrayLayer` itself relates to `ArrayLayer`.

    momentum is a required constructor argument, no default, the same posture MomentumArrayLayer
    already has.
    """

    def __init__(self, size: int, input_size: int, momentum: float) -> None:
        super().__init__(size, input_size)
        self._momentum = momentum
        self._prev_delta_W = pa.Array.zeros((size, input_size))
        self._prev_delta_b = pa.Array.zeros(size)

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        (
            self.W,
            self.b,
            self._prev_delta_W,
            self._prev_delta_b,
        ) = pa.layer_momentum_apply_accumulated_gradient(
            self.W,
            self.b,
            self._grad_W,
            self._grad_b,
            self._prev_delta_W,
            self._prev_delta_b,
            self._momentum,
            learning_rate,
            batch_size,
        )
        self._reset_gradient_accum()

    def sgd_step(self, input_activation: "pa.Array", learning_rate: float) -> None:
        # not plain SGD, so not RustArrayLayer's fused step
        unfused_sgd_step(self, input_activation, learning_rate)
