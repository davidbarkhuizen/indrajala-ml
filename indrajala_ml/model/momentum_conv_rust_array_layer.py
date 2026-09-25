from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.conv_rust_array_layer import ConvRustArrayLayer


class MomentumConvRustArrayLayer(ConvRustArrayLayer):
    """
    MomentumConvArrayLayer on the Rust backend: the same update and velocity state, applied by
    MomentumRustArrayLayer's fused call (layer_momentum_apply_accumulated_gradient), which takes a
    (W, b) pair of any matching shapes, so a conv (channel_count, fan_in) W too. momentum is
    required.
    """

    hyperparameters = ("momentum",)

    def __init__(
        self,
        input_height: int,
        input_width: int,
        input_channels: int,
        kernel_size: int,
        channel_count: int,
        stride: int = 1,
        *,
        momentum: float,
    ) -> None:
        super().__init__(input_height, input_width, input_channels, kernel_size, channel_count, stride)
        self._momentum = momentum
        self._velocity_W = pa.Array.zeros((channel_count, self.fan_in))
        self._velocity_b = pa.Array.zeros(channel_count)

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
