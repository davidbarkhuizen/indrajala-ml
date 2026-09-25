from __future__ import annotations

import numpy as np

from indrajala_ml.model.conv_array_layer import ConvArrayLayer
from indrajala_ml.model.momentum_array_layer import momentum_update


class MomentumConvArrayLayer(ConvArrayLayer):
    """
    ConvArrayLayer with MomentumArrayLayer's update, Goyal et al. 2017's eq. (9), as
    make_momentum_kernel_cls: a velocity shaped as W (channel_count, fan_in) and as b
    (channel_count,), zero-initialized. g is ConvArrayLayer's accumulator, summed over positions and
    examples, so g / B averages examples only. momentum is required.
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
        self._velocity_W = np.zeros((channel_count, self.fan_in))
        self._velocity_b = np.zeros(channel_count)

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
