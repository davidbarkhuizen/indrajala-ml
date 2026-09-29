from __future__ import annotations

import numpy as np

from indrajala_ml.model.conv_array_layer import ConvArrayLayer
from indrajala_ml.model.optimizers import momentum_update


class MomentumConvArrayLayer(ConvArrayLayer):
    """
    ConvArrayLayer with the Momentum rule's update (optimizers.momentum_update), Goyal et al.
    2017's eq. (9), as the pure-Python optimizer applies it to a ConvKernel: a velocity shaped as
    W (channel_count, fan_in) and as b (channel_count,), zero-initialized. g is ConvArrayLayer's accumulator, summed over
    positions and examples, so g / B averages examples only. momentum is required. The network's
    optimizer calls this layer's own update until stage 3 of docs/composable-layers-workplan.md.
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
            self.grad_W,
            self.grad_b,
            self._velocity_W,
            self._velocity_b,
            self._momentum,
            learning_rate,
            batch_size,
        )
        self.reset_gradient_accum()
