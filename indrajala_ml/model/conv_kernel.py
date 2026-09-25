from __future__ import annotations

from typing import Sequence

from indrajala_ml.model.backprop_network_base import fan_in_aware_weights_and_bias


class ConvKernel:
    """
    One conv output channel's shared weights, a flat in_channels x kernel_size x kernel_size
    list (in that order), and bias, read by every ConvUnit of the channel. Not a BackpropNode: its
    input_node_weights is owned per node and rebound on update, which doesn't suit a list many units
    share.

    accumulate_gradient() is called once per contributing position (every unit of the channel, for
    every example) and apply_accumulated_gradient() once per kernel, dividing by batch_size only:
    positions are summed, examples averaged.
    """

    def __init__(
        self,
        kernel_size: int,
        in_channels: int = 1,
        weights: list[float] | None = None,
        bias: float = 0.0,
    ) -> None:

        assert kernel_size >= 1, f"kernel_size must be at least 1; got {kernel_size}"
        assert in_channels >= 1, f"in_channels must be at least 1; got {in_channels}"

        self.kernel_size = kernel_size
        self.in_channels = in_channels

        fan_in = kernel_size * kernel_size * in_channels
        self.weights: list[float] = weights if weights is not None else [0.0 for _ in range(fan_in)]
        assert len(self.weights) == fan_in, f"expected {fan_in} weights (kernel_size**2 * in_channels); got {len(self.weights)}"

        self.bias: float = bias

        self._weight_gradient_accum: list[float] = [0.0 for _ in range(fan_in)]
        self._bias_gradient_accum: float = 0.0

    def randomize_fan_in_aware(self) -> None:
        # a kernel's fan-in is its receptive field, kernel_size**2 * in_channels
        self.weights, self.bias = fan_in_aware_weights_and_bias(len(self.weights))

    def accumulate_gradient(self, delta: float, receptive_field_values: Sequence[float]) -> None:
        assert len(receptive_field_values) == len(self.weights)
        for i, value in enumerate(receptive_field_values):
            self._weight_gradient_accum[i] += delta * value
        self._bias_gradient_accum += delta

    def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
        self.weights = [
            weight - learning_rate * (accum / batch_size)
            for weight, accum in zip(self.weights, self._weight_gradient_accum)
        ]
        self.bias = self.bias - learning_rate * (self._bias_gradient_accum / batch_size)
        self._reset_gradient_accum()

    def _reset_gradient_accum(self) -> None:
        self._weight_gradient_accum = [0.0 for _ in self.weights]
        self._bias_gradient_accum = 0.0
