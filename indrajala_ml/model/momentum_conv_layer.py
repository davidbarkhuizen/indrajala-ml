from __future__ import annotations

from indrajala_ml.model.conv_kernel import ConvKernel
from indrajala_ml.model.conv_layer import ConvLayer


def make_momentum_kernel_cls(momentum: float) -> type[ConvKernel]:
    """
    A ConvKernel subclass with make_momentum_node_cls's update, Goyal et al. 2017's eq. (9): a
    velocity per weight and for the bias (zero-initialized), u = m * u + g / B, and steps w - lr * u.
    g is ConvKernel's accumulator, summed over positions and examples, so g / B averages examples
    only. A factory because momentum has no default.
    """

    class MomentumConvKernel(ConvKernel):
        def __init__(
            self,
            kernel_size: int,
            in_channels: int = 1,
            weights: list[float] | None = None,
            bias: float = 0.0,
        ) -> None:
            super().__init__(kernel_size, in_channels, weights, bias)
            self._weight_velocities = [0.0] * len(self.weights)
            self._bias_velocity = 0.0

        def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
            self._weight_velocities = [
                momentum * velocity + accum / batch_size
                for velocity, accum in zip(self._weight_velocities, self._weight_gradient_accum)
            ]
            self.weights = [
                weight - learning_rate * velocity for weight, velocity in zip(self.weights, self._weight_velocities)
            ]
            self._bias_velocity = momentum * self._bias_velocity + self._bias_gradient_accum / batch_size
            self.bias = self.bias - learning_rate * self._bias_velocity
            self._reset_gradient_accum()

    return MomentumConvKernel


def make_momentum_conv_layer_cls(momentum: float) -> type[ConvLayer]:
    """A ConvLayer of make_momentum_kernel_cls kernels."""

    class MomentumConvLayer(ConvLayer):
        _kernel_cls = make_momentum_kernel_cls(momentum)

    return MomentumConvLayer
