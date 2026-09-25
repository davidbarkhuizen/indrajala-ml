from __future__ import annotations

from collections.abc import Sequence

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_node import BackpropNode
from indrajala_ml.model.base_node import AbstractNode


def make_momentum_node_cls(momentum: float) -> type[BackpropNode]:
    """
    A BackpropNode subclass with momentum as Goyal et al. 2017's eq. (9): each node keeps a velocity
    per weight and for its bias (zero-initialized), u = m * u + g / B, and steps w - lr * u. It
    replaces Rumelhart, Hinton & Williams (1986)'s generalized delta rule, Δw(n) = η·δ·a + α·Δw(n-1)
    (the paper's eq. (10)), which folds the rate into the velocity and so needs a correction when
    the rate changes. At a constant rate the two are equivalent.

    A factory because momentum has no default: α = 0.9 hurt across a learning-rate sweep, and no
    value in 0.3-0.7 beat no momentum once enough seeds ruled out noise.
    """

    class MomentumBackpropNode(BackpropNode):
        def __init__(
            self,
            input_nodes: Sequence[AbstractNode],
            input_node_weights: Sequence[float] | None = None,
            bias: float = 0.0,
        ) -> None:
            super().__init__(input_nodes, input_node_weights, bias)
            self._weight_velocities = [0.0] * len(self.input_nodes)
            self._bias_velocity = 0.0

        def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
            self._weight_velocities = [
                momentum * velocity + accum / batch_size
                for velocity, accum in zip(self._weight_velocities, self._weight_gradient_accum)
            ]
            self.update_input_weights(
                [
                    weight - learning_rate * velocity
                    for weight, velocity in zip(self.input_node_weights, self._weight_velocities)
                ]
            )
            self._bias_velocity = momentum * self._bias_velocity + self._bias_gradient_accum / batch_size
            self.bias = self.bias - learning_rate * self._bias_velocity
            self._reset_gradient_accum()

    return MomentumBackpropNode


def make_momentum_layer_cls(momentum: float) -> type[BackpropLayer]:
    """
    A BackpropLayer of make_momentum_node_cls nodes, used for hidden and output layers alike:
    momentum changes the weight update, which every trainable layer shares.
    """

    class MomentumLayer(BackpropLayer):
        _node_cls = make_momentum_node_cls(momentum)

    return MomentumLayer
