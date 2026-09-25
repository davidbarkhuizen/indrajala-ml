from __future__ import annotations

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_node import BackpropNode


def make_momentum_node_cls(momentum: float) -> type[BackpropNode]:
    """
    A BackpropNode subclass with the momentum term of Rumelhart, Hinton & Williams (1986)'s
    generalized delta rule, Δw(n) = η·δ·a + α·Δw(n-1): each node keeps its previous weight and bias
    deltas (zero-initialized) and adds momentum times them to each step.

    A factory because momentum has no default: α = 0.9 hurt across a learning-rate sweep, and no
    value in 0.3-0.7 beat no momentum once enough seeds ruled out noise.
    """

    class MomentumBackpropNode(BackpropNode):
        def __init__(self, input_nodes, input_node_weights=None, bias: float = 0.0) -> None:
            super().__init__(input_nodes, input_node_weights, bias)
            self._prev_weight_deltas = [0.0] * len(self.input_nodes)
            self._prev_bias_delta = 0.0

        def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
            # the averaged gradient takes the single-example gradient's place; the momentum term
            # depends only on the previous update
            new_weights = []
            new_prev = []
            for weight, accum, prev in zip(
                self.input_node_weights, self._weight_gradient_accum, self._prev_weight_deltas
            ):
                delta_w = learning_rate * accum / batch_size + momentum * prev
                new_weights.append(weight - delta_w)
                new_prev.append(delta_w)
            self.update_input_weights(new_weights)
            self._prev_weight_deltas = new_prev

            bias_delta = learning_rate * self._bias_gradient_accum / batch_size + momentum * self._prev_bias_delta
            self.bias = self.bias - bias_delta
            self._prev_bias_delta = bias_delta

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
