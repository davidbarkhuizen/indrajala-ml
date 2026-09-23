from __future__ import annotations

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_node import BackpropNode


def make_momentum_node_cls(momentum: float) -> type[BackpropNode]:
    """
    Returns a BackpropNode subclass whose apply_gradient adds the momentum term from Rumelhart,
    Hinton & Williams (1986)'s own generalized delta rule - Δw(n) = η·δ·a + α·Δw(n-1) - which
    BackpropNode.apply_gradient never had. Each node tracks its own previous weight/bias delta
    (zero-initialized) and folds momentum * that previous delta into the current step, then
    remembers the new delta for next time.

    A factory, not a fixed class, because momentum is a genuinely tunable coefficient - unlike
    every other node variant in this codebase (softmax, cross-entropy, ReLU), which differ by a
    fixed formula with no free parameter, there is no single momentum value this codebase's own
    measurements support recommending as a default: the canonical α=0.9 robustly hurts across a
    learning-rate sweep, and no coefficient in 0.3-0.7 measurably beats no momentum at all, once
    enough seeds rule out noise.
    """

    class MomentumBackpropNode(BackpropNode):
        def __init__(self, input_nodes, input_node_weights=None, bias: float = 0.0) -> None:
            super().__init__(input_nodes, input_node_weights, bias)
            self._prev_weight_deltas = [0.0] * len(self.input_nodes)
            self._prev_bias_delta = 0.0

        def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
            # the averaged accumulated gradient (accum / batch_size) plugs in exactly where a
            # single-example gradient (self.delta * node.value()) would in the unbatched formula -
            # the momentum term itself (momentum * prev) is unaffected by batching, since it's a
            # function of the *previous update*, not of how this one's gradient was computed
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
    The layer-level counterpart to make_momentum_node_cls - a BackpropLayer whose nodes are all
    MomentumBackpropNodes at the given coefficient. Unlike ReLULayer (hidden-only) or
    CrossEntropyOutputLayer (output-only), this is used for both hidden and output layers - the
    momentum term modifies the weight-update rule itself, which every trainable layer shares,
    not the activation or loss (see MomentumBackpropClassifierNetwork, which sets both
    hidden_layer_cls and output_layer_cls to the same momentum-configured layer class).
    """

    class MomentumLayer(BackpropLayer):
        _node_cls = make_momentum_node_cls(momentum)

    return MomentumLayer
