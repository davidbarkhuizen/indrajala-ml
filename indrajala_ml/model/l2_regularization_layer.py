from __future__ import annotations

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_node import BackpropNode


def make_l2_node_cls(l2_lambda: float) -> type[BackpropNode]:
    """
    A BackpropNode subclass with L2 weight decay: minimizing C + (l2_lambda/2)*sum(w^2) adds
    l2_lambda*w to each weight's gradient, so w -= learning_rate*(gradient + l2_lambda*w), a
    shrinkage by (1 - learning_rate*l2_lambda) on top of the gradient step.

    The bias isn't regularized, as is standard (Goodfellow, Bengio & Courville, Deep Learning,
    ch. 7): large weights, not a large intercept, are what signal overfitting. l2_lambda is
    required.
    """

    class L2RegularizedBackpropNode(BackpropNode):
        def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
            # the penalty is added once, to the averaged gradient: the weight doesn't move within
            # a batch, so it is the same for every example
            self.update_input_weights(
                [
                    weight - learning_rate * (accum / batch_size + l2_lambda * weight)
                    for weight, accum in zip(self.input_node_weights, self._weight_gradient_accum)
                ]
            )
            self.bias = self.bias - learning_rate * (self._bias_gradient_accum / batch_size)
            self._reset_gradient_accum()

    return L2RegularizedBackpropNode


def make_l2_layer_cls(l2_lambda: float) -> type[BackpropLayer]:
    """
    A BackpropLayer of make_l2_node_cls nodes, used for hidden and output layers alike: L2 changes
    the weight update, which every trainable layer shares.
    """

    class L2Layer(BackpropLayer):
        _node_cls = make_l2_node_cls(l2_lambda)

    return L2Layer
