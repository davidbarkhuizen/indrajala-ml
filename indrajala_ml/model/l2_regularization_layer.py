from __future__ import annotations

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_node import BackpropNode


def make_l2_node_cls(l2_lambda: float) -> type[BackpropNode]:
    """
    Returns a BackpropNode subclass whose apply_gradient adds an L2 (weight decay) penalty term
    to every incoming weight's update: minimizing C + (l2_lambda/2)*sum(w^2) instead of just C
    adds l2_lambda*w to that weight's gradient, so the update becomes
    w -= learning_rate*(delta*input_value + l2_lambda*w) - equivalently, an extra multiplicative
    shrinkage of w by (1 - learning_rate*l2_lambda) on top of the ordinary gradient step, which is
    where "weight decay" gets its name.

    Deliberately does NOT regularize bias - standard practice (e.g. Goodfellow, Bengio &
    Courville's Deep Learning, ch. 7): penalizing bias doesn't serve L2's actual purpose
    (discouraging large weights, which is what actually controls a model's effective
    complexity) and can needlessly hurt fitting the intercept, since a large bias isn't itself a
    sign of overfitting the way large weights are.

    A factory, not a fixed class, for the same reason as make_momentum_node_cls: there is no
    single l2_lambda value this codebase has measured and can recommend.
    """

    class L2RegularizedBackpropNode(BackpropNode):
        def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
            # l2_lambda*weight is added once here, against the batch-averaged data gradient -
            # not accumulated per example in accumulate_gradient() (inherited unchanged from
            # BackpropNode), since the weight itself doesn't move during a batch's forward/
            # backward passes: the penalty term is the same value at every example in the
            # batch, so accumulating it per example and then averaging would just reproduce
            # this same single term, at the cost of doing so the confusing way
            self.update_input_weights(
                [
                    weight - learning_rate * (accum / batch_size + l2_lambda * weight)
                    for weight, accum in zip(self.input_node_weights, self._weight_gradient_accum)
                ]
            )
            self.bias = self.bias - learning_rate * self._bias_gradient_accum / batch_size
            self._reset_gradient_accum()

    return L2RegularizedBackpropNode


def make_l2_layer_cls(l2_lambda: float) -> type[BackpropLayer]:
    """
    The layer-level counterpart to make_l2_node_cls - a BackpropLayer whose nodes are all
    L2RegularizedBackpropNodes at the given l2_lambda. Like momentum's MomentumLayer (and unlike
    ReLULayer or CrossEntropyOutputLayer), this is used for both hidden and output layers: L2
    modifies the weight-update rule itself, which every trainable layer shares, not the
    activation or loss (see L2RegularizedBackpropClassifierNetwork, which sets both
    hidden_layer_cls and output_layer_cls to the same L2-configured layer class).
    """

    class L2Layer(BackpropLayer):
        _node_cls = make_l2_node_cls(l2_lambda)

    return L2Layer
