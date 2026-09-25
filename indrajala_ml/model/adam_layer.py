from __future__ import annotations

import math
from collections.abc import Sequence

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_node import BackpropNode
from indrajala_ml.model.base_node import AbstractNode


def make_adam_node_cls(beta1: float, beta2: float, epsilon: float) -> type[BackpropNode]:
    """
    A BackpropNode subclass whose apply_accumulated_gradient is Adam (Kingma & Ba, 2014): a
    per-parameter adaptive learning rate from bias-corrected running estimates of the gradient mean
    (m) and uncentered variance (v). Every weight and the bias get their own zero-initialized m/v,
    and each node counts its own steps t.

    A per-node t equals a global one: every trainable node gets exactly one
    apply_accumulated_gradient call per learn()/learn_batch().
    """

    class AdamBackpropNode(BackpropNode):
        def __init__(
            self,
            input_nodes: Sequence[AbstractNode],
            input_node_weights: Sequence[float] | None = None,
            bias: float = 0.0,
        ) -> None:
            super().__init__(input_nodes, input_node_weights, bias)
            self._weight_m = [0.0] * len(self.input_nodes)
            self._weight_v = [0.0] * len(self.input_nodes)
            self._bias_m = 0.0
            self._bias_v = 0.0
            self._t = 0

        def apply_accumulated_gradient(self, learning_rate: float, batch_size: int) -> None:
            self._t += 1
            bias_correction1 = 1 - beta1**self._t
            bias_correction2 = 1 - beta2**self._t

            new_weights: list[float] = []
            new_m: list[float] = []
            new_v: list[float] = []
            for weight, accum, m, v in zip(
                self.input_node_weights, self._weight_gradient_accum, self._weight_m, self._weight_v
            ):
                g = accum / batch_size
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * g * g
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2
                new_weights.append(weight - learning_rate * m_hat / (math.sqrt(v_hat) + epsilon))
                new_m.append(m)
                new_v.append(v)
            self.update_input_weights(new_weights)
            self._weight_m = new_m
            self._weight_v = new_v

            g_bias = self._bias_gradient_accum / batch_size
            self._bias_m = beta1 * self._bias_m + (1 - beta1) * g_bias
            self._bias_v = beta2 * self._bias_v + (1 - beta2) * g_bias * g_bias
            bias_m_hat = self._bias_m / bias_correction1
            bias_v_hat = self._bias_v / bias_correction2
            self.bias = self.bias - learning_rate * bias_m_hat / (math.sqrt(bias_v_hat) + epsilon)

            self._reset_gradient_accum()

    return AdamBackpropNode


def make_adam_layer_cls(beta1: float, beta2: float, epsilon: float) -> type[BackpropLayer]:
    """
    A BackpropLayer of make_adam_node_cls nodes. Used for hidden and output layers alike: Adam
    changes the weight update, which every trainable layer shares.
    """

    class AdamLayer(BackpropLayer):
        _node_cls = make_adam_node_cls(beta1, beta2, epsilon)

    return AdamLayer
