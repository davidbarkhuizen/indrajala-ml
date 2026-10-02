# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, which strict mode takes for constants)
"""
The numpy networks' optimizer (array_optimizer_base.py): each rule's formulas on numpy arrays.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.optimizers.array_optimizer_base import ArrayOptimizerBase, momentum_update
from indrajala_ml.model.protocols.array_protocols import ArrayNetworkLayer, TrainedArrayLayer
from indrajala_ml.model.specs.update_rules import Adam, Momentum, WeightDecay


class NumpyOptimizer(ArrayOptimizerBase[FloatArray, TrainedArrayLayer[FloatArray], None]):
    """
    The optimizer of the numpy networks: each rule's formulas on numpy arrays, each of a layer's
    parameters (TrainedArrayLayer.parameters(): W and b, a linear layer's W, or batch norm's gamma
    and beta) stepped in place, one after another. Each rule's formula is elementwise, so a
    parameter's step doesn't depend on the others'.
    """

    def apply(self, index: int, layer: ArrayNetworkLayer[FloatArray], learning_rate: float, batch_size: int) -> None:
        if not hasattr(layer, "parameters"):
            return  # a pool layer: nothing trained
        trained = cast("TrainedArrayLayer[FloatArray]", layer)
        self._apply_rule(index, trained, learning_rate, batch_size)
        trained.reset_gradient_accum()

    def step_single(
        self, index: int, layer: ArrayNetworkLayer[FloatArray], input_activation: FloatArray, learning_rate: float
    ) -> None:
        # accumulate, then apply at batch_size=1
        layer.accumulate_gradient(input_activation)
        self.apply(index, layer, learning_rate, 1)

    def _apply_sgd(
        self, _index: int, layer: TrainedArrayLayer[FloatArray], learning_rate: float, batch_size: int
    ) -> None:
        for parameter, gradient in zip(layer.parameters(), layer.gradients()):
            parameter -= learning_rate * (gradient / batch_size)

    def _apply_weight_decay(
        self, _index: int, layer: TrainedArrayLayer[FloatArray], learning_rate: float, batch_size: int
    ) -> None:
        l2_lambda = cast(WeightDecay, self.rule).l2_lambda
        for parameter, gradient, decayed in zip(layer.parameters(), layer.gradients(), layer.decayed):
            if decayed:
                parameter -= learning_rate * (gradient / batch_size + l2_lambda * parameter)
            else:
                parameter -= learning_rate * (gradient / batch_size)  # a bias, gamma or beta

    def _apply_momentum(
        self, index: int, layer: TrainedArrayLayer[FloatArray], learning_rate: float, batch_size: int
    ) -> None:
        momentum = cast(Momentum, self.rule).momentum
        # state: a velocity per parameter
        state = self._zeros(index, layer, 1)
        for i, (parameter, gradient) in enumerate(zip(layer.parameters(), layer.gradients())):
            state[i] = momentum_update(parameter, gradient, state[i], momentum, learning_rate, batch_size)

    def _apply_adam(
        self, index: int, layer: TrainedArrayLayer[FloatArray], learning_rate: float, batch_size: int
    ) -> None:
        rule = cast(Adam, self.rule)
        beta1, beta2, epsilon = rule.beta1, rule.beta2, rule.epsilon
        # state: m, then v, per parameter
        state = self._zeros(index, layer, 2)

        bias_correction1 = 1 - beta1**self.t
        bias_correction2 = 1 - beta2**self.t

        for i, (parameter, gradient) in enumerate(zip(layer.parameters(), layer.gradients())):
            g = gradient / batch_size
            m = beta1 * state[2 * i] + (1 - beta1) * g
            v = beta2 * state[2 * i + 1] + (1 - beta2) * g * g
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2
            parameter -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            state[2 * i], state[2 * i + 1] = m, v
