"""
The optimizer of the pure-Python networks, one per network (the composable-layers workplan, D5): an
update rule (update_rules.py) applied to each weight set (layer_protocols.WeightSet: a BackpropNode
or ConvKernel) of a layer, with the rule's state per weight set, keyed by the layer's index in
trainable_layers and the weight set's index in the layer, and one step count t for the whole
network. The nodes and kernels keep their weights and gradient accumulators; the per-weight formulas
are here, in Python floats with the source's grouping (README, Update rules), as the array
optimizers' (optimizers.py) are on arrays. A module of its own so the pure-Python networks import
neither numpy nor the Rust extension.

BackpropNetworkBase calls begin_step() once per learn* call, then per layer in forward order
either apply() after accumulating a batch or step_single() for one example. state() and
load_state() copy t and the state out and back in (checkpoint.py).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

from indrajala_ml.model.checkpoint import OptimizerState
from indrajala_ml.model.layer_protocols import TrainableLayer, WeightSet
from indrajala_ml.model.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay

# one weight set's state: lists shaped as its weights (momentum's velocities; Adam's m, then v),
# and the bias's values in the same order, none for a weight set without a bias (a linear layer's
# node)
WeightSetState = tuple[list[list[float]], list[float]]


def _copy_layers(layers: dict[int, list[WeightSetState]]) -> dict[int, list[WeightSetState]]:
    # deep: _apply_momentum and _apply_adam step the bias's state in place
    return {
        index: [([list(values) for values in weight_state], list(bias_state)) for weight_state, bias_state in sets]
        for index, sets in layers.items()
    }


class PythonOptimizer:
    """The optimizer of the pure-Python networks: each rule's formulas per weight, in Python floats."""

    def __init__(self, rule: UpdateRule) -> None:
        self.rule = rule
        self.t = 0
        # per layer index: each weight set's state, in the layer's weight_sets() order
        self._state: dict[int, list[WeightSetState]] = {}
        match rule:
            case SGD():
                self._apply_rule = self._apply_sgd
            case Momentum():
                self._apply_rule = self._apply_momentum
            case Adam():
                self._apply_rule = self._apply_adam
            case WeightDecay():
                self._apply_rule = self._apply_weight_decay

    def begin_step(self) -> None:
        self.t += 1

    def state(self) -> OptimizerState[list[WeightSetState]]:
        return OptimizerState(self.t, _copy_layers(self._state))

    def load_state(self, state: OptimizerState[list[WeightSetState]]) -> None:
        self.t = state.t
        self._state = _copy_layers(state.layers)

    def apply(self, index: int, layer: TrainableLayer, learning_rate: float, batch_size: int) -> None:
        weight_sets = layer.weight_sets()
        self._apply_rule(index, weight_sets, learning_rate, batch_size)
        for weight_set in weight_sets:
            weight_set.reset_gradient_accum()

    def step_single(self, index: int, layer: TrainableLayer, learning_rate: float) -> None:
        # accumulate, then apply at batch_size=1
        layer.accumulate_gradients()
        self.apply(index, layer, learning_rate, 1)

    def _zeros(self, index: int, weight_sets: Sequence[WeightSet], count: int) -> list[WeightSetState]:
        # per weight set, count zero lists shaped as its weights and count zeros for its bias: the
        # rule's state for a layer, made on the layer's first step
        state = self._state.get(index)
        if state is None:
            state = self._state[index] = [
                ([[0.0] * len(weight_set.weights) for _ in range(count)], [0.0] * count if weight_set.has_bias else [])
                for weight_set in weight_sets
            ]
        return state

    def _apply_sgd(self, _index: int, weight_sets: Sequence[WeightSet], learning_rate: float, batch_size: int) -> None:
        for weight_set in weight_sets:
            weight_set.set_weights(
                [
                    weight - learning_rate * (accum / batch_size)
                    for weight, accum in zip(weight_set.weights, weight_set.weight_gradient_accum)
                ]
            )
            if weight_set.has_bias:
                weight_set.bias = weight_set.bias - learning_rate * (weight_set.bias_gradient_accum / batch_size)

    def _apply_weight_decay(
        self, _index: int, weight_sets: Sequence[WeightSet], learning_rate: float, batch_size: int
    ) -> None:
        l2_lambda = cast(WeightDecay, self.rule).l2_lambda
        for weight_set in weight_sets:
            if not weight_set.weights_decayed:
                # batch norm's gamma, as its beta, steps with plain SGD (D7)
                self._apply_sgd(_index, [weight_set], learning_rate, batch_size)
                continue
            # the penalty is added once, to the averaged gradient: the weight doesn't move within
            # a batch, so it is the same for every example
            weight_set.set_weights(
                [
                    weight - learning_rate * (accum / batch_size + l2_lambda * weight)
                    for weight, accum in zip(weight_set.weights, weight_set.weight_gradient_accum)
                ]
            )
            # bias unregularized
            if weight_set.has_bias:
                weight_set.bias = weight_set.bias - learning_rate * (weight_set.bias_gradient_accum / batch_size)

    def _apply_momentum(
        self, index: int, weight_sets: Sequence[WeightSet], learning_rate: float, batch_size: int
    ) -> None:
        momentum = cast(Momentum, self.rule).momentum
        # state: the weights' velocities, the bias's velocity
        for weight_set, (weight_state, bias_state) in zip(weight_sets, self._zeros(index, weight_sets, 1)):
            weight_state[0] = [
                momentum * velocity + accum / batch_size
                for velocity, accum in zip(weight_state[0], weight_set.weight_gradient_accum)
            ]
            weight_set.set_weights(
                [weight - learning_rate * velocity for weight, velocity in zip(weight_set.weights, weight_state[0])]
            )
            if weight_set.has_bias:
                bias_state[0] = momentum * bias_state[0] + weight_set.bias_gradient_accum / batch_size
                weight_set.bias = weight_set.bias - learning_rate * bias_state[0]

    def _apply_adam(self, index: int, weight_sets: Sequence[WeightSet], learning_rate: float, batch_size: int) -> None:
        rule = cast(Adam, self.rule)
        beta1, beta2, epsilon = rule.beta1, rule.beta2, rule.epsilon
        bias_correction1 = 1 - beta1**self.t
        bias_correction2 = 1 - beta2**self.t

        # state: the weights' m and v, the bias's m and v
        for weight_set, (weight_state, bias_state) in zip(weight_sets, self._zeros(index, weight_sets, 2)):
            new_weights: list[float] = []
            new_m: list[float] = []
            new_v: list[float] = []
            for weight, accum, m, v in zip(
                weight_set.weights, weight_set.weight_gradient_accum, weight_state[0], weight_state[1]
            ):
                g = accum / batch_size
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * g * g
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2
                new_weights.append(weight - learning_rate * m_hat / (math.sqrt(v_hat) + epsilon))
                new_m.append(m)
                new_v.append(v)
            weight_set.set_weights(new_weights)
            weight_state[0] = new_m
            weight_state[1] = new_v

            if not weight_set.has_bias:
                continue
            g_bias = weight_set.bias_gradient_accum / batch_size
            bias_state[0] = beta1 * bias_state[0] + (1 - beta1) * g_bias
            bias_state[1] = beta2 * bias_state[1] + (1 - beta2) * g_bias * g_bias
            bias_m_hat = bias_state[0] / bias_correction1
            bias_v_hat = bias_state[1] / bias_correction2
            weight_set.bias = weight_set.bias - learning_rate * bias_m_hat / (math.sqrt(bias_v_hat) + epsilon)
