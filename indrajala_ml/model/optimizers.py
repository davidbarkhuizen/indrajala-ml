# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, which strict mode takes for constants)
"""
The optimizers of the array networks, one per network (docs/composable-layers-workplan.md, D5):
an update rule (update_rules.py) applied to each weighted layer's (W, b, grad_W, grad_b), with
the rule's state per layer, keyed by the layer's index in network.layers, and one step count t
for the whole network. The layers keep their weights and gradient accumulators; the formulas are
here, with the source's grouping (README, Update rules).

ArrayNetworkBase calls begin_step() once per learn* call, then per layer in forward order either
apply() after accumulating a batch or step_single() for one example. A dense or conv layer is
stepped alike: every rule's formula, and every fused Rust op, takes W and b of any matching shapes.
A layer without W (a pool layer) has nothing to step.
"""

from __future__ import annotations

from typing import cast

import indrajala_math_rust as pa
import numpy as np

from indrajala_ml.model.array_layer import FloatArray
from indrajala_ml.model.array_protocols import ArrayNetworkLayer, WeightedArrayLayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from indrajala_ml.model.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay


def momentum_update(
    W: FloatArray,
    b: FloatArray,
    grad_W: FloatArray,
    grad_b: FloatArray,
    velocity_W: FloatArray,
    velocity_b: FloatArray,
    momentum: float,
    learning_rate: float,
    batch_size: int,
) -> tuple[FloatArray, FloatArray]:
    """
    Goyal et al. 2017's eq. (9) for one layer's (W, b), whatever their shape: u = m * u + g / B,
    then w - lr * u. Steps W and b in place and returns the new velocities. The numpy counterpart
    of the fused layer_momentum_apply_accumulated_gradient.
    """
    velocity_W = momentum * velocity_W + grad_W / batch_size
    velocity_b = momentum * velocity_b + grad_b / batch_size
    W -= learning_rate * velocity_W
    b -= learning_rate * velocity_b
    return velocity_W, velocity_b


class NumpyOptimizer:
    """The optimizer of the numpy networks: each rule's formulas on numpy arrays, W and b stepped in place."""

    def __init__(self, rule: UpdateRule) -> None:
        self.rule = rule
        self.t = 0
        # per layer index: the rule's state arrays (momentum's velocities, Adam's m and v)
        self._state: dict[int, list[FloatArray]] = {}
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

    def apply(self, index: int, layer: ArrayNetworkLayer[FloatArray], learning_rate: float, batch_size: int) -> None:
        if not hasattr(layer, "W"):
            return  # a pool layer: no weights
        weighted = cast("WeightedArrayLayer[FloatArray]", layer)
        self._apply_rule(index, weighted, learning_rate, batch_size)
        weighted.reset_gradient_accum()

    def step_single(
        self, index: int, layer: ArrayNetworkLayer[FloatArray], input_activation: FloatArray, learning_rate: float
    ) -> None:
        # accumulate, then apply at batch_size=1
        layer.accumulate_gradient(input_activation)
        self.apply(index, layer, learning_rate, 1)

    def _zeros(self, index: int, layer: WeightedArrayLayer[FloatArray], count: int) -> list[FloatArray]:
        # count zero arrays shaped as W, then as many shaped as b: the rule's state for a layer,
        # made on the layer's first step
        state = self._state.get(index)
        if state is None:
            state = self._state[index] = [np.zeros(layer.W.shape) for _ in range(count)] + [
                np.zeros(layer.b.shape) for _ in range(count)
            ]
        return state

    def _apply_sgd(
        self, _index: int, layer: WeightedArrayLayer[FloatArray], learning_rate: float, batch_size: int
    ) -> None:
        layer.W -= learning_rate * (layer.grad_W / batch_size)
        layer.b -= learning_rate * (layer.grad_b / batch_size)

    def _apply_weight_decay(
        self, _index: int, layer: WeightedArrayLayer[FloatArray], learning_rate: float, batch_size: int
    ) -> None:
        l2_lambda = cast(WeightDecay, self.rule).l2_lambda
        layer.W -= learning_rate * (layer.grad_W / batch_size + l2_lambda * layer.W)
        layer.b -= learning_rate * (layer.grad_b / batch_size)  # bias unregularized

    def _apply_momentum(
        self, index: int, layer: WeightedArrayLayer[FloatArray], learning_rate: float, batch_size: int
    ) -> None:
        # state: velocity_W, velocity_b
        state = self._zeros(index, layer, 1)
        state[0], state[1] = momentum_update(
            layer.W,
            layer.b,
            layer.grad_W,
            layer.grad_b,
            state[0],
            state[1],
            cast(Momentum, self.rule).momentum,
            learning_rate,
            batch_size,
        )

    def _apply_adam(
        self, index: int, layer: WeightedArrayLayer[FloatArray], learning_rate: float, batch_size: int
    ) -> None:
        rule = cast(Adam, self.rule)
        beta1, beta2, epsilon = rule.beta1, rule.beta2, rule.epsilon
        # state: m_W, v_W, m_b, v_b
        state = self._zeros(index, layer, 2)
        m_W, v_W, m_b, v_b = state

        bias_correction1 = 1 - beta1**self.t
        bias_correction2 = 1 - beta2**self.t

        g_W = layer.grad_W / batch_size
        m_W = beta1 * m_W + (1 - beta1) * g_W
        v_W = beta2 * v_W + (1 - beta2) * g_W * g_W
        m_hat_W = m_W / bias_correction1
        v_hat_W = v_W / bias_correction2
        layer.W -= learning_rate * m_hat_W / (np.sqrt(v_hat_W) + epsilon)

        g_b = layer.grad_b / batch_size
        m_b = beta1 * m_b + (1 - beta1) * g_b
        v_b = beta2 * v_b + (1 - beta2) * g_b * g_b
        m_hat_b = m_b / bias_correction1
        v_hat_b = v_b / bias_correction2
        layer.b -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

        state[:] = [m_W, v_W, m_b, v_b]


def _rust_zeros(like: pa.Array) -> pa.Array:
    shape = like.shape
    return pa.Array.zeros(shape[0] if len(shape) == 1 else (shape[0], shape[1]))


class RustOptimizer:
    """
    NumpyOptimizer on the Rust backend: each rule is one fused call per layer (fused.rs), taking W
    and b together, and the layer's W and b are rebound to its result. For SGD the single-example
    step is one fused call too (step_single).
    """

    def __init__(self, rule: UpdateRule) -> None:
        self.rule = rule
        self.t = 0
        self._state: dict[int, list[pa.Array]] = {}
        self._fused_sgd_step = isinstance(rule, SGD)
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

    def apply(self, index: int, layer: ArrayNetworkLayer[pa.Array], learning_rate: float, batch_size: int) -> None:
        if not hasattr(layer, "W"):
            return  # a pool layer: no weights
        weighted = cast("WeightedArrayLayer[pa.Array]", layer)
        self._apply_rule(index, weighted, learning_rate, batch_size)
        weighted.reset_gradient_accum()

    def step_single(
        self, index: int, layer: ArrayNetworkLayer[pa.Array], input_activation: pa.Array, learning_rate: float
    ) -> None:
        if self._fused_sgd_step and isinstance(layer, RustArrayLayer):
            # accumulate then apply at batch_size=1 as one fused call, bit-identical to that pair
            # (tests/test_rust_array_layer_sgd_step.py). It relies on the accumulators being fresh
            # zeros, which they always are here: apply resets them after every step, and this
            # doesn't touch them. Dense layers only: a conv gradient sums over output positions,
            # which layer_sgd_step's outer product doesn't.
            layer.W, layer.b = pa.layer_sgd_step(layer.W, layer.b, layer.delta, input_activation, learning_rate)
            return
        layer.accumulate_gradient(input_activation)
        self.apply(index, layer, learning_rate, 1)

    def _zeros(self, index: int, layer: WeightedArrayLayer[pa.Array], count: int) -> list[pa.Array]:
        # as NumpyOptimizer._zeros
        state = self._state.get(index)
        if state is None:
            state = self._state[index] = [_rust_zeros(layer.W) for _ in range(count)] + [
                _rust_zeros(layer.b) for _ in range(count)
            ]
        return state

    def _apply_sgd(
        self, _index: int, layer: WeightedArrayLayer[pa.Array], learning_rate: float, batch_size: int
    ) -> None:
        layer.W, layer.b = pa.layer_apply_accumulated_gradient(
            layer.W, layer.b, layer.grad_W, layer.grad_b, learning_rate, batch_size
        )

    def _apply_weight_decay(
        self, _index: int, layer: WeightedArrayLayer[pa.Array], learning_rate: float, batch_size: int
    ) -> None:
        layer.W, layer.b = pa.layer_l2_apply_accumulated_gradient(
            layer.W,
            layer.b,
            layer.grad_W,
            layer.grad_b,
            cast(WeightDecay, self.rule).l2_lambda,
            learning_rate,
            batch_size,
        )

    def _apply_momentum(
        self, index: int, layer: WeightedArrayLayer[pa.Array], learning_rate: float, batch_size: int
    ) -> None:
        # state: velocity_W, velocity_b
        state = self._zeros(index, layer, 1)
        layer.W, layer.b, state[0], state[1] = pa.layer_momentum_apply_accumulated_gradient(
            layer.W,
            layer.b,
            layer.grad_W,
            layer.grad_b,
            state[0],
            state[1],
            cast(Momentum, self.rule).momentum,
            learning_rate,
            batch_size,
        )

    def _apply_adam(
        self, index: int, layer: WeightedArrayLayer[pa.Array], learning_rate: float, batch_size: int
    ) -> None:
        rule = cast(Adam, self.rule)
        # state: m_W, v_W, m_b, v_b
        state = self._zeros(index, layer, 2)
        layer.W, layer.b, state[0], state[1], state[2], state[3] = pa.layer_adam_apply_accumulated_gradient(
            layer.W,
            layer.b,
            layer.grad_W,
            layer.grad_b,
            state[0],
            state[1],
            state[2],
            state[3],
            self.t,
            rule.beta1,
            rule.beta2,
            rule.epsilon,
            learning_rate,
            batch_size,
        )
