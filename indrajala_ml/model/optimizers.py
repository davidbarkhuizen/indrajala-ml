# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, which strict mode takes for constants)
"""
The optimizers of the array networks, one per network (the composable-layers workplan, D5): an
update rule (update_rules.py) applied to each trained layer's parameters and gradients, with the
rule's state per layer, keyed by the layer's index in network.layers, and one step count t for the
whole network. The layers keep their weights and gradient accumulators; the formulas are here, with
the source's grouping (README, Update rules).

ArrayNetworkBase calls begin_step() once per learn* call, then per layer in forward order either
apply() after accumulating a batch or step_single() for one example. Every layer is stepped alike:
the numpy optimizer steps each of its parameters (array_protocols.TrainedArrayLayer), and every
rule's formula, and every fused Rust op, takes arrays of any matching shapes. A pool layer has
nothing to step.

state() and load_state() copy t and the state out and back in (checkpoint.py), so a network's
checkpoint resumes training by bits.
"""

from __future__ import annotations

from typing import Any, cast

import indrajala_math_rust as pa
import numpy as np

from indrajala_ml.model.array_layer import FloatArray
from indrajala_ml.model.array_protocols import ArrayNetworkLayer, TrainedArrayLayer
from indrajala_ml.model.checkpoint import OptimizerState
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from indrajala_ml.model.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay


def momentum_update(
    parameter: FloatArray,
    gradient: FloatArray,
    velocity: FloatArray,
    momentum: float,
    learning_rate: float,
    batch_size: int,
) -> FloatArray:
    """
    Goyal et al. 2017's eq. (9) for one parameter, whatever its shape: u = m * u + g / B, then
    w - lr * u. Steps the parameter in place and returns the new velocity. The numpy counterpart
    of the fused layer_momentum_apply_accumulated_gradient, which takes W and b together.
    """
    velocity = momentum * velocity + gradient / batch_size
    parameter -= learning_rate * velocity
    return velocity


class NumpyOptimizer:
    """
    The optimizer of the numpy networks: each rule's formulas on numpy arrays, each of a layer's
    parameters (TrainedArrayLayer.parameters(): W and b, a linear layer's W, or batch norm's gamma
    and beta) stepped in place, one after another. Each rule's formula is elementwise, so a
    parameter's step doesn't depend on the others'.
    """

    def __init__(self, rule: UpdateRule) -> None:
        self.rule = rule
        self.t = 0
        # per layer index: the rule's state arrays, per parameter in parameters() order
        # (momentum's velocity; Adam's m, then v)
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

    def state(self) -> OptimizerState[list[FloatArray]]:
        return OptimizerState(
            self.t, {index: [array.copy() for array in arrays] for index, arrays in self._state.items()}
        )

    def load_state(self, state: OptimizerState[Any]) -> None:
        # numpy arrays or nested lists (a checkpoint pickled across a worker boundary)
        self.t = state.t
        self._state = {
            index: [np.array(values, dtype=np.float64) for values in arrays] for index, arrays in state.layers.items()
        }

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

    def _zeros(self, index: int, layer: TrainedArrayLayer[FloatArray], count: int) -> list[FloatArray]:
        # count zero arrays shaped as each parameter, parameter by parameter: the rule's state for
        # a layer, made on the layer's first step
        state = self._state.get(index)
        if state is None:
            state = self._state[index] = [
                np.zeros(parameter.shape) for parameter in layer.parameters() for _ in range(count)
            ]
        return state

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


def _rust_zeros(like: pa.Array) -> pa.Array:
    shape = like.shape
    return pa.Array.zeros(shape[0] if len(shape) == 1 else (shape[0], shape[1]))


# a bias-free layer's missing second parameter: the fused ops step each parameter of their pair on
# its own, and step an empty array to an empty array
_EMPTY = pa.Array.zeros(0)


def _pair(arrays: tuple[pa.Array, ...] | list[pa.Array]) -> tuple[pa.Array, pa.Array]:
    # a layer's parameters (or gradients, or one of the rule's state arrays) as the fused ops'
    # (W, b) pair: (W, b), (gamma, beta), or a linear layer's (W, empty)
    return (arrays[0], arrays[1]) if len(arrays) == 2 else (arrays[0], _EMPTY)


class RustOptimizer:
    """
    NumpyOptimizer on the Rust backend: each rule is one fused call per layer (fused.rs), taking a
    pair of parameters, and the layer's parameters are rebound to its result. The pair is a dense
    or conv layer's (W, b), a batch-norm layer's (gamma, beta), or a linear layer's W with an empty
    array in b's place: every fused op checks each parameter against its own gradient and state
    only. The state is per parameter, as NumpyOptimizer's. For SGD the single-example step is one
    fused call too (step_single).
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

    def state(self) -> OptimizerState[list[pa.Array]]:
        return OptimizerState(
            self.t, {index: [array.copy() for array in arrays] for index, arrays in self._state.items()}
        )

    def load_state(self, state: OptimizerState[Any]) -> None:
        # Rust arrays or nested lists (a checkpoint pickled across a worker boundary)
        self.t = state.t
        self._state = {
            index: [values.copy() if isinstance(values, pa.Array) else pa.Array(values) for values in arrays]
            for index, arrays in state.layers.items()
        }

    def apply(self, index: int, layer: ArrayNetworkLayer[pa.Array], learning_rate: float, batch_size: int) -> None:
        if not hasattr(layer, "parameters"):
            return  # a pool layer: nothing trained
        trained = cast("TrainedArrayLayer[pa.Array]", layer)
        parameters = self._apply_rule(index, trained, learning_rate, batch_size)
        trained.set_parameters(parameters[: len(trained.decayed)])
        trained.reset_gradient_accum()

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

    def _zeros(self, index: int, layer: TrainedArrayLayer[pa.Array], count: int) -> list[pa.Array]:
        # as NumpyOptimizer._zeros
        state = self._state.get(index)
        if state is None:
            state = self._state[index] = [
                _rust_zeros(parameter) for parameter in layer.parameters() for _ in range(count)
            ]
        return state

    def _apply_sgd(
        self, _index: int, layer: TrainedArrayLayer[pa.Array], learning_rate: float, batch_size: int
    ) -> tuple[pa.Array, pa.Array]:
        return pa.layer_apply_accumulated_gradient(
            *_pair(layer.parameters()), *_pair(layer.gradients()), learning_rate, batch_size
        )

    def _apply_weight_decay(
        self, index: int, layer: TrainedArrayLayer[pa.Array], learning_rate: float, batch_size: int
    ) -> tuple[pa.Array, pa.Array]:
        if not layer.decayed[0]:
            # gamma and beta: plain SGD, as NumpyOptimizer steps a parameter that isn't decayed
            return self._apply_sgd(index, layer, learning_rate, batch_size)
        return pa.layer_l2_apply_accumulated_gradient(
            *_pair(layer.parameters()),
            *_pair(layer.gradients()),
            cast(WeightDecay, self.rule).l2_lambda,
            learning_rate,
            batch_size,
        )

    def _apply_momentum(
        self, index: int, layer: TrainedArrayLayer[pa.Array], learning_rate: float, batch_size: int
    ) -> tuple[pa.Array, pa.Array]:
        # state: a velocity per parameter
        state = self._zeros(index, layer, 1)
        w, b, velocity_w, velocity_b = pa.layer_momentum_apply_accumulated_gradient(
            *_pair(layer.parameters()),
            *_pair(layer.gradients()),
            *_pair(state),
            cast(Momentum, self.rule).momentum,
            learning_rate,
            batch_size,
        )
        state[:] = [velocity_w, velocity_b][: len(state)]
        return w, b

    def _apply_adam(
        self, index: int, layer: TrainedArrayLayer[pa.Array], learning_rate: float, batch_size: int
    ) -> tuple[pa.Array, pa.Array]:
        rule = cast(Adam, self.rule)
        # state: m, then v, per parameter
        state = self._zeros(index, layer, 2)
        m_w, v_w, m_b, v_b = state if len(state) == 4 else (*state, _EMPTY, _EMPTY)
        w, b, m_w, v_w, m_b, v_b = pa.layer_adam_apply_accumulated_gradient(
            *_pair(layer.parameters()),
            *_pair(layer.gradients()),
            m_w,
            v_w,
            m_b,
            v_b,
            self.t,
            rule.beta1,
            rule.beta2,
            rule.epsilon,
            learning_rate,
            batch_size,
        )
        state[:] = [m_w, v_w, m_b, v_b][: len(state)]
        return w, b
