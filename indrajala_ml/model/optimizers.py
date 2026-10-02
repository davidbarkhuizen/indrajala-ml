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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import indrajala_math_rust as pa
import numpy as np

from indrajala_ml.model.array_layer import FloatArray
from indrajala_ml.model.array_protocols import ArrayBackend, ArrayNetworkLayer, BackendArray, TrainedArrayLayer
from indrajala_ml.model.checkpoint import OptimizerState
from indrajala_ml.model.optimizer_base import OptimizerBase
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


class ArrayOptimizerBase[A: BackendArray, T, R](OptimizerBase[list[A], T, R]):
    """
    What NumpyOptimizer and RustOptimizer share: their state per layer, the rule's state arrays per
    parameter in parameters() order (momentum's velocity; Adam's m, then v), copied out and in
    through the backend's arrays. T is what an _apply_* method steps: a layer on numpy, a pair of
    its parameters on Rust.
    """

    def __init__(self, rule: UpdateRule, backend: ArrayBackend[A]) -> None:
        super().__init__(rule)
        # the backend whose arrays the state is made of (array_backend.py, which builds this
        # optimizer as backend.optimizer(rule))
        self._backend = backend

    def state(self) -> OptimizerState[list[A]]:
        return OptimizerState(
            self.t, {index: [array.copy() for array in arrays] for index, arrays in self._state.items()}
        )

    def load_state(self, state: OptimizerState[Any]) -> None:
        # this backend's arrays or nested lists (a checkpoint pickled across a worker boundary)
        self.t = state.t
        self._state = {
            index: [self._backend.owned(values) for values in arrays] for index, arrays in state.layers.items()
        }

    def _zeros(self, index: int, layer: TrainedArrayLayer[A], count: int) -> list[A]:
        # count zero arrays shaped as each parameter, parameter by parameter: the rule's state for
        # a layer, made on the layer's first step
        state = self._state.get(index)
        if state is None:
            state = self._state[index] = [
                self._backend.zeros(_shape(parameter)) for parameter in layer.parameters() for _ in range(count)
            ]
        return state


def _shape(array: BackendArray) -> int | tuple[int, int]:
    # an array's shape as the backends' zeros takes it: every parameter is a vector or a matrix
    shape = array.shape
    return shape[0] if len(shape) == 1 else (shape[0], shape[1])


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


# a lone parameter's missing partner: the fused ops step each parameter of their pair on its own,
# and step an empty array to an empty array
_EMPTY = pa.Array.zeros(0)


def _pair(arrays: Sequence[pa.Array]) -> tuple[pa.Array, pa.Array]:
    # one or two parameters (or gradients, or one of the rule's state arrays) as the fused ops' (W, b)
    # pair: (W, b), (gamma, beta), or a lone parameter (a linear layer's W, a position table) with
    # an empty array
    return (arrays[0], arrays[1]) if len(arrays) == 2 else (arrays[0], _EMPTY)


@dataclass(frozen=True)
class _Pair:
    """
    A layer's parameters first and first + 1, or first alone (the last of an odd count), with their
    gradients: what one fused call steps. Every layer but attention is one pair, its first 0;
    attention's eight parameters are its four projections' (W, b) pairs.
    """

    layer: TrainedArrayLayer[pa.Array]
    first: int
    parameters: Sequence[pa.Array]
    gradients: Sequence[pa.Array]
    # whether WeightDecay decays the pair: a weight and its bias are stepped together, the bias as
    # the fused L2 op steps it, so only the first parameter's flag counts
    decayed: bool


def _pairs(layer: TrainedArrayLayer[pa.Array]) -> list[_Pair]:
    parameters, gradients = layer.parameters(), layer.gradients()
    return [
        _Pair(layer, first, parameters[first : first + 2], gradients[first : first + 2], layer.decayed[first])
        for first in range(0, len(parameters), 2)
    ]


class RustOptimizer(ArrayOptimizerBase[pa.Array, _Pair, tuple[pa.Array, pa.Array]]):
    """
    NumpyOptimizer on the Rust backend: each rule is one fused call (fused.rs) per pair of a layer's
    parameters (_Pair), and the layer's parameters are rebound to the results. A pair is a dense or
    conv layer's (W, b), a batch-norm or layer-norm layer's (gamma, beta), one of attention's
    projections' (W, b), or a lone parameter (a linear layer's W, a position table) with an empty
    array in b's place: every fused op checks each parameter against its own gradient and state
    only. The state is per parameter, as NumpyOptimizer's, and each pair steps its own slice of it.
    For SGD the single-example step of a dense layer is one fused call too (step_single).
    """

    def __init__(self, rule: UpdateRule, backend: ArrayBackend[pa.Array]) -> None:
        super().__init__(rule, backend)
        self._fused_sgd_step = isinstance(rule, SGD)

    def apply(self, index: int, layer: ArrayNetworkLayer[pa.Array], learning_rate: float, batch_size: int) -> None:
        if not hasattr(layer, "parameters"):
            return  # a pool layer: nothing trained
        trained = cast("TrainedArrayLayer[pa.Array]", layer)
        # a comprehension, so no pair outlives it: a pair still holding the old parameters and
        # gradients when the layer rebinds them keeps their memory from being reused for the new
        # gradients, and every step then pays fresh pages (the SGD step and Array.zeros measured 3x
        # slower, conv B=32 epochs +11%)
        parameters = [
            parameter
            for pair in _pairs(trained)
            for parameter in self._apply_rule(index, pair, learning_rate, batch_size)[: len(pair.parameters)]
        ]
        trained.set_parameters(parameters)
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

    def _pair_state(self, index: int, pair: _Pair, count: int) -> list[pa.Array]:
        # the pair's slice of the layer's state: count arrays per parameter, parameter by parameter
        state = self._zeros(index, pair.layer, count)
        return state[pair.first * count : (pair.first + len(pair.parameters)) * count]

    def _store_pair_state(self, index: int, pair: _Pair, count: int, arrays: Sequence[pa.Array]) -> None:
        start = pair.first * count
        self._state[index][start : start + len(pair.parameters) * count] = arrays[: len(pair.parameters) * count]

    def _apply_sgd(self, _index: int, pair: _Pair, learning_rate: float, batch_size: int) -> tuple[pa.Array, pa.Array]:
        return pa.layer_apply_accumulated_gradient(
            *_pair(pair.parameters), *_pair(pair.gradients), learning_rate, batch_size
        )

    def _apply_weight_decay(
        self, index: int, pair: _Pair, learning_rate: float, batch_size: int
    ) -> tuple[pa.Array, pa.Array]:
        if not pair.decayed:
            # gamma and beta, a position table: plain SGD, as NumpyOptimizer steps a parameter that
            # isn't decayed
            return self._apply_sgd(index, pair, learning_rate, batch_size)
        return pa.layer_l2_apply_accumulated_gradient(
            *_pair(pair.parameters),
            *_pair(pair.gradients),
            cast(WeightDecay, self.rule).l2_lambda,
            learning_rate,
            batch_size,
        )

    def _apply_momentum(
        self, index: int, pair: _Pair, learning_rate: float, batch_size: int
    ) -> tuple[pa.Array, pa.Array]:
        # state: a velocity per parameter
        state = self._pair_state(index, pair, 1)
        w, b, velocity_w, velocity_b = pa.layer_momentum_apply_accumulated_gradient(
            *_pair(pair.parameters),
            *_pair(pair.gradients),
            *_pair(state),
            cast(Momentum, self.rule).momentum,
            learning_rate,
            batch_size,
        )
        self._store_pair_state(index, pair, 1, [velocity_w, velocity_b])
        return w, b

    def _apply_adam(self, index: int, pair: _Pair, learning_rate: float, batch_size: int) -> tuple[pa.Array, pa.Array]:
        rule = cast(Adam, self.rule)
        # state: m, then v, per parameter
        state = self._pair_state(index, pair, 2)
        m_w, v_w, m_b, v_b = state if len(state) == 4 else (*state, _EMPTY, _EMPTY)
        w, b, m_w, v_w, m_b, v_b = pa.layer_adam_apply_accumulated_gradient(
            *_pair(pair.parameters),
            *_pair(pair.gradients),
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
        self._store_pair_state(index, pair, 2, [m_w, v_w, m_b, v_b])
        return w, b
