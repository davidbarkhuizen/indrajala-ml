"""
The Rust networks' optimizer (array_optimizer_base.py): each rule one fused call per pair of a
layer's parameters.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import indrajala_math_rust as pa

from indrajala_ml.model.layers.forward_context import NO_CONTEXT, ForwardContext
from indrajala_ml.model.layers.rust.rust_array_layer import RustArrayLayer
from indrajala_ml.model.optimizers.array_optimizer_base import ArrayOptimizerBase
from indrajala_ml.model.protocols.array_protocols import ArrayBackend, ArrayNetworkLayer, TrainedArrayLayer
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay

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
    A layer's parameters first and first + 1, or first alone, with their gradients: what one fused
    call steps. Most layers are one pair, its first 0; attention's eight parameters are its four
    projections' (W, b) pairs, and the GRU cell's (W, U, b) are W alone, then (U, b).
    """

    layer: TrainedArrayLayer[pa.Array]
    first: int
    parameters: Sequence[pa.Array]
    gradients: Sequence[pa.Array]
    # whether WeightDecay and AdamW decay the pair's first parameter: the fused ops never decay the
    # second, as a bias, so _pairs gives a pair a second parameter only when it isn't decayed
    decayed: bool


def _pairs(layer: TrainedArrayLayer[pa.Array], scale: float | None) -> list[_Pair]:
    # scale: the clip factor the gradients are read times during a step that clips
    # (OptimizerBase.step), None otherwise. A parameter takes the next one as its pair's second
    # only when the next isn't decayed (a bias, beta), since the fused ops step the second as a
    # bias; a decayed next parameter (the GRU's U after its W) starts a pair of its own, so every
    # parameter is decayed by its own flag, as NumpyOptimizer decays it
    parameters, gradients = layer.parameters(), layer.gradients()
    if scale is not None:
        gradients = tuple(gradient * scale for gradient in gradients)
    decayed = layer.decayed
    pairs: list[_Pair] = []
    first = 0
    while first < len(parameters):
        end = first + 2 if first + 1 < len(parameters) and not decayed[first + 1] else first + 1
        pairs.append(_Pair(layer, first, parameters[first:end], gradients[first:end], decayed[first]))
        first = end
    return pairs


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
            for pair in _pairs(trained, self._gradient_scale)
            for parameter in self._apply_rule(index, pair, learning_rate, batch_size)[: len(pair.parameters)]
        ]
        trained.set_parameters(parameters)
        trained.reset_gradient_accum()

    def _squared_gradient(self, layer: ArrayNetworkLayer[pa.Array]) -> float:
        if not hasattr(layer, "parameters"):
            return 0.0  # a pool layer
        squares = 0.0
        for gradient in cast("TrainedArrayLayer[pa.Array]", layer).gradients():
            flat = gradient.reshape((1, math.prod(gradient.shape)))
            squares += (flat @ flat.T)[0, 0]
        return squares

    def step_single(
        self,
        index: int,
        layer: ArrayNetworkLayer[pa.Array],
        input_activation: pa.Array,
        learning_rate: float,
        *,
        context: ForwardContext = NO_CONTEXT,
    ) -> None:
        if self._fused_sgd_step and isinstance(layer, RustArrayLayer):
            # accumulate then apply at batch_size=1 as one fused call, bit-identical to that pair
            # (tests/model/test_rust_array_layer_sgd_step.py). It relies on the accumulators being fresh
            # zeros, which they always are here: apply resets them after every step, and this
            # doesn't touch them. Dense layers only: a conv gradient sums over output positions,
            # which layer_sgd_step's outer product doesn't.
            layer.W, layer.b = pa.layer_sgd_step(layer.W, layer.b, layer.delta, input_activation, learning_rate)
            return
        layer.accumulate_gradient(input_activation, context=context)
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
            # AdamW's decay on the pair's weight; the op never decays its b (D9)
            weight_decay=rule.weight_decay if pair.decayed else 0.0,
        )
        self._store_pair_state(index, pair, 2, [m_w, v_w, m_b, v_b])
        return w, b
