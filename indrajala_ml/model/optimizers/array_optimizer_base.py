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

from typing import Any

from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.optimizers.optimizer_base import OptimizerBase
from indrajala_ml.model.persistence.checkpoint import OptimizerState
from indrajala_ml.model.protocols.array_protocols import ArrayBackend, BackendArray, TrainedArrayLayer
from indrajala_ml.model.specs.update_rules import UpdateRule


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
