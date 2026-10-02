"""
The structural interfaces of the pure-Python networks' layers: BackpropLayer and its siblings,
ConvLayer and MaxPoolLayer satisfy them without a common base class.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Protocol, runtime_checkable

from indrajala_ml.model.base_node import AbstractNode
from indrajala_ml.pcg64 import Pcg64Generator


class InputLayer(Protocol):
    """What a layer reads its input from: a StateLayer, or any layer before it."""

    @property
    def nodes(self) -> Sequence[AbstractNode]: ...


class WeightSet(Protocol):
    """
    One BackpropNode's or ConvKernel's trained weights and bias, with their gradients accumulated
    over a batch: what the pure-Python optimizer (python_optimizer.py) steps. weights is read-only
    here, since a node rebinds its weights through set_weights. A batch-norm feature's weights are
    [gamma] and its bias is beta.
    """

    # weight decay applies to the weights (not to batch norm's gamma, the batch-norm workplan's D7),
    # and there is a bias to step (not a linear layer's node, D2)
    weights_decayed: ClassVar[bool]
    has_bias: ClassVar[bool]

    weight_gradient_accum: list[float]
    bias_gradient_accum: float

    # a property, which a node's bias is and a kernel's plain attribute satisfies
    @property
    def bias(self) -> float: ...

    @bias.setter
    def bias(self, value: float) -> None: ...

    @property
    def weights(self) -> Sequence[float]: ...

    def set_weights(self, weights: list[float]) -> None: ...

    def reset_gradient_accum(self) -> None: ...


class TrainableLayer(InputLayer, Protocol):
    """What BackpropNetworkBase drives layer by layer: the forward and backward passes and training."""

    def forward(self) -> None: ...

    def set_training_mode(self, training: bool) -> None: ...

    # the next layer's own type: a dense layer reads its nodes, a conv or pool layer its
    # downstream_sum
    def compute_hidden_deltas(self, next_layer: Any) -> None: ...

    def downstream_sum(self, own_index: int) -> float: ...

    def accumulate_gradients(self) -> None: ...

    # what the optimizer steps: a dense layer's nodes, a conv layer's kernels, none for a pool layer
    def weight_sets(self) -> Sequence[WeightSet]: ...

    # fan_in_aware_weights_and_bias per node or kernel, in order; a pool layer draws nothing
    def randomize_fan_in_aware(self, rng: Pcg64Generator) -> None: ...

    # a dense or conv layer's (weights, bias) per node or kernel, a linear layer's (weights,) per node,
    # a batch-norm layer's ([gamma], beta, running_mean, running_var) per feature; empty for a pool
    # layer
    def snapshot_state(self) -> list[Any]: ...

    def restore_state(self, layer_snapshot: Any) -> None: ...


@runtime_checkable
class GeneratorLayer(Protocol):
    """
    A layer that draws in training (dropout's masks), pure-Python or array, from the generator its
    network owns: the network's rng setter hands it over (the RNG generators workplan, D8).
    """

    def set_rng(self, rng: Any, /) -> None: ...
