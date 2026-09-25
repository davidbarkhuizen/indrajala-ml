"""
The structural interfaces of the pure-Python networks' layers: BackpropLayer and its siblings,
ConvLayer and MaxPoolLayer satisfy them without a common base class.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from indrajala_ml.model.base_node import AbstractNode


class InputLayer(Protocol):
    """What a layer reads its input from: a StateLayer, or any layer before it."""

    @property
    def nodes(self) -> Sequence[AbstractNode]: ...


class TrainableLayer(InputLayer, Protocol):
    """What BackpropNetworkBase drives layer by layer: the forward and backward passes and training."""

    def forward(self) -> None: ...

    def set_training_mode(self, training: bool) -> None: ...

    # the next layer's own type: a dense layer reads its nodes, a conv or pool layer its
    # downstream_sum
    def compute_hidden_deltas(self, next_layer: Any) -> None: ...

    def downstream_sum(self, own_index: int) -> float: ...

    def apply_gradients(self, learning_rate: float) -> None: ...

    def accumulate_gradients(self) -> None: ...

    def apply_accumulated_gradients(self, learning_rate: float, batch_size: int) -> None: ...

    # a dense or conv layer's (weights, bias) per node or kernel; empty for a pool layer
    def snapshot_state(self) -> list[Any]: ...

    def restore_state(self, layer_snapshot: Any) -> None: ...
