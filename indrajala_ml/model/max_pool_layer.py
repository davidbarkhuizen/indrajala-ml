from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from indrajala_ml.model.base_node import AbstractNode


@dataclass(frozen=True)
class PoolSpec:
    """One MaxPoolLayer's own hyperparameters - stride defaults to pool_size (non-overlapping
    windows), the standard choice. The input shape comes from the previous layer, as for
    ConvSpec (see ConvMultiClassBackpropClassifierNetwork)."""

    pool_size: int
    stride: int | None = None


class PoolUnit(AbstractNode):
    """
    One max-pooling output position - no weights, no bias, no activation function. forward()
    caches which window slot held the maximum (the first one, on an exact tie), and that slot is
    the only input this unit's delta flows back to: d max(x) / d x_i is 1 for the argmax and 0
    for every other slot, since nudging a non-maximal input doesn't change the output at all.
    """

    def __init__(self, input_nodes: Sequence[AbstractNode]) -> None:
        self.input_nodes: Sequence[AbstractNode] = input_nodes

        # populated by forward()/compute_hidden_delta() - see BackpropNode's own identical
        # convention (backprop_node.py) for why these have no default
        self._activation: float
        self.argmax_slot: int
        self.delta: float

    def forward(self) -> float:
        values = [node.value() for node in self.input_nodes]
        self._activation = max(values)
        self.argmax_slot = values.index(self._activation)
        return self._activation

    def value(self) -> float:
        return self._activation

    def compute_hidden_delta(self, downstream_sum: float) -> None:
        # max is the identity on its winning input, so the delta is the downstream sum itself -
        # no activation derivative to multiply in
        self.delta = downstream_sum


class MaxPoolLayer:
    """
    A max-pooling hidden layer over input_channels channel-major planes of input_height x
    input_width (the same layout ConvLayer reads and produces, so either can follow the
    other). Each channel is pooled independently - channel_count == input_channels, and .nodes
    is channel-major like ConvLayer's own.

    Weight-free, but implements the same duck-typed layer surface ConvLayer does, so
    BackpropNetworkBase's generic machinery drives it unchanged: forward and the two backward
    hooks (compute_hidden_deltas/downstream_sum) do real work; every gradient/persistence hook
    is a no-op, and snapshot_state() is an empty list - it still sits in trainable_layers, since
    both the forward pass and snapshot/restore iterate over that one list.

    downstream_sum(i) uses the same reverse-map shape ConvLayer's does - input index -> every
    (unit, window slot) pair that reads it - but a unit only contributes its delta when that
    slot won its last forward pass. With overlapping windows (stride < pool_size) one input can
    win several windows, and then receives every one of their deltas.
    """

    def __init__(
        self,
        input_layer,
        input_height: int,
        input_width: int,
        input_channels: int,
        pool_size: int,
        stride: int | None = None,
    ) -> None:

        stride = pool_size if stride is None else stride

        assert pool_size >= 1, f"pool_size must be at least 1; got {pool_size}"
        assert stride >= 1, f"stride must be at least 1; got {stride}"
        assert input_channels >= 1, f"input_channels must be at least 1; got {input_channels}"
        assert input_channels * input_height * input_width == len(input_layer.nodes), (
            f"input_channels*input_height*input_width "
            f"({input_channels * input_height * input_width}) must match input_layer's own node "
            f"count ({len(input_layer.nodes)})"
        )
        assert pool_size <= input_height and pool_size <= input_width, (
            f"pool_size ({pool_size}) must fit within input_height x input_width "
            f"({input_height}x{input_width})"
        )

        self.input_layer = input_layer
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.pool_size = pool_size
        self.stride = stride
        self.channel_count = input_channels

        self.out_height = (input_height - pool_size) // stride + 1
        self.out_width = (input_width - pool_size) // stride + 1

        self.nodes: list[PoolUnit] = []
        self._fan_out: list[list[tuple[PoolUnit, int]]] = [[] for _ in input_layer.nodes]
        for channel in range(input_channels):
            for row in range(self.out_height):
                for col in range(self.out_width):
                    indices = self._window_indices(channel, row, col)
                    unit = PoolUnit(input_nodes=[input_layer.nodes[i] for i in indices])
                    self.nodes.append(unit)
                    for slot, input_index in enumerate(indices):
                        self._fan_out[input_index].append((unit, slot))

    def _window_indices(self, channel: int, row: int, col: int) -> list[int]:
        plane = self.input_height * self.input_width
        return [
            channel * plane + (row * self.stride + pr) * self.input_width + (col * self.stride + pc)
            for pr in range(self.pool_size)
            for pc in range(self.pool_size)
        ]

    def forward(self) -> None:
        for unit in self.nodes:
            unit.forward()

    def compute_hidden_deltas(self, next_layer) -> None:
        for own_index, unit in enumerate(self.nodes):
            unit.compute_hidden_delta(next_layer.downstream_sum(own_index))

    def downstream_sum(self, own_index: int) -> float:
        return sum(unit.delta for unit, slot in self._fan_out[own_index] if unit.argmax_slot == slot)

    # weight-free: every gradient/persistence hook below is a deliberate no-op

    def set_training_mode(self, training: bool) -> None:
        pass

    def accumulate_gradients(self) -> None:
        pass

    def apply_accumulated_gradients(self, learning_rate: float, batch_size: int) -> None:
        pass

    def apply_gradients(self, learning_rate: float) -> None:
        pass

    def snapshot_state(self) -> list:
        return []

    def restore_state(self, layer_snapshot: list) -> None:
        assert layer_snapshot == [], f"a MaxPoolLayer has no state to restore; got {layer_snapshot!r}"

    def randomize_fan_in_aware(self) -> None:
        pass
