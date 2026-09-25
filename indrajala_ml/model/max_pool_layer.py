from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from indrajala_ml.model.base_node import AbstractNode
from indrajala_ml.model.layer_protocols import InputLayer, TrainableLayer


@dataclass(frozen=True)
class PoolSpec:
    """
    One MaxPoolLayer's hyperparameters; stride defaults to pool_size (non-overlapping windows). The
    input shape comes from the previous layer.
    """

    pool_size: int
    stride: int | None = None


class PoolUnit(AbstractNode):
    """
    One max-pooling output position, with no weights. forward() records which window slot held the
    maximum (the first, on a tie); the unit's delta flows back to that input only, since
    d max(x) / d x_i is 0 for every other slot.
    """

    def __init__(self, input_nodes: Sequence[AbstractNode]) -> None:
        self.input_nodes: Sequence[AbstractNode] = input_nodes

        # set by forward()/compute_hidden_delta(); no default, as in BackpropNode
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
    input_width, the layout ConvLayer reads and writes, so either can follow the other. Each
    channel is pooled separately (channel_count == input_channels); .nodes is channel-major.

    No weights, but it implements ConvLayer's layer methods: forward and the backward hooks do work,
    the gradient and snapshot methods are no-ops (snapshot_state() is []). It sits in
    trainable_layers because the forward pass and snapshot/restore both walk that list.

    downstream_sum(i) uses ConvLayer's reverse map, from each input to the (unit, slot) pairs that
    read it, counting a unit's delta only if that slot won. With overlapping windows (stride <
    pool_size) an input can win several windows and receives each one's delta.
    """

    def __init__(
        self,
        input_layer: InputLayer,
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
            f"pool_size ({pool_size}) must fit within input_height x input_width ({input_height}x{input_width})"
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

    def compute_hidden_deltas(self, next_layer: TrainableLayer) -> None:
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

    def snapshot_state(self) -> list[tuple[list[float], float]]:
        return []

    def restore_state(self, layer_snapshot: list[tuple[list[float], float]]) -> None:
        assert layer_snapshot == [], f"a MaxPoolLayer has no state to restore; got {layer_snapshot!r}"

    def randomize_fan_in_aware(self) -> None:
        pass
