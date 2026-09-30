"""
The pure-Python layer-major batch path (the batch-norm workplan, D3): a batch forward layer by
layer, then backward layer by layer, which batch norm needs, since each of its values depends on
the whole batch. Only a network with a BatchNorm layer takes it; every other network keeps
BackpropNetworkBase's example-major loop, so its bits don't change.

Every other layer runs its own per-example code unchanged: a layer's nodes hold one example's
state at a time (their example_fields: the activation, the delta, dropout's mask, a pool unit's
winning slot), so this path keeps each layer's per example, one "lane" per example, and loads a
lane back before the layer or its neighbour reads it. A BatchNormLayer keeps its own per-example
lists and is selected with select_example instead.

The gradients accumulate per weight in example order, as the example-major loop's do.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from indrajala_ml.model.batch_norm_layer import BatchNormLayer
from indrajala_ml.model.layer_protocols import TrainableLayer
from indrajala_ml.model.state_layer import StateLayer

# one layer's state for one example: each node's example_fields
Lane = list[dict[str, Any]]


def _capture(layer: TrainableLayer) -> Lane:
    lane: Lane = []
    for node in layer.nodes:
        fields = vars(node)
        lane.append({name: fields[name] for name in node.example_fields if name in fields})
    return lane


def _load(layer: TrainableLayer, lane: Lane) -> None:
    for node, fields in zip(layer.nodes, lane):
        vars(node).update(fields)


class LayerMajorBatch:
    """
    One training batch through layers (a network's trainable_layers) behind input_layer: forward(),
    then outputs() or backward() and accumulate_gradients(). The caller sets training mode.
    """

    def __init__(
        self, input_layer: StateLayer, layers: Sequence[TrainableLayer], states: Sequence[tuple[float, ...]]
    ) -> None:
        self.input_layer = input_layer
        self.layers = layers
        self.states = states
        # lanes[i][e]: layer i's state for example e; empty for a batch-norm layer, which keeps its own
        self.lanes: list[list[Lane]] = []

    def _select(self, index: int, example: int) -> None:
        # layer index's nodes (the input layer's at -1) take example's values
        if index < 0:
            self.input_layer.update_state(self.states[example])
            return
        layer = self.layers[index]
        if isinstance(layer, BatchNormLayer):
            layer.select_example(example)
        else:
            _load(layer, self.lanes[index][example])

    def forward(self) -> None:
        count = len(self.states)
        for index, layer in enumerate(self.layers):
            if isinstance(layer, BatchNormLayer):
                # the linear layer's z for every example, then the batch's statistics
                inputs: list[list[float]] = []
                for example in range(count):
                    self._select(index - 1, example)
                    inputs.append([node.value() for node in layer.input_layer.nodes])
                layer.forward_batch(inputs)
                self.lanes.append([])
                continue
            lanes: list[Lane] = []
            for example in range(count):
                self._select(index - 1, example)
                layer.forward()
                lanes.append(_capture(layer))
            self.lanes.append(lanes)

    def outputs(self) -> list[list[float]]:
        """The output layer's values, example by example."""
        last = len(self.layers) - 1
        rows: list[list[float]] = []
        for example in range(len(self.states)):
            self._select(last, example)
            rows.append([node.value() for node in self.layers[last].nodes])
        return rows

    def backward(self, output_deltas: Callable[[int], None]) -> None:
        """
        The deltas of every layer, output_deltas(e) setting the output layer's for example e (the
        network's shape: a one-hot class or a float target).
        """
        count = len(self.states)
        last = len(self.layers) - 1
        for example in range(count):
            self._select(last, example)
            output_deltas(example)
            self.lanes[last][example] = _capture(self.layers[last])

        for index in reversed(range(last)):
            layer, next_layer = self.layers[index], self.layers[index + 1]
            if isinstance(layer, BatchNormLayer):
                downstream: list[list[float]] = []
                for example in range(count):
                    self._select(index + 1, example)
                    downstream.append([next_layer.downstream_sum(j) for j in range(layer.size)])
                layer.backward_batch(downstream)
                continue
            for example in range(count):
                self._select(index + 1, example)
                self._select(index, example)
                layer.compute_hidden_deltas(next_layer)
                self.lanes[index][example] = _capture(layer)

    def accumulate_gradients(self) -> None:
        for index, layer in enumerate(self.layers):
            if isinstance(layer, BatchNormLayer):
                layer.accumulate_gradients()
                continue
            for example in range(len(self.states)):
                self._select(index - 1, example)
                self._select(index, example)
                layer.accumulate_gradients()
