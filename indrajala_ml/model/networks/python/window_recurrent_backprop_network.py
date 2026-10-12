# pyright: reportConstantRedefinition=false
# (the states are named as in the array network, C, which strict mode takes for a constant)
"""
The window-recurrent network in pure Python (the window-recurrent adder workplan, stage 6),
WindowRecurrentShape's counterpart (window_recurrent_array_network.py): its window, a sequence
model's layers with StateInputs (recurrent_layer.StateInputLayer), run over each window of k
columns in turn, and its own parts, which hand a state from window to window: the attention
pooling over the tokens the output layer reads, the GRU cell and the codes, starting from the
learned initial state, snapped (W8).

    network = WindowRecurrentBackpropNetwork.randomized((k,), window_specs, WindowRecurrence(4, Codes(8)), Adam(), seed=7)

An input is W windows of k column token ids, flat, W * k of them, and its label the digit of each
column, W * k of them; every example of a batch has the same W. The loss, the training step and the
order of the weights are the array network's: the mean of every column's cross-entropy plus 1 / W
of each snapped state's code losses (W14); the windows forward, then backward from the last, each
window's forward pass re-run, its generator restored, before its backward pass (W20); the window's
layers, then the pooling, the cell, the codes and the initial state, for the optimizer, snapshot()
and format 2.

Every window runs through the layer-major batch path (layer_major.py), as a pure-Python network
with dropout trains, so the dropout layers draw numpy's masks, and the network draws the array
network's weights from a seed by bits. Its sums are folds and its products Python floats, so it
trains as the array networks do within their products' rounding.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar, Self

from indrajala_ml.model.layers.python.batch_norm_layer import fold
from indrajala_ml.model.layers.python.layer_major import LayerMajorBatch
from indrajala_ml.model.layers.python.recurrent_layer import (
    AttentionPoolingLayer,
    CodesLayer,
    GRULayer,
    InitialStateLayer,
    Rows,
    StateInputLayer,
)
from indrajala_ml.model.layers.python.token_layer import TokenSoftmaxLayer
from indrajala_ml.model.networks.python.backprop_network_base import BackpropNetworkBase, randomize_fan_in_aware
from indrajala_ml.model.persistence.format2 import WINDOW_RECURRENT, NetworkFile
from indrajala_ml.model.protocols.classification import argmax_first_occurrence
from indrajala_ml.model.protocols.layer_protocols import TrainableLayer
from indrajala_ml.model.specs.bounds import validate_batch, validate_class_count
from indrajala_ml.model.specs.layer_specs import Dense, LayerSpec
from indrajala_ml.model.specs.spec_shapes import InputShape, spec_shapes, token_shape
from indrajala_ml.model.specs.spec_validation import validate_window_recurrent
from indrajala_ml.model.specs.update_rules import UpdateRule
from indrajala_ml.model.specs.window_recurrence import WindowRecurrence
from indrajala_ml.pcg64 import generator_state, set_generator_state

Batch = Sequence[tuple[tuple[float, ...], Sequence[int]]]


class WindowRecurrentBackpropNetwork(BackpropNetworkBase[TrainableLayer, TokenSoftmaxLayer]):
    """
    The module's network: its window's layers (trainable_layers, built behind a StateLayer of k
    inputs) and its own parts (parts). It saves and loads in format 2 only, with the shape
    "window_recurrent"; each part's entry in the weights is its rows', [weights] or [weights, bias]
    per row, as a layer's (format2.py).
    """

    format2_shape: ClassVar[str] = WINDOW_RECURRENT
    preset_arguments: ClassVar[tuple[str, ...] | None] = None

    def __init__(
        self,
        input_shape: InputShape,
        layers: Sequence[LayerSpec],
        recurrence: WindowRecurrence,
        update_rule: UpdateRule,
        input_bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        validate_window_recurrent(layers, recurrence, input_shape)
        output = layers[-1]
        assert isinstance(output, Dense)
        validate_class_count(output.size)
        self.recurrence = recurrence
        self.update_rule = update_rule
        self.class_count = output.size
        self.columns = input_shape[0]
        # the window's token ids, (0, 1) per input as a sequence network's unless given
        bounds = [(0.0, 1.0)] * math.prod(input_shape) if input_bounds is None else input_bounds
        super().__init__(layers, input_shape, bounds)

        # the token width the output layer reads, which the pooling reads too
        _, features = token_shape(spec_shapes(layers, input_shape)[-1].input_shape)
        s = recurrence.state_size
        self.pooling = AttentionPoolingLayer(self.columns, features)
        self.cell = GRULayer(features, s, recurrence.cell.update_bias)
        self.codes = CodesLayer(recurrence.codes.count, s, recurrence.codes.commitment)
        self.initial_state = InitialStateLayer(s)
        self.parts: list[Any] = [self.pooling, self.cell, self.codes, self.initial_state]
        self._state_inputs = [layer for layer in self.trainable_layers if isinstance(layer, StateInputLayer)]

    def _window_state_size(self) -> int | None:
        return self.recurrence.state_size

    def _update_rule(self) -> UpdateRule:
        return self.update_rule

    def randomize(self) -> None:
        # the window's layers as a sequence network's, its StateInputs' P among them in order;
        # then the cell's gates and the codes. The pooling's query and the initial state start at
        # zero and draw nothing.
        randomize_fan_in_aware(self)
        self.cell.draw(self.rng)
        self.codes.draw(self.rng)

    # the forward pass

    def _windows(self, states: Sequence[Sequence[float]]) -> int:
        widths = {len(state) for state in states}
        assert len(widths) == 1, f"every example of a batch has the same window count; got widths {sorted(widths)}"
        count, remainder = divmod(widths.pop(), self.columns)
        assert remainder == 0 and count >= 1, (
            f"a window-recurrent network reads whole windows of {self.columns} columns; got {len(states[0])} columns"
        )
        return count

    def _window(self, states: Sequence[Sequence[float]], w: int) -> list[tuple[float, ...]]:
        k = self.columns
        return [tuple(state[w * k : (w + 1) * k]) for state in states]

    def _first_state(self, examples: int) -> Rows:
        return self.codes.forward_batch(self.initial_state.forward_batch(examples))

    def _window_forward(self, window: Sequence[tuple[float, ...]], C: Rows) -> tuple[LayerMajorBatch, Rows]:
        # one window's pass through its layers, and its next state, from its columns and state C
        for state_input in self._state_inputs:
            state_input.states = C
        batch = LayerMajorBatch(self.input_layer, self.trainable_layers, window)
        batch.forward()
        u = self.pooling.forward_batch(batch.values(len(self.trainable_layers) - 2))
        return batch, self.codes.forward_batch(self.cell.forward_batch(u, C))

    def forward_windows(self, states: Sequence[Sequence[float]]) -> list[list[float]]:
        """The inference forward pass: each column's digit probabilities, W * k * classes per example."""
        C = self._first_state(len(states))
        outputs: list[list[float]] = [[] for _ in states]
        for w in range(self._windows(states)):
            batch, C = self._window_forward(self._window(states, w), C)
            for row, output in zip(outputs, batch.outputs(), strict=True):
                row.extend(output)
        return outputs

    def _forward(self, state: tuple[float, ...]) -> list[float]:
        return self.forward_windows([state])[0]

    def predict_probabilities(self, state: tuple[float, ...]) -> list[list[float]]:
        """Each column's digit probabilities."""
        values = self._forward(state)
        width = self.class_count
        return [values[i : i + width] for i in range(0, len(values), width)]

    def classify_state(self, state: tuple[float, ...]) -> tuple[int, ...]:
        return tuple(argmax_first_occurrence(column) for column in self.predict_probabilities(state))

    # the training step

    def learn(self, learning_rate: float, state: tuple[float, ...], category: Sequence[int]) -> None:
        self.learn_batch(learning_rate, [(state, category)])

    def learn_batch(self, learning_rate: float, batch: Batch) -> None:
        validate_batch(batch)
        self._accumulate_batch_gradients([state for state, _ in batch], [label for _, label in batch])
        optimizer = self.optimizer
        optimizer.begin_step()
        for index, layer in enumerate([*self.trainable_layers, *self.parts]):
            optimizer.apply(index, layer, learning_rate, len(batch))

    def _accumulate_batch_gradients(self, states: Sequence[Sequence[float]], labels: Sequence[Sequence[int]]) -> None:
        # every layer's and part's gradient of the batch's loss, accumulated (W14, W20)
        windows = self._windows(states)
        for label in labels:
            assert len(label) == windows * self.columns, (
                f"a label is a digit per column, {windows * self.columns}; got {len(label)}"
            )
        weight = 1.0 / windows
        C_by_window: list[Rows] = []
        generators: list[dict[str, Any]] = []
        self._set_training_mode(True)
        try:
            C = self._first_state(len(states))
            for w in range(windows):
                C_by_window.append(C)
                generators.append(generator_state(self.rng))
                _, C = self._window_forward(self._window(states, w), C)
            after = generator_state(self.rng)

            dC: Rows = [[0.0] * self.recurrence.state_size for _ in states]
            for w in reversed(range(windows)):
                set_generator_state(self.rng, generators[w])
                batch, _ = self._window_forward(self._window(states, w), C_by_window[w])
                k = self.columns
                dC = self._window_backward(batch, [label[w * k : (w + 1) * k] for label in labels], dC, weight)
            set_generator_state(self.rng, after)
        finally:
            self._set_training_mode(False)

        self._first_state(len(states))
        self.initial_state.backward_batch(self.codes.backward_batch(dC, weight))

    def _window_backward(
        self, batch: LayerMajorBatch, labels: Sequence[Sequence[int]], dC_next: Rows, weight: float
    ) -> Rows:
        # one window's backward pass after its forward pass: its gradients accumulated, and its
        # state's gradient returned, from the next state's dC_next and its digits' labels
        dU, dC = self.cell.backward_batch(self.codes.backward_batch(dC_next, weight))
        pooled = self.pooling.backward_batch(dU)

        width = self.class_count

        def output_deltas(example: int) -> None:
            # one-hot per column, then the mean over every window's columns: 1 / W of each
            # window's mean over its own
            label = labels[example]
            for i, node in enumerate(self.output_layer.nodes):
                t, c = divmod(i, width)
                node.compute_output_delta(1.0 if c == label[t] else 0.0)
                node.delta = node.delta * weight

        batch.backward(output_deltas, pooled)
        batch.accumulate_gradients()

        for state_input in self._state_inputs:
            dC = [[a + b for a, b in zip(row, delta, strict=True)] for row, delta in zip(dC, state_input.state_deltas)]
        return dC

    def loss_batch(self, states: Sequence[Sequence[float]], categories: Sequence[Sequence[int]]) -> float:
        """
        The batch's digits' loss, summed over the examples, each the mean of its columns'
        cross-entropies, in inference: the loss whose gradient a training step takes, but for
        the codes' losses.
        """
        width = self.class_count
        losses = [
            -math.log(output[t * width + label])
            for output, labels in zip(self.forward_windows(states), categories, strict=True)
            for t, label in enumerate(labels)
        ]
        return fold(losses) / len(categories[0])

    # the weights: the window's layers', then the parts'

    def snapshot(self) -> list[list[Any]]:
        return [*super().snapshot(), *(part.snapshot_state() for part in self.parts)]

    def restore(self, snapshot: list[list[Any]]) -> None:
        count = len(self.trainable_layers)
        assert len(snapshot) == count + len(self.parts), (
            f"a window-recurrent network restores {count} layers and {len(self.parts)} parts; got {len(snapshot)} entries"
        )
        super().restore(snapshot[:count])
        for part, entry in zip(self.parts, snapshot[count:]):
            part.restore_state(entry)

    @classmethod
    def _from_file(cls, file: NetworkFile) -> Self:
        assert file.recurrence is not None
        return cls(file.input_shape, file.layers, file.recurrence, file.update_rule, file.input_bounds)

    @classmethod
    def _load_legacy(cls, state: dict[str, Any]) -> Self:
        raise ValueError(f"{cls.__name__} saves in format 2 only; this file has format {state.get('format')!r}")
