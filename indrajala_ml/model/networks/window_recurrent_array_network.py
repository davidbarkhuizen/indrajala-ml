# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, X, Y, H, which strict mode takes for constants)
"""
A window-recurrent network on either array backend (the window-recurrent adder workplan, stages 4
and 5; The network):
its window, a sequence model's layers with StateInputs, run over each window of k columns in turn,
and its own parts, which hand a state from window to window: the attention pooling over the tokens
the output layer reads, the GRU cell and the codes, starting from the learned initial state,
snapped (W8). Without codes (WindowRecurrence(s, None)) the state is continuous: the network has
no codes part, the cell's output is the next state, and the codes' losses are gone (the
window-recurrent claims workplan, stage 4).

    network = WindowRecurrentArrayNetwork((k,), window_specs, WindowRecurrence(4, Codes(8)), Adam())
    network = WindowRecurrentRustArrayNetwork((k,), window_specs, WindowRecurrence(4, Codes(8)), Adam())

An input is W windows of k column token ids, flat, W * k of them, and its label the digit of each
column, W * k of them; every example of a batch has the same W, and the network takes any W.
Aligning the columns in their windows (W6-i) and the final window of zero columns (W9) are the
caller's (the harness, stage 8).

The loss per example (W14): the mean of every column's cross-entropy, over all W windows, plus, for
each snapped state (the initial one and each window's), 1 / W times the codes' losses. A training
step runs the windows forward, keeping each window's state and the generator's state before it,
then backward from the last window, re-running each window's forward pass (its generator
restored, so its dropout masks repeat) before its backward pass (W20); the gradient of a window's
state flows into the window before it, through the codes, the cell and the StateInputs.

The window's layers keep the indices they have in a sequence network; the network's own parts
follow them, in the order pooling, cell, codes (if any), initial state, for the optimizer,
snapshot() and format 2 (format2_json.recurrence_parameters).

WindowRecurrentShape is the network over ArrayNetworkBase, for either backend; its parts are the
backend's (recurrent_array_layer.py, recurrent_rust_array_layer.py), whose products are their
backend's, so the two networks agree to within those products' rounding.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, Self

import indrajala_math_rust as pa
import numpy as np

from indrajala_ml.data.prepared_dataset import CLASSIFY_CHUNK_ROWS, PreparedDataset
from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.layers.numpy import recurrent_array_layer as numpy_parts
from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.layers.rust import recurrent_rust_array_layer as rust_parts
from indrajala_ml.model.networks.numpy.numpy_array_network_base import NumpyArrayNetworkBase
from indrajala_ml.model.networks.rust.rust_array_network_base import RustArrayNetworkBase
from indrajala_ml.model.persistence.format2 import WINDOW_RECURRENT, NetworkFile
from indrajala_ml.model.protocols.array_protocols import BackendArray
from indrajala_ml.model.specs.bounds import validate_class_count
from indrajala_ml.model.specs.layer_specs import Dense, LayerSpec
from indrajala_ml.model.specs.spec_shapes import InputShape, spec_shapes, token_shape
from indrajala_ml.model.specs.spec_validation import validate_window_recurrent
from indrajala_ml.model.specs.update_rules import UpdateRule
from indrajala_ml.model.specs.window_recurrence import WindowRecurrence
from indrajala_ml.pcg64 import generator_state, set_generator_state

if TYPE_CHECKING:
    from indrajala_ml.model.networks.array_network_base import ArrayNetworkBase

    # the mixin subclasses ArrayNetworkBase for the type checker only, as array_network_shapes.py's
    # do, so the registry walks over ArrayNetworkBase's subclasses find the two networks, not it
    _ShapeBase = ArrayNetworkBase
else:

    class _ShapeBase[A]:
        pass


class _Parts(NamedTuple):
    """A backend's classes of the network's parts, and of the StateInput its window holds."""

    pooling: Any
    cell: Any
    codes: Any
    initial_state: Any
    state_input: Any


_PARTS = {
    NUMPY.name: _Parts(
        numpy_parts.AttentionPoolingArrayLayer,
        numpy_parts.GRUArrayLayer,
        numpy_parts.CodesArrayLayer,
        numpy_parts.InitialStateArrayLayer,
        numpy_parts.StateInputArrayLayer,
    ),
    RUST.name: _Parts(
        rust_parts.AttentionPoolingRustArrayLayer,
        rust_parts.GRURustArrayLayer,
        rust_parts.CodesRustArrayLayer,
        rust_parts.InitialStateRustArrayLayer,
        rust_parts.StateInputRustArrayLayer,
    ),
}


class _WithPooling:
    """
    The output layer as the layer before it reads it, with the pooling's gradient of the same
    tokens added: both read the tokens the output layer reads. Every layer that may stand before
    a window's token-wise output layer reads only its downstream_batch(), on either backend.
    """

    def __init__(self, output_layer: Any, pooled: Any) -> None:
        self._output_layer = output_layer
        self._pooled = pooled

    def downstream_batch(self) -> Any:
        return self._output_layer.downstream_batch() + self._pooled


class WindowRecurrentShape[A: BackendArray](_ShapeBase[A]):
    """
    The module's network over ArrayNetworkBase, for either backend: its window's layers
    (self.layers) and its own parts (self.parts). A mixin, listed before the backend's base.
    """

    format2_shape: ClassVar[str] = WINDOW_RECURRENT
    preset_arguments: ClassVar[tuple[str, ...] | None] = None

    def __init__(
        self,
        input_shape: InputShape,
        layers: Sequence[LayerSpec],
        recurrence: WindowRecurrence,
        update_rule: UpdateRule,
    ) -> None:
        validate_window_recurrent(layers, recurrence, input_shape)
        output = layers[-1]
        assert isinstance(output, Dense)
        validate_class_count(output.size)
        self.recurrence = recurrence
        self.update_rule = update_rule
        self.class_count = output.size
        self.columns = input_shape[0]
        super().__init__(layers, input_shape)

        # the token width the output layer reads, which the pooling reads too
        _, features = token_shape(spec_shapes(layers, input_shape)[-1].input_shape)
        s = recurrence.state_size
        parts = _PARTS[self.backend.name]
        self.pooling = parts.pooling(self.columns, features)
        self.cell = parts.cell(features, s, recurrence.cell.update_bias)
        codes = recurrence.codes
        self.codes: Any = None if codes is None else parts.codes(codes.count, s, codes.commitment)
        self.initial_state = parts.initial_state(s)
        self.parts: list[Any] = [
            part for part in (self.pooling, self.cell, self.codes, self.initial_state) if part is not None
        ]
        self._state_inputs: list[Any] = [layer for layer in self.layers if isinstance(layer, parts.state_input)]

    def _window_state_size(self) -> int | None:
        return self.recurrence.state_size

    def _update_rule(self) -> UpdateRule:
        return self.update_rule

    def randomize(self) -> None:
        # the window's layers as a sequence network's, its StateInputs' P among them in order;
        # then the cell's gates and the codes. The pooling's query and the initial state start at
        # zero and draw nothing.
        super().randomize()
        self.cell.draw(self.rng)
        if self.codes is not None:
            self.codes.draw(self.rng)

    # the forward pass

    def _snap(self, H: Any) -> Any:
        # the next state: H snapped to its code, or H itself for a continuous state
        return H if self.codes is None else self.codes.forward_batch(H)

    def _snap_backward(self, dQ: Any, weight: float) -> Any:
        # the next state's gradient to H's: straight through and the codes' losses', or dQ itself
        return dQ if self.codes is None else self.codes.backward_batch(dQ, weight)

    def _snapped(self) -> Any:
        assert self.codes is not None, "a continuous state has no codes (WindowRecurrence(s, None))"
        return self.codes

    def _windows(self, X: Any) -> int:
        count, remainder = divmod(X.shape[1], self.columns)
        assert remainder == 0 and count >= 1, (
            f"a window-recurrent network reads whole windows of {self.columns} columns; got {X.shape[1]} columns"
        )
        return count

    def _window(self, X: Any, w: int) -> Any:
        return self.backend.columns(X, w * self.columns, (w + 1) * self.columns)

    def _first_state(self, examples: int) -> Any:
        return self._snap(self.initial_state.forward_batch(examples))

    def _window_forward(self, X: Any, C: Any) -> tuple[Any, Any]:
        # one window's digits' probabilities and its next state, from its columns X and state C
        for state_input in self._state_inputs:
            state_input.set_state(C)
        self._activations: list[Any] = [X]
        for layer in self.layers:
            self._activations.append(layer.forward_batch(self._activations[-1]))
        u = self.pooling.forward_batch(self._activations[-2])
        return self._activations[-1], self._snap(self.cell.forward_batch(u, C))

    def forward_windows(self, X: A) -> A:
        """The inference forward pass: each column's digit probabilities, (N, W * k * classes)."""
        C = self._first_state(X.shape[0])
        outputs: list[Any] = []
        for w in range(self._windows(X)):
            output, C = self._window_forward(self._window(X, w), C)
            outputs.append(output)
        return self.backend.concatenate_columns(outputs)

    def state_codes(self, X: A) -> list[list[int]]:
        """
        Each state's code, per example: the initial state's, then each window's next state's, W + 1
        of them (the report's codes used, W4-i). A snapped state's only.
        """
        snapped = self._snapped()
        C = self._first_state(X.shape[0])
        codes = [[int(i) for i in snapped.indices]]
        for w in range(self._windows(X)):
            _, C = self._window_forward(self._window(X, w), C)
            codes.append([int(i) for i in snapped.indices])
        return [list(example) for example in zip(*codes, strict=True)]

    def first_step(self) -> tuple[A, list[int]]:
        """The initial state before it is snapped (1, state_size), and the code it snaps to."""
        snapped = self._snapped()
        self._first_state(1)
        return snapped._H, [int(i) for i in snapped.indices]

    def window_step(self, X: A, C: A) -> tuple[A, A, list[int]]:
        """
        One window in inference, from its columns X (N, k) and states C (N, state_size): its digits'
        probabilities (N, k * classes), the cell's output before it is snapped (N, state_size) and
        the codes it snaps to (the window-recurrent claims workplan's certificate, E1).
        """
        snapped = self._snapped()
        output, _ = self._window_forward(X, C)
        return output, snapped._H, [int(i) for i in snapped.indices]

    def _forward_input(self, x: A) -> A:
        return self.backend.row(self.forward_windows(self.backend.matrix([x.tolist()])), 0)

    def forward_rows(self, prepared: PreparedDataset) -> Iterator[A]:
        states = self._prepared_states(prepared)
        for start in range(0, len(prepared), CLASSIFY_CHUNK_ROWS):
            yield self.forward_windows(
                self.backend.row_range(states, start, min(start + CLASSIFY_CHUNK_ROWS, len(prepared)))
            )

    # the shape's hooks: a digit per column, as a sequence network's per token

    def predict_probabilities(self, state: tuple[float, ...]) -> list[list[float]]:
        """Each column's digit probabilities."""
        values = self._forward(state).tolist()
        width = self.class_count
        return [values[i : i + width] for i in range(0, len(values), width)]

    def classify_state(self, state: tuple[float, ...]) -> tuple[int, ...]:
        return self._classify_output(self._forward(state))

    def _classify_output(self, output: A) -> tuple[int, ...]:
        return self._classify_output_batch(self.backend.matrix([output.tolist()]))[0]

    def _classify_output_batch(self, output_batch: A) -> list[tuple[int, ...]]:
        return self.backend.argmax_token_rows(output_batch, self.class_count)

    def _target_batch_array(self, categories: Sequence[Sequence[int]]) -> A:
        columns = len(categories[0])
        target_batch = self.backend.zeros((len(categories), columns * self.class_count))
        for row, category in enumerate(categories):
            assert len(category) == columns, (
                f"a batch's labels are a digit per column, {columns} for every example; got {len(category)}"
            )
            for t, label in enumerate(category):
                target_batch[row, t * self.class_count + label] = 1.0
        return target_batch

    def _target_array(self, category: Sequence[int]) -> A:
        return self.backend.row(self._target_batch_array([category]), 0)

    # the training step

    def _learn_input(self, learning_rate: float, x: A, category: Any) -> None:
        self._learn_batch_input(learning_rate, self.backend.matrix([x.tolist()]), [category])

    def _learn_batch_input(self, learning_rate: float, X: A, categories: Sequence[Any]) -> None:
        examples = len(categories)
        self._accumulate_gradients(X, self._target_batch_array(categories))
        optimizer = self.optimizer
        optimizer.begin_step()
        for index, layer in enumerate([*self.layers, *self.parts]):
            optimizer.apply(index, layer, learning_rate, examples)

    def _accumulate_gradients(self, X: Any, Y: Any) -> None:
        # every layer's and part's gradient of the batch's loss, accumulated (W14, W20)
        windows = self._windows(X)
        weight = 1.0 / windows
        width = self.columns * self.class_count
        states: list[Any] = []
        generators: list[dict[str, Any]] = []
        self._set_training_mode(True)
        try:
            C = self._first_state(X.shape[0])
            for w in range(windows):
                states.append(C)
                generators.append(generator_state(self.rng))
                _, C = self._window_forward(self._window(X, w), C)
            after = generator_state(self.rng)

            dC = self.backend.zeros((X.shape[0], self.recurrence.state_size))
            for w in reversed(range(windows)):
                set_generator_state(self.rng, generators[w])
                self._window_forward(self._window(X, w), states[w])
                dC = self._window_backward(self.backend.columns(Y, w * width, (w + 1) * width), dC, weight)
            set_generator_state(self.rng, after)
        finally:
            self._set_training_mode(False)

        self._first_state(X.shape[0])
        self.initial_state.backward_batch(self._snap_backward(dC, weight))

    def _window_backward(self, Y: Any, dC_next: Any, weight: float) -> Any:
        # one window's backward pass after its forward pass: its gradients accumulated, and its
        # state's gradient returned, from the next state's dC_next and its digits' targets Y
        dU, dC = self.cell.backward_batch(self._snap_backward(dC_next, weight))
        pooled = self.pooling.backward_batch(dU)

        output_layer: Any = self.layers[-1]
        output_layer.compute_output_delta_batch(Y)
        # the mean over every window's columns: each window's mean over its own, times 1 / W
        output_layer.delta_batch = output_layer.delta_batch * weight
        self.layers[-2].compute_hidden_delta_batch(_WithPooling(output_layer, pooled))
        for i in reversed(range(len(self.layers) - 2)):
            self.layers[i].compute_hidden_delta_batch(self.layers[i + 1])
        for layer, input_activation_batch in zip(self.layers, self._activations):
            layer.accumulate_gradient_batch(input_activation_batch)

        for state_input in self._state_inputs:
            dC = dC + state_input.state_delta
        return dC

    def loss_batch(self, X: A, categories: Sequence[Sequence[int]]) -> float:
        """
        The batch's digits' loss, summed over the examples, each the mean of its columns'
        cross-entropies, in inference: the loss whose gradient a training step takes, but for
        the codes' losses.
        """
        P = np.array(self.forward_windows(X).tolist()).reshape(len(categories), -1, self.class_count)
        labels = np.array(categories)
        chosen = np.take_along_axis(P, labels[:, :, np.newaxis], axis=2)[:, :, 0]
        return float(-np.sum(np.log(chosen)) / labels.shape[1])

    # the weights: the window's layers', then the parts'

    def snapshot(self) -> list[tuple[A, ...]]:
        return [*super().snapshot(), *(tuple(array.copy() for array in part.parameters()) for part in self.parts)]

    def restore(self, snapshot: Sequence[tuple[Any, ...]]) -> None:
        count = len(self.layers)
        assert len(snapshot) == count + len(self.parts), (
            f"a window-recurrent network restores {count} layers and {len(self.parts)} parts; got {len(snapshot)} entries"
        )
        super().restore(snapshot[:count])
        for part, entry in zip(self.parts, snapshot[count:]):
            expected = len(part.parameters())
            assert len(entry) == expected, f"a {type(part).__name__} restores {expected} arrays; got {entry!r}"
            part.set_parameters([self.backend.owned(values) for values in entry])

    @classmethod
    def _from_file(cls, file: NetworkFile) -> Self:
        assert file.recurrence is not None
        return cls(file.input_shape, file.layers, file.recurrence, file.update_rule)


class WindowRecurrentArrayNetwork(WindowRecurrentShape[FloatArray], NumpyArrayNetworkBase):
    """WindowRecurrentShape on the numpy backend."""


class WindowRecurrentRustArrayNetwork(WindowRecurrentShape[pa.Array], RustArrayNetworkBase):
    """WindowRecurrentShape on the Rust backend."""
