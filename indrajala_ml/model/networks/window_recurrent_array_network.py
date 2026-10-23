# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, X, Y, H, which strict mode takes for constants)
"""
A window-recurrent network on either array backend (the window-recurrent adder workplan, stages 4
and 5; The network):
its window, a sequence model's layers with StateInputs, run over each window of k columns in turn,
and its own parts, which hand a state from window to window: for each of the state's factors (the
factored-state workplan, stage 4), the attention pooling over the tokens the output layer reads,
the GRU cell and the codes, starting from the learned initial state, snapped (W8); the backend's
recurrence (RecurrenceArrayLayer, RecurrenceRustArrayLayer) runs them. A factor without codes
(Factor(s, None)) is continuous: it has no codes part, its cell's output is its next state, and its
codes' losses are gone (the window-recurrent claims workplan, stage 4). A machine's state, from
state_codes, first_step and window_step, is a tuple of codes, one per factor (D4).

    recurrence = WindowRecurrence((Factor(4, Codes(8)),))
    network = WindowRecurrentArrayNetwork((k,), window_specs, recurrence, Adam())
    network = WindowRecurrentRustArrayNetwork((k,), window_specs, recurrence, Adam())

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
follow them, factor by factor, each in the order pooling, cell, codes (if any), initial state, for
the optimizer, snapshot() and format 2 (format2_json.recurrence_parameters).

WindowRecurrentShape is the network over ArrayNetworkBase, for either backend; its parts are the
backend's (recurrent_array_layer.py, recurrent_rust_array_layer.py), whose products are their
backend's, so the two networks agree to within those products' rounding.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, Self, cast

import indrajala_math_rust as pa
import numpy as np

from indrajala_ml.data.prepared_dataset import CLASSIFY_CHUNK_ROWS, PreparedDataset
from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.layers.array.branch_layers import BranchArrayLayer, SplitArrayLayer
from indrajala_ml.model.layers.numpy import recurrent_array_layer as numpy_parts
from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.layers.rust import recurrent_rust_array_layer as rust_parts
from indrajala_ml.model.layers.rust import residual_rust_array_layer as residual_layers
from indrajala_ml.model.layers.rust import token_rust_array_layer as token_layers
from indrajala_ml.model.layers.rust.attention_rust_array_layer import AttentionRustArrayLayer
from indrajala_ml.model.layers.rust.layer_norm_rust_array_layer import LayerNormRustArrayLayer
from indrajala_ml.model.networks.numpy.numpy_array_network_base import NumpyArrayNetworkBase
from indrajala_ml.model.networks.rust.rust_array_network_base import RustArrayNetworkBase
from indrajala_ml.model.persistence.format2 import WINDOW_RECURRENT, NetworkFile
from indrajala_ml.model.protocols.array_protocols import BackendArray, TrainedArrayLayer
from indrajala_ml.model.specs.bounds import validate_class_count
from indrajala_ml.model.specs.layer_specs import Dense, LayerSpec, factor_reads
from indrajala_ml.model.specs.spec_shapes import InputShape, spec_shapes, token_shape
from indrajala_ml.model.specs.spec_validation import validate_window_recurrent
from indrajala_ml.model.specs.update_rules import UpdateRule
from indrajala_ml.model.specs.window_recurrence import (
    Consolidation,
    WindowRecurrence,
    pooling_reads,
    validate_consolidation,
)
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
    """A backend's class of the network's recurrence, and of the StateInput its window holds."""

    recurrence: Any
    state_input: Any


_PARTS = {
    NUMPY.name: _Parts(numpy_parts.RecurrenceArrayLayer, numpy_parts.StateInputArrayLayer),
    RUST.name: _Parts(rust_parts.RecurrenceRustArrayLayer, rust_parts.StateInputRustArrayLayer),
}


class _WithPooling:
    """
    The layer whose tokens the poolings read (the output layer, or the first StateInput, D10), as
    the layer before it reads it, with the poolings' gradient of the same tokens added. Every
    layer that may stand before either reads only its downstream_batch(), on either backend.
    """

    def __init__(self, layer: Any, pooled: Any) -> None:
        self._layer = layer
        self._pooled = pooled

    def downstream_batch(self) -> Any:
        return self._layer.downstream_batch() + self._pooled


class _Pooled:
    """
    A branch's end (the window branches workplan) as the layer before it reads it: its poolings'
    gradient alone, since nothing else reads the branch's result. A LayerNorm ends each branch.
    """

    def __init__(self, pooled: Any) -> None:
        self._pooled = pooled

    def downstream_batch(self) -> Any:
        return self._pooled


def _summed(pooled: Any, readers: Sequence[int]) -> Any:
    # the readers' poolings' gradient: the step's one array if it read one, else each reader's
    # summed in factor order (as the crate's window program sums them)
    if not isinstance(pooled, list):
        return pooled
    parts = cast("list[Any]", pooled)
    total = parts[readers[0]]
    for g in readers[1:]:
        total = total + parts[g]
    return total


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

        # the layer whose input tokens the poolings read (D10), and their width; with branches, each
        # factor's branch's end (the window branches workplan)
        reads = pooling_reads(layers, recurrence)
        ends = factor_reads(layers)  # each factor's branch's end, or none
        _, features = token_shape(spec_shapes(layers, input_shape)[ends[0] if ends else reads].input_shape)
        parts = _PARTS[self.backend.name]
        sizes = [
            (f.state_size, None, 0.0) if f.codes is None else (f.state_size, f.codes.count, f.codes.commitment)
            for f in recurrence.factors
        ]
        self.recurrence_layer: Any = parts.recurrence(
            sizes, self.columns, features, recurrence.cross, recurrence.cell.update_bias
        )
        self.parts: list[Any] = self.recurrence_layer.parts()
        self._state_inputs: list[Any] = [layer for layer in self.layers if isinstance(layer, parts.state_input)]
        # that layer among the built layers, each factor's: the output layer, the first StateInput,
        # or its branch's end (a built layer a spec, expanded)
        read = len(self.layers) - 1 if reads == len(layers) - 1 else self.layers.index(self._state_inputs[0])
        self._pooling_reads: list[int] = ends or [read] * len(recurrence.factors)

    def _window_state_size(self) -> int | None:
        return self.recurrence.state_size

    def _update_rule(self) -> UpdateRule:
        return self.update_rule

    @property
    def consolidation(self) -> Consolidation:
        """The consolidation training holds the states to (empty: none; consolidate)."""
        return self.recurrence_layer.consolidation

    def set_commitments(self, commitments: Sequence[float]) -> None:
        """
        Each factor's commitment from the next training step on, in factor order (the
        commitment-control workplan): training state, not the network's, so not saved with it; a
        loaded network has its specs' commitments.
        """
        self.recurrence_layer.set_commitments(commitments)

    def set_temperatures(self, temperatures: Sequence[float | None]) -> None:
        """
        Each factor's codes' temperature from the next training step on, in factor order (None:
        hard; the commitment-control workplan, stage 3): soft in the training step's windows only,
        every other pass, the initial state's included, snapped (D6, D11). Training state, as the
        commitments: not saved with the network.
        """
        self.recurrence_layer.set_temperatures(temperatures)

    def set_continuous(self, continuous: Sequence[bool]) -> None:
        """
        Each snapped factor run unsnapped, in factor order (the state formation workplan's warm-up, D4):
        its cell's output passed on and its codes unread, in the training step's windows and in
        forward_windows (the screen); every machine reading (window_states, state_codes, the
        certificate) and the initial state stay snapped. Training state, as the temperatures: not
        saved with the network.
        """
        self.recurrence_layer.set_continuous(continuous)

    def set_couplings(self, couplings: Sequence[float]) -> None:
        """
        Each snapped factor's coupling, in factor order (the state formation workplan's gradual
        handover, D25): its next state `(1 - l) h + l q` in the training step's windows and in
        forward_windows (the screen), 1 its code (as without), 0 its cell's output with its codes still
        learning from it; every machine reading and the initial state stay snapped. Training state, as
        the temperatures: not saved with the network.
        """
        self.recurrence_layer.set_couplings(couplings)

    def set_straight(self, straight: bool) -> None:
        """
        Every soft factor's gradient to its cell straight through, `dq` and the commitment's (the state
        formation workplan's D51: as the temperature falls, the soft path's gradient becomes the hard
        snap's); its codes' gradients as without it. Training state, as the temperatures: not saved.
        """
        self.recurrence_layer.set_straight(straight)

    def consolidate(self, consolidation: Consolidation) -> None:
        """
        From the next training step on, a state snapped to a duplicate joint code is held by the
        commitment to its representative's codes (the code-consolidation workplan, D1); empty, to
        its own, as without one. Training state, not the network's: not saved with it.
        """
        validate_consolidation(self.recurrence, consolidation)
        self.recurrence_layer.consolidation = dict(consolidation)

    def randomize(self) -> None:
        # the window's layers as a sequence network's, its StateInputs' P among them in order;
        # then each factor's cell's gates and codes, factor by factor. The poolings' queries and
        # the initial states start at zero and draw nothing.
        super().randomize()
        self.recurrence_layer.draw(self.rng)

    # the forward pass

    def _machine_states(self, indices: Sequence[Any] | None = None) -> list[tuple[int, ...]]:
        # each example's state as a machine's, a code per factor, from a snap's indices (each
        # factor's), the last snap's by default
        assert self.recurrence.snapped, (
            "a continuous factor has no codes (Factor(s, None)): the state isn't a machine's"
        )
        snap: Sequence[Any] = self.recurrence_layer.indices if indices is None else indices
        return [tuple(int(i) for i in codes) for codes in zip(*snap, strict=True)]

    def _windows(self, X: Any) -> int:
        count, remainder = divmod(X.shape[1], self.columns)
        assert remainder == 0 and count >= 1, (
            f"a window-recurrent network reads whole windows of {self.columns} columns; got {X.shape[1]} columns"
        )
        return count

    def _window(self, X: Any, w: int) -> Any:
        return self.backend.columns(X, w * self.columns, (w + 1) * self.columns)

    def _first_state(self, examples: int) -> Any:
        return self.recurrence_layer.first_state(examples)

    def _window_forward(self, X: Any, C: Any) -> tuple[Any, Any]:
        # one window's digits' probabilities and its next state, from its columns X and state C
        for state_input in self._state_inputs:
            state_input.set_state(C)
        self._activations: list[Any] = [X]
        for layer in self.layers:
            self._activations.append(layer.forward_batch(self._activations[-1]))
        reads = self._pooling_reads
        tokens = (
            self._activations[reads[0]]
            if len(set(reads)) == 1
            else [self._activations[r] for r in reads]  # each factor's branch's (the window branches workplan)
        )
        return self._activations[-1], self.recurrence_layer.forward_batch(tokens, C)

    def _run_windows(
        self, X: Any, C: Any, states: bool, codes: bool, live: bool = False
    ) -> tuple[Any, list[Any], list[Any], list[Any]]:
        # the inference passes' windows: the per-layer path here, the window program on Rust; live, the
        # continuous factors read unsnapped (forward_windows)
        return self.per_layer_windows(X, C, states, codes, live)

    def per_layer_windows(
        self, X: Any, C: Any, states: bool, codes: bool, live: bool = False
    ) -> tuple[Any, list[Any], list[Any], list[Any]]:
        """
        The inference forward pass over every window of X from the snapped state C: the outputs
        (N, W * k * classes), and, as kept, each window's next state and its cells' outputs before
        the snap (states) and each factor's code indices (codes), W of each; what isn't kept is an
        empty list. The per-layer path, window by window; the Rust network's window program runs it
        in one call (the window-program workplan, stage 3), and this stays its reference.
        """
        outputs: list[Any] = []
        kept_states: list[Any] = []
        kept_cells: list[Any] = []
        kept_indices: list[Any] = []
        self.recurrence_layer.reading = live  # live: the continuous factors read unsnapped (the screen)
        try:
            for w in range(self._windows(X)):
                output, C = self._window_forward(self._window(X, w), C)
                outputs.append(output)
                if states:
                    kept_states.append(C)
                    kept_cells.append(self.recurrence_layer.H)
                if codes:
                    kept_indices.append(list(self.recurrence_layer.indices))
        finally:
            self.recurrence_layer.reading = False
        return self.backend.concatenate_columns(outputs), kept_states, kept_cells, kept_indices

    def forward_windows(self, X: A) -> A:
        """The inference forward pass: each column's digit probabilities, (N, W * k * classes)."""
        outputs, _, _, _ = self._run_windows(X, self._first_state(X.shape[0]), False, False, live=True)
        return outputs

    def window_states(self, X: A, live: bool = False) -> tuple[A, list[A], list[A]]:
        """
        The inference forward pass with its states: each column's digit probabilities, as
        forward_windows's; the states (N, state_size), the initial state's then each window's
        next state's, W + 1 of them, a snapped state as its code's vector (the claims workplan's
        drift profile, E2b); and the cells' outputs before the snap, the same W + 1 (the snap
        wall's margins, docs/workplans/snap-wall-workplan.md). `live`: the continuous factors (set_continuous)
        read unsnapped, as forward_windows (the state formation workplan's warm-up readings).
        """
        C = self._first_state(X.shape[0])
        H = self.recurrence_layer.H
        outputs, states, cells, _ = self._run_windows(X, C, True, False, live=live)
        return outputs, [C, *states], [H, *cells]

    def state_codes(self, X: A) -> list[list[tuple[int, ...]]]:
        """
        Each state, per example, as a machine's, a code per factor: the initial state's, then each
        window's next state's, W + 1 of them (the report's codes used, W4-i). Every factor snapped.
        """
        C = self._first_state(X.shape[0])
        first = self._machine_states()
        _, _, _, indices = self._run_windows(X, C, False, True)
        states = [first, *(self._machine_states(window) for window in indices)]
        return [list(example) for example in zip(*states, strict=True)]

    def first_step(self) -> tuple[A, list[tuple[int, ...]]]:
        """The initial state before it is snapped (1, state_size), and the state it snaps to."""
        self._first_state(1)
        return self.recurrence_layer.H, self._machine_states()

    def window_step(self, X: A, C: A) -> tuple[A, A, list[tuple[int, ...]]]:
        """
        One window in inference, from its columns X (N, k) and states C (N, state_size): its digits'
        probabilities (N, k * classes), the cells' outputs before they are snapped (N, state_size)
        and the states they snap to (the window-recurrent claims workplan's certificate, E1).
        """
        output, _, cells, indices = self._run_windows(X, C, True, True)
        return output, cells[0], self._machine_states(indices[0])

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
        # every layer's and part's gradient of the batch's loss: the per-layer path here, the window
        # program on Rust
        self.per_layer_gradients(X, Y)

    def per_layer_gradients(self, X: Any, Y: Any) -> None:
        """
        Every layer's and part's gradient of the batch's loss, accumulated (W14, W20), window by
        window: numpy's training step, and the reference of the Rust network's window program
        (the window-program workplan, stage 5).
        """
        windows = self._windows(X)
        weight = 1.0 / windows
        width = self.columns * self.class_count
        states: list[Any] = []
        generators: list[dict[str, Any]] = []
        walled = self._walled_codes()
        right: list[bool] = []
        self._set_training_mode(True)
        recurrence = self.recurrence_layer
        recurrence.soft = True
        try:
            C = self._first_state(X.shape[0])
            for w in range(windows):
                states.append(C)
                generators.append(generator_state(self.rng))
                output, C = self._window_forward(self._window(X, w), C)
                if walled:
                    right = self._all_right(right, output, self.backend.columns(Y, w * width, (w + 1) * width))
            after = generator_state(self.rng)
            for codes in walled:
                codes.gate = np.array(right, dtype=float)

            dC = self.backend.zeros((X.shape[0], self.recurrence.state_size))
            for w in reversed(range(windows)):
                set_generator_state(self.rng, generators[w])
                self._window_forward(self._window(X, w), states[w])
                dC = self._window_backward(self.backend.columns(Y, w * width, (w + 1) * width), dC, weight)
            set_generator_state(self.rng, after)
        finally:
            recurrence.soft = False
            self._set_training_mode(False)

        self._first_state(X.shape[0])
        self.recurrence_layer.first_state_backward(dC, weight)
        for codes in walled:
            codes.gate = None  # this batch's only

    def _walled_codes(self) -> list[Any]:
        # the factors' codes with the snap wall on (docs/workplans/snap-wall-workplan.md; numpy's only, D8)
        factors = getattr(self.recurrence_layer, "factors", ())
        return [f.codes for f in factors if f.codes is not None and getattr(f.codes, "wall", 0.0) > 0.0]

    def _all_right(self, right: list[bool], output: Any, Y: Any) -> list[bool]:
        # each example's every digit right so far, this window's columns included (the wall's gate, D1)
        predicted = self.backend.argmax_token_rows(output, self.class_count)
        targets = self.backend.argmax_token_rows(Y, self.class_count)
        now = [p == t for p, t in zip(predicted, targets, strict=True)]
        return now if not right else [a and b for a, b in zip(right, now, strict=True)]

    def _window_backward(self, Y: Any, dC_next: Any, weight: float) -> Any:
        # one window's backward pass after its forward pass: its gradients accumulated, and its
        # state's gradient returned, from the next state's dC_next and its digits' targets Y
        pooled, dC = self.recurrence_layer.backward_batch(dC_next, weight)

        output_layer: Any = self.layers[-1]
        output_layer.compute_output_delta_batch(Y)
        # the mean over every window's columns: each window's mean over its own, times 1 / W
        output_layer.delta_batch = output_layer.delta_batch * weight
        reads = self._pooling_reads
        for i in reversed(range(len(self.layers) - 1)):
            following = self.layers[i + 1]
            readers = [g for g, r in enumerate(reads) if r == i + 1]
            if isinstance(following, BranchArrayLayer):
                self.layers[i].compute_hidden_delta_batch(_Pooled(_summed(pooled, readers)))
            elif readers:
                self.layers[i].compute_hidden_delta_batch(_WithPooling(following, _summed(pooled, readers)))
            else:
                self.layers[i].compute_hidden_delta_batch(following)
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


@contextmanager
def per_layer_path(network: Any) -> Generator[None]:
    """
    A Rust network's inference passes and training steps through the per-layer path
    (per_layer_windows, per_layer_gradients) while open, the window program's reference: its parity
    tests and scripts/window_program_check.py.
    """
    # instance attributes shadow the class's methods while open
    setattr(network, "_run_windows", network.per_layer_windows)  # noqa: B010
    setattr(network, "_accumulate_gradients", network.per_layer_gradients)  # noqa: B010
    try:
        yield
    finally:
        delattr(network, "_run_windows")
        delattr(network, "_accumulate_gradients")


def _program_op(layer: Any) -> tuple[Any, ...]:
    # a window's Rust layer as the window program's op (the crate's program.rs); every layer kind a
    # window can hold has one (a window holds neither Patches nor TokenMean: validate_window_specs)
    if isinstance(layer, token_layers.EmbeddingRustArrayLayer):
        return ("embedding",)
    if isinstance(layer, token_layers.PositionRustArrayLayer):
        return ("position",)
    if isinstance(layer, rust_parts.StateInputRustArrayLayer):
        return ("state_input", layer.tokens)
    if isinstance(layer, residual_layers.ForkRustArrayLayer):
        return ("fork",)
    if isinstance(layer, residual_layers.AddRustArrayLayer):
        return ("add",)
    if isinstance(layer, SplitArrayLayer):
        return ("split",)
    if isinstance(layer, BranchArrayLayer):
        return ("branch",)
    if isinstance(layer, LayerNormRustArrayLayer):
        return ("layer_norm", layer.epsilon)
    # the softmax output layer is a token-wise dense layer too: it goes first
    if isinstance(layer, token_layers.TokenSoftmaxRustArrayLayer):
        return ("token_output", layer.tokens)
    if isinstance(layer, token_layers.TokenDenseRustArrayLayer):
        return ("token_dense", layer.tokens, layer.activation)
    if isinstance(layer, AttentionRustArrayLayer):
        return ("attention", layer.heads, layer.causal, layer.dropout, *layer.rotary_table)
    if isinstance(layer, token_layers.TokenDropoutRustArrayLayer):
        return ("dropout", layer.drop_probability)
    raise ValueError(f"the window program has no op for a window's {type(layer).__name__}")


class WindowRecurrentRustArrayNetwork(WindowRecurrentShape[pa.Array], RustArrayNetworkBase):
    """
    WindowRecurrentShape on the Rust backend. Its inference passes (forward_windows,
    window_states, state_codes, window_step, and so loss_batch and forward_rows) and its training
    step's gradients run every window in one crate call, the window program built from its layers
    at construction (the window-program workplan, stages 3 and 5), by bits the per-layer path's
    (per_layer_windows, per_layer_gradients): one forward pass in training, each window's results
    and dropout masks kept, so no window's forward pass is recomputed (D5, D6).
    """

    def __init__(
        self,
        input_shape: InputShape,
        layers: Sequence[LayerSpec],
        recurrence: WindowRecurrence,
        update_rule: UpdateRule,
    ) -> None:
        super().__init__(input_shape, layers, recurrence, update_rule)
        self._program = pa.WindowProgram(
            [_program_op(layer) for layer in self.layers],
            self.columns,
            self._pooling_reads,
            [f.state_size for f in recurrence.factors],
            [f.codes is not None for f in recurrence.factors],
            cross=recurrence.cross,
        )
        # the consolidation as the program takes it, (duplicate, representative) pairs
        self._consolidation: list[tuple[list[int], list[int]]] = []

    def _program_parameters(self) -> list[pa.Array]:
        # the network's parameters in order, as the program reads them: the window's layers', then
        # the parts' (each factor's pooling, cell, codes and initial state)
        layers = [
            array for layer in self.layers if isinstance(layer, TrainedArrayLayer) for array in layer.parameters()
        ]
        return [*layers, *(array for part in self.parts for array in part.parameters())]

    def _program_gradients(self) -> list[pa.Array]:
        # each parameter's gradient, in _program_parameters' order
        layers = [array for layer in self.layers if isinstance(layer, TrainedArrayLayer) for array in layer.gradients()]
        return [*layers, *(array for part in self.parts for array in part.gradients())]

    def _accumulate_gradients(self, X: Any, Y: Any) -> None:
        # the windows forward in training and back in one call, each gradient accumulated in place;
        # then the initial state's, through its codes (W14, W20; D5, D6)
        assert not self._walled_codes(), "the snap wall is numpy's (docs/workplans/snap-wall-workplan.md, D8)"
        windows = self._windows(X)
        weight = 1.0 / windows
        recurrence = self.recurrence_layer
        dC = pa.window_program_learn(
            self._program,
            X,
            Y,
            self._program_parameters(),
            self._program_gradients(),
            self._first_state(X.shape[0]),
            self.rng,
            weight,
            recurrence.commitments,
            consolidation=self._consolidation,
            temperatures=recurrence.temperatures,
            continuous=recurrence.continuous if any(recurrence.continuous) else None,
            couplings=recurrence.couplings if min(recurrence.couplings) < 1.0 else None,
            straight=recurrence.straight,
        )
        recurrence.first_state_backward(dC, weight)

    def consolidate(self, consolidation: Consolidation) -> None:
        super().consolidate(consolidation)
        self._consolidation = [(list(d), list(r)) for d, r in consolidation.items()]

    def _run_windows(
        self, X: Any, C: Any, states: bool, codes: bool, live: bool = False
    ) -> tuple[Any, list[Any], list[Any], list[Any]]:
        dropping = (AttentionRustArrayLayer, token_layers.TokenDropoutRustArrayLayer)
        assert not any(layer.training for layer in self.layers if isinstance(layer, dropping)), (
            "the window program runs inference only: training's forward pass is the per-layer path"
        )
        self._windows(X)  # whole windows, refused as the per-layer path refuses them
        continuous = self.recurrence_layer.continuous
        couplings = self.recurrence_layer.couplings
        return pa.window_program_forward(
            self._program,
            X,
            self._program_parameters(),
            C,
            states=states,
            codes=codes,
            continuous=continuous if live and any(continuous) else None,
            couplings=couplings if live and min(couplings) < 1.0 else None,
        )
