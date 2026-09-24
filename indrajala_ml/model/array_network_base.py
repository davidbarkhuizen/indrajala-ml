from __future__ import annotations

from typing import Sequence

from indrajala_ml.model.array_backend import NUMPY
from indrajala_ml.model.array_layer import ArrayLayer
from indrajala_ml.model.bounds import validate_batch, validate_layer_sizes
from indrajala_ml.prepared_dataset import CLASSIFY_CHUNK_ROWS, PreparedDataset


class ArrayNetworkBase:
    """
    Shared machinery behind every array-backed sibling network in this codebase, numpy and Rust
    (RustArrayNetworkBase sets the backend) - plain, momentum, L2, Adam, ReLU, softmax, dropout,
    cross-entropy, both the multiclass and single-output shapes: layer assembly, the forward
    pass, learn/learn_batch, fan-in-aware randomize, and snapshot/restore.

    Mirrors BackpropNetworkBase's own hidden_layer_cls/output_layer_cls extension-point design
    exactly, one level up: every method here is identical across every array-based sibling in
    this codebase except for which layer class gets constructed, one constructor hyperparameter,
    and the target-array shape - a purely mechanical collapse, not a redesign of any hand-derived
    formula. Every hand-derived formula lives entirely in the ArrayLayer subclass a sibling plugs
    in via hidden_layer_cls/output_layer_cls (e.g. MomentumArrayLayer.apply_accumulated_gradient).

    What stays out of this base, in the two "shape" subclasses instead
    (VectorizedMultiClassBackpropClassifierNetwork / ArrayBackpropClassifierNetwork):
    predict_probabilities/predict_probability and classify_state (argmax vs. 0.5-threshold),
    class_count handling, and save/load (the JSON envelope and its class_count presence differ
    between the two shapes) - genuinely different concerns, not duplicated ones, matching where
    BackpropClassifierNetwork/MultiClassBackpropClassifierNetwork already draw the same line over
    BackpropNetworkBase.
    """

    # override points for a sibling whose hidden/output layers need different per-layer math
    # (e.g. MomentumArrayLayer) - every "plain" shape class leaves these as ArrayLayer, so this
    # is a pure extension point with zero behavior change for them. A hyperparameter-free
    # sibling (ReLU, softmax, cross-entropy) can set one of these as a plain class attribute; a
    # hyperparameter-bearing sibling (momentum, L2, Adam, dropout) sets it as an *instance*
    # attribute in its own __init__ (a closure capturing the hyperparameter) before calling
    # super().__init__() - the same pattern AdamBackpropClassifierNetwork already uses one layer
    # down, over BackpropNetworkBase's own hidden_layer_cls/output_layer_cls.
    hidden_layer_cls: type = ArrayLayer
    output_layer_cls: type = ArrayLayer

    # the array operations that differ between numpy and Rust (array_backend.py);
    # RustArrayNetworkBase sets RUST
    backend = NUMPY

    def __init__(self, layer_sizes: list[int], dimension: int, output_size: int) -> None:

        validate_layer_sizes(layer_sizes)

        self.layer_sizes = layer_sizes
        self.dimension = dimension

        self.layers: list = []
        previous_size = dimension
        for size in layer_sizes:
            self.layers.append(self.hidden_layer_cls(size, previous_size))
            previous_size = size

        self.output_layer = self.output_layer_cls(output_size, previous_size)
        self.layers.append(self.output_layer)

    def _forward(self, state: tuple[float, ...]):
        return self._forward_input(self.backend.vector(state))

    def _forward_input(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def classify_row(self, prepared: PreparedDataset, index: int):
        # classify_state for row index of a prepared dataset; _classify_output is the shape
        # class's argmax or 0.5 threshold, shared with classify_state
        return self._classify_output(self._forward_input(self.backend.row(self._prepared_states(prepared), index)))

    def classify_rows(self, prepared: PreparedDataset) -> list:
        # classify_row for every row, as the trainers' accuracy pass needs it, through
        # forward_batch over chunks of rows (docs/optimizations/implemented.md). The
        # predictions are classify_row's, but only by construction on Rust, where a batched
        # forward row equals the single-example forward exactly: numpy's X @ W.T can differ from
        # W @ x in the last ULP, so an argmax between outputs an ULP apart could differ
        states = self._prepared_states(prepared)
        predictions = []
        for start in range(0, len(prepared), CLASSIFY_CHUNK_ROWS):
            X = self.backend.row_range(states, start, min(start + CLASSIFY_CHUNK_ROWS, len(prepared)))
            for layer in self.layers:
                X = layer.forward_batch(X)
            predictions.extend(self._classify_output_batch(X))
        return predictions

    def prepare_dataset(self, rows: Sequence[tuple[tuple[float, ...], object]]) -> PreparedDataset:
        return PreparedDataset.from_rows(rows, self.backend.name)

    def _prepared_states(self, prepared: PreparedDataset):
        name = self.backend.name
        assert prepared.backend == name, f"a {name} network needs a {name} dataset; got {prepared.backend!r}"
        return prepared.states

    def _set_training_mode(self, training: bool) -> None:
        # no-op for every sibling except dropout's own override - the array-level counterpart to
        # BackpropNetworkBase._set_training_mode, and to DropoutArrayLayer.set_training_mode
        # which this hook delegates to once dropout overrides it.
        pass

    def _target_array(self, category):
        raise NotImplementedError

    def _target_batch_array(self, categories: Sequence):
        raise NotImplementedError

    def _classify_output(self, output):
        raise NotImplementedError

    def _classify_output_batch(self, output_batch) -> list:
        # _classify_output for each row of a (batch, output_size) forward_batch output
        raise NotImplementedError

    # learn/learn_row and learn_batch/learn_batch_rows only differ in where the input array
    # comes from (a converted tuple, or a prepared dataset's rows); the training step itself is
    # _learn_input/_learn_batch_input, shared, so the two paths can't drift

    def learn(self, learning_rate: float, state: tuple[float, ...], category) -> None:
        self._learn_input(learning_rate, self.backend.vector(state), category)

    def learn_row(self, learning_rate: float, prepared: PreparedDataset, index: int) -> None:
        self._learn_input(
            learning_rate, self.backend.row(self._prepared_states(prepared), index), prepared.labels[index]
        )

    def _learn_input(self, learning_rate: float, x, category) -> None:
        activations = [x]
        self._set_training_mode(True)
        try:
            for layer in self.layers:
                x = layer.forward(x)
                activations.append(x)
        finally:
            self._set_training_mode(False)

        target = self._target_array(category)
        self.output_layer.compute_output_delta(target)

        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].compute_hidden_delta(self.layers[i + 1])

        # sgd_step is accumulate_gradient then apply_accumulated_gradient at batch_size=1, which
        # RustArrayLayer fuses into one call where the layer's update is plain SGD
        for layer, input_activation in zip(self.layers, activations):
            layer.sgd_step(input_activation, learning_rate)

    def learn_batch(self, learning_rate: float, batch: Sequence[tuple[tuple[float, ...], object]]) -> None:
        validate_batch(batch)
        X = self.backend.matrix([state for state, _target in batch])
        self._learn_batch_input(learning_rate, X, [category for _state, category in batch])

    def learn_batch_rows(self, learning_rate: float, prepared: PreparedDataset, indices: Sequence[int]) -> None:
        validate_batch(indices)
        X = self.backend.rows(self._prepared_states(prepared), indices)
        self._learn_batch_input(learning_rate, X, [prepared.labels[i] for i in indices])

    def _learn_batch_input(self, learning_rate: float, X, categories: Sequence) -> None:
        batch_size = len(categories)

        activations = [X]
        self._set_training_mode(True)
        try:
            for layer in self.layers:
                X = layer.forward_batch(X)
                activations.append(X)
        finally:
            self._set_training_mode(False)

        target_batch = self._target_batch_array(categories)
        self.output_layer.compute_output_delta_batch(target_batch)

        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].compute_hidden_delta_batch(self.layers[i + 1])

        for layer, input_activation_batch in zip(self.layers, activations):
            layer.accumulate_gradient_batch(input_activation_batch)
            layer.apply_accumulated_gradient(learning_rate, batch_size)

    def randomize(self) -> None:
        # the same fan-in-aware scheme (limit = 1/sqrt(fan_in)) every array-based sibling in
        # this codebase would otherwise have to reimplement independently, drawn from the
        # backend's own RNG
        previous_size = self.dimension
        for layer in self.layers:
            layer.W, layer.b = self.backend.random_layer(layer.size, previous_size)
            previous_size = layer.size

    def snapshot(self) -> list[tuple]:
        return [(layer.W.copy(), layer.b.copy()) for layer in self.layers]

    def restore(self, snapshot: list[tuple]) -> None:
        # accepts this backend's arrays or nested lists (a loaded file, or a snapshot pickled
        # across a worker boundary)
        for layer, (W, b) in zip(self.layers, snapshot):
            layer.W = self.backend.owned(W)
            layer.b = self.backend.owned(b)
