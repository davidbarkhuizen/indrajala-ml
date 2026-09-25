# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, X, A, which strict mode takes for constants)
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Generic, Protocol, cast

from typing_extensions import Self

from indrajala_ml.model.array_protocols import A, ArrayBackend, ArrayNetworkLayer, WeightedArrayLayer
from indrajala_ml.model.bounds import validate_batch, validate_layer_sizes
from indrajala_ml.prepared_dataset import CLASSIFY_CHUNK_ROWS, PreparedDataset


class DenseArrayLayerClass(Protocol[A]):
    """A dense layer class as a network builds it: (size, input_size, *its hyperparameters)."""

    hyperparameters: tuple[str, ...]

    def __call__(self, *args: Any, **kwargs: Any) -> WeightedArrayLayer[A]: ...


def as_weighted_array_layers(layers: Sequence[ArrayNetworkLayer[A]]) -> list[WeightedArrayLayer[A]]:
    """
    layers, checked to all have weights (W, b, size): a dense network's, whose randomize and
    snapshot/restore read them (the conv networks, with weightless pool layers, override both).
    """
    dense = [layer for layer in layers if isinstance(layer, WeightedArrayLayer)]
    assert len(dense) == len(layers), f"expected only dense layers; got {[type(layer).__name__ for layer in layers]}"
    return dense


class ArrayNetworkBase(Generic[A]):
    """
    What every array-backed network shares, numpy and Rust (NumpyArrayNetworkBase and
    RustArrayNetworkBase set the backend, A being its array type):
    layer assembly, the forward pass, learn/learn_batch and their prepared-dataset forms,
    classify_rows, fan-in-aware randomize, and snapshot/restore.

    A sibling differs only in its layer classes (hidden_layer_cls/output_layer_cls) and their
    hyperparameters; every hand-derived formula lives in the layer class (e.g.
    MomentumArrayLayer.apply_accumulated_gradient). What differs between the multiclass and
    single-output networks (classify_state, targets, class_count, save/load) is in the shape mixins
    in array_network_shapes.py, listed before the backend's base.
    """

    # the layer classes a sibling overrides for different per-layer math (e.g. MomentumArrayLayer);
    # a layer class's hyperparameters come from the network's attributes of the same names
    # (_new_layer). Set by the backend's base.
    hidden_layer_cls: DenseArrayLayerClass[A]
    output_layer_cls: DenseArrayLayerClass[A]

    # the array operations that differ between numpy and Rust (array_backend.py), set by the
    # backend's base
    backend: ArrayBackend[A]

    # the constructor keyword arguments a sibling stores under the same attribute names (e.g.
    # ("beta1", "beta2", "epsilon")): its layers read them (_new_layer), and the shapes'
    # save/load round-trip them through _extra_state/_extra_init_kwargs
    hyperparameters: tuple[str, ...] = ()

    def __init__(self, layer_sizes: list[int], dimension: int, output_size: int) -> None:

        validate_layer_sizes(layer_sizes)

        self.layer_sizes = layer_sizes
        self.dimension = dimension

        self.layers: list[ArrayNetworkLayer[A]] = []
        previous_size = dimension
        for size in layer_sizes:
            self.layers.append(self._new_layer(self.hidden_layer_cls, size, previous_size))
            previous_size = size

        self.output_layer = self._new_layer(self.output_layer_cls, output_size, previous_size)
        self.layers.append(self.output_layer)

    def _new_layer(self, layer_cls: DenseArrayLayerClass[A], size: int, input_size: int) -> WeightedArrayLayer[A]:
        # a hyperparameter-bearing sibling stores its hyperparameters before super().__init__()
        return layer_cls(size, input_size, **{name: getattr(self, name) for name in layer_cls.hyperparameters})

    def _forward(self, state: tuple[float, ...]) -> A:
        return self._forward_input(self.backend.vector(state))

    def _forward_input(self, x: A) -> A:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def classify_row(self, prepared: PreparedDataset, index: int) -> Any:
        # classify_state for row index of a prepared dataset; _classify_output is the shape
        # class's argmax or 0.5 threshold, shared with classify_state
        return self._classify_output(self._forward_input(self.backend.row(self._prepared_states(prepared), index)))

    def classify_rows(self, prepared: PreparedDataset) -> list[Any]:
        # classify_row for every row, as the trainers' accuracy pass needs it, through
        # forward_batch over chunks of rows (docs/optimizations/implemented.md). The
        # predictions are classify_row's, but only by construction on Rust, where a batched
        # forward row equals the single-example forward exactly: numpy's X @ W.T can differ from
        # W @ x in the last ULP, so an argmax between outputs an ULP apart could differ
        states = self._prepared_states(prepared)
        predictions: list[Any] = []
        for start in range(0, len(prepared), CLASSIFY_CHUNK_ROWS):
            X = self.backend.row_range(states, start, min(start + CLASSIFY_CHUNK_ROWS, len(prepared)))
            for layer in self.layers:
                X = layer.forward_batch(X)
            predictions.extend(self._classify_output_batch(X))
        return predictions

    def prepare_dataset(self, rows: Sequence[tuple[tuple[float, ...], object]]) -> PreparedDataset:
        return PreparedDataset.from_rows(rows, self.backend.name)

    def _prepared_states(self, prepared: PreparedDataset) -> A:
        name = self.backend.name
        assert prepared.backend == name, f"a {name} network needs a {name} dataset; got {prepared.backend!r}"
        return cast(A, prepared.states)  # the assert: this backend's array type

    def _set_training_mode(self, training: bool) -> None:
        # a no-op; the dropout networks override it to toggle their dropout layers
        pass

    # the shape's hooks (array_network_shapes.py), over its label type: a float (single output) or
    # a class index (multiclass)

    def _target_array(self, category: Any) -> A:
        raise NotImplementedError

    def _target_batch_array(self, categories: Sequence[Any]) -> A:
        raise NotImplementedError

    def _classify_output(self, output: A) -> Any:
        raise NotImplementedError

    def _classify_output_batch(self, output_batch: A) -> list[Any]:
        # _classify_output for each row of a (batch, output_size) forward_batch output
        raise NotImplementedError

    # learn/learn_row and learn_batch/learn_batch_rows only differ in where the input array
    # comes from (a converted tuple, or a prepared dataset's rows); the training step itself is
    # _learn_input/_learn_batch_input, shared, so the two paths can't drift

    def learn(self, learning_rate: float, state: tuple[float, ...], category: Any) -> None:
        self._learn_input(learning_rate, self.backend.vector(state), category)

    def learn_row(self, learning_rate: float, prepared: PreparedDataset, index: int) -> None:
        self._learn_input(
            learning_rate, self.backend.row(self._prepared_states(prepared), index), prepared.labels[index]
        )

    def _learn_input(self, learning_rate: float, x: A, category: Any) -> None:
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

    def _learn_batch_input(self, learning_rate: float, X: A, categories: Sequence[Any]) -> None:
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

    @classmethod
    def randomized(cls, *args: Any, **kwargs: Any) -> Self:
        # every sibling's randomized signature is its __init__ signature, so one pass-through
        # serves them all, positional or keyword, defaults included
        network = cls(*args, **kwargs)
        network.randomize()
        return network

    def randomize(self) -> None:
        # fan-in-aware (limit = 1/sqrt(fan_in)), drawn from the backend's RNG
        previous_size = self.dimension
        for layer in as_weighted_array_layers(self.layers):
            layer.W, layer.b = self.backend.random_layer(layer.size, previous_size)
            previous_size = layer.size

    def _extra_state(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.hyperparameters}

    @classmethod
    def _extra_init_kwargs(cls, state: dict[str, Any]) -> dict[str, Any]:
        # the inverse of _extra_state: a loaded state dict's hyperparameters, as constructor kwargs
        return {name: state[name] for name in cls.hyperparameters}

    def snapshot(self) -> list[tuple[A, ...]]:  # (W, b) per layer; the conv networks add () for a pool layer
        return [(layer.W.copy(), layer.b.copy()) for layer in as_weighted_array_layers(self.layers)]

    def restore(self, snapshot: Sequence[tuple[Any, ...]]) -> None:
        # accepts this backend's arrays or nested lists (a loaded file, or a snapshot pickled
        # across a worker boundary)
        for layer, (W, b) in zip(as_weighted_array_layers(self.layers), snapshot):
            layer.W = self.backend.owned(W)
            layer.b = self.backend.owned(b)
