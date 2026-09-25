from __future__ import annotations

from typing import Sequence

from indrajala_ml.model.bounds import validate_class_count, validate_layer_sizes
from indrajala_ml.model.conv_front_end import (
    build_conv_array_network_layers,
    load_conv_model_json,
    save_conv_array_model_json,
)
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.model_io import (
    load_array_model_json,
    load_single_output_array_model_json,
    save_array_model_json,
    save_single_output_array_model_json,
)


class ArrayMultiClassShape:
    """
    The multiclass shape over ArrayNetworkBase, for either backend: argmax classify_state,
    predict_probabilities, one-hot targets, and the save/load envelope with class_count.

    A mixin, listed before the backend's base, which supplies self.backend:
    VectorizedMultiClassBackpropClassifierNetwork and RustArrayMultiClassBackpropClassifierNetwork
    are this shape on numpy and on Rust.
    """

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int) -> None:
        validate_class_count(class_count)
        self.class_count = class_count
        super().__init__(layer_sizes, dimension, class_count)

    def predict_probabilities(self, state: tuple[float, ...]) -> list[float]:
        return self._forward(state).tolist()

    def classify_state(self, state: tuple[float, ...]) -> int:
        return self._classify_output(self._forward(state))

    def _classify_output(self, output) -> int:
        return self.backend.argmax(output)

    def _classify_output_batch(self, output_batch) -> list[int]:
        return self.backend.argmax_rows(output_batch)

    def _target_array(self, category: int):
        target = self.backend.zeros(self.class_count)
        target[category] = 1.0
        return target

    def _target_batch_array(self, categories: Sequence[int]):
        target_batch = self.backend.zeros((len(categories), self.class_count))
        for row, category in enumerate(categories):
            target_batch[row, category] = 1.0
        return target_batch

    def save(self, path: str) -> None:
        # not save_model_json, whose envelope carries input_bounds, which the array networks lack
        save_array_model_json(
            path,
            layer_sizes=self.layer_sizes,
            dimension=self.dimension,
            class_count=self.class_count,
            snapshot=self.snapshot(),
            extra=self._extra_state(),
        )

    @classmethod
    def load(cls, path: str):
        state = load_array_model_json(path)
        network = cls(
            state["layer_sizes"],
            state["dimension"],
            state["class_count"],
            **cls._extra_init_kwargs(state),
        )
        # restore converts the file's nested lists through the backend
        network.restore(state["snapshot"])
        return network


class ArraySingleOutputShape:
    """
    The single-output shape over ArrayNetworkBase, for either backend: 0.5-threshold classify_state,
    predict_probability, a scalar target, and the save/load envelope without class_count. It hosts
    the ensembles' sub-networks, one binary classifier per class.

    A mixin like ArrayMultiClassShape: ArrayBackpropClassifierNetwork and
    RustArrayBackpropClassifierNetwork are this shape on numpy and on Rust.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        # input_bounds is accepted and ignored: ensemble_train.py constructs every classifier_cls
        # as classifier_cls(layer_sizes, dimension, input_bounds)
        super().__init__(layer_sizes, dimension, 1)

    def predict_probability(self, state: tuple[float, ...]) -> float:
        return self._forward(state).tolist()[0]

    def classify_state(self, state: tuple[float, ...]) -> float:
        return self._classify_output(self._forward(state))

    def _classify_output(self, output) -> float:
        return 1.0 if output.tolist()[0] > 0.5 else 0.0

    def _classify_output_batch(self, output_batch) -> list[float]:
        return [1.0 if row[0] > 0.5 else 0.0 for row in output_batch.tolist()]

    def _target_array(self, category: float):
        return self.backend.vector([category])

    def _target_batch_array(self, categories: Sequence[float]):
        return self.backend.matrix([[category] for category in categories])

    def save(self, path: str) -> None:
        # the envelope without class_count
        save_single_output_array_model_json(
            path,
            layer_sizes=self.layer_sizes,
            dimension=self.dimension,
            snapshot=self.snapshot(),
            extra=self._extra_state(),
        )

    @classmethod
    def load(cls, path: str):
        state = load_single_output_array_model_json(path)
        network = cls(state["layer_sizes"], state["dimension"], **cls._extra_init_kwargs(state))
        network.restore(state["snapshot"])
        return network


class ArrayConvShape:
    """
    The convolutional shape over the multiclass shape, for either backend: a front end of conv and
    max-pool layers (one ConvSpec or PoolSpec each, in order), one or more sigmoid dense layers, and
    a one-vs-rest sigmoid output layer.

    A mixin, listed before the backend's plain multiclass network, which supplies self.backend and
    hidden_layer_cls (the dense layer class). ConvVectorizedMultiClassBackpropClassifierNetwork and
    ConvRustArrayMultiClassBackpropClassifierNetwork are this shape on numpy and on Rust, and set
    conv_layer_cls and pool_layer_cls.

    __init__ doesn't call super().__init__(), whose flat layer_sizes can't describe conv layers; it
    builds self.layers directly. Everything that only walks self.layers through the per-layer hooks
    (the forward pass, learn*, the multiclass shape's outputs and targets) is inherited. Overridden
    is what assumes a dense W in every layer: randomize, snapshot/restore (an empty entry for a pool
    layer) and save/load (the pure-Python conv network's envelope, so a model saved by any of the
    three loads into the others).
    """

    conv_layer_cls: type
    pool_layer_cls: type

    def __init__(
        self,
        input_height: int,
        input_width: int,
        conv_specs: list[ConvSpec | PoolSpec],
        dense_layer_sizes: list[int],
        class_count: int,
    ) -> None:

        validate_class_count(class_count)
        validate_layer_sizes(dense_layer_sizes, label="dense_layer_sizes", noun="dense hidden layer")

        self.class_count = class_count
        self.dimension = input_height * input_width
        self.input_height = input_height
        self.input_width = input_width
        self.conv_specs = list(conv_specs)
        self.dense_layer_sizes = dense_layer_sizes

        self.conv_layers, dense_layers, self.output_layer = build_conv_array_network_layers(
            input_height,
            input_width,
            self.conv_specs,
            dense_layer_sizes,
            class_count,
            conv_cls=self.conv_layer_cls,
            pool_cls=self.pool_layer_cls,
            dense_cls=self.hidden_layer_cls,
        )
        self.layers = self.conv_layers + dense_layers + [self.output_layer]

    def randomize(self) -> None:
        # forward order, as ConvMultiClassBackpropClassifierNetwork.randomize: conv layers from
        # their kernel fan-in (a conv W is (channel_count, fan_in)), pool layers draw nothing,
        # and the dense tail starts from the front end's flattened output size. Each backend
        # draws from its own RNG, so numpy and Rust never reproduce each other from a seed.
        for layer in self.conv_layers:
            if isinstance(layer, self.conv_layer_cls):
                layer.W, layer.b = self.backend.random_layer(layer.channel_count, layer.fan_in)

        previous_size = self.conv_layers[-1].size
        for layer in self.layers[len(self.conv_layers) :]:
            layer.W, layer.b = self.backend.random_layer(layer.size, previous_size)
            previous_size = layer.size

    def snapshot(self) -> list[tuple]:
        return [
            () if isinstance(layer, self.pool_layer_cls) else (layer.W.copy(), layer.b.copy())
            for layer in self.layers
        ]

    def restore(self, snapshot: list[tuple]) -> None:
        # accepts either backend's snapshot, or a loaded file's lists
        for layer, entry in zip(self.layers, snapshot):
            if isinstance(layer, self.pool_layer_cls):
                assert len(entry) == 0, f"a {type(layer).__name__} has no state to restore; got {entry!r}"
                continue
            W, b = entry
            layer.W = self.backend.owned(W)
            layer.b = self.backend.owned(b)

    def save(self, path: str) -> None:
        save_conv_array_model_json(path, self)

    @classmethod
    def load(cls, path: str):
        return load_conv_model_json(cls, path)
