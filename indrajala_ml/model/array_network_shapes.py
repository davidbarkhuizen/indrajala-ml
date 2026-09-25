from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic

from typing_extensions import Self

from indrajala_ml.model.array_network_base import as_weighted_array_layers
from indrajala_ml.model.array_protocols import A, WeightedArrayLayer
from indrajala_ml.model.bounds import validate_class_count, validate_layer_sizes
from indrajala_ml.model.conv_front_end import (
    ArrayConvLayer,
    ArrayFrontEndLayer,
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

if TYPE_CHECKING:
    from indrajala_ml.model.array_network_base import ArrayNetworkBase

    # For the type checker only, each mixin subclasses what it's mixed into, so the attributes
    # and methods its host supplies (backend, layers, _forward, snapshot, ...) resolve; at
    # runtime each is a plain Generic class ahead of the backend's base in the MRO.
    _ShapeBase = ArrayNetworkBase
else:
    _ShapeBase = Generic


class ArrayMultiClassShape(_ShapeBase[A]):
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

    def _classify_output(self, output: A) -> int:
        return self.backend.argmax(output)

    def _classify_output_batch(self, output_batch: A) -> list[int]:
        return self.backend.argmax_rows(output_batch)

    def _target_array(self, category: int) -> A:
        target = self.backend.zeros(self.class_count)
        target[category] = 1.0
        return target

    def _target_batch_array(self, categories: Sequence[int]) -> A:
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
    def load(cls, path: str) -> Self:
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


class ArraySingleOutputShape(_ShapeBase[A]):
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

    def _classify_output(self, output: A) -> float:
        return 1.0 if output.tolist()[0] > 0.5 else 0.0

    def _classify_output_batch(self, output_batch: A) -> list[float]:
        return [1.0 if row[0] > 0.5 else 0.0 for row in output_batch.tolist()]

    def _target_array(self, category: float) -> A:
        return self.backend.vector([category])

    def _target_batch_array(self, categories: Sequence[float]) -> A:
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
    def load(cls, path: str) -> Self:
        state = load_single_output_array_model_json(path)
        network = cls(state["layer_sizes"], state["dimension"], **cls._extra_init_kwargs(state))
        network.restore(state["snapshot"])
        return network


# as _ShapeBase: the conv shape's host is a backend's multiclass network
if TYPE_CHECKING:
    _ConvShapeBase = ArrayMultiClassShape
else:
    _ConvShapeBase = Generic


class ArrayConvShape(_ConvShapeBase[A]):
    """
    The convolutional shape over the multiclass shape, for either backend: a front end of conv and
    max-pool layers (one ConvSpec or PoolSpec each, in order), one or more sigmoid dense layers, and
    a one-vs-rest sigmoid output layer.

    A mixin, listed before the backend's plain multiclass network, which supplies self.backend,
    hidden_layer_cls and output_layer_cls (the dense layer classes). ConvVectorizedMultiClassBackpropClassifierNetwork and
    ConvRustArrayMultiClassBackpropClassifierNetwork are this shape on numpy and on Rust, and set
    conv_layer_cls and pool_layer_cls.

    __init__ doesn't call super().__init__(), whose flat layer_sizes can't describe conv layers; it
    builds self.layers directly. Everything that only walks self.layers through the per-layer hooks
    (the forward pass, learn*, the multiclass shape's outputs and targets) is inherited. Overridden
    is what assumes a dense W in every layer: randomize, snapshot/restore (an empty entry for a pool
    layer) and save/load (the pure-Python conv network's envelope, so a model saved by any of the
    three loads into the others), with the network's hyperparameters in it.

    The conv and dense layers are built through _new_layer, so a sibling's layer classes take the
    network's hyperparameters as the dense networks' do.
    """

    conv_layer_cls: type[ArrayConvLayer[A]]
    pool_layer_cls: type[ArrayFrontEndLayer[A]]

    def __init__(
        self,
        input_height: int,
        input_width: int,
        conv_specs: Sequence[ConvSpec | PoolSpec],
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
            hidden_cls=self.hidden_layer_cls,
            output_cls=self.output_layer_cls,
            new_layer=self._new_layer,
        )
        self.layers = [*self.conv_layers, *dense_layers, self.output_layer]

    def randomize(self) -> None:
        # forward order, as ConvMultiClassBackpropClassifierNetwork.randomize: conv layers from
        # their kernel fan-in (a conv W is (channel_count, fan_in)), pool layers draw nothing,
        # and the dense tail starts from the front end's flattened output size. Each backend
        # draws from its own RNG, so numpy and Rust never reproduce each other from a seed.
        for layer in self.conv_layers:
            if isinstance(layer, self.conv_layer_cls):
                layer.W, layer.b = self.backend.random_layer(layer.channel_count, layer.fan_in)

        previous_size = self.conv_layers[-1].size
        for layer in as_weighted_array_layers(self.layers[len(self.conv_layers) :]):
            layer.W, layer.b = self.backend.random_layer(layer.size, previous_size)
            previous_size = layer.size

    def snapshot(self) -> list[tuple[A, ...]]:
        entries: list[tuple[A, ...]] = []
        for layer in self.layers:
            if isinstance(layer, self.pool_layer_cls):
                entries.append(())
            else:
                assert isinstance(layer, WeightedArrayLayer)  # conv and dense layers have W, b
                entries.append((layer.W.copy(), layer.b.copy()))
        return entries

    def restore(self, snapshot: Sequence[tuple[Any, ...]]) -> None:
        # accepts either backend's snapshot, or a loaded file's lists
        for layer, entry in zip(self.layers, snapshot):
            if isinstance(layer, self.pool_layer_cls):
                assert len(entry) == 0, f"a {type(layer).__name__} has no state to restore; got {entry!r}"
                continue
            assert isinstance(layer, WeightedArrayLayer)
            W, b = entry
            layer.W = self.backend.owned(W)
            layer.b = self.backend.owned(b)

    def save(self, path: str) -> None:
        save_conv_array_model_json(path, self, extra=self._extra_state())

    @classmethod
    def load(cls, path: str) -> Self:
        return load_conv_model_json(cls, path, cls._extra_init_kwargs)
