from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, NoReturn, Self, cast

from indrajala_ml.model.array_layer_builder import InputShape
from indrajala_ml.model.array_protocols import BackendArray
from indrajala_ml.model.bounds import validate_class_count, validate_layer_sizes
from indrajala_ml.model.conv_front_end import ArrayFrontEndLayer, load_conv_model_json, save_conv_array_model_json
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.layer_specs import Dense, LayerSpec
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.model_io import (
    load_array_model_json,
    load_single_output_array_model_json,
    save_array_model_json,
    save_single_output_array_model_json,
)
from indrajala_ml.model.update_rules import UpdateRule

if TYPE_CHECKING:
    from indrajala_ml.model.array_network_base import ArrayNetworkBase

    # For the type checker only, each mixin subclasses what it's mixed into, so the attributes
    # and methods its host supplies (backend, layers, _forward, snapshot, ...) resolve; at
    # runtime each is a plain generic class ahead of the backend's base in the MRO. (The runtime
    # base can't be Generic itself: a class with type parameters already derives from it.)
    _ShapeBase = ArrayNetworkBase
else:

    class _ShapeBase[A]:
        pass


class ArrayMultiClassShape[A: BackendArray](_ShapeBase[A]):
    """
    The multiclass shape over ArrayNetworkBase, for either backend: argmax classify_state,
    predict_probabilities, one-hot targets, and the save/load envelope with class_count.

    A mixin, listed before the backend's base, which supplies self.backend:
    VectorizedMultiClassBackpropClassifierNetwork and RustArrayMultiClassBackpropClassifierNetwork
    are this shape on numpy and on Rust.
    """

    def __init__(self, layer_sizes: list[int], dimension: int, class_count: int) -> None:
        validate_class_count(class_count)
        validate_layer_sizes(layer_sizes)
        self.class_count = class_count
        self.layer_sizes = layer_sizes
        self.dimension = dimension
        super().__init__(self._dense_specs(layer_sizes, class_count), (dimension,))

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


class ArraySingleOutputShape[A: BackendArray](_ShapeBase[A]):
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
        validate_layer_sizes(layer_sizes)
        self.layer_sizes = layer_sizes
        self.dimension = dimension
        super().__init__(self._dense_specs(layer_sizes, 1), (dimension,))

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


# as _ShapeBase: the conv and sequential shapes' host is a backend's multiclass or single-output
# network
if TYPE_CHECKING:
    _ConvShapeBase = ArrayMultiClassShape
    _SingleOutputHostBase = ArraySingleOutputShape
else:
    _ConvShapeBase = _ShapeBase
    _SingleOutputHostBase = _ShapeBase


class ArrayConvShape[A: BackendArray](_ConvShapeBase[A]):
    """
    The convolutional shape over the multiclass shape, for either backend: a front end of conv and
    max-pool layers (one ConvSpec or PoolSpec each, in order), one or more dense hidden layers, and
    the output layer, on a single-channel input_height x input_width image.

    A mixin, listed before the backend's plain multiclass network, which supplies self.backend,
    the targets and the classification. ConvVectorizedMultiClassBackpropClassifierNetwork and
    ConvRustArrayMultiClassBackpropClassifierNetwork are this shape on numpy and on Rust.

    __init__ builds the layers from its own arguments, not the multiclass shape's flat
    layer_sizes: the conv specs, then _dense_specs. Its save/load is the pure-Python conv
    network's envelope with one (W, b) entry per layer (an empty one for a pool layer), so a model
    saved by either backend loads into the other, though not into the pure-Python network, with
    the network's hyperparameters in it.
    """

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
        assert any(isinstance(spec, ConvSpec) for spec in conv_specs), "conv_specs must contain at least one ConvSpec"

        self.class_count = class_count
        self.dimension = input_height * input_width
        self.input_height = input_height
        self.input_width = input_width
        self.conv_specs = list(conv_specs)
        self.dense_layer_sizes = dense_layer_sizes

        # past the multiclass shape's __init__, whose flat layer_sizes can't describe conv layers,
        # to the backend's base
        super(ArrayMultiClassShape, self).__init__(
            [*self.conv_specs, *self._dense_specs(dense_layer_sizes, class_count)],
            (input_height, input_width, 1),
        )
        self.conv_layers = cast("list[ArrayFrontEndLayer[A]]", self.layers[: len(self.conv_specs)])

    def save(self, path: str) -> None:
        save_conv_array_model_json(path, self, extra=self._extra_state())

    @classmethod
    def load(cls, path: str) -> Self:
        return load_conv_model_json(cls, path, cls._extra_init_kwargs)


def _sequential_save_not_yet(network: object) -> NoReturn:
    raise NotImplementedError(
        f"a {type(network).__name__} saves in format 2, which records its layer specs "
        "(docs/composable-layers-workplan.md, stage 5); no legacy envelope can describe them"
    )


class SequentialMultiClassShape[A: BackendArray](_ConvShapeBase[A]):
    """
    A multiclass network of any accepted layer specs (layer_specs.py) and update rule, over the
    multiclass shape's argmax classification and one-hot targets: the generic counterpart of the
    presets, which build the same layers from their own constructor arguments. class_count is the
    output layer's size.

    A mixin, listed before the backend's plain multiclass network, as ArrayConvShape. It has no
    save/load until format 2 (stage 5 of docs/composable-layers-workplan.md).
    """

    def __init__(self, input_shape: InputShape, layers: Sequence[LayerSpec], update_rule: UpdateRule) -> None:
        output = layers[-1] if layers else None
        assert isinstance(output, Dense), f"the last layer must be the output layer, a Dense; got {output!r}"
        validate_class_count(output.size)

        self.class_count = output.size
        self.input_shape = input_shape
        self.dimension = math.prod(input_shape)
        self.layer_specs = list(layers)
        self.update_rule = update_rule

        # past the multiclass shape's __init__, whose flat layer_sizes can't describe these layers,
        # to the backend's base
        super(ArrayMultiClassShape, self).__init__(self.layer_specs, input_shape)

    def _update_rule(self) -> UpdateRule:
        return self.update_rule

    def save(self, path: str) -> None:
        _sequential_save_not_yet(self)

    @classmethod
    def load(cls, path: str) -> Self:
        _sequential_save_not_yet(cls)


class SequentialSingleOutputShape[A: BackendArray](_SingleOutputHostBase[A]):
    """SequentialMultiClassShape over the single-output shape: its output layer has one node."""

    def __init__(self, input_shape: InputShape, layers: Sequence[LayerSpec], update_rule: UpdateRule) -> None:
        output = layers[-1] if layers else None
        assert isinstance(output, Dense) and output.size == 1, (
            f"a single-output network's last layer is a one-node Dense; got {output!r}"
        )

        self.input_shape = input_shape
        self.dimension = math.prod(input_shape)
        self.layer_specs = list(layers)
        self.update_rule = update_rule

        super(ArraySingleOutputShape, self).__init__(self.layer_specs, input_shape)

    def _update_rule(self) -> UpdateRule:
        return self.update_rule

    def save(self, path: str) -> None:
        _sequential_save_not_yet(self)

    @classmethod
    def load(cls, path: str) -> Self:
        _sequential_save_not_yet(cls)
