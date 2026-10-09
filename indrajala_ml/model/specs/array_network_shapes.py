from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

from indrajala_ml.model.layers.python.conv_front_end import ArrayFrontEndLayer, load_conv_model_state
from indrajala_ml.model.layers.python.conv_layer import ConvSpec
from indrajala_ml.model.layers.python.max_pool_layer import PoolSpec
from indrajala_ml.model.persistence.format2 import NetworkFile
from indrajala_ml.model.protocols.array_protocols import BackendArray
from indrajala_ml.model.specs.bounds import validate_class_count, validate_layer_sizes
from indrajala_ml.model.specs.layer_specs import Dense, LayerSpec, refuse_token_wise_output, token_wise_output
from indrajala_ml.model.specs.spec_shapes import InputShape, spec_shapes
from indrajala_ml.model.specs.update_rules import UpdateRule

if TYPE_CHECKING:
    from indrajala_ml.model.networks.array_network_base import ArrayNetworkBase

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
    predict_probabilities, one-hot targets, and the legacy save envelope with class_count, which
    load still reads. Every array network saves in format 2 (ArrayNetworkBase.save, format2.py).

    A mixin, listed before the backend's base, which supplies self.backend:
    VectorizedMultiClassBackpropClassifierNetwork and RustArrayMultiClassBackpropClassifierNetwork
    are this shape on numpy and on Rust.
    """

    format2_shape: ClassVar[str] = "multiclass"
    preset_arguments: ClassVar[tuple[str, ...] | None] = ("layer_sizes", "dimension", "class_count")

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

    @classmethod
    def _load_legacy(cls, state: dict[str, Any]) -> Self:
        # the legacy envelope: layer_sizes, dimension, class_count and the hyperparameters
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
    predict_probability, a scalar target, and the legacy envelope without class_count. It hosts
    the ensembles' sub-networks, one binary classifier per class.

    A mixin like ArrayMultiClassShape: ArrayBackpropClassifierNetwork and
    RustArrayBackpropClassifierNetwork are this shape on numpy and on Rust.
    """

    format2_shape: ClassVar[str] = "single_output"
    preset_arguments: ClassVar[tuple[str, ...] | None] = ("layer_sizes", "dimension")

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

    @classmethod
    def _load_legacy(cls, state: dict[str, Any]) -> Self:
        # the multiclass shape's legacy envelope without class_count
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
    layer_sizes: the conv specs, then _dense_specs. Its legacy envelope is the pure-Python conv
    network's with one (W, b) entry per layer (an empty one for a pool layer), so a model saved by
    either backend loads into the other, though not into the pure-Python network, with the
    network's hyperparameters in it.
    """

    preset_arguments: ClassVar[tuple[str, ...] | None] = (
        "input_height",
        "input_width",
        "conv_specs",
        "dense_layer_sizes",
        "class_count",
    )

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

    @classmethod
    def _load_legacy(cls, state: dict[str, Any]) -> Self:
        return load_conv_model_state(cls, state, cls._extra_init_kwargs(state))


class SequentialMultiClassShape[A: BackendArray](_ConvShapeBase[A]):
    """
    A multiclass network of any accepted layer specs (layer_specs.py) and update rule, over the
    multiclass shape's argmax classification and one-hot targets: the generic counterpart of the
    presets, which build the same layers from their own constructor arguments. class_count is the
    output layer's size.

    A mixin, listed before the backend's plain multiclass network, as ArrayConvShape. It has no
    legacy envelope: it saves in format 2 (format2.py), without a preset, and loads any format-2
    file of its shape, a preset's included.
    """

    preset_arguments: ClassVar[tuple[str, ...] | None] = None

    def __init__(self, input_shape: InputShape, layers: Sequence[LayerSpec], update_rule: UpdateRule) -> None:
        output = layers[-1] if layers else None
        assert isinstance(output, Dense), f"the last layer must be the output layer, a Dense; got {output!r}"
        refuse_token_wise_output(layers, "multiclass")
        validate_class_count(output.size)

        self.class_count = output.size
        self.dimension = math.prod(input_shape)
        # read by _update_rule, while the base builds the optimizer
        self.update_rule = update_rule

        # past the multiclass shape's __init__, whose flat layer_sizes can't describe these layers,
        # to the backend's base
        super(ArrayMultiClassShape, self).__init__(layers, input_shape)

    def _update_rule(self) -> UpdateRule:
        return self.update_rule

    @classmethod
    def _from_file(cls, file: NetworkFile) -> Self:
        # any format-2 file of its shape, preset or not, numpy or Rust
        return cls(file.input_shape, file.layers, file.update_rule)

    @classmethod
    def _load_legacy(cls, state: dict[str, Any]) -> Self:
        # not the preset parent's envelope, which no Sequential network ever wrote
        raise ValueError(f"{cls.__name__} saves in format 2 only; this file has format {state.get('format')!r}")


class SequentialSingleOutputShape[A: BackendArray](_SingleOutputHostBase[A]):
    """SequentialMultiClassShape over the single-output shape: its output layer has one node."""

    preset_arguments: ClassVar[tuple[str, ...] | None] = None

    def __init__(self, input_shape: InputShape, layers: Sequence[LayerSpec], update_rule: UpdateRule) -> None:
        refuse_token_wise_output(layers, "single-output")
        output = layers[-1] if layers else None
        assert isinstance(output, Dense) and output.size == 1, (
            f"a single-output network's last layer is a one-node Dense; got {output!r}"
        )

        self.dimension = math.prod(input_shape)
        self.update_rule = update_rule

        super(ArraySingleOutputShape, self).__init__(layers, input_shape)

    def _update_rule(self) -> UpdateRule:
        return self.update_rule

    @classmethod
    def _from_file(cls, file: NetworkFile) -> Self:
        # any format-2 file of its shape, preset or not, numpy or Rust
        return cls(file.input_shape, file.layers, file.update_rule)

    @classmethod
    def _load_legacy(cls, state: dict[str, Any]) -> Self:
        # not the preset parent's envelope, which no Sequential network ever wrote
        raise ValueError(f"{cls.__name__} saves in format 2 only; this file has format {state.get('format')!r}")


class SequentialSequenceShape[A: BackendArray](_ShapeBase[A]):
    """
    The sequence shape over ArrayNetworkBase, for either backend (the sequence task workplan, D6):
    a network of any accepted layer specs whose output layer is token-wise (token_wise_output), a
    softmax over class_count classes for each of its tokens tokens. Its label is one class per
    token, a tuple; classify_state is the per-token argmax, and its targets are one-hot per token.
    The loss is the mean of the tokens' cross-entropies (the token-wise output layer's).

    A mixin, listed before the backend's base. It saves in format 2 only, as the other Sequential
    shapes, with the shape "sequence".
    """

    format2_shape: ClassVar[str] = "sequence"
    preset_arguments: ClassVar[tuple[str, ...] | None] = None

    def __init__(self, input_shape: InputShape, layers: Sequence[LayerSpec], update_rule: UpdateRule) -> None:
        assert layers and token_wise_output(layers), (
            "a sequence network's token part ends in its output layer, applied to each token (the sequence task "
            f"workplan, D6); got {list(layers)!r}"
        )
        output = layers[-1]
        assert isinstance(output, Dense)
        validate_class_count(output.size)

        self.class_count = output.size
        self.dimension = math.prod(input_shape)
        self.update_rule = update_rule

        super().__init__(layers, input_shape)
        self.tokens = spec_shapes(layers, input_shape)[-1].output_shape[0]

    def _update_rule(self) -> UpdateRule:
        return self.update_rule

    def predict_probabilities(self, state: tuple[float, ...]) -> list[list[float]]:
        """Each token's class probabilities."""
        values = self._forward(state).tolist()
        width = self.class_count
        return [values[i : i + width] for i in range(0, len(values), width)]

    def classify_state(self, state: tuple[float, ...]) -> tuple[int, ...]:
        return self._classify_output(self._forward(state))

    def _classify_output(self, output: A) -> tuple[int, ...]:
        return self._classify_output_batch(self.backend.matrix([output.tolist()]))[0]

    def _classify_output_batch(self, output_batch: A) -> list[tuple[int, ...]]:
        return self.backend.argmax_token_rows(output_batch, self.class_count)

    def _target_array(self, category: Sequence[int]) -> A:
        target = self.backend.zeros(self.tokens * self.class_count)
        for t, label in enumerate(self._labels(category)):
            target[t * self.class_count + label] = 1.0
        return target

    def _target_batch_array(self, categories: Sequence[Sequence[int]]) -> A:
        target_batch = self.backend.zeros((len(categories), self.tokens * self.class_count))
        for row, category in enumerate(categories):
            for t, label in enumerate(self._labels(category)):
                target_batch[row, t * self.class_count + label] = 1.0
        return target_batch

    def _labels(self, category: Sequence[int]) -> Sequence[int]:
        assert len(category) == self.tokens, (
            f"a sequence network's label is one class per token, {self.tokens}; got {len(category)}"
        )
        return category

    @classmethod
    def _from_file(cls, file: NetworkFile) -> Self:
        return cls(file.input_shape, file.layers, file.update_rule)

    @classmethod
    def _load_legacy(cls, state: dict[str, Any]) -> Self:
        raise ValueError(f"{cls.__name__} saves in format 2 only; this file has format {state.get('format')!r}")
