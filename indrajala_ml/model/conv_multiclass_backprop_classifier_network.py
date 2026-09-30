from __future__ import annotations

from collections.abc import Sequence
from typing import Self, cast

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.bounds import validate_class_count, validate_layer_sizes
from indrajala_ml.model.conv_front_end import load_conv_model_json, save_conv_model_json
from indrajala_ml.model.conv_layer import ConvLayer, ConvSpec
from indrajala_ml.model.max_pool_layer import MaxPoolLayer, PoolSpec
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork


class ConvMultiClassBackpropClassifierNetwork(
    MultiClassBackpropClassifierNetwork[ConvLayer | MaxPoolLayer | BackpropLayer]
):
    """
    A convolutional MultiClassBackpropClassifierNetwork: a front end of ConvLayers and
    MaxPoolLayers (one ConvSpec or PoolSpec each, in order; the first reads the single-channel
    image, each later one the previous layer's channels), then one or more dense hidden layers and a
    one-vs-rest output layer. conv_specs/conv_layers name the whole front end, pooling included.

    Its layer specs are conv_specs followed by the dense specs (_dense_specs), built by
    BackpropNetworkBase over a (input_height, input_width, 1) input, past the multiclass network's
    __init__, whose flat layer_sizes can't describe conv layers. learn/learn_batch/_backward/
    classify_state/predict_probabilities and the fan-in-aware randomize are inherited: they go
    through per-layer hooks (compute_hidden_deltas, downstream_sum, the gradient, randomize and
    snapshot methods, weight_sets) that ConvLayer and MaxPoolLayer implement.

    Inputs are normalized pixels, so input_bounds is fixed at [(0.0, 1.0)] * dimension, not a
    parameter.

    The update is the optimizer's: a sibling overrides _update_rule, as
    MomentumConvMultiClassBackpropClassifierNetwork does.

    The numpy and Rust networks (ArrayConvShape) are parity-tested against this one step by step
    (tests/test_conv_array_multiclass_backprop_model.py); all three build their layers from the
    same specs (python_layer_builder.py, array_layer_builder.py).
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
        self.input_height = input_height
        self.input_width = input_width
        self.conv_specs = list(conv_specs)
        self.dense_layer_sizes = dense_layer_sizes

        dimension = input_height * input_width
        # past the multiclass network's __init__ to BackpropNetworkBase's
        super(MultiClassBackpropClassifierNetwork, self).__init__(
            [*self.conv_specs, *self._dense_specs(dense_layer_sizes, class_count)],
            (input_height, input_width, 1),
            [(0.0, 1.0)] * dimension,
        )
        self.conv_layers = cast("list[ConvLayer | MaxPoolLayer]", self.hidden_layers[: len(self.conv_specs)])

    def save(self, path: str) -> None:
        # the inherited snapshot() covers conv layers through their snapshot_state()
        save_conv_model_json(path, self, self.snapshot())

    @classmethod
    def load(cls, path: str) -> Self:
        return load_conv_model_json(cls, path)
