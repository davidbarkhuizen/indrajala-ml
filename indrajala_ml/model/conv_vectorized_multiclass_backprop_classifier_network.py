from __future__ import annotations

import numpy as np

from indrajala_ml.model.array_layer import ArrayLayer, fan_in_aware_random_layer
from indrajala_ml.model.bounds import validate_class_count, validate_layer_sizes
from indrajala_ml.model.conv_array_layer import ConvArrayLayer
from indrajala_ml.model.conv_front_end import build_conv_front_end, spec_from_json, spec_to_json
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.max_pool_array_layer import MaxPoolArrayLayer
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.model_io import load_json, save_json
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class ConvVectorizedMultiClassBackpropClassifierNetwork(VectorizedMultiClassBackpropClassifierNetwork):
    """
    The numpy sibling of ConvMultiClassBackpropClassifierNetwork: a convolutional front end of
    ConvArrayLayers and MaxPoolArrayLayers (one ConvSpec or PoolSpec each, in order), then one or
    more sigmoid dense ArrayLayers, then a one-vs-rest sigmoid output ArrayLayer.

    Like the pure-Python conv class, this doesn't call super().__init__(): ArrayNetworkBase's
    constructor builds every layer from a flat layer_sizes list, which can't express conv
    hyperparameters. It builds self.layers/self.output_layer directly instead, and everything
    else - predict_probabilities/classify_state/the one-hot targets from
    VectorizedMultiClassBackpropClassifierNetwork, _forward/learn/learn_batch from
    ArrayNetworkBase - is inherited unchanged. Those only iterate self.layers through the
    per-layer hooks (forward*/compute_*_delta*/downstream*/accumulate_gradient*/
    apply_accumulated_gradient) the conv and pool array layers implement.

    What differs from the dense array classes is anything assuming every layer has a dense
    (size, previous_size) W: randomize (conv layers draw from their kernel fan-in, pool layers
    draw nothing), snapshot/restore (an empty entry for a pool layer, mirroring MaxPoolLayer's
    own [] snapshot), and save/load (the pure-Python conv class's JSON envelope keys).
    """

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

        self.conv_layers: list[ConvArrayLayer | MaxPoolArrayLayer] = build_conv_front_end(
            input_height,
            input_width,
            self.conv_specs,
            make_conv=lambda spec, _previous, height, width, channels: ConvArrayLayer(
                input_height=height,
                input_width=width,
                input_channels=channels,
                kernel_size=spec.kernel_size,
                channel_count=spec.channel_count,
                stride=spec.stride,
            ),
            make_pool=lambda spec, _previous, height, width, channels: MaxPoolArrayLayer(
                input_height=height,
                input_width=width,
                input_channels=channels,
                pool_size=spec.pool_size,
                stride=spec.stride,
            ),
        )

        dense_layers: list[ArrayLayer] = []
        previous_size = self.conv_layers[-1].size
        for size in dense_layer_sizes:
            dense_layers.append(ArrayLayer(size, previous_size))
            previous_size = size

        self.output_layer = ArrayLayer(class_count, previous_size)
        self.layers = self.conv_layers + dense_layers + [self.output_layer]

    def randomize(self) -> None:
        # forward order, as ConvMultiClassBackpropClassifierNetwork.randomize: each conv layer
        # scoped to its kernel fan-in (fan_in_aware_random_layer's (size, previous_size) shape is
        # exactly a conv W's (channel_count, input_channels * kernel_size**2)), pool layers draw
        # nothing, and the dense tail's fan-in starts from the last conv/pool layer's flattened
        # output size
        for layer in self.conv_layers:
            if isinstance(layer, ConvArrayLayer):
                layer.W, layer.b = fan_in_aware_random_layer(layer.channel_count, layer.fan_in)

        previous_size = self.conv_layers[-1].size
        for layer in self.layers[len(self.conv_layers) :]:
            layer.W, layer.b = fan_in_aware_random_layer(layer.size, previous_size)
            previous_size = layer.size

    @classmethod
    def randomized(
        cls,
        input_height: int,
        input_width: int,
        conv_specs: list[ConvSpec | PoolSpec],
        dense_layer_sizes: list[int],
        class_count: int,
    ) -> "ConvVectorizedMultiClassBackpropClassifierNetwork":
        network = cls(input_height, input_width, conv_specs, dense_layer_sizes, class_count)
        network.randomize()
        return network

    def snapshot(self) -> list[tuple]:
        return [
            () if isinstance(layer, MaxPoolArrayLayer) else (layer.W.copy(), layer.b.copy()) for layer in self.layers
        ]

    def restore(self, snapshot: list[tuple]) -> None:
        for layer, entry in zip(self.layers, snapshot):
            if isinstance(layer, MaxPoolArrayLayer):
                assert len(entry) == 0, f"a MaxPoolArrayLayer has no state to restore; got {entry!r}"
                continue
            W, b = entry
            layer.W = np.array(W, dtype=np.float64).copy()
            layer.b = np.array(b, dtype=np.float64).copy()

    def save(self, path: str) -> None:
        # the same envelope keys as ConvMultiClassBackpropClassifierNetwork.save, but the
        # snapshot entries are this class's own (W, b) per layer, not per-kernel lists - a file
        # saved by one class doesn't load into the other
        save_json(
            path,
            {
                "input_height": self.input_height,
                "input_width": self.input_width,
                "conv_layers": [spec_to_json(spec) for spec in self.conv_specs],
                "dense_layer_sizes": self.dense_layer_sizes,
                "class_count": self.class_count,
                "snapshot": [[entry[0].tolist(), entry[1].tolist()] if entry else [] for entry in self.snapshot()],
            },
        )

    @classmethod
    def load(cls, path: str) -> "ConvVectorizedMultiClassBackpropClassifierNetwork":
        state = load_json(path)
        network = cls(
            input_height=state["input_height"],
            input_width=state["input_width"],
            conv_specs=[spec_from_json(spec) for spec in state["conv_layers"]],
            dense_layer_sizes=state["dense_layer_sizes"],
            class_count=state["class_count"],
        )
        network.restore(state["snapshot"])
        return network
