from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.bounds import validate_class_count, validate_layer_sizes
from indrajala_ml.model.conv_front_end import (
    build_conv_array_network_layers,
    load_conv_array_model_json,
    save_conv_array_model_json,
)
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.conv_rust_array_layer import ConvRustArrayLayer
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.max_pool_rust_array_layer import MaxPoolRustArrayLayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.prepared_dataset import PreparedDataset


def _to_rust_array(values) -> "pa.Array":
    # a pa.Array, a numpy array or nested lists (a loaded file) - all have, or are, lists
    return pa.Array(values.tolist() if hasattr(values, "tolist") else values)


class ConvRustArrayMultiClassBackpropClassifierNetwork(RustArrayMultiClassBackpropClassifierNetwork):
    """
    The Rust sibling of ConvVectorizedMultiClassBackpropClassifierNetwork: a convolutional front
    end of ConvRustArrayLayers and MaxPoolRustArrayLayers (one ConvSpec or PoolSpec each, in
    order), then one or more sigmoid dense RustArrayLayers, then a one-vs-rest sigmoid output
    RustArrayLayer.

    Like the other two conv networks, this doesn't call super().__init__(), whose flat
    layer_sizes list can't express conv hyperparameters; it builds self.layers directly.
    _forward/learn/learn_batch (RustArrayNetworkBase) and predict_probabilities/classify_state/
    the one-hot targets (RustArrayMultiClassBackpropClassifierNetwork) are inherited unchanged.

    What's overridden is anything assuming every layer has a dense (size, previous_size) W:
    randomize (conv layers draw from their kernel fan-in, pool layers draw nothing),
    snapshot/restore (an empty entry for a pool layer), and save/load. The save format is the
    numpy conv network's (save_conv_array_model_json), so a model saved by either backend loads
    into the other.
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

        self.conv_layers, dense_layers, self.output_layer = build_conv_array_network_layers(
            input_height,
            input_width,
            self.conv_specs,
            dense_layer_sizes,
            class_count,
            conv_cls=ConvRustArrayLayer,
            pool_cls=MaxPoolRustArrayLayer,
            dense_cls=RustArrayLayer,
        )
        self.layers = self.conv_layers + dense_layers + [self.output_layer]

    def classify_rows(self, prepared: PreparedDataset) -> list[int]:
        # row by row, not batched: even after candidate 4 (docs/optimizations.md) a batched pass
        # measured a tie for one conv layer and slower with pooling or a second conv layer
        # (candidate 4's stage 2), since the batched conv forward still writes cols it doesn't need
        return [self.classify_row(prepared, index) for index in range(len(prepared))]

    def randomize(self) -> None:
        # the numpy conv network's scheme, drawn from pa.uniform: each conv layer scoped to its
        # kernel fan-in, pool layers draw nothing, and the dense tail's fan-in starts from the
        # last conv/pool layer's flattened output size. Never seed-reproducible against numpy.
        for layer in self.conv_layers:
            if isinstance(layer, ConvRustArrayLayer):
                limit = 1.0 / (layer.fan_in**0.5)
                layer.W = pa.uniform(-limit, limit, (layer.channel_count, layer.fan_in))
                layer.b = pa.uniform(-limit, limit, layer.channel_count)

        previous_size = self.conv_layers[-1].size
        for layer in self.layers[len(self.conv_layers) :]:
            limit = 1.0 / (previous_size**0.5)
            layer.W = pa.uniform(-limit, limit, (layer.size, previous_size))
            layer.b = pa.uniform(-limit, limit, layer.size)
            previous_size = layer.size

    @classmethod
    def randomized(
        cls,
        input_height: int,
        input_width: int,
        conv_specs: list[ConvSpec | PoolSpec],
        dense_layer_sizes: list[int],
        class_count: int,
    ) -> "ConvRustArrayMultiClassBackpropClassifierNetwork":
        network = cls(input_height, input_width, conv_specs, dense_layer_sizes, class_count)
        network.randomize()
        return network

    def snapshot(self) -> list[tuple]:
        return [
            () if isinstance(layer, MaxPoolRustArrayLayer) else (layer.W.copy(), layer.b.copy())
            for layer in self.layers
        ]

    def restore(self, snapshot: list[tuple]) -> None:
        # accepts this network's own snapshot, the numpy conv network's, or a loaded file's lists
        for layer, entry in zip(self.layers, snapshot):
            if isinstance(layer, MaxPoolRustArrayLayer):
                assert len(entry) == 0, f"a MaxPoolRustArrayLayer has no state to restore; got {entry!r}"
                continue
            W, b = entry
            layer.W = _to_rust_array(W)
            layer.b = _to_rust_array(b)

    def save(self, path: str) -> None:
        save_conv_array_model_json(path, self)

    @classmethod
    def load(cls, path: str) -> "ConvRustArrayMultiClassBackpropClassifierNetwork":
        return load_conv_array_model_json(cls, path)
