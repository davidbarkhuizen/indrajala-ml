from __future__ import annotations

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_network_base import fan_in_aware_weights_and_bias
from indrajala_ml.model.bounds import validate_class_count, validate_layer_sizes
from indrajala_ml.model.conv_front_end import build_conv_front_end, load_conv_model_json, save_conv_model_json
from indrajala_ml.model.conv_layer import ConvLayer, ConvSpec
from indrajala_ml.model.max_pool_layer import MaxPoolLayer, PoolSpec
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.state_layer import StateLayer


class ConvMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    A convolutional MultiClassBackpropClassifierNetwork: a front end of ConvLayers and
    MaxPoolLayers (one ConvSpec or PoolSpec each, in order; the first reads the single-channel
    image, each later one the previous layer's channels), then one or more dense hidden layers and a
    one-vs-rest output layer. conv_specs/conv_layers name the whole front end, pooling included.

    __init__ doesn't call super().__init__(), whose flat layer_sizes and single hidden_layer_cls
    can't describe conv layers; it builds input_layer/hidden_layers/output_layer/trainable_layers
    directly, in the shape BackpropNetworkBase's methods expect. Those, and learn/learn_batch/
    _backward/classify_state/predict_probabilities, are inherited: they go through per-layer hooks
    (compute_hidden_deltas, downstream_sum, the gradient and snapshot methods) that ConvLayer
    implements.

    Inputs are normalized pixels, so input_bounds is fixed at [(0.0, 1.0)] * dimension, not a
    parameter.

    The numpy and Rust networks (ArrayConvShape) are parity-tested against this one step by step
    (tests/test_conv_array_multiclass_backprop_model.py); all three build their front end
    through conv_front_end.build_conv_front_end.
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

        self.input_bounds = [(0.0, 1.0)] * self.dimension
        self.input_layer = StateLayer(self.dimension, self.input_bounds)

        self.conv_layers: list[ConvLayer | MaxPoolLayer] = build_conv_front_end(
            input_height,
            input_width,
            self.conv_specs,
            make_conv=lambda spec, previous, height, width, channels: ConvLayer(
                input_layer=previous,
                input_height=height,
                input_width=width,
                kernel_size=spec.kernel_size,
                channel_count=spec.channel_count,
                stride=spec.stride,
                input_channels=channels,
            ),
            make_pool=lambda spec, previous, height, width, channels: MaxPoolLayer(
                input_layer=previous,
                input_height=height,
                input_width=width,
                input_channels=channels,
                pool_size=spec.pool_size,
                stride=spec.stride,
            ),
            input_layer=self.input_layer,
        )

        dense_layers: list[BackpropLayer] = []
        previous_layer: ConvLayer | MaxPoolLayer | BackpropLayer = self.conv_layers[-1]
        for size in dense_layer_sizes:
            layer = BackpropLayer(size=size, input_layer=previous_layer)
            dense_layers.append(layer)
            previous_layer = layer

        self.output_layer = BackpropLayer(size=class_count, input_layer=previous_layer)

        self.hidden_layers: list[ConvLayer | MaxPoolLayer | BackpropLayer] = self.conv_layers + dense_layers
        self.trainable_layers: list[ConvLayer | MaxPoolLayer | BackpropLayer] = self.hidden_layers + [self.output_layer]

    def randomize(self) -> None:
        # conv layers first, in forward order, each from its kernel fan-in
        # (kernel_size**2 * input_channels); a MaxPoolLayer draws nothing, so adding pooling
        # never shifts a conv layer's draws
        for conv_layer in self.conv_layers:
            conv_layer.randomize_fan_in_aware()

        # the dense tail, not randomize_fan_in_aware(network), which assumes every trainable
        # layer is dense; the first dense layer's fan-in is the front end's flattened output
        previous_size = len(self.conv_layers[-1].nodes)
        for layer in self.hidden_layers[len(self.conv_layers):] + [self.output_layer]:
            for node in layer.nodes:
                weights, bias = fan_in_aware_weights_and_bias(previous_size)
                node.update_input_weights(weights)
                node.bias = bias
            previous_size = layer.size

    def save(self, path: str) -> None:
        # the inherited snapshot() covers conv layers through their snapshot_state()
        save_conv_model_json(path, self, self.snapshot())

    @classmethod
    def load(cls, path: str) -> "ConvMultiClassBackpropClassifierNetwork":
        return load_conv_model_json(cls, path)
