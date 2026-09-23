from __future__ import annotations

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_network_base import fan_in_aware_weights_and_bias
from indrajala_ml.model.bounds import validate_class_count, validate_layer_sizes
from indrajala_ml.model.conv_layer import ConvLayer
from indrajala_ml.model.model_io import load_json, save_json
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.state_layer import StateLayer


class ConvMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    A convolutional sibling of MultiClassBackpropClassifierNetwork - one ConvLayer (v1 scope:
    directly after the input, single input channel, 'valid' padding) feeding one or more
    ordinary dense hidden layers, then a plain one-vs-rest output layer, exactly like the
    dense-only base class.

    A new class, not a retrofit, for the same reason as every other sibling in this codebase
    (see MultiClassBackpropClassifierNetwork's own docstring) - here specifically because the
    constructor shape genuinely differs: MultiClassBackpropClassifierNetwork.__init__ (via
    BackpropNetworkBase.__init__) assumes every hidden layer is built the same way from a flat
    layer_sizes: list[int] and one hidden_layer_cls; a ConvLayer's own constructor needs conv
    hyperparameters (kernel_size, channel_count, stride, input_height/width), not a single int
    size, so this class does not call super().__init__() at all - it builds
    input_layer/hidden_layers/output_layer/trainable_layers directly, in the exact shape
    BackpropNetworkBase's inherited methods (_forward_outputs, _backward_hidden_layers, the
    five gradient/persistence methods) already expect. Those methods, and learn/learn_batch/
    _backward/classify_state/predict_probabilities, are inherited completely unchanged - none
    of them reach into layer internals directly, they go through the same per-layer hooks
    (accumulate_gradients/apply_accumulated_gradients/snapshot_state/restore_state) ConvLayer
    itself implements.

    Every real use case here is a normalized-pixel image (UCI digits, MNIST), so input_bounds
    is not a constructor parameter the way it is for the dense-only base class's more general
    geometric targets - it's fixed internally to [(0.0, 1.0)] * dimension, the same convention
    demo_mnist_ensemble_recognition.py's own MNIST training already uses.
    """

    def __init__(
        self,
        input_height: int,
        input_width: int,
        kernel_size: int,
        channel_count: int,
        dense_layer_sizes: list[int],
        class_count: int,
        stride: int = 1,
    ) -> None:

        validate_class_count(class_count)
        validate_layer_sizes(dense_layer_sizes, label="dense_layer_sizes", noun="dense hidden layer")

        self.class_count = class_count
        self.dimension = input_height * input_width
        self.input_height = input_height
        self.input_width = input_width
        self.kernel_size = kernel_size
        self.channel_count = channel_count
        self.dense_layer_sizes = dense_layer_sizes
        self.stride = stride

        self.input_bounds = [(0.0, 1.0)] * self.dimension
        self.input_layer = StateLayer(self.dimension, self.input_bounds)

        self.conv_layer = ConvLayer(
            input_layer=self.input_layer,
            input_height=input_height,
            input_width=input_width,
            kernel_size=kernel_size,
            channel_count=channel_count,
            stride=stride,
        )

        dense_layers: list[BackpropLayer] = []
        previous_layer: ConvLayer | BackpropLayer = self.conv_layer
        for size in dense_layer_sizes:
            layer = BackpropLayer(size=size, input_layer=previous_layer)
            dense_layers.append(layer)
            previous_layer = layer

        self.output_layer = BackpropLayer(size=class_count, input_layer=previous_layer)

        self.hidden_layers: list[ConvLayer | BackpropLayer] = [self.conv_layer] + dense_layers
        self.trainable_layers: list[ConvLayer | BackpropLayer] = self.hidden_layers + [self.output_layer]

    def randomize(self) -> None:
        self.conv_layer.randomize_fan_in_aware()

        # fan_in_aware_weights_and_bias (backprop_network_base.py) applied directly here rather
        # than via randomize_fan_in_aware(network), since that function assumes every trainable
        # layer is a plain BackpropLayer with a .size attribute and nodes with
        # update_input_weights - true for every dense layer here, but not for self.conv_layer,
        # which needs its own kernel-fan-in-scoped randomize_fan_in_aware() above instead.
        # previous_size starts at the conv layer's own flattened output size (its true fan-out
        # into the first dense layer), not network.dimension.
        previous_size = len(self.conv_layer.nodes)
        for layer in self.hidden_layers[1:] + [self.output_layer]:
            for node in layer.nodes:
                weights, bias = fan_in_aware_weights_and_bias(previous_size)
                node.update_input_weights(weights)
                node.bias = bias
            previous_size = layer.size

    @classmethod
    def randomized(
        cls,
        input_height: int,
        input_width: int,
        kernel_size: int,
        channel_count: int,
        dense_layer_sizes: list[int],
        class_count: int,
        stride: int = 1,
    ) -> "ConvMultiClassBackpropClassifierNetwork":
        network = cls(input_height, input_width, kernel_size, channel_count, dense_layer_sizes, class_count, stride)
        network.randomize()
        return network

    def save(self, path: str) -> None:
        # not save_model_json (model_io.py) - that envelope hardcodes layer_sizes: list[int],
        # which has no way to express conv hyperparameters. self.snapshot() (inherited
        # unchanged from BackpropNetworkBase) already works correctly here, conv layer
        # included, purely because ConvLayer implements snapshot_state() itself - the
        # per-layer hook BackpropLayer defines for exactly this kind of sibling.
        save_json(
            path,
            {
                "input_height": self.input_height,
                "input_width": self.input_width,
                "kernel_size": self.kernel_size,
                "channel_count": self.channel_count,
                "stride": self.stride,
                "dense_layer_sizes": self.dense_layer_sizes,
                "class_count": self.class_count,
                "snapshot": self.snapshot(),
            },
        )

    @classmethod
    def load(cls, path: str) -> "ConvMultiClassBackpropClassifierNetwork":
        state = load_json(path)

        network = cls(
            input_height=state["input_height"],
            input_width=state["input_width"],
            kernel_size=state["kernel_size"],
            channel_count=state["channel_count"],
            dense_layer_sizes=state["dense_layer_sizes"],
            class_count=state["class_count"],
            stride=state["stride"],
        )
        network.restore(state["snapshot"])
        return network
