from __future__ import annotations

from dataclasses import asdict

from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_network_base import fan_in_aware_weights_and_bias
from indrajala_ml.model.bounds import validate_class_count, validate_layer_sizes
from indrajala_ml.model.conv_layer import ConvLayer, ConvSpec
from indrajala_ml.model.model_io import load_json, save_json
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.state_layer import StateLayer


class ConvMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    A convolutional sibling of MultiClassBackpropClassifierNetwork - one or more stacked
    ConvLayers ('valid' padding, one ConvSpec each, the first reading the single-channel input
    image, each later one reading the previous layer's channel_count channels) feeding one or
    more ordinary dense hidden layers, then a plain one-vs-rest output layer, exactly like the
    dense-only base class.

    A new class, not a retrofit, for the same reason as every other sibling in this codebase
    (see MultiClassBackpropClassifierNetwork's own docstring) - here specifically because the
    constructor shape genuinely differs: MultiClassBackpropClassifierNetwork.__init__ (via
    BackpropNetworkBase.__init__) assumes every hidden layer is built the same way from a flat
    layer_sizes: list[int] and one hidden_layer_cls; a ConvLayer's own constructor needs conv
    hyperparameters (a ConvSpec plus the input shape it's chained from), not a single int size,
    so this class does not call super().__init__() at all - it builds
    input_layer/hidden_layers/output_layer/trainable_layers directly, in the exact shape
    BackpropNetworkBase's inherited methods (_forward_outputs, _backward_hidden_layers, the
    five gradient/persistence methods) already expect. Those methods, and learn/learn_batch/
    _backward/classify_state/predict_probabilities, are inherited completely unchanged - none
    of them reach into layer internals directly, they go through the same per-layer hooks
    (compute_hidden_deltas/downstream_sum/accumulate_gradients/apply_accumulated_gradients/
    snapshot_state/restore_state) ConvLayer itself implements.

    Every real use case here is a normalized-pixel image (UCI digits, MNIST), so input_bounds
    is not a constructor parameter the way it is for the dense-only base class's more general
    geometric targets - it's fixed internally to [(0.0, 1.0)] * dimension, the same convention
    demo_mnist_ensemble_recognition.py's own MNIST training already uses.
    """

    def __init__(
        self,
        input_height: int,
        input_width: int,
        conv_specs: list[ConvSpec],
        dense_layer_sizes: list[int],
        class_count: int,
    ) -> None:

        validate_class_count(class_count)
        validate_layer_sizes(dense_layer_sizes, label="dense_layer_sizes", noun="dense hidden layer")
        assert len(conv_specs) >= 1, "conv_specs must contain at least one ConvSpec"

        self.class_count = class_count
        self.dimension = input_height * input_width
        self.input_height = input_height
        self.input_width = input_width
        self.conv_specs = list(conv_specs)
        self.dense_layer_sizes = dense_layer_sizes

        self.input_bounds = [(0.0, 1.0)] * self.dimension
        self.input_layer = StateLayer(self.dimension, self.input_bounds)

        self.conv_layers: list[ConvLayer] = []
        previous: StateLayer | ConvLayer = self.input_layer
        height, width, channels = input_height, input_width, 1
        for spec in self.conv_specs:
            layer = ConvLayer(
                input_layer=previous,
                input_height=height,
                input_width=width,
                kernel_size=spec.kernel_size,
                channel_count=spec.channel_count,
                stride=spec.stride,
                input_channels=channels,
            )
            self.conv_layers.append(layer)
            previous = layer
            height, width, channels = layer.out_height, layer.out_width, layer.channel_count

        dense_layers: list[BackpropLayer] = []
        previous_layer: ConvLayer | BackpropLayer = self.conv_layers[-1]
        for size in dense_layer_sizes:
            layer = BackpropLayer(size=size, input_layer=previous_layer)
            dense_layers.append(layer)
            previous_layer = layer

        self.output_layer = BackpropLayer(size=class_count, input_layer=previous_layer)

        self.hidden_layers: list[ConvLayer | BackpropLayer] = self.conv_layers + dense_layers
        self.trainable_layers: list[ConvLayer | BackpropLayer] = self.hidden_layers + [self.output_layer]

    def randomize(self) -> None:
        # every conv layer first, in forward order, each scoped to its own kernel fan-in
        # (kernel_size**2 * input_channels - see ConvKernel.randomize_fan_in_aware)
        for conv_layer in self.conv_layers:
            conv_layer.randomize_fan_in_aware()

        # fan_in_aware_weights_and_bias (backprop_network_base.py) applied directly here rather
        # than via randomize_fan_in_aware(network), since that function assumes every trainable
        # layer is a plain BackpropLayer with a .size attribute and nodes with
        # update_input_weights - true for every dense layer here, but not for the conv layers,
        # which need their own kernel-fan-in-scoped randomize_fan_in_aware() above instead.
        # previous_size starts at the last conv layer's own flattened output size (its true
        # fan-out into the first dense layer), not network.dimension.
        previous_size = len(self.conv_layers[-1].nodes)
        for layer in self.hidden_layers[len(self.conv_layers):] + [self.output_layer]:
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
        conv_specs: list[ConvSpec],
        dense_layer_sizes: list[int],
        class_count: int,
    ) -> "ConvMultiClassBackpropClassifierNetwork":
        network = cls(input_height, input_width, conv_specs, dense_layer_sizes, class_count)
        network.randomize()
        return network

    def save(self, path: str) -> None:
        # not save_model_json (model_io.py) - that envelope hardcodes layer_sizes: list[int],
        # which has no way to express conv hyperparameters. self.snapshot() (inherited
        # unchanged from BackpropNetworkBase) already works correctly here, conv layers
        # included, purely because ConvLayer implements snapshot_state() itself - the
        # per-layer hook BackpropLayer defines for exactly this kind of sibling.
        save_json(
            path,
            {
                "input_height": self.input_height,
                "input_width": self.input_width,
                "conv_layers": [asdict(spec) for spec in self.conv_specs],
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
            conv_specs=[ConvSpec(**spec) for spec in state["conv_layers"]],
            dense_layer_sizes=state["dense_layer_sizes"],
            class_count=state["class_count"],
        )
        network.restore(state["snapshot"])
        return network
