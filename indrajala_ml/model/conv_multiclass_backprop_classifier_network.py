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
    A convolutional sibling of MultiClassBackpropClassifierNetwork - a convolutional front end
    of stacked ConvLayers and MaxPoolLayers (one ConvSpec or PoolSpec each, in order; the first
    reads the single-channel input image, each later one reads the previous layer's
    channel_count channels) feeding one or more ordinary dense hidden layers, then a plain
    one-vs-rest output layer, exactly like the dense-only base class. conv_specs/conv_layers
    name that whole front end, pooling layers included.

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

    ConvVectorizedMultiClassBackpropClassifierNetwork and
    ConvRustArrayMultiClassBackpropClassifierNetwork are the array-backed siblings (ArrayConvShape
    on each backend), parity-tested against this class step by step
    (tests/test_conv_vectorized_multiclass_backprop_model.py). All three chain their front end
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
        # every conv layer first, in forward order, each scoped to its own kernel fan-in
        # (kernel_size**2 * input_channels - see ConvKernel.randomize_fan_in_aware); a
        # MaxPoolLayer's own randomize_fan_in_aware is a no-op that draws nothing, so adding
        # pooling never shifts any conv layer's random draws
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

    def save(self, path: str) -> None:
        # self.snapshot() (inherited unchanged from BackpropNetworkBase) already works here, conv
        # layers included, because ConvLayer implements snapshot_state() itself - the per-layer
        # hook BackpropLayer defines for exactly this kind of sibling.
        save_conv_model_json(path, self, self.snapshot())

    @classmethod
    def load(cls, path: str) -> "ConvMultiClassBackpropClassifierNetwork":
        return load_conv_model_json(cls, path)
