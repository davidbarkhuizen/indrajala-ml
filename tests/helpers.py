import random
from typing import Callable

import numpy as np
import pytest

from indrajala_ml.model.adam_layer import make_adam_layer_cls
from indrajala_ml.model.binary_cross_entropy_backprop_classifier_network import (
    BinaryCrossEntropyBackpropClassifierNetwork,
    CrossEntropyOutputLayer,
)
from indrajala_ml.model.conv_array_layer import ConvArrayLayer
from indrajala_ml.model.conv_multiclass_backprop_classifier_network import ConvMultiClassBackpropClassifierNetwork
from indrajala_ml.model.conv_rust_array_multiclass_backprop_classifier_network import (
    ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.conv_vectorized_multiclass_backprop_classifier_network import (
    ConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.dropout_layer import make_dropout_layer_cls
from indrajala_ml.model.fan_in_aware_backprop_classifier_network import FanInAwareBackpropClassifierNetwork
from indrajala_ml.model.l2_regularization_layer import make_l2_layer_cls
from indrajala_ml.model.linear_classifier_network import LinearClassifierNetwork
from indrajala_ml.model.momentum_layer import make_momentum_layer_cls
from indrajala_ml.model.multiclass_backprop_classifier_network import MultiClassBackpropClassifierNetwork
from indrajala_ml.model.relu_layer import ReLULayer
from indrajala_ml.model.softmax_multiclass_backprop_classifier_network import (
    SoftmaxMultiClassBackpropClassifierNetwork,
)


def assert_save_and_load_round_trip(network, load_fn, tmp_path, filename: str, states):
    """
    Shared core of every save/load round-trip test in this codebase (the *multiclass/softmax/
    ensemble/conv sibling model classes): save network to a temp path, reload it via load_fn,
    and confirm the reloaded network's snapshot and predictions match the original exactly.
    Returns the loaded network so each call site can layer its own class-specific assertions
    (dimension, class_count, input_bounds, conv hyperparameters, ...) on top.
    """
    path = str(tmp_path / filename)
    network.save(path)
    loaded = load_fn(path)

    assert loaded.snapshot() == network.snapshot()
    for state in states:
        assert loaded.classify_state(state) == network.classify_state(state)
        assert loaded.predict_probabilities(state) == pytest.approx(network.predict_probabilities(state))

    return loaded


def assert_randomize_breaks_symmetry(network) -> None:
    """
    Shared by every *_backprop_model.py's test_randomize_breaks_symmetry_between_nodes_...:
    identical starting weights across nodes in the same layer would receive identical gradients
    forever and the layer would collapse to one effective unit, so randomize() must give each
    node independent random weights, not a shared default.
    """
    weight_sets = [tuple(node.input_node_weights) for node in network.hidden_layers[0].nodes]
    assert len(set(weight_sets)) == len(weight_sets)


def assert_snapshot_restore_round_trip(network, step, times: int = 5) -> None:
    """
    Shared by every *_backprop_model.py's/ensemble's/conv sibling's
    test_snapshot_and_restore_round_trip: snapshot, apply `step` `times` times (confirming the
    network actually moved), restore, and confirm the snapshot matches the pre-training one
    exactly again.
    """
    before = network.snapshot()

    for _ in range(times):
        step()

    assert network.snapshot() != before

    network.restore(before)

    assert network.snapshot() == before


def wire_fixed_single_hidden_node(network) -> None:
    """
    Shared single-hidden-node/single-output-node weight wiring behind every
    *_backprop_model.py's own _fixed_network(): hidden weight=0.5, bias=0.1, output weight=0.8,
    bias=-0.2 - the exact values test_backprop_model.py's own hand-computed forward/backward
    tests derive their expected numbers from, reused by every sibling that needs the identical
    starting point for a side-by-side comparison.
    """
    hidden_node = network.hidden_layers[0].nodes[0]
    output_node = network.output_layer.nodes[0]
    hidden_node.update_input_weights([0.5])
    hidden_node.bias = 0.1
    output_node.update_input_weights([0.8])
    output_node.bias = -0.2


def matching_array_backprop_networks(
    rng: random.Random,
    array_network_cls,
    wrap: Callable,
    layer_sizes: list[int],
    dimension: int,
    class_count: int,
    bounds: float = 10.0,
):
    """
    Shared by test_vectorized_multiclass_backprop_model.py's and
    test_rust_array_multiclass_backprop_model.py's own _matching_networks: builds a
    MultiClassBackpropClassifierNetwork and an array-backed sibling
    (VectorizedMultiClassBackpropClassifierNetwork / RustArrayMultiClassBackpropClassifierNetwork,
    passed as array_network_cls) with identical injected weights. Neither array backend's RNG
    stream is meaningfully comparable to Python's random module, so initial weights are always
    forced identical explicitly here instead of via each network's own randomize(). `wrap`
    converts a nested Python list of weights (or a
    flat list of biases) into the array backend's own array type - np.array for the numpy
    sibling, pa.Array for the Rust one.
    """
    node_network = MultiClassBackpropClassifierNetwork(
        layer_sizes, dimension, [(-bounds, bounds)] * dimension, class_count
    )
    array_network = array_network_cls(layer_sizes, dimension, class_count)

    previous_size = dimension
    for layer_index, size in enumerate([*layer_sizes, class_count]):
        weights = [[rng.uniform(-2.0, 2.0) for _ in range(previous_size)] for _ in range(size)]
        biases = [rng.uniform(-2.0, 2.0) for _ in range(size)]

        node_layer = node_network.trainable_layers[layer_index]
        for node, node_weights, bias in zip(node_layer.nodes, weights, biases):
            node.update_input_weights(node_weights)
            node.bias = bias

        array_network.layers[layer_index].W = wrap(weights)
        array_network.layers[layer_index].b = wrap(biases)

        previous_size = size

    return node_network, array_network


def matching_single_output_array_backprop_networks(
    rng: random.Random,
    array_network_cls,
    wrap: Callable,
    layer_sizes: list[int],
    dimension: int,
    bounds: float = 10.0,
):
    """
    The single-output analogue of matching_array_backprop_networks above, for
    ArrayBackpropClassifierNetwork/RustArrayBackpropClassifierNetwork: builds a
    FanInAwareBackpropClassifierNetwork per-node reference (not plain
    BackpropClassifierNetwork - the array sibling's own randomize() is fan-in-aware only, per its
    own docstring, so this is the genuinely matching per-node scheme) and an array-backed
    sibling with identical injected weights, the same "force identical, never rely on
    randomize()" reasoning matching_array_backprop_networks's own docstring gives.
    """
    node_network = FanInAwareBackpropClassifierNetwork(layer_sizes, dimension, [(-bounds, bounds)] * dimension)
    array_network = array_network_cls(layer_sizes, dimension)

    previous_size = dimension
    for layer_index, size in enumerate([*layer_sizes, 1]):
        weights = [[rng.uniform(-2.0, 2.0) for _ in range(previous_size)] for _ in range(size)]
        biases = [rng.uniform(-2.0, 2.0) for _ in range(size)]

        node_layer = node_network.trainable_layers[layer_index]
        for node, node_weights, bias in zip(node_layer.nodes, weights, biases):
            node.update_input_weights(node_weights)
            node.bias = bias

        array_network.layers[layer_index].W = wrap(weights)
        array_network.layers[layer_index].b = wrap(biases)

        previous_size = size

    return node_network, array_network


def matching_cross_entropy_array_backprop_networks(
    rng: random.Random,
    array_network_cls,
    wrap: Callable,
    layer_sizes: list[int],
    dimension: int,
    bounds: float = 10.0,
):
    """
    The cross-entropy analogue of matching_single_output_array_backprop_networks above, for
    CrossEntropyArrayBackpropClassifierNetwork/CrossEntropyRustArrayBackpropClassifierNetwork:
    builds a BinaryCrossEntropyBackpropClassifierNetwork per-node reference (the genuine per-node
    counterpart, unlike matching_single_output_array_backprop_networks's own
    FanInAwareBackpropClassifierNetwork, which only matches on init scheme, not loss function) and
    a cross-entropy array-backed sibling with identical injected weights - the same "force
    identical, never rely on randomize()" reasoning matching_array_backprop_networks's own
    docstring gives.
    """
    node_network = BinaryCrossEntropyBackpropClassifierNetwork(layer_sizes, dimension, [(-bounds, bounds)] * dimension)
    array_network = array_network_cls(layer_sizes, dimension)

    previous_size = dimension
    for layer_index, size in enumerate([*layer_sizes, 1]):
        weights = [[rng.uniform(-2.0, 2.0) for _ in range(previous_size)] for _ in range(size)]
        biases = [rng.uniform(-2.0, 2.0) for _ in range(size)]

        node_layer = node_network.trainable_layers[layer_index]
        for node, node_weights, bias in zip(node_layer.nodes, weights, biases):
            node.update_input_weights(node_weights)
            node.bias = bias

        array_network.layers[layer_index].W = wrap(weights)
        array_network.layers[layer_index].b = wrap(biases)

        previous_size = size

    return node_network, array_network


class AdamMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    Test-only per-node Adam reference: MultiClassBackpropClassifierNetwork with its
    hidden_layer_cls/output_layer_cls extension points (BackpropNetworkBase) set to
    make_adam_layer_cls's node class, the exact same construction
    AdamBackpropClassifierNetwork uses for the single-output case. Not a new production class,
    since this codebase has no per-node multi-class Adam sibling to build against otherwise. Gives
    every array-based Adam sibling (AdamVectorizedMultiClassBackpropClassifierNetwork,
    AdamRustArrayMultiClassBackpropClassifierNetwork) a genuine parity reference instead of a
    hand-derived fixture, shared here rather than duplicated per test module, since both
    siblings need the identical reference.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        class_count: int,
        beta1: float,
        beta2: float,
        epsilon: float,
    ) -> None:
        layer_cls = make_adam_layer_cls(beta1, beta2, epsilon)
        self.hidden_layer_cls = layer_cls
        self.output_layer_cls = layer_cls
        super().__init__(layer_sizes, dimension, input_bounds, class_count)


def matching_adam_array_backprop_networks(
    rng: random.Random,
    array_network_cls,
    wrap: Callable,
    layer_sizes: list[int],
    dimension: int,
    class_count: int,
    beta1: float,
    beta2: float,
    epsilon: float,
    bounds: float = 10.0,
):
    """
    The Adam-sibling analogue of matching_array_backprop_networks above: builds an
    AdamMultiClassBackpropClassifierNetwork per-node reference and an Adam array-backed sibling
    (AdamVectorizedMultiClassBackpropClassifierNetwork / AdamRustArrayMultiClassBackpropClassifierNetwork,
    passed as array_network_cls) with identical injected weights - kept separate from the non-Adam
    helper since the reference class and constructor signature both differ (beta1/beta2/epsilon).
    """
    node_network = AdamMultiClassBackpropClassifierNetwork(
        layer_sizes, dimension, [(-bounds, bounds)] * dimension, class_count, beta1, beta2, epsilon
    )
    array_network = array_network_cls(layer_sizes, dimension, class_count, beta1, beta2, epsilon)

    previous_size = dimension
    for layer_index, size in enumerate([*layer_sizes, class_count]):
        weights = [[rng.uniform(-2.0, 2.0) for _ in range(previous_size)] for _ in range(size)]
        biases = [rng.uniform(-2.0, 2.0) for _ in range(size)]

        node_layer = node_network.trainable_layers[layer_index]
        for node, node_weights, bias in zip(node_layer.nodes, weights, biases):
            node.update_input_weights(node_weights)
            node.bias = bias

        array_network.layers[layer_index].W = wrap(weights)
        array_network.layers[layer_index].b = wrap(biases)

        previous_size = size

    return node_network, array_network


class L2MultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    Test-only per-node L2 reference: MultiClassBackpropClassifierNetwork with its
    hidden_layer_cls/output_layer_cls extension points set to make_l2_layer_cls's node class -
    the same construction L2RegularizedBackpropClassifierNetwork uses for the single-output
    case. Gives L2VectorizedMultiClassBackpropClassifierNetwork a genuine parity reference.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        class_count: int,
        l2_lambda: float,
    ) -> None:
        layer_cls = make_l2_layer_cls(l2_lambda)
        self.hidden_layer_cls = layer_cls
        self.output_layer_cls = layer_cls
        super().__init__(layer_sizes, dimension, input_bounds, class_count)


def matching_l2_array_backprop_networks(
    rng: random.Random,
    array_network_cls,
    wrap: Callable,
    layer_sizes: list[int],
    dimension: int,
    class_count: int,
    l2_lambda: float,
    bounds: float = 10.0,
):
    """
    The L2-sibling analogue of matching_array_backprop_networks above - see
    matching_adam_array_backprop_networks's own docstring for the general shape this follows.
    """
    node_network = L2MultiClassBackpropClassifierNetwork(
        layer_sizes, dimension, [(-bounds, bounds)] * dimension, class_count, l2_lambda
    )
    array_network = array_network_cls(layer_sizes, dimension, class_count, l2_lambda)

    previous_size = dimension
    for layer_index, size in enumerate([*layer_sizes, class_count]):
        weights = [[rng.uniform(-2.0, 2.0) for _ in range(previous_size)] for _ in range(size)]
        biases = [rng.uniform(-2.0, 2.0) for _ in range(size)]

        node_layer = node_network.trainable_layers[layer_index]
        for node, node_weights, bias in zip(node_layer.nodes, weights, biases):
            node.update_input_weights(node_weights)
            node.bias = bias

        array_network.layers[layer_index].W = wrap(weights)
        array_network.layers[layer_index].b = wrap(biases)

        previous_size = size

    return node_network, array_network


class MomentumMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    Test-only per-node momentum reference: MultiClassBackpropClassifierNetwork with its
    hidden_layer_cls/output_layer_cls extension points set to make_momentum_layer_cls's node
    class - the same construction MomentumBackpropClassifierNetwork uses for the single-output
    case. Gives MomentumVectorizedMultiClassBackpropClassifierNetwork a genuine parity reference.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        class_count: int,
        momentum: float,
    ) -> None:
        layer_cls = make_momentum_layer_cls(momentum)
        self.hidden_layer_cls = layer_cls
        self.output_layer_cls = layer_cls
        super().__init__(layer_sizes, dimension, input_bounds, class_count)


def matching_momentum_array_backprop_networks(
    rng: random.Random,
    array_network_cls,
    wrap: Callable,
    layer_sizes: list[int],
    dimension: int,
    class_count: int,
    momentum: float,
    bounds: float = 10.0,
):
    """
    The momentum-sibling analogue of matching_array_backprop_networks above - see
    matching_adam_array_backprop_networks's own docstring for the general shape this follows.
    """
    node_network = MomentumMultiClassBackpropClassifierNetwork(
        layer_sizes, dimension, [(-bounds, bounds)] * dimension, class_count, momentum
    )
    array_network = array_network_cls(layer_sizes, dimension, class_count, momentum)

    previous_size = dimension
    for layer_index, size in enumerate([*layer_sizes, class_count]):
        weights = [[rng.uniform(-2.0, 2.0) for _ in range(previous_size)] for _ in range(size)]
        biases = [rng.uniform(-2.0, 2.0) for _ in range(size)]

        node_layer = node_network.trainable_layers[layer_index]
        for node, node_weights, bias in zip(node_layer.nodes, weights, biases):
            node.update_input_weights(node_weights)
            node.bias = bias

        array_network.layers[layer_index].W = wrap(weights)
        array_network.layers[layer_index].b = wrap(biases)

        previous_size = size

    return node_network, array_network


class ReLUMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    Test-only per-node ReLU reference: MultiClassBackpropClassifierNetwork with its
    hidden_layer_cls extension point set to ReLULayer - the output layer stays the default
    plain BackpropLayer (sigmoid), matching ReLUNode's hidden-layer-only convention (the same
    construction ReLUBackpropClassifierNetwork uses for the single-output case). Gives
    ReLUVectorizedMultiClassBackpropClassifierNetwork a genuine parity reference.
    """

    hidden_layer_cls = ReLULayer


def matching_relu_array_backprop_networks(
    rng: random.Random,
    array_network_cls,
    wrap: Callable,
    layer_sizes: list[int],
    dimension: int,
    class_count: int,
    bounds: float = 10.0,
):
    """
    The ReLU-sibling analogue of matching_array_backprop_networks above - see
    matching_adam_array_backprop_networks's own docstring for the general shape this follows. No
    extra coefficient argument, unlike the Adam/L2/momentum analogues - ReLU has none.
    """
    node_network = ReLUMultiClassBackpropClassifierNetwork(
        layer_sizes, dimension, [(-bounds, bounds)] * dimension, class_count
    )
    array_network = array_network_cls(layer_sizes, dimension, class_count)

    previous_size = dimension
    for layer_index, size in enumerate([*layer_sizes, class_count]):
        weights = [[rng.uniform(-2.0, 2.0) for _ in range(previous_size)] for _ in range(size)]
        biases = [rng.uniform(-2.0, 2.0) for _ in range(size)]

        node_layer = node_network.trainable_layers[layer_index]
        for node, node_weights, bias in zip(node_layer.nodes, weights, biases):
            node.update_input_weights(node_weights)
            node.bias = bias

        array_network.layers[layer_index].W = wrap(weights)
        array_network.layers[layer_index].b = wrap(biases)

        previous_size = size

    return node_network, array_network


class DropoutMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    Test-only per-node dropout reference: MultiClassBackpropClassifierNetwork with its
    hidden_layer_cls extension point set to make_dropout_layer_cls(drop_probability)'s layer
    class - the output layer stays the default plain BackpropLayer (sigmoid), matching
    DropoutNode's hidden-layer-only convention. No genuine per-node
    DropoutMultiClassBackpropClassifierNetwork sibling exists in this codebase
    (DropoutBackpropClassifierNetwork is scoped to the single-output case only), so this exists
    purely to give DropoutVectorizedMultiClassBackpropClassifierNetwork/
    DropoutRustArrayMultiClassBackpropClassifierNetwork an eval-mode parity reference (dropout is
    a deterministic no-op at eval mode - training defaults to False on both sides, and neither
    predict_probabilities nor classify_state ever toggles it on). A genuine training-time
    comparison isn't achievable across two independent RNG streams.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dimension: int,
        input_bounds: list[tuple[float, float]],
        class_count: int,
        drop_probability: float,
    ) -> None:
        self.hidden_layer_cls = make_dropout_layer_cls(drop_probability)
        super().__init__(layer_sizes, dimension, input_bounds, class_count)


def matching_dropout_array_backprop_networks(
    rng: random.Random,
    array_network_cls,
    wrap: Callable,
    layer_sizes: list[int],
    dimension: int,
    class_count: int,
    drop_probability: float,
    bounds: float = 10.0,
):
    """
    The dropout-sibling analogue of matching_array_backprop_networks above - see
    matching_adam_array_backprop_networks's own docstring for the general shape this follows.
    Only meaningful for eval-mode comparisons (predict_probabilities/classify_state): see
    DropoutMultiClassBackpropClassifierNetwork's own docstring for why a training-mode/learn()
    comparison isn't attempted here.
    """
    node_network = DropoutMultiClassBackpropClassifierNetwork(
        layer_sizes, dimension, [(-bounds, bounds)] * dimension, class_count, drop_probability
    )
    array_network = array_network_cls(layer_sizes, dimension, class_count, drop_probability)

    previous_size = dimension
    for layer_index, size in enumerate([*layer_sizes, class_count]):
        weights = [[rng.uniform(-2.0, 2.0) for _ in range(previous_size)] for _ in range(size)]
        biases = [rng.uniform(-2.0, 2.0) for _ in range(size)]

        node_layer = node_network.trainable_layers[layer_index]
        for node, node_weights, bias in zip(node_layer.nodes, weights, biases):
            node.update_input_weights(node_weights)
            node.bias = bias

        array_network.layers[layer_index].W = wrap(weights)
        array_network.layers[layer_index].b = wrap(biases)

        previous_size = size

    return node_network, array_network


def matching_softmax_array_backprop_networks(
    rng: random.Random,
    array_network_cls,
    wrap: Callable,
    layer_sizes: list[int],
    dimension: int,
    class_count: int,
    bounds: float = 10.0,
):
    """
    The softmax-sibling analogue of matching_array_backprop_networks above - see
    matching_adam_array_backprop_networks's own docstring for the general shape this follows.
    Unlike momentum/L2/ReLU (which need a test-only per-node reference subclass built via
    make_*_layer_cls/hidden_layer_cls), a genuine per-node softmax reference already exists
    (SoftmaxMultiClassBackpropClassifierNetwork), so this uses it directly.
    """
    node_network = SoftmaxMultiClassBackpropClassifierNetwork(
        layer_sizes, dimension, [(-bounds, bounds)] * dimension, class_count
    )
    array_network = array_network_cls(layer_sizes, dimension, class_count)

    previous_size = dimension
    for layer_index, size in enumerate([*layer_sizes, class_count]):
        weights = [[rng.uniform(-2.0, 2.0) for _ in range(previous_size)] for _ in range(size)]
        biases = [rng.uniform(-2.0, 2.0) for _ in range(size)]

        node_layer = node_network.trainable_layers[layer_index]
        for node, node_weights, bias in zip(node_layer.nodes, weights, biases):
            node.update_input_weights(node_weights)
            node.bias = bias

        array_network.layers[layer_index].W = wrap(weights)
        array_network.layers[layer_index].b = wrap(biases)

        previous_size = size

    return node_network, array_network


class CrossEntropyMultiClassBackpropClassifierNetwork(MultiClassBackpropClassifierNetwork):
    """
    Test-only per-node cross-entropy reference: MultiClassBackpropClassifierNetwork with its
    output_layer_cls extension point set to CrossEntropyOutputLayer - the hidden layers stay the
    default plain BackpropLayer (sigmoid), the same construction
    BinaryCrossEntropyBackpropClassifierNetwork uses for the single-output case (a single
    class-attribute override, needing no factory function since CrossEntropyOutputLayer takes no
    extra tunable coefficient - unlike L2/momentum's own make_*_layer_cls factories). Gives
    CrossEntropyVectorizedMultiClassBackpropClassifierNetwork a genuine parity reference.
    """

    output_layer_cls = CrossEntropyOutputLayer


def matching_cross_entropy_multiclass_array_backprop_networks(
    rng: random.Random,
    array_network_cls,
    wrap: Callable,
    layer_sizes: list[int],
    dimension: int,
    class_count: int,
    bounds: float = 10.0,
):
    """
    The cross-entropy-multiclass-sibling analogue of matching_array_backprop_networks above - see
    matching_adam_array_backprop_networks's own docstring for the general shape this follows. No
    extra coefficient argument, matching matching_softmax_array_backprop_networks's own posture.
    """
    node_network = CrossEntropyMultiClassBackpropClassifierNetwork(
        layer_sizes, dimension, [(-bounds, bounds)] * dimension, class_count
    )
    array_network = array_network_cls(layer_sizes, dimension, class_count)

    previous_size = dimension
    for layer_index, size in enumerate([*layer_sizes, class_count]):
        weights = [[rng.uniform(-2.0, 2.0) for _ in range(previous_size)] for _ in range(size)]
        biases = [rng.uniform(-2.0, 2.0) for _ in range(size)]

        node_layer = node_network.trainable_layers[layer_index]
        for node, node_weights, bias in zip(node_layer.nodes, weights, biases):
            node.update_input_weights(node_weights)
            node.bias = bias

        array_network.layers[layer_index].W = wrap(weights)
        array_network.layers[layer_index].b = wrap(biases)

        previous_size = size

    return node_network, array_network


def matching_conv_array_backprop_networks(
    rng: random.Random,
    input_height: int,
    input_width: int,
    conv_specs: list,
    dense_layer_sizes: list[int],
    class_count: int,
):
    """
    The conv counterpart of matching_array_backprop_networks: builds a
    ConvMultiClassBackpropClassifierNetwork and a ConvVectorizedMultiClassBackpropClassifierNetwork
    with identical injected weights, conv kernels included (kernel.weights = W[c] - the layouts
    ConvArrayLayer documents). Pool layers have nothing to inject.
    """
    node_network = ConvMultiClassBackpropClassifierNetwork(
        input_height, input_width, conv_specs, dense_layer_sizes, class_count
    )
    array_network = ConvVectorizedMultiClassBackpropClassifierNetwork(
        input_height, input_width, conv_specs, dense_layer_sizes, class_count
    )

    for node_layer, array_layer in zip(node_network.trainable_layers, array_network.layers):
        if isinstance(array_layer, ConvArrayLayer):
            for c, kernel in enumerate(node_layer.kernels):
                kernel.weights = [rng.uniform(-1.0, 1.0) for _ in range(array_layer.fan_in)]
                kernel.bias = rng.uniform(-0.5, 0.5)
                array_layer.W[c] = kernel.weights
                array_layer.b[c] = kernel.bias
        elif hasattr(array_layer, "W"):
            for i, node in enumerate(node_layer.nodes):
                node.update_input_weights([rng.uniform(-1.0, 1.0) for _ in range(array_layer.input_size)])
                node.bias = rng.uniform(-1.0, 1.0)
                array_layer.W[i] = node.input_node_weights
                array_layer.b[i] = node.bias

    return node_network, array_network


def copy_conv_network_weights_into_array_network(node_network, array_network) -> None:
    """Copies a ConvMultiClassBackpropClassifierNetwork's weights (e.g. after its own seeded
    randomize()) into a same-shaped ConvVectorizedMultiClassBackpropClassifierNetwork."""
    for node_layer, array_layer in zip(node_network.trainable_layers, array_network.layers):
        if hasattr(node_layer, "kernels"):
            array_layer.W = np.array([kernel.weights for kernel in node_layer.kernels])
            array_layer.b = np.array([kernel.bias for kernel in node_layer.kernels])
        elif hasattr(array_layer, "W"):
            array_layer.W = np.array([node.input_node_weights for node in node_layer.nodes])
            array_layer.b = np.array([node.bias for node in node_layer.nodes])


def assert_conv_array_network_weights_match(node_network, array_network, rtol=1e-9, atol=1e-9) -> None:
    """The conv counterpart of assert_array_network_weights_match: conv layers compare per
    kernel, pool layers have nothing to compare."""
    for node_layer, array_layer in zip(node_network.trainable_layers, array_network.layers):
        if hasattr(node_layer, "kernels"):
            expected_W = np.array([kernel.weights for kernel in node_layer.kernels])
            expected_b = np.array([kernel.bias for kernel in node_layer.kernels])
        elif hasattr(array_layer, "W"):
            expected_W = np.array([node.input_node_weights for node in node_layer.nodes])
            expected_b = np.array([node.bias for node in node_layer.nodes])
        else:
            continue
        np.testing.assert_allclose(array_layer.W, expected_W, rtol=rtol, atol=atol)
        np.testing.assert_allclose(array_layer.b, expected_b, rtol=rtol, atol=atol)


def matching_conv_numpy_rust_networks(
    rng: random.Random,
    input_height: int,
    input_width: int,
    conv_specs: list,
    dense_layer_sizes: list[int],
    class_count: int,
):
    """
    A ConvVectorizedMultiClassBackpropClassifierNetwork and a
    ConvRustArrayMultiClassBackpropClassifierNetwork with identical injected weights - drawn from
    rng into the numpy network, then restored into the Rust one from its snapshot (the two
    backends' RNGs aren't comparable, so neither network's own randomize() is used).
    """
    numpy_network = ConvVectorizedMultiClassBackpropClassifierNetwork(
        input_height, input_width, conv_specs, dense_layer_sizes, class_count
    )
    rust_network = ConvRustArrayMultiClassBackpropClassifierNetwork(
        input_height, input_width, conv_specs, dense_layer_sizes, class_count
    )
    for layer in numpy_network.layers:
        if hasattr(layer, "W"):
            rows, cols = layer.W.shape
            layer.W = np.array([[rng.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)])
            layer.b = np.array([rng.uniform(-0.5, 0.5) for _ in range(layer.b.shape[0])])
    rust_network.restore(numpy_network.snapshot())
    return numpy_network, rust_network


def assert_array_network_snapshots_match(numpy_network, rust_network, rtol=1e-12, atol=1e-13) -> None:
    """Every (W, b) entry of two array-backed networks' snapshots, compared via .tolist() (both
    backends have it); pool layers' empty entries must both be empty. The default tolerance is
    tight on purpose: over the conv parity runs the numpy and Rust weights differ by at most
    ~7e-16 (the two backends' summation orders), not by anything a 1e-9 tolerance would need."""
    numpy_snapshot, rust_snapshot = numpy_network.snapshot(), rust_network.snapshot()
    assert len(numpy_snapshot) == len(rust_snapshot)
    for numpy_entry, rust_entry in zip(numpy_snapshot, rust_snapshot):
        assert len(numpy_entry) == len(rust_entry)
        for numpy_array, rust_array in zip(numpy_entry, rust_entry):
            np.testing.assert_allclose(rust_array.tolist(), numpy_array.tolist(), rtol=rtol, atol=atol)


def assert_array_network_weights_match(node_network, array_network, rtol=1e-9, atol=1e-9) -> None:
    """
    Shared by both array-backed siblings' own _assert_networks_match: both numpy ndarrays and
    pa.Array support .tolist(), so the same comparison works against either backend without
    needing a backend-specific read path.
    """
    for node_layer, array_layer in zip(node_network.trainable_layers, array_network.layers):
        expected_W = np.array([node.input_node_weights for node in node_layer.nodes])
        expected_b = np.array([node.bias for node in node_layer.nodes])
        assert np.allclose(array_layer.W.tolist(), expected_W, rtol=rtol, atol=atol)
        assert np.allclose(array_layer.b.tolist(), expected_b, rtol=rtol, atol=atol)


def assert_array_network_snapshot_restore_round_trip(
    array_network_cls, layer_sizes: list[int], dimension: int, class_count: int
) -> None:
    """
    Shared by both array-backed siblings' own test_snapshot_restore_round_trips_weights - same
    .tolist()-based equality reasoning as assert_array_network_weights_match above.
    """
    network = array_network_cls.randomized(layer_sizes, dimension, class_count)
    snapshot = network.snapshot()

    other = array_network_cls(layer_sizes, dimension, class_count)
    other.restore(snapshot)

    for (W1, b1), (W2, b2) in zip(network.snapshot(), other.snapshot()):
        assert W1.tolist() == W2.tolist()
        assert b1.tolist() == b2.tolist()


def assert_single_output_array_network_snapshot_restore_round_trip(
    array_network_cls, layer_sizes: list[int], dimension: int
) -> None:
    """
    The single-output analogue of assert_array_network_snapshot_restore_round_trip above, for
    ArrayBackpropClassifierNetwork/RustArrayBackpropClassifierNetwork (no class_count argument).
    """
    network = array_network_cls.randomized(layer_sizes, dimension)
    snapshot = network.snapshot()

    other = array_network_cls(layer_sizes, dimension)
    other.restore(snapshot)

    for (W1, b1), (W2, b2) in zip(network.snapshot(), other.snapshot()):
        assert W1.tolist() == W2.tolist()
        assert b1.tolist() == b2.tolist()


def assert_single_output_array_network_save_load_round_trip(network, load_fn, tmp_path, filename: str, state):
    """
    The single-output analogue of assert_array_network_save_load_round_trip above, for
    ArrayBackpropClassifierNetwork/RustArrayBackpropClassifierNetwork: predict_probability, not
    predict_probabilities, and no class_count field to compare.
    """
    path = str(tmp_path / filename)
    network.save(path)
    loaded = load_fn(path)

    assert loaded.layer_sizes == network.layer_sizes
    assert loaded.dimension == network.dimension
    assert loaded.predict_probability(state) == pytest.approx(network.predict_probability(state))

    return loaded


def assert_array_network_save_load_round_trip(network, load_fn, tmp_path, filename: str, state):
    """
    Array-backed analogue of assert_save_and_load_round_trip above: a numpy/pa.Array snapshot
    element doesn't support a plain == equality check the way a node network's snapshot() does
    (it's elementwise, not a single bool), so this checks the JSON envelope's scalar fields plus
    predict_probabilities via pytest.approx instead of snapshot() equality. Shared by both
    test_vectorized_multiclass_backprop_model.py's and
    test_rust_array_multiclass_backprop_model.py's own test_save_load_round_trips_weights_and_predictions.
    """
    path = str(tmp_path / filename)
    network.save(path)
    loaded = load_fn(path)

    assert loaded.layer_sizes == network.layer_sizes
    assert loaded.dimension == network.dimension
    assert loaded.class_count == network.class_count
    assert loaded.predict_probabilities(state) == pytest.approx(network.predict_probabilities(state))

    return loaded


def classifier_with_bounded_square_region(bounds: list[tuple[float, float]]) -> LinearClassifierNetwork:

    # positive region is exactly the square [-1, 1] x [-1, 1]: x > -1, x < 1, y > -1, y < 1
    classifier = LinearClassifierNetwork(4, 2, bounds)
    for node, (weights, threshold) in zip(
        classifier.hidden_layer.nodes,
        [([1.0, 0.0], 1.0), ([-1.0, 0.0], 1.0), ([0.0, 1.0], 1.0), ([0.0, -1.0], 1.0)],
    ):
        node.update_input_weights(weights)
        node.threshold = threshold
    return classifier


def classifier_with_tiny_bounded_region(bounds: list[tuple[float, float]]) -> LinearClassifierNetwork:

    # positive region is exactly [-0.1, 0.1] x [-0.1, 0.1] - a 0.04 unit^2 square, a tiny
    # fraction of a square_bounds(10.0)-sized (400 unit^2) box (0.01%), for exercising the
    # tight-box positive-region sampling optimisation (see geometry.positive_region_bounding_box)
    classifier = LinearClassifierNetwork(4, 2, bounds)
    for node, (weights, threshold) in zip(
        classifier.hidden_layer.nodes,
        [([1.0, 0.0], 0.1), ([-1.0, 0.0], 0.1), ([0.0, 1.0], 0.1), ([0.0, -1.0], 0.1)],
    ):
        node.update_input_weights(weights)
        node.threshold = threshold
    return classifier


def unreachable_class_classifier(bounds: list[tuple[float, float]]) -> LinearClassifierNetwork:

    # tiny weights + a large threshold mean the decision boundary never crosses these bounds,
    # so one class can never be sampled - used to exercise the safety guard against an
    # unreachable-class sampling loop hanging forever (see
    # demo_unreachable_class_safety_guard.py for the same idea as a standalone demo)
    classifier = LinearClassifierNetwork(1, 2, bounds)
    node = classifier.hidden_layer.nodes[0]
    node.update_input_weights([0.01, 0.01])
    node.threshold = -5.0
    return classifier


def network_with_hidden_thresholds(
    dimension: int,
    bounds: list[tuple[float, float]],
    thresholds: list[float],
    required_active: int | None = None,
) -> LinearClassifierNetwork:

    # zero input weights make each node's z() equal to its threshold alone, regardless of
    # state - so the thresholds directly pick each node's active/inactive status
    network = LinearClassifierNetwork(len(thresholds), dimension, bounds, required_active)
    for node, threshold in zip(network.hidden_layer.nodes, thresholds):
        node.update_input_weights([0.0] * dimension)
        node.threshold = threshold
    return network
