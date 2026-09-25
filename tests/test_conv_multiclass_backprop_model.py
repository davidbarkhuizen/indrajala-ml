import random
from pathlib import Path

import pytest

from indrajala_ml.digits_data import load_digits_dataset, split_train_test
from indrajala_ml.model.backprop_layer import BackpropLayer
from indrajala_ml.model.backprop_network_base import as_dense_layers
from indrajala_ml.model.conv_kernel import ConvKernel
from indrajala_ml.model.conv_layer import ConvLayer, ConvSpec
from indrajala_ml.model.conv_multiclass_backprop_classifier_network import (
    ConvMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.max_pool_layer import MaxPoolLayer, PoolSpec
from indrajala_ml.multiclass_evaluate import accuracy
from indrajala_ml.train import train_linear_classifier_network
from tests.helpers import (
    approx,
    assert_save_and_load_round_trip,
    assert_snapshot_restore_round_trip,
    conv_layer,
    conv_layers_only,
)


def _small_network() -> ConvMultiClassBackpropClassifierNetwork:
    return ConvMultiClassBackpropClassifierNetwork(
        input_height=8,
        input_width=8,
        conv_specs=[ConvSpec(kernel_size=3, channel_count=4)],
        dense_layer_sizes=[16],
        class_count=10,
    )


def test_construction_shape():

    network = _small_network()

    assert network.dimension == 64
    assert network.input_bounds == [(0.0, 1.0)] * 64
    assert isinstance(network.hidden_layers[0], ConvLayer)
    assert network.hidden_layers[0] is network.conv_layers[0]
    assert isinstance(network.hidden_layers[1], BackpropLayer)
    assert network.hidden_layers[1].size == 16
    assert network.output_layer.size == 10
    assert network.trainable_layers == network.hidden_layers + [network.output_layer]

    # 8x8 input, kernel_size=3, stride=1 -> out_height=out_width=6; channel_count=4
    assert network.conv_layers[0].out_height == 6
    assert network.conv_layers[0].out_width == 6
    assert len(network.conv_layers[0].nodes) == 4 * 6 * 6

    # the first dense layer's own input_nodes must be exactly the conv layer's flattened output
    assert network.hidden_layers[1].nodes[0].input_nodes is network.conv_layers[0].nodes


def test_class_count_and_dense_layer_sizes_are_validated():

    with pytest.raises(AssertionError):
        ConvMultiClassBackpropClassifierNetwork(8, 8, [ConvSpec(3, 4)], [16], class_count=1)

    with pytest.raises(AssertionError):
        ConvMultiClassBackpropClassifierNetwork(8, 8, [ConvSpec(3, 4)], [], class_count=10)


def test_randomize_randomizes_conv_kernels_and_every_dense_layer():

    random.seed(0)
    network = _small_network()
    network.randomize()

    for kernel in conv_layer(network, 0).kernels:
        assert len(set(kernel.weights)) > 1

    for layer in as_dense_layers(network.hidden_layers[1:] + [network.output_layer]):
        for node in layer.nodes:
            assert len(set(node.input_node_weights)) > 1


def test_randomized_classmethod_uses_this_classs_own_constructor_signature():

    # MultiClassBackpropClassifierNetwork.randomized would call cls() with the wrong signature
    random.seed(0)
    network = ConvMultiClassBackpropClassifierNetwork.randomized(
        input_height=8, input_width=8, conv_specs=[ConvSpec(3, 4)], dense_layer_sizes=[16], class_count=10
    )

    assert conv_layer(network, 0).kernel_size == 3
    assert len(set(conv_layer(network, 0).kernels[0].weights)) > 1  # actually randomized, not left at zero


def test_forward_and_backward_run_without_error_and_move_every_weight():

    random.seed(0)
    network = _small_network()
    network.randomize()

    conv_weights_before = [list(k.weights) for k in conv_layer(network, 0).kernels]
    dense_weights_before = [
        [list(node.input_node_weights) for node in layer.nodes]
        for layer in as_dense_layers(network.hidden_layers[1:] + [network.output_layer])
    ]

    state = tuple(random.uniform(0.0, 1.0) for _ in range(64))
    network.learn(learning_rate=0.1, state=state, category=3)

    assert any(
        after != before
        for kernel, before in zip(conv_layer(network, 0).kernels, conv_weights_before)
        for after, before in [(kernel.weights, before)]
    )
    for layer, before_layer in zip(
        as_dense_layers(network.hidden_layers[1:] + [network.output_layer]), dense_weights_before
    ):
        for node, before in zip(layer.nodes, before_layer):
            assert node.input_node_weights != before


def test_learn_batch_also_moves_every_weight():

    # learn_batch directly: train_backprop_network_mini_batch's pocket rollback could restore
    # the starting weights
    random.seed(0)
    network = _small_network()
    network.randomize()
    before = network.snapshot()

    batch = [(tuple(random.uniform(0.0, 1.0) for _ in range(64)), i % 10) for i in range(4)]
    network.learn_batch(learning_rate=0.1, batch=batch)

    assert network.snapshot() != before


def test_classify_state_returns_a_valid_class_index():

    random.seed(0)
    network = _small_network()
    network.randomize()

    state = tuple(random.uniform(0.0, 1.0) for _ in range(64))
    predicted = network.classify_state(state)

    assert 0 <= predicted < 10
    probabilities = network.predict_probabilities(state)
    assert len(probabilities) == 10


def test_snapshot_and_restore_round_trip_through_the_conv_layer_too():

    random.seed(0)
    network = _small_network()
    network.randomize()

    snapshot = network.snapshot()
    assert len(snapshot) == len(network.trainable_layers)  # one entry per layer, conv included
    spec = network.conv_specs[0]
    assert isinstance(spec, ConvSpec)
    assert len(snapshot[0]) == spec.channel_count  # the conv layer's own entry: one per kernel

    state = tuple(random.uniform(0.0, 1.0) for _ in range(64))
    assert_snapshot_restore_round_trip(
        network, lambda: network.learn(learning_rate=0.1, state=state, category=3), times=1
    )


class _Kernel(ConvKernel):
    pass


class _ConvLayer(ConvLayer):
    _kernel_cls = _Kernel


class _HiddenLayer(BackpropLayer):
    pass


class _OutputLayer(BackpropLayer):
    pass


class _HookedNetwork(ConvMultiClassBackpropClassifierNetwork):
    conv_layer_cls = _ConvLayer
    hidden_layer_cls = _HiddenLayer
    output_layer_cls = _OutputLayer


def test_layer_and_kernel_class_hooks_build_every_layer():

    network = _HookedNetwork(8, 8, [ConvSpec(3, 2), PoolSpec(2), ConvSpec(2, 3)], [6, 5], class_count=10)
    first, pool, last = network.conv_layers

    for layer in (first, last):
        assert type(layer) is _ConvLayer
        assert [type(kernel) for kernel in layer.kernels] == [_Kernel] * layer.channel_count
    assert type(pool) is MaxPoolLayer
    assert [type(layer) for layer in network.hidden_layers[3:]] == [_HiddenLayer, _HiddenLayer]
    assert type(network.output_layer) is _OutputLayer


def test_the_hooks_default_to_the_plain_classes():

    network = _small_network()

    assert type(network.conv_layers[0]) is ConvLayer
    assert all(type(kernel) is ConvKernel for kernel in conv_layer(network, 0).kernels)
    assert type(network.hidden_layers[1]) is BackpropLayer and type(network.output_layer) is BackpropLayer


def test_save_and_load_round_trip(tmp_path: Path):

    random.seed(0)
    network = _small_network()
    network.randomize()

    state = tuple(random.uniform(0.0, 1.0) for _ in range(64))

    loaded = assert_save_and_load_round_trip(
        network, ConvMultiClassBackpropClassifierNetwork.load, tmp_path, "conv_model.json", [state]
    )

    assert loaded.input_height == network.input_height
    assert loaded.input_width == network.input_width
    assert loaded.conv_specs == network.conv_specs
    assert loaded.dense_layer_sizes == network.dense_layer_sizes
    assert loaded.class_count == network.class_count


def test_trains_on_a_real_uci_digits_subset():

    # 200 of the 1797 rows, 15 epochs: train_linear_classifier_network drives the conv network,
    # and backprop through the conv layer really learns. The pinned values are measured
    random.seed(0)

    dataset = load_digits_dataset()
    subset = dataset[:200]
    train_data, test_data = split_train_test(subset, test_fraction=0.2, seed=1)

    student = ConvMultiClassBackpropClassifierNetwork.randomized(
        input_height=8, input_width=8, conv_specs=[ConvSpec(3, 4)], dense_layer_sizes=[16], class_count=10
    )
    result = train_linear_classifier_network(student, train_data, learning_rate=0.5, epochs=15)

    diagnostic = result.diagnostic
    assert diagnostic.best_training_accuracy == 0.9875
    assert diagnostic.best_epoch_index == 10
    assert diagnostic.plateaued is True
    assert diagnostic.converged is False
    assert diagnostic.still_improving is False

    assert accuracy(student, test_data) == 0.925


def _two_conv_layer_network() -> ConvMultiClassBackpropClassifierNetwork:
    return ConvMultiClassBackpropClassifierNetwork(
        input_height=8,
        input_width=8,
        conv_specs=[ConvSpec(kernel_size=3, channel_count=3), ConvSpec(kernel_size=2, channel_count=4, stride=2)],
        dense_layer_sizes=[8],
        class_count=10,
    )


def test_empty_conv_specs_are_rejected():

    with pytest.raises(AssertionError):
        ConvMultiClassBackpropClassifierNetwork(8, 8, [], [16], class_count=10)


def test_two_conv_layers_chain_shape_and_channels():

    network = _two_conv_layer_network()
    first, second = conv_layers_only(network)

    # 8x8 -> k3 s1 -> 6x6x3 -> k2 s2 -> 3x3x4
    assert (first.out_height, first.out_width, first.channel_count) == (6, 6, 3)
    assert second.input_layer is first
    assert (second.input_height, second.input_width, second.input_channels) == (6, 6, 3)
    assert (second.out_height, second.out_width, second.channel_count) == (3, 3, 4)
    assert all(len(kernel.weights) == 2 * 2 * 3 for kernel in second.kernels)

    assert network.hidden_layers[:2] == network.conv_layers
    assert network.hidden_layers[2].nodes[0].input_nodes is second.nodes
    assert network.trainable_layers == network.hidden_layers + [network.output_layer]


def test_randomize_draws_one_rng_sequence_per_layer_in_forward_order():

    # the first conv layer's kernels must be drawn first, exactly as a one-conv-layer network
    # would draw them - the property that keeps the pinned single-layer digits result below
    # reproducible when the network grows a second conv layer spec
    random.seed(0)
    one_layer = ConvMultiClassBackpropClassifierNetwork(8, 8, [ConvSpec(3, 3)], [8], class_count=10)
    one_layer.conv_layers[0].randomize_fan_in_aware()

    random.seed(0)
    two_layer = _two_conv_layer_network()
    two_layer.randomize()

    assert [k.weights for k in conv_layer(two_layer, 0).kernels] == [
        k.weights for k in conv_layer(one_layer, 0).kernels
    ]
    for kernel in conv_layer(two_layer, 1).kernels:
        assert len(set(kernel.weights)) > 1


def test_learn_moves_every_kernel_in_every_conv_layer():

    random.seed(0)
    network = _two_conv_layer_network()
    network.randomize()
    before = [[list(k.weights) for k in layer.kernels] for layer in conv_layers_only(network)]

    state = tuple(random.uniform(0.0, 1.0) for _ in range(64))
    network.learn(learning_rate=0.1, state=state, category=3)

    for layer, layer_before in zip(conv_layers_only(network), before):
        assert any(kernel.weights != kernel_before for kernel, kernel_before in zip(layer.kernels, layer_before))


def test_network_gradient_check_from_output_loss_back_to_the_first_conv_layer():

    # end-to-end: the one-vs-rest sigmoid output delta (a - t) * a * (1 - a) is the gradient of
    # L = 0.5 * sum((a - t)**2), so after _forward/_backward/_accumulate_gradients every kernel's
    # accumulator must match a finite-difference estimate of that L - through the dense tail,
    # the second conv layer's downstream_sum, and into the first conv layer's kernels
    random.seed(3)
    network = _two_conv_layer_network()
    network.randomize()
    state = tuple(random.uniform(0.0, 1.0) for _ in range(64))
    category = 7

    def loss() -> float:
        outputs = network._forward(state)
        return 0.5 * sum((a - (1.0 if i == category else 0.0)) ** 2 for i, a in enumerate(outputs))

    network._forward(state)
    network._backward(category)
    network._accumulate_gradients()

    assert any(unit.delta != 0.0 for unit in network.conv_layers[0].nodes)

    epsilon = 1e-6
    for layer in conv_layers_only(network):
        for kernel in layer.kernels:
            for i in range(len(kernel.weights)):
                original = kernel.weights[i]
                kernel.weights[i] = original + epsilon
                loss_plus = loss()
                kernel.weights[i] = original - epsilon
                loss_minus = loss()
                kernel.weights[i] = original
                assert kernel._weight_gradient_accum[i] == approx((loss_plus - loss_minus) / (2 * epsilon), abs=1e-7)

            original_bias = kernel.bias
            kernel.bias = original_bias + epsilon
            loss_plus = loss()
            kernel.bias = original_bias - epsilon
            loss_minus = loss()
            kernel.bias = original_bias
            assert kernel._bias_gradient_accum == approx((loss_plus - loss_minus) / (2 * epsilon), abs=1e-7)


def test_two_conv_layer_save_and_load_round_trip(tmp_path: Path):

    random.seed(0)
    network = _two_conv_layer_network()
    network.randomize()
    state = tuple(random.uniform(0.0, 1.0) for _ in range(64))

    loaded = assert_save_and_load_round_trip(
        network, ConvMultiClassBackpropClassifierNetwork.load, tmp_path, "two_conv_model.json", [state]
    )

    assert loaded.conv_specs == network.conv_specs
    assert len(loaded.conv_layers) == 2


def _pooled_network() -> ConvMultiClassBackpropClassifierNetwork:
    # 8x8 -> conv k3 -> 6x6x4 -> pool 2 -> 3x3x4 -> conv k2 -> 2x2x6
    return ConvMultiClassBackpropClassifierNetwork(
        input_height=8,
        input_width=8,
        conv_specs=[ConvSpec(3, 4), PoolSpec(2), ConvSpec(2, 6)],
        dense_layer_sizes=[8],
        class_count=10,
    )


def test_pool_spec_builds_a_max_pool_layer_in_the_chain():

    network = _pooled_network()
    _first, pool, last = network.conv_layers
    assert isinstance(last, ConvLayer)

    assert isinstance(pool, MaxPoolLayer)
    assert (pool.out_height, pool.out_width, pool.channel_count) == (3, 3, 4)
    assert last.input_layer is pool and last.input_channels == 4
    assert (last.out_height, last.out_width, last.channel_count) == (2, 2, 6)
    assert network.trainable_layers.index(pool) == 1


def test_conv_specs_of_only_pooling_are_rejected():

    with pytest.raises(AssertionError):
        ConvMultiClassBackpropClassifierNetwork(8, 8, [PoolSpec(2)], [16], class_count=10)


def test_pooling_does_not_shift_any_conv_layers_random_draws():

    random.seed(0)
    unpooled = ConvMultiClassBackpropClassifierNetwork(8, 8, [ConvSpec(3, 4)], [8], class_count=10)
    unpooled.conv_layers[0].randomize_fan_in_aware()

    random.seed(0)
    pooled = _pooled_network()
    pooled.randomize()

    assert [k.weights for k in conv_layer(pooled, 0).kernels] == [k.weights for k in conv_layer(unpooled, 0).kernels]


def test_network_gradient_check_through_a_pooling_layer():

    # the same end-to-end finite-difference check as the two-conv-layer one above, with a max
    # pool between the two conv layers
    random.seed(3)
    network = _pooled_network()
    network.randomize()
    state = tuple(random.uniform(0.0, 1.0) for _ in range(64))
    category = 2

    def loss() -> float:
        outputs = network._forward(state)
        return 0.5 * sum((a - (1.0 if i == category else 0.0)) ** 2 for i, a in enumerate(outputs))

    network._forward(state)
    network._backward(category)
    network._accumulate_gradients()

    first, _pool, last = network.conv_layers
    assert isinstance(first, ConvLayer) and isinstance(last, ConvLayer)
    assert any(unit.delta != 0.0 for unit in first.nodes)

    epsilon = 1e-6
    for layer in (first, last):
        for kernel in layer.kernels:
            for i in range(len(kernel.weights)):
                original = kernel.weights[i]
                kernel.weights[i] = original + epsilon
                loss_plus = loss()
                kernel.weights[i] = original - epsilon
                loss_minus = loss()
                kernel.weights[i] = original
                assert kernel._weight_gradient_accum[i] == approx((loss_plus - loss_minus) / (2 * epsilon), abs=1e-7)


def test_pooled_snapshot_has_an_empty_pool_entry_and_save_load_round_trips(tmp_path: Path):

    random.seed(0)
    network = _pooled_network()
    network.randomize()
    assert network.snapshot()[1] == []

    state = tuple(random.uniform(0.0, 1.0) for _ in range(64))
    loaded = assert_save_and_load_round_trip(
        network, ConvMultiClassBackpropClassifierNetwork.load, tmp_path, "pooled_model.json", [state]
    )

    assert loaded.conv_specs == network.conv_specs
    assert isinstance(loaded.conv_specs[1], PoolSpec)
