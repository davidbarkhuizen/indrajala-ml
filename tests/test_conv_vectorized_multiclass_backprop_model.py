import random

import numpy as np
import pytest

from indrajala_ml.digits_data import load_digits_dataset, split_train_test
from indrajala_ml.model.array_layer import ArrayLayer
from indrajala_ml.model.conv_array_layer import ConvArrayLayer
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.conv_multiclass_backprop_classifier_network import ConvMultiClassBackpropClassifierNetwork
from indrajala_ml.model.conv_vectorized_multiclass_backprop_classifier_network import (
    ConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.max_pool_array_layer import MaxPoolArrayLayer
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.multiclass_evaluate import accuracy
from indrajala_ml.train import train_linear_classifier_network
from tests.helpers import (
    assert_conv_array_network_weights_match,
    copy_conv_network_weights_into_array_network,
    matching_conv_array_backprop_networks,
)

CLASS_COUNT = 10

# every architecture the step-by-step parity gates below run over, on 8x8 inputs
ARCHITECTURES = {
    "one_conv": ([ConvSpec(3, 4)], [8]),
    "conv_conv_stride_2": ([ConvSpec(3, 3), ConvSpec(2, 4, stride=2)], [8]),  # multi-channel 2nd layer
    "conv_pool_conv": ([ConvSpec(3, 4), PoolSpec(2), ConvSpec(2, 6)], [8]),
    "conv_overlapping_pool": ([ConvSpec(3, 3), PoolSpec(2, stride=1)], [8, 6]),
    "conv_conv_conv": ([ConvSpec(2, 2), ConvSpec(3, 3), ConvSpec(2, 2, stride=2)], [8]),
}


def _digits_rows() -> list[tuple[tuple[float, ...], int]]:
    # real UCI digits: [0, 1] pixels with many exact zeros, so ReLU zeros and pooling ties
    # actually occur in the parity runs rather than only in the layer-level tie tests
    return load_digits_dataset()[:120]


def _matching_networks(rng: random.Random, architecture: str):
    conv_specs, dense_layer_sizes = ARCHITECTURES[architecture]
    return matching_conv_array_backprop_networks(rng, 8, 8, conv_specs, dense_layer_sizes, CLASS_COUNT)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_predict_probabilities_and_classify_state_match_across_a_sweep(architecture):

    rng = random.Random(0)
    node_network, array_network = _matching_networks(rng, architecture)

    states = [state for state, _label in _digits_rows()[:30]]
    states += [tuple(rng.uniform(0.0, 1.0) for _ in range(64)) for _ in range(20)]
    for state in states:
        np.testing.assert_allclose(
            array_network.predict_probabilities(state), node_network.predict_probabilities(state), rtol=1e-9, atol=1e-12
        )
        assert array_network.classify_state(state) == node_network.classify_state(state)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_learn_matches_after_every_step_not_just_at_the_end(architecture):

    # the regression gate every numpy sibling has: one silently-wrong intermediate step fails
    # loudly instead of being averaged away over many steps
    rng = random.Random(1)
    node_network, array_network = _matching_networks(rng, architecture)

    for state, label in _digits_rows()[:60]:
        node_network.learn(0.5, state, label)
        array_network.learn(0.5, state, label)
        assert_conv_array_network_weights_match(node_network, array_network)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_learn_batch_matches_after_every_batch_not_just_at_the_end(architecture):

    rng = random.Random(2)
    node_network, array_network = _matching_networks(rng, architecture)
    rows = _digits_rows()

    for start in range(0, len(rows), 8):
        batch = rows[start : start + 8]
        node_network.learn_batch(0.5, batch)
        array_network.learn_batch(0.5, batch)
        assert_conv_array_network_weights_match(node_network, array_network)


def test_the_parity_runs_really_exercise_relu_zeros_and_pooling_ties():

    # guards the claim the parity tests above rest on: on real digits rows, some conv outputs
    # are exactly zero and some pooling windows hold a tied maximum. Measured, not assumed: these
    # network-level runs still pass with last-occurrence tie-breaking, because a tie after a ReLU
    # layer is between exact zeros and the ReLU mask zeroes the gradient into any of them, so
    # the winner doesn't change any weight. First-occurrence tie parity is pinned at the layer
    # level instead (tests/test_max_pool_array_layer.py).
    rng = random.Random(0)
    _node_network, array_network = _matching_networks(rng, "conv_pool_conv")
    X = np.array([state for state, _label in _digits_rows()])
    conv, pool = array_network.layers[0], array_network.layers[1]
    array_network.layers[2].forward_batch(pool.forward_batch(conv.forward_batch(X)))

    assert np.any(conv.A == 0.0)
    windows = conv.A.reshape(len(X), 4, 3, 2, 3, 2).transpose(0, 1, 2, 4, 3, 5)
    window_values = windows.reshape(len(X), 4, 3, 3, 4)
    tied = (window_values == window_values.max(axis=-1, keepdims=True)).sum(axis=-1) > 1
    assert np.any(tied)


def test_reproduces_the_pinned_pure_python_uci_digits_result():

    # test_conv_multiclass_backprop_model.py's test_trains_on_a_real_uci_digits_subset pins
    # best_training_accuracy 0.9875 at epoch index 10 and test accuracy 0.925 for the
    # pure-Python network. Same seeded initial weights, same training call, numpy network only.
    random.seed(0)

    dataset = load_digits_dataset()
    subset = dataset[:200]
    train_data, test_data = split_train_test(subset, test_fraction=0.2, seed=1)

    reference = ConvMultiClassBackpropClassifierNetwork.randomized(
        input_height=8, input_width=8, conv_specs=[ConvSpec(3, 4)], dense_layer_sizes=[16], class_count=10
    )
    student = ConvVectorizedMultiClassBackpropClassifierNetwork(
        input_height=8, input_width=8, conv_specs=[ConvSpec(3, 4)], dense_layer_sizes=[16], class_count=10
    )
    copy_conv_network_weights_into_array_network(reference, student)

    result = train_linear_classifier_network(student, train_data, learning_rate=0.5, epochs=15)

    diagnostic = result.diagnostic
    assert diagnostic.best_training_accuracy == 0.9875
    assert diagnostic.best_epoch_index == 10
    assert diagnostic.plateaued is True
    assert diagnostic.converged is False
    assert diagnostic.still_improving is False

    assert accuracy(student, test_data) == 0.925


def test_construction_shape_chains_conv_and_pool_layers():

    network = ConvVectorizedMultiClassBackpropClassifierNetwork(
        8, 8, [ConvSpec(3, 4), PoolSpec(2), ConvSpec(2, 6)], [8], class_count=10
    )
    first, pool, last = network.conv_layers

    assert network.dimension == 64
    assert isinstance(first, ConvArrayLayer) and isinstance(pool, MaxPoolArrayLayer) and isinstance(last, ConvArrayLayer)
    assert (first.out_height, first.out_width, first.channel_count) == (6, 6, 4)
    assert (pool.out_height, pool.out_width, pool.channel_count) == (3, 3, 4)
    assert (last.input_channels, last.out_height, last.out_width, last.channel_count) == (4, 2, 2, 6)
    assert last.W.shape == (6, 2 * 2 * 4)

    dense, output = network.layers[3:]
    assert isinstance(dense, ArrayLayer) and dense.W.shape == (8, 2 * 2 * 6)
    assert output is network.output_layer and output.W.shape == (10, 8)
    assert network.layers == network.conv_layers + [dense, output]


@pytest.mark.parametrize(
    "arguments",
    [
        (8, 8, [], [8], 10),  # no conv layers
        (8, 8, [PoolSpec(2)], [8], 10),  # only pooling
        (8, 8, [ConvSpec(3, 4)], [], 10),  # no dense layer
        (8, 8, [ConvSpec(3, 4)], [0], 10),  # empty dense layer
        (8, 8, [ConvSpec(3, 4)], [8], 1),  # class_count
    ],
)
def test_construction_rejects_invalid_arguments(arguments):

    with pytest.raises(AssertionError):
        ConvVectorizedMultiClassBackpropClassifierNetwork(*arguments)


def test_randomized_breaks_symmetry_and_builds_a_usable_network():

    network = ConvVectorizedMultiClassBackpropClassifierNetwork.randomized(
        8, 8, [ConvSpec(3, 4), PoolSpec(2), ConvSpec(2, 6)], [8], class_count=10
    )
    first, _pool, last = network.conv_layers

    for layer in (first, last):
        assert len({tuple(row) for row in layer.W}) == layer.channel_count
        # fan-in-aware: every draw within 1/sqrt(input_channels * kernel_size**2)
        assert np.all(np.abs(layer.W) <= 1.0 / np.sqrt(layer.fan_in))
    assert np.all(np.abs(network.layers[3].W) <= 1.0 / np.sqrt(last.size))

    state = tuple(0.01 * i for i in range(64))
    probabilities = network.predict_probabilities(state)
    assert len(probabilities) == 10
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert 0 <= network.classify_state(state) < 10


def _assert_snapshots_equal(a, b) -> None:
    assert len(a) == len(b)
    for entry_a, entry_b in zip(a, b):
        assert len(entry_a) == len(entry_b)
        for array_a, array_b in zip(entry_a, entry_b):
            np.testing.assert_array_equal(array_a, array_b)


def test_snapshot_has_an_empty_pool_entry_and_restore_round_trips():

    network = ConvVectorizedMultiClassBackpropClassifierNetwork.randomized(
        8, 8, [ConvSpec(3, 4), PoolSpec(2), ConvSpec(2, 6)], [8], class_count=10
    )
    before = network.snapshot()
    assert before[1] == ()

    for state, label in _digits_rows()[:5]:
        network.learn(0.5, state, label)
    assert not np.array_equal(network.snapshot()[0][0], before[0][0])

    network.restore(before)
    _assert_snapshots_equal(network.snapshot(), before)

    with pytest.raises(AssertionError):
        network.restore([before[0], before[0], *before[2:]])


@pytest.mark.parametrize("conv_specs", [[ConvSpec(3, 4)], [ConvSpec(3, 4), PoolSpec(2, stride=1), ConvSpec(2, 6, stride=2)]])
def test_save_load_round_trips_specs_weights_and_predictions(tmp_path, conv_specs):

    network = ConvVectorizedMultiClassBackpropClassifierNetwork.randomized(8, 8, conv_specs, [8], class_count=10)
    path = str(tmp_path / "conv_vectorized_model.json")
    network.save(path)
    loaded = ConvVectorizedMultiClassBackpropClassifierNetwork.load(path)

    assert loaded.conv_specs == network.conv_specs
    assert (loaded.input_height, loaded.input_width, loaded.dense_layer_sizes, loaded.class_count) == (8, 8, [8], 10)
    _assert_snapshots_equal(loaded.snapshot(), network.snapshot())
    for state, _label in _digits_rows()[:10]:
        assert loaded.predict_probabilities(state) == network.predict_probabilities(state)
