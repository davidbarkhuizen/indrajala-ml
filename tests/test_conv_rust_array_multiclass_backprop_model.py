import random

import numpy as np
import pytest

from indrajala_ml.digits_data import load_digits_dataset, split_train_test
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.conv_multiclass_backprop_classifier_network import ConvMultiClassBackpropClassifierNetwork
from indrajala_ml.model.conv_rust_array_layer import ConvRustArrayLayer
from indrajala_ml.model.conv_rust_array_multiclass_backprop_classifier_network import (
    ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.conv_vectorized_multiclass_backprop_classifier_network import (
    ConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.max_pool_rust_array_layer import MaxPoolRustArrayLayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from indrajala_ml.multiclass_evaluate import accuracy
from indrajala_ml.train import train_linear_classifier_network
from tests.helpers import (
    assert_array_network_snapshots_match,
    copy_conv_network_weights_into_array_network,
    matching_conv_numpy_rust_networks,
)
from tests.test_conv_vectorized_multiclass_backprop_model import ARCHITECTURES, CLASS_COUNT, _digits_rows

POOLED = [ConvSpec(3, 4), PoolSpec(2), ConvSpec(2, 6)]
OVERLAPPING_POOL_STRIDED = [ConvSpec(3, 4), PoolSpec(2, stride=1), ConvSpec(2, 6, stride=2)]


def _matching_networks(rng: random.Random, architecture: str):
    conv_specs, dense_layer_sizes = ARCHITECTURES[architecture]
    return matching_conv_numpy_rust_networks(rng, 8, 8, conv_specs, dense_layer_sizes, CLASS_COUNT)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_predict_probabilities_and_classify_state_match_the_numpy_network(architecture):

    rng = random.Random(0)
    numpy_network, rust_network = _matching_networks(rng, architecture)

    states = [state for state, _label in _digits_rows()[:30]]
    states += [tuple(rng.uniform(0.0, 1.0) for _ in range(64)) for _ in range(20)]
    for state in states:
        expected = numpy_network.predict_probabilities(state)
        np.testing.assert_allclose(rust_network.predict_probabilities(state), expected, rtol=1e-12, atol=1e-14)
        assert rust_network.classify_state(state) == numpy_network.classify_state(state)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_learn_matches_the_numpy_network_after_every_step(architecture):

    # one silently-wrong intermediate step fails loudly instead of being averaged away
    rng = random.Random(1)
    numpy_network, rust_network = _matching_networks(rng, architecture)

    for state, label in _digits_rows()[:60]:
        numpy_network.learn(0.5, state, label)
        rust_network.learn(0.5, state, label)
        assert_array_network_snapshots_match(numpy_network, rust_network)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_learn_batch_matches_the_numpy_network_after_every_batch(architecture):

    rng = random.Random(2)
    numpy_network, rust_network = _matching_networks(rng, architecture)
    rows = _digits_rows()

    for start in range(0, len(rows), 8):
        batch = rows[start : start + 8]
        numpy_network.learn_batch(0.5, batch)
        rust_network.learn_batch(0.5, batch)
        assert_array_network_snapshots_match(numpy_network, rust_network)


def test_reproduces_the_pinned_pure_python_uci_digits_result():

    # test_conv_multiclass_backprop_model.py pins best_training_accuracy 0.9875 at epoch index 10
    # and test accuracy 0.925 for the pure-Python network, and the numpy network reproduces it.
    # Same seeded initial weights, same training call, Rust network only.
    random.seed(0)

    dataset = load_digits_dataset()
    subset = dataset[:200]
    train_data, test_data = split_train_test(subset, test_fraction=0.2, seed=1)

    reference = ConvMultiClassBackpropClassifierNetwork.randomized(
        input_height=8, input_width=8, conv_specs=[ConvSpec(3, 4)], dense_layer_sizes=[16], class_count=10
    )
    bridge = ConvVectorizedMultiClassBackpropClassifierNetwork(8, 8, [ConvSpec(3, 4)], [16], class_count=10)
    copy_conv_network_weights_into_array_network(reference, bridge)
    student = ConvRustArrayMultiClassBackpropClassifierNetwork(8, 8, [ConvSpec(3, 4)], [16], class_count=10)
    student.restore(bridge.snapshot())

    result = train_linear_classifier_network(student, train_data, learning_rate=0.5, epochs=15)

    diagnostic = result.diagnostic
    assert diagnostic.best_training_accuracy == 0.9875
    assert diagnostic.best_epoch_index == 10
    assert diagnostic.plateaued is True
    assert diagnostic.converged is False
    assert diagnostic.still_improving is False

    assert accuracy(student, test_data) == 0.925


def test_construction_shape_chains_conv_and_pool_layers():

    network = ConvRustArrayMultiClassBackpropClassifierNetwork(8, 8, POOLED, [8], class_count=10)
    first, pool, last = network.conv_layers

    assert network.dimension == 64
    assert isinstance(first, ConvRustArrayLayer) and isinstance(pool, MaxPoolRustArrayLayer)
    assert isinstance(last, ConvRustArrayLayer)
    assert (first.out_height, first.out_width, first.channel_count) == (6, 6, 4)
    assert (pool.out_height, pool.out_width, pool.channel_count) == (3, 3, 4)
    assert (last.input_channels, last.out_height, last.out_width, last.channel_count) == (4, 2, 2, 6)
    assert last.W.shape == (6, 2 * 2 * 4)

    dense, output = network.layers[3:]
    assert isinstance(dense, RustArrayLayer) and dense.W.shape == (8, 2 * 2 * 6)
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
        (8, 8, [ConvSpec(9, 4)], [8], 10),  # kernel larger than the input
    ],
)
def test_construction_rejects_invalid_arguments(arguments):

    with pytest.raises(AssertionError):
        ConvRustArrayMultiClassBackpropClassifierNetwork(*arguments)


def test_randomized_breaks_symmetry_and_builds_a_usable_network():

    network = ConvRustArrayMultiClassBackpropClassifierNetwork.randomized(8, 8, POOLED, [8], class_count=10)
    first, _pool, last = network.conv_layers

    for layer in (first, last):
        W = np.array(layer.W.tolist())
        assert len({tuple(row) for row in W}) == layer.channel_count
        # fan-in-aware: every draw within 1/sqrt(input_channels * kernel_size**2)
        limit = 1.0 / np.sqrt(layer.fan_in)
        assert np.all(np.abs(W) <= limit) and np.all(np.abs(layer.b.tolist()) <= limit)
    assert np.all(np.abs(network.layers[3].W.tolist()) <= 1.0 / np.sqrt(last.size))
    assert np.all(np.abs(network.output_layer.W.tolist()) <= 1.0 / np.sqrt(8))

    state = tuple(0.01 * i for i in range(64))
    probabilities = network.predict_probabilities(state)
    assert len(probabilities) == 10
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert 0 <= network.classify_state(state) < 10


def test_snapshot_has_an_empty_pool_entry_and_restore_round_trips():

    network = ConvRustArrayMultiClassBackpropClassifierNetwork.randomized(8, 8, POOLED, [8], class_count=10)
    before = network.snapshot()
    assert before[1] == ()

    for state, label in _digits_rows()[:5]:
        network.learn(0.5, state, label)
    assert network.snapshot()[0][0].tolist() != before[0][0].tolist()

    network.restore(before)
    assert [[array.tolist() for array in entry] for entry in network.snapshot()] == [
        [array.tolist() for array in entry] for entry in before
    ]
    # restore copies: training afterwards leaves the snapshot itself untouched
    network.learn(0.5, *_digits_rows()[0])
    assert network.snapshot()[0][0].tolist() != before[0][0].tolist()

    with pytest.raises(AssertionError):
        network.restore([before[0], before[0], *before[2:]])


@pytest.mark.parametrize("conv_specs", [[ConvSpec(3, 4)], OVERLAPPING_POOL_STRIDED])
def test_save_load_round_trips_specs_weights_and_predictions(tmp_path, conv_specs):

    network = ConvRustArrayMultiClassBackpropClassifierNetwork.randomized(8, 8, conv_specs, [8], class_count=10)
    path = str(tmp_path / "conv_rust_array_model.json")
    network.save(path)
    loaded = ConvRustArrayMultiClassBackpropClassifierNetwork.load(path)

    assert loaded.conv_specs == network.conv_specs
    assert (loaded.input_height, loaded.input_width, loaded.dense_layer_sizes, loaded.class_count) == (8, 8, [8], 10)
    assert_array_network_snapshots_match(network, loaded, rtol=0, atol=0)
    for state, _label in _digits_rows()[:10]:
        assert loaded.predict_probabilities(state) == network.predict_probabilities(state)


@pytest.mark.parametrize("save_backend", ["numpy", "rust"])
def test_a_model_saved_by_either_backend_loads_into_the_other_with_the_same_predictions(tmp_path, save_backend):

    rng = random.Random(3)
    numpy_network, rust_network = matching_conv_numpy_rust_networks(
        rng, 8, 8, OVERLAPPING_POOL_STRIDED, [8], CLASS_COUNT
    )
    for state, label in _digits_rows()[:20]:
        numpy_network.learn(0.5, state, label)
        rust_network.learn(0.5, state, label)

    path = str(tmp_path / "conv_model.json")
    if save_backend == "numpy":
        saved, loaded = numpy_network, ConvRustArrayMultiClassBackpropClassifierNetwork
    else:
        saved, loaded = rust_network, ConvVectorizedMultiClassBackpropClassifierNetwork
    saved.save(path)
    other = loaded.load(path)

    assert isinstance(other, loaded) and other.conv_specs == saved.conv_specs
    # weights round-trip exactly through JSON; predictions agree to the backends' reduction order
    for entry_saved, entry_other in zip(saved.snapshot(), other.snapshot()):
        assert [array.tolist() for array in entry_other] == [array.tolist() for array in entry_saved]
    for state, _label in _digits_rows()[20:60]:
        np.testing.assert_allclose(
            other.predict_probabilities(state), saved.predict_probabilities(state), rtol=1e-12, atol=1e-14
        )
        assert other.classify_state(state) == saved.classify_state(state)
