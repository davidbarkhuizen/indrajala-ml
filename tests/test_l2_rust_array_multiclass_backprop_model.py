import random

import pytest

import indrajala_ml_array as pa
from indrajala_ml.model.l2_rust_array_multiclass_backprop_classifier_network import (
    L2RustArrayMultiClassBackpropClassifierNetwork,
)
from tests.helpers import (
    assert_array_network_save_load_round_trip,
    assert_array_network_weights_match,
    matching_l2_array_backprop_networks,
)

DIMENSION = 6
LAYER_SIZES = [5]
CLASS_COUNT = 3
L2_LAMBDA = 0.05


def _matching_networks(rng: random.Random, bounds: float = 10.0):
    # tier 1: identical fixed weights/inputs injected directly, never randomize(), since
    # indrajala_ml_array.uniform's RNG can never be seed-comparable against Python's random
    # module.
    return matching_l2_array_backprop_networks(
        rng,
        L2RustArrayMultiClassBackpropClassifierNetwork,
        pa.Array,
        LAYER_SIZES,
        DIMENSION,
        CLASS_COUNT,
        L2_LAMBDA,
        bounds,
    )


def test_predict_probabilities_matches_across_a_random_sweep():

    rng = random.Random(0)
    node_network, rust_network = _matching_networks(rng)

    for _ in range(50):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        expected = node_network.predict_probabilities(state)
        actual = rust_network.predict_probabilities(state)
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_classify_state_matches_across_a_random_sweep():

    rng = random.Random(1)
    node_network, rust_network = _matching_networks(rng)

    for _ in range(50):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        assert rust_network.classify_state(state) == node_network.classify_state(state)


def test_learn_matches_after_every_step_not_just_at_the_end():

    rng = random.Random(2)
    node_network, rust_network = _matching_networks(rng)
    learning_rate = 0.1

    for step in range(30):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        category = rng.randrange(CLASS_COUNT)

        node_network.learn(learning_rate, state, category)
        rust_network.learn(learning_rate, state, category)

        assert_array_network_weights_match(node_network, rust_network)


def test_learn_batch_matches_after_every_batch_not_just_at_the_end():

    rng = random.Random(3)
    node_network, rust_network = _matching_networks(rng)
    learning_rate = 0.1
    batch_size = 8

    for _ in range(15):
        batch = [
            (tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION)), rng.randrange(CLASS_COUNT))
            for _ in range(batch_size)
        ]

        node_network.learn_batch(learning_rate, batch)
        rust_network.learn_batch(learning_rate, batch)

        assert_array_network_weights_match(node_network, rust_network)


def test_randomized_builds_a_usable_network():

    network = L2RustArrayMultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, L2_LAMBDA
    )
    state = tuple(0.1 * i for i in range(DIMENSION))

    probabilities = network.predict_probabilities(state)
    assert len(probabilities) == CLASS_COUNT
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert 0 <= network.classify_state(state) < CLASS_COUNT


def test_snapshot_restore_round_trips_weights():

    # not assert_array_network_snapshot_restore_round_trip (helpers.py): l2_lambda is required,
    # no default - same reasoning as test_l2_vectorized_multiclass_backprop_model.py's own
    # equivalent test.
    network = L2RustArrayMultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, L2_LAMBDA
    )
    snapshot = network.snapshot()

    other = L2RustArrayMultiClassBackpropClassifierNetwork(LAYER_SIZES, DIMENSION, CLASS_COUNT, L2_LAMBDA)
    other.restore(snapshot)

    for (W1, b1), (W2, b2) in zip(network.snapshot(), other.snapshot()):
        assert W1.tolist() == W2.tolist()
        assert b1.tolist() == b2.tolist()


def test_save_load_round_trips_weights_and_predictions(tmp_path):

    network = L2RustArrayMultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, L2_LAMBDA
    )
    state = tuple(0.1 * i for i in range(DIMENSION))

    assert_array_network_save_load_round_trip(
        network,
        L2RustArrayMultiClassBackpropClassifierNetwork.load,
        tmp_path,
        "l2_rust_array_model.json",
        state,
    )


def test_save_load_round_trips_the_l2_lambda(tmp_path):

    network = L2RustArrayMultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, l2_lambda=0.02
    )
    path = str(tmp_path / "l2_rust_array_hyperparams.json")
    network.save(path)

    loaded = L2RustArrayMultiClassBackpropClassifierNetwork.load(path)
    assert loaded.l2_lambda == 0.02


def test_construction_rejects_invalid_arguments():

    with pytest.raises(AssertionError):
        L2RustArrayMultiClassBackpropClassifierNetwork([], DIMENSION, CLASS_COUNT, L2_LAMBDA)

    with pytest.raises(AssertionError):
        L2RustArrayMultiClassBackpropClassifierNetwork([0], DIMENSION, CLASS_COUNT, L2_LAMBDA)

    with pytest.raises(AssertionError):
        L2RustArrayMultiClassBackpropClassifierNetwork(LAYER_SIZES, DIMENSION, class_count=1, l2_lambda=L2_LAMBDA)


def test_learn_batch_rejects_an_empty_batch():

    network = L2RustArrayMultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, L2_LAMBDA
    )
    with pytest.raises(AssertionError):
        network.learn_batch(0.1, [])
