import random

import pytest

import indrajala_ml_array as pa
from indrajala_ml.model.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork
from tests.helpers import (
    assert_array_network_weights_match,
    assert_single_output_array_network_save_load_round_trip,
    assert_single_output_array_network_snapshot_restore_round_trip,
    matching_single_output_array_backprop_networks,
)

DIMENSION = 6
LAYER_SIZES = [5]


def _matching_networks(rng: random.Random):
    # tier 1 - identical fixed weights/inputs injected directly, never randomize(), since
    # indrajala_ml_array.uniform's RNG can never be seed-comparable against Python's random
    # module
    return matching_single_output_array_backprop_networks(
        rng, RustArrayBackpropClassifierNetwork, pa.Array, LAYER_SIZES, DIMENSION
    )


def test_predict_probability_matches_across_a_random_sweep():

    rng = random.Random(0)
    node_network, rust_network = _matching_networks(rng)

    for _ in range(50):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        expected = node_network.predict_probability(state)
        actual = rust_network.predict_probability(state)
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
    learning_rate = 0.3

    for step in range(100):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        category = float(rng.randrange(2))

        node_network.learn(learning_rate, state, category)
        rust_network.learn(learning_rate, state, category)

        assert_array_network_weights_match(node_network, rust_network)


def test_learn_batch_matches_after_every_batch_not_just_at_the_end():

    rng = random.Random(3)
    node_network, rust_network = _matching_networks(rng)
    learning_rate = 0.3
    batch_size = 8

    for _ in range(20):
        batch = [
            (tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION)), float(rng.randrange(2)))
            for _ in range(batch_size)
        ]

        node_network.learn_batch(learning_rate, batch)
        rust_network.learn_batch(learning_rate, batch)

        assert_array_network_weights_match(node_network, rust_network)


def test_randomized_builds_a_usable_network():

    network = RustArrayBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION)
    state = tuple(0.1 * i for i in range(DIMENSION))

    probability = network.predict_probability(state)
    assert 0.0 <= probability <= 1.0
    assert network.classify_state(state) in (0.0, 1.0)


def test_randomized_accepts_and_discards_input_bounds_for_duck_type_compatibility():

    network = RustArrayBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION, [(-1.0, 1.0)] * DIMENSION)
    state = tuple(0.1 * i for i in range(DIMENSION))
    assert 0.0 <= network.predict_probability(state) <= 1.0


def test_snapshot_restore_round_trips_weights():

    assert_single_output_array_network_snapshot_restore_round_trip(RustArrayBackpropClassifierNetwork, LAYER_SIZES, DIMENSION)


def test_save_load_round_trips_weights_and_predictions(tmp_path):

    network = RustArrayBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION)
    state = tuple(0.1 * i for i in range(DIMENSION))

    assert_single_output_array_network_save_load_round_trip(
        network,
        RustArrayBackpropClassifierNetwork.load,
        tmp_path,
        "rust_array_backprop_model.json",
        state,
    )


def test_construction_rejects_invalid_arguments():

    with pytest.raises(AssertionError):
        RustArrayBackpropClassifierNetwork([], DIMENSION)

    with pytest.raises(AssertionError):
        RustArrayBackpropClassifierNetwork([0], DIMENSION)


def test_learn_batch_rejects_an_empty_batch():

    network = RustArrayBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION)
    with pytest.raises(AssertionError):
        network.learn_batch(0.1, [])
