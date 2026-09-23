import random

import numpy as np
import pytest

from indrajala_ml.model.array_backprop_classifier_network import ArrayBackpropClassifierNetwork
from tests.helpers import (
    assert_array_network_weights_match,
    assert_single_output_array_network_save_load_round_trip,
    assert_single_output_array_network_snapshot_restore_round_trip,
    matching_single_output_array_backprop_networks,
)

DIMENSION = 6
LAYER_SIZES = [5]


def _matching_networks(rng: random.Random):
    return matching_single_output_array_backprop_networks(
        rng, ArrayBackpropClassifierNetwork, np.array, LAYER_SIZES, DIMENSION
    )


def test_predict_probability_matches_across_a_random_sweep():

    rng = random.Random(0)
    node_network, array_network = _matching_networks(rng)

    for _ in range(50):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        expected = node_network.predict_probability(state)
        actual = array_network.predict_probability(state)
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_classify_state_matches_across_a_random_sweep():

    rng = random.Random(1)
    node_network, array_network = _matching_networks(rng)

    for _ in range(50):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        assert array_network.classify_state(state) == node_network.classify_state(state)


def test_learn_matches_after_every_step_not_just_at_the_end():

    # one silently-wrong intermediate step should fail loudly rather than being averaged away
    # by many steps - the same "required regression gate" discipline every array-layer sibling's
    # test suite in this codebase applies
    rng = random.Random(2)
    node_network, array_network = _matching_networks(rng)
    learning_rate = 0.3

    for step in range(100):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        category = float(rng.randrange(2))

        node_network.learn(learning_rate, state, category)
        array_network.learn(learning_rate, state, category)

        assert_array_network_weights_match(node_network, array_network)


def test_learn_batch_matches_after_every_batch_not_just_at_the_end():

    rng = random.Random(3)
    node_network, array_network = _matching_networks(rng)
    learning_rate = 0.3
    batch_size = 8

    for _ in range(20):
        batch = [
            (tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION)), float(rng.randrange(2)))
            for _ in range(batch_size)
        ]

        node_network.learn_batch(learning_rate, batch)
        array_network.learn_batch(learning_rate, batch)

        assert_array_network_weights_match(node_network, array_network)


def test_randomized_builds_a_usable_network():

    network = ArrayBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION)
    state = tuple(0.1 * i for i in range(DIMENSION))

    probability = network.predict_probability(state)
    assert 0.0 <= probability <= 1.0
    assert network.classify_state(state) in (0.0, 1.0)


def test_randomized_accepts_and_discards_input_bounds_for_duck_type_compatibility():

    # ensemble_train.py's classifier_cls contract always calls
    # classifier_cls.randomized(layer_sizes, dimension, input_bounds) - this class has no real
    # use for it (no StateLayer/input_bounds notion), but must accept it without raising
    network = ArrayBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION, [(-1.0, 1.0)] * DIMENSION)
    state = tuple(0.1 * i for i in range(DIMENSION))
    assert 0.0 <= network.predict_probability(state) <= 1.0


def test_snapshot_restore_round_trips_weights():

    assert_single_output_array_network_snapshot_restore_round_trip(ArrayBackpropClassifierNetwork, LAYER_SIZES, DIMENSION)


def test_save_load_round_trips_weights_and_predictions(tmp_path):

    network = ArrayBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION)
    state = tuple(0.1 * i for i in range(DIMENSION))

    assert_single_output_array_network_save_load_round_trip(
        network,
        ArrayBackpropClassifierNetwork.load,
        tmp_path,
        "array_backprop_model.json",
        state,
    )


def test_construction_rejects_invalid_arguments():

    with pytest.raises(AssertionError):
        ArrayBackpropClassifierNetwork([], DIMENSION)

    with pytest.raises(AssertionError):
        ArrayBackpropClassifierNetwork([0], DIMENSION)


def test_learn_batch_rejects_an_empty_batch():

    network = ArrayBackpropClassifierNetwork.randomized(LAYER_SIZES, DIMENSION)
    with pytest.raises(AssertionError):
        network.learn_batch(0.1, [])
