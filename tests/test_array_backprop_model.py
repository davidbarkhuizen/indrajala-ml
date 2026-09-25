import random

import pytest

from indrajala_ml.model.array_backprop_classifier_network import ArrayBackpropClassifierNetwork
from indrajala_ml.model.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork
from tests.helpers import (
    assert_array_network_weights_match,
    assert_single_output_array_network_save_load_round_trip,
    assert_single_output_array_network_snapshot_restore_round_trip,
    matching_single_output_array_backprop_networks,
)

DIMENSION = 6
LAYER_SIZES = [5]

NETWORK_CLS = {"numpy": ArrayBackpropClassifierNetwork, "rust": RustArrayBackpropClassifierNetwork}


@pytest.fixture
def network_cls(backend):
    return NETWORK_CLS[backend.name]


def _matching_networks(rng: random.Random, backend):
    return matching_single_output_array_backprop_networks(
        rng, NETWORK_CLS[backend.name], backend.owned, LAYER_SIZES, DIMENSION
    )


def test_predict_probability_matches_across_a_random_sweep(backend):

    rng = random.Random(0)
    node_network, array_network = _matching_networks(rng, backend)

    for _ in range(50):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        expected = node_network.predict_probability(state)
        actual = array_network.predict_probability(state)
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_classify_state_matches_across_a_random_sweep(backend):

    rng = random.Random(1)
    node_network, array_network = _matching_networks(rng, backend)

    for _ in range(50):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        assert array_network.classify_state(state) == node_network.classify_state(state)


def test_learn_matches_after_every_step_not_just_at_the_end(backend):

    rng = random.Random(2)
    node_network, array_network = _matching_networks(rng, backend)
    learning_rate = 0.3

    for step in range(100):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        category = float(rng.randrange(2))

        node_network.learn(learning_rate, state, category)
        array_network.learn(learning_rate, state, category)

        assert_array_network_weights_match(node_network, array_network)


def test_learn_batch_matches_after_every_batch_not_just_at_the_end(backend):

    rng = random.Random(3)
    node_network, array_network = _matching_networks(rng, backend)
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


def test_randomized_builds_a_usable_network(network_cls):

    network = network_cls.randomized(LAYER_SIZES, DIMENSION)
    state = tuple(0.1 * i for i in range(DIMENSION))

    probability = network.predict_probability(state)
    assert 0.0 <= probability <= 1.0
    assert network.classify_state(state) in (0.0, 1.0)


def test_randomized_accepts_and_discards_input_bounds_for_duck_type_compatibility(network_cls):

    # ensemble_train.py calls classifier_cls.randomized(layer_sizes, dimension, input_bounds)
    network = network_cls.randomized(LAYER_SIZES, DIMENSION, [(-1.0, 1.0)] * DIMENSION)
    state = tuple(0.1 * i for i in range(DIMENSION))
    assert 0.0 <= network.predict_probability(state) <= 1.0


def test_snapshot_restore_round_trips_weights(network_cls):

    assert_single_output_array_network_snapshot_restore_round_trip(network_cls, LAYER_SIZES, DIMENSION)


def test_save_load_round_trips_weights_and_predictions(network_cls, tmp_path):

    network = network_cls.randomized(LAYER_SIZES, DIMENSION)
    state = tuple(0.1 * i for i in range(DIMENSION))

    assert_single_output_array_network_save_load_round_trip(
        network, network_cls.load, tmp_path, "model.json", state
    )


def test_construction_rejects_invalid_arguments(network_cls):

    with pytest.raises(AssertionError):
        network_cls([], DIMENSION)

    with pytest.raises(AssertionError):
        network_cls([0], DIMENSION)


def test_learn_batch_rejects_an_empty_batch(network_cls):

    network = network_cls.randomized(LAYER_SIZES, DIMENSION)
    with pytest.raises(AssertionError):
        network.learn_batch(0.1, [])
