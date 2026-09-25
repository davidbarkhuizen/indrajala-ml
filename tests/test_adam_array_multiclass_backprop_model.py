import random

import pytest

from indrajala_ml.model.adam_rust_array_multiclass_backprop_classifier_network import (
    AdamRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.adam_vectorized_multiclass_backprop_classifier_network import (
    AdamVectorizedMultiClassBackpropClassifierNetwork,
)
from tests.helpers import (
    assert_array_network_save_load_round_trip,
    assert_array_network_snapshot_restore_round_trip,
    assert_array_network_weights_match,
    matching_adam_array_backprop_networks,
)

DIMENSION = 6
LAYER_SIZES = [5]
CLASS_COUNT = 3
BETA1, BETA2, EPSILON = 0.9, 0.999, 1e-8

NETWORK_CLS = {
    "numpy": AdamVectorizedMultiClassBackpropClassifierNetwork,
    "rust": AdamRustArrayMultiClassBackpropClassifierNetwork,
}


@pytest.fixture
def network_cls(backend):
    return NETWORK_CLS[backend.name]


def _matching_networks(rng: random.Random, backend):
    return matching_adam_array_backprop_networks(
        rng, NETWORK_CLS[backend.name], backend.owned, LAYER_SIZES, DIMENSION, CLASS_COUNT, BETA1, BETA2, EPSILON
    )


def test_predict_probabilities_matches_across_a_random_sweep(backend):

    rng = random.Random(0)
    node_network, array_network = _matching_networks(rng, backend)

    for _ in range(50):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        expected = node_network.predict_probabilities(state)
        actual = array_network.predict_probabilities(state)
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_classify_state_matches_across_a_random_sweep(backend):

    rng = random.Random(1)
    node_network, array_network = _matching_networks(rng, backend)

    for _ in range(50):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        assert array_network.classify_state(state) == node_network.classify_state(state)


def test_learn_matches_after_every_step_not_just_at_the_end(backend):

    # checked after every step: Adam's m/v/t state only shows a mistake across repeated steps
    rng = random.Random(2)
    node_network, array_network = _matching_networks(rng, backend)
    learning_rate = 0.1

    for step in range(30):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        category = rng.randrange(CLASS_COUNT)

        node_network.learn(learning_rate, state, category)
        array_network.learn(learning_rate, state, category)

        assert_array_network_weights_match(node_network, array_network)


def test_learn_batch_matches_after_every_batch_not_just_at_the_end(backend):

    rng = random.Random(3)
    node_network, array_network = _matching_networks(rng, backend)
    learning_rate = 0.1
    batch_size = 8

    for _ in range(15):
        batch = [
            (tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION)), rng.randrange(CLASS_COUNT))
            for _ in range(batch_size)
        ]

        node_network.learn_batch(learning_rate, batch)
        array_network.learn_batch(learning_rate, batch)

        assert_array_network_weights_match(node_network, array_network)


def test_randomized_builds_a_usable_network(network_cls):

    network = network_cls.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT)
    state = tuple(0.1 * i for i in range(DIMENSION))

    probabilities = network.predict_probabilities(state)
    assert len(probabilities) == CLASS_COUNT
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert 0 <= network.classify_state(state) < CLASS_COUNT


def test_snapshot_restore_round_trips_weights(network_cls):

    assert_array_network_snapshot_restore_round_trip(network_cls, LAYER_SIZES, DIMENSION, CLASS_COUNT)


def test_save_load_round_trips_weights_and_predictions(network_cls, tmp_path):

    network = network_cls.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT)
    state = tuple(0.1 * i for i in range(DIMENSION))

    assert_array_network_save_load_round_trip(network, network_cls.load, tmp_path, "model.json", state)


def test_save_load_round_trips_the_adam_hyperparameters(network_cls, tmp_path):

    network = network_cls.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT, beta1=0.8, beta2=0.99, epsilon=1e-6)
    path = str(tmp_path / "hyperparameters.json")
    network.save(path)

    loaded = network_cls.load(path)
    assert loaded.beta1 == 0.8
    assert loaded.beta2 == 0.99
    assert loaded.epsilon == 1e-6


def test_construction_rejects_invalid_arguments(network_cls):

    with pytest.raises(AssertionError):
        network_cls([], DIMENSION, CLASS_COUNT)

    with pytest.raises(AssertionError):
        network_cls([0], DIMENSION, CLASS_COUNT)

    with pytest.raises(AssertionError):
        network_cls(LAYER_SIZES, DIMENSION, 1)


def test_learn_batch_rejects_an_empty_batch(network_cls):

    network = network_cls.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT)
    with pytest.raises(AssertionError):
        network.learn_batch(0.1, [])
