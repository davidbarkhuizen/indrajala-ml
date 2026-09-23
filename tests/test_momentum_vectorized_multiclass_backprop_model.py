import random

import numpy as np
import pytest

from indrajala_ml.model.momentum_vectorized_multiclass_backprop_classifier_network import (
    MomentumVectorizedMultiClassBackpropClassifierNetwork,
)
from tests.helpers import (
    assert_array_network_save_load_round_trip,
    assert_array_network_weights_match,
    matching_momentum_array_backprop_networks,
)

DIMENSION = 6
LAYER_SIZES = [5]
CLASS_COUNT = 3
MOMENTUM = 0.5


def _matching_networks(rng: random.Random, bounds: float = 10.0):
    return matching_momentum_array_backprop_networks(
        rng,
        MomentumVectorizedMultiClassBackpropClassifierNetwork,
        np.array,
        LAYER_SIZES,
        DIMENSION,
        CLASS_COUNT,
        MOMENTUM,
        bounds,
    )


def test_predict_probabilities_matches_across_a_random_sweep():

    rng = random.Random(0)
    node_network, array_network = _matching_networks(rng)

    for _ in range(50):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        expected = node_network.predict_probabilities(state)
        actual = array_network.predict_probabilities(state)
        assert np.allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_classify_state_matches_across_a_random_sweep():

    rng = random.Random(1)
    node_network, array_network = _matching_networks(rng)

    for _ in range(50):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        assert array_network.classify_state(state) == node_network.classify_state(state)


def test_learn_matches_after_every_step_not_just_at_the_end():

    rng = random.Random(2)
    node_network, array_network = _matching_networks(rng)
    learning_rate = 0.1

    for step in range(30):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        category = rng.randrange(CLASS_COUNT)

        node_network.learn(learning_rate, state, category)
        array_network.learn(learning_rate, state, category)

        assert_array_network_weights_match(node_network, array_network)


def test_learn_batch_matches_after_every_batch_not_just_at_the_end():

    rng = random.Random(3)
    node_network, array_network = _matching_networks(rng)
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


def test_randomized_builds_a_usable_network():

    network = MomentumVectorizedMultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, MOMENTUM
    )
    state = tuple(0.1 * i for i in range(DIMENSION))

    probabilities = network.predict_probabilities(state)
    assert len(probabilities) == CLASS_COUNT
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert 0 <= network.classify_state(state) < CLASS_COUNT


def test_snapshot_restore_round_trips_weights():

    # not assert_array_network_snapshot_restore_round_trip (helpers.py): that helper's own
    # .randomized(layer_sizes, dimension, class_count)/(layer_sizes, dimension, class_count)
    # calls assume no further required constructor argument, which doesn't hold here - momentum
    # is required, no default, matching MomentumBackpropClassifierNetwork's own posture.
    network = MomentumVectorizedMultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, MOMENTUM
    )
    snapshot = network.snapshot()

    other = MomentumVectorizedMultiClassBackpropClassifierNetwork(LAYER_SIZES, DIMENSION, CLASS_COUNT, MOMENTUM)
    other.restore(snapshot)

    for (W1, b1), (W2, b2) in zip(network.snapshot(), other.snapshot()):
        assert W1.tolist() == W2.tolist()
        assert b1.tolist() == b2.tolist()


def test_save_load_round_trips_weights_and_predictions(tmp_path):

    network = MomentumVectorizedMultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, MOMENTUM
    )
    state = tuple(0.1 * i for i in range(DIMENSION))

    assert_array_network_save_load_round_trip(
        network,
        MomentumVectorizedMultiClassBackpropClassifierNetwork.load,
        tmp_path,
        "momentum_vectorized_model.json",
        state,
    )


def test_save_load_round_trips_the_momentum_coefficient(tmp_path):

    network = MomentumVectorizedMultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, momentum=0.7
    )
    path = str(tmp_path / "momentum_vectorized_hyperparams.json")
    network.save(path)

    loaded = MomentumVectorizedMultiClassBackpropClassifierNetwork.load(path)
    assert loaded.momentum == 0.7


def test_construction_rejects_invalid_arguments():

    with pytest.raises(AssertionError):
        MomentumVectorizedMultiClassBackpropClassifierNetwork([], DIMENSION, CLASS_COUNT, MOMENTUM)

    with pytest.raises(AssertionError):
        MomentumVectorizedMultiClassBackpropClassifierNetwork([0], DIMENSION, CLASS_COUNT, MOMENTUM)

    with pytest.raises(AssertionError):
        MomentumVectorizedMultiClassBackpropClassifierNetwork(
            LAYER_SIZES, DIMENSION, class_count=1, momentum=MOMENTUM
        )


def test_learn_batch_rejects_an_empty_batch():

    network = MomentumVectorizedMultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, MOMENTUM
    )
    with pytest.raises(AssertionError):
        network.learn_batch(0.1, [])
