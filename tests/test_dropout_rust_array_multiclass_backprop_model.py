import random

import pytest

import indrajala_math_rust as pa
from indrajala_ml.model.dropout_rust_array_multiclass_backprop_classifier_network import (
    DropoutRustArrayMultiClassBackpropClassifierNetwork,
)
from tests.helpers import (
    assert_array_network_save_load_round_trip,
    matching_dropout_array_backprop_networks,
)

DIMENSION = 6
LAYER_SIZES = [5]
CLASS_COUNT = 3
DROP_PROBABILITY = 0.5


def _matching_networks(rng: random.Random, bounds: float = 10.0):
    return matching_dropout_array_backprop_networks(
        rng,
        DropoutRustArrayMultiClassBackpropClassifierNetwork,
        pa.Array,
        LAYER_SIZES,
        DIMENSION,
        CLASS_COUNT,
        DROP_PROBABILITY,
        bounds,
    )


def test_predict_probabilities_at_eval_mode_matches_across_a_random_sweep():

    rng = random.Random(0)
    node_network, rust_network = _matching_networks(rng)

    for _ in range(50):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        expected = node_network.predict_probabilities(state)
        actual = rust_network.predict_probabilities(state)
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_classify_state_at_eval_mode_matches_across_a_random_sweep():

    rng = random.Random(1)
    node_network, rust_network = _matching_networks(rng)

    for _ in range(50):
        state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
        assert rust_network.classify_state(state) == node_network.classify_state(state)


def test_predict_probabilities_is_deterministic_run_to_run_no_stochasticity_at_inference():

    network = DropoutRustArrayMultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, DROP_PROBABILITY
    )
    state = tuple(0.1 * i for i in range(DIMENSION))

    predictions = [tuple(network.predict_probabilities(state)) for _ in range(20)]

    assert len(set(predictions)) == 1


def test_predict_probabilities_between_learn_calls_is_unaffected_by_training_mode():

    network = DropoutRustArrayMultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, DROP_PROBABILITY
    )
    state = tuple(0.1 * i for i in range(DIMENSION))

    network.learn(0.1, state, 1)

    assert all(not layer.training for layer in network.hidden_layers)
    probabilities = network.predict_probabilities(state)
    assert len(probabilities) == CLASS_COUNT
    assert all(0.0 <= p <= 1.0 for p in probabilities)


def test_learn_batch_draws_an_independent_mask_per_example_and_leaves_training_mode_off():

    network = DropoutRustArrayMultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, DROP_PROBABILITY
    )
    batch = [
        (tuple(0.1 * i + 0.01 * j for i in range(DIMENSION)), j % CLASS_COUNT) for j in range(6)
    ]

    network.learn_batch(0.1, batch)

    assert all(not layer.training for layer in network.hidden_layers)
    for layer in network.hidden_layers:
        assert layer._mask_batch.shape == (len(batch), layer.size)


def test_learn_moves_the_weights():

    network = DropoutRustArrayMultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, DROP_PROBABILITY
    )
    before = network.snapshot()

    for i in range(10):
        state = tuple(0.1 * i + 0.01 * j for j in range(DIMENSION))
        network.learn(0.1, state, i % CLASS_COUNT)

    after = network.snapshot()
    assert any(W1.tolist() != W2.tolist() for (W1, _b1), (W2, _b2) in zip(before, after))


def test_randomized_builds_a_usable_network():

    network = DropoutRustArrayMultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, DROP_PROBABILITY
    )
    state = tuple(0.1 * i for i in range(DIMENSION))

    probabilities = network.predict_probabilities(state)
    assert len(probabilities) == CLASS_COUNT
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert 0 <= network.classify_state(state) < CLASS_COUNT


def test_snapshot_restore_round_trips_weights():

    network = DropoutRustArrayMultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, DROP_PROBABILITY
    )
    snapshot = network.snapshot()

    other = DropoutRustArrayMultiClassBackpropClassifierNetwork(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, DROP_PROBABILITY
    )
    other.restore(snapshot)

    for (W1, b1), (W2, b2) in zip(network.snapshot(), other.snapshot()):
        assert W1.tolist() == W2.tolist()
        assert b1.tolist() == b2.tolist()


def test_save_load_round_trips_weights_and_predictions(tmp_path):

    network = DropoutRustArrayMultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, DROP_PROBABILITY
    )
    state = tuple(0.1 * i for i in range(DIMENSION))

    assert_array_network_save_load_round_trip(
        network,
        DropoutRustArrayMultiClassBackpropClassifierNetwork.load,
        tmp_path,
        "dropout_rust_array_model.json",
        state,
    )


def test_save_load_round_trips_the_drop_probability(tmp_path):

    network = DropoutRustArrayMultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, drop_probability=0.3
    )
    path = str(tmp_path / "dropout_rust_array_hyperparams.json")
    network.save(path)

    loaded = DropoutRustArrayMultiClassBackpropClassifierNetwork.load(path)
    assert loaded.drop_probability == 0.3


def test_construction_rejects_invalid_arguments():

    with pytest.raises(AssertionError):
        DropoutRustArrayMultiClassBackpropClassifierNetwork([], DIMENSION, CLASS_COUNT, DROP_PROBABILITY)

    with pytest.raises(AssertionError):
        DropoutRustArrayMultiClassBackpropClassifierNetwork([0], DIMENSION, CLASS_COUNT, DROP_PROBABILITY)

    with pytest.raises(AssertionError):
        DropoutRustArrayMultiClassBackpropClassifierNetwork(
            LAYER_SIZES, DIMENSION, class_count=1, drop_probability=DROP_PROBABILITY
        )

    with pytest.raises(AssertionError):
        DropoutRustArrayMultiClassBackpropClassifierNetwork(
            LAYER_SIZES, DIMENSION, CLASS_COUNT, drop_probability=1.0
        )


def test_drop_probability_is_a_required_constructor_argument():

    with pytest.raises(TypeError):
        DropoutRustArrayMultiClassBackpropClassifierNetwork(LAYER_SIZES, DIMENSION, CLASS_COUNT)  # type: ignore[call-arg]


def test_learn_batch_rejects_an_empty_batch():

    network = DropoutRustArrayMultiClassBackpropClassifierNetwork.randomized(
        LAYER_SIZES, DIMENSION, CLASS_COUNT, DROP_PROBABILITY
    )
    with pytest.raises(AssertionError):
        network.learn_batch(0.1, [])
