import random

import pytest

from indrajala_ml.evaluate import class_balanced_disagreement_rate
from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.linear_classifier_network import LinearClassifierNetwork
from indrajala_ml.train import (
    random_alternating_training_data,
    reachable_reference_and_training_data,
    train_linear_classifier_network,
)
from tests.helpers import approx, network_with_hidden_thresholds


def test_half_widths_of_asymmetric_bounds():

    network = LinearClassifierNetwork(1, 2, [(-10.0, 10.0), (-2.0, 8.0)])

    assert network.half_widths() == [10.0, 5.0]


def test_cardinality_must_be_at_least_one():

    with pytest.raises(AssertionError):
        LinearClassifierNetwork(0, 2, [(-1.0, 1.0), (-1.0, 1.0)])


def test_dimension_must_match_bounds_length():

    with pytest.raises(AssertionError):
        LinearClassifierNetwork(1, 2, [(-1.0, 1.0)])


def test_input_bounds_must_all_have_positive_width():

    with pytest.raises(AssertionError):
        LinearClassifierNetwork(1, 2, [(-10.0, 10.0), (5.0, 5.0)])

    # inverted bounds (hi < lo) are equally nonsensical
    with pytest.raises(AssertionError):
        LinearClassifierNetwork(1, 2, [(-10.0, 10.0), (5.0, -5.0)])


def test_hidden_layer_snapshot_and_restore_round_trip():

    network = LinearClassifierNetwork.randomized(3, 2, square_bounds(10.0))
    before = network.snapshot()

    # perturb every hidden node so restoring is actually exercised, not a no-op
    for node in network.hidden_layer.nodes:
        node.update_input_weights([w + 1.0 for w in node.input_node_weights])
        node.threshold += 1.0

    assert network.snapshot() != before

    network.restore(before)

    assert network.snapshot() == before


def test_randomized_returns_an_already_randomized_classifier():

    bounds = square_bounds(10.0)
    classifier = LinearClassifierNetwork.randomized(2, 2, bounds)

    assert classifier.cardinality == 2
    assert classifier.dimension == 2
    # a fresh (non-randomized) node always starts at threshold=0.0, weights=[1.0, 1.0] -
    # confirm randomize() actually ran, not just construction
    assert any(
        node.threshold != 0.0 or list(node.input_node_weights) != [1.0, 1.0] for node in classifier.hidden_layer.nodes
    )


def test_randomize_scales_weight_range_with_input_bounds_half_width(monkeypatch: pytest.MonkeyPatch):

    # each dimension's weight range is 20 / half_width, so w_i * x_i has a similar magnitude
    # whatever its bounds, asymmetric bounds included; the threshold's range stays fixed
    calls: list[tuple[float, float]] = []
    original_uniform = random.uniform

    def recording_uniform(a: float, b: float) -> float:
        calls.append((a, b))
        return original_uniform(a, b)

    monkeypatch.setattr(random, "uniform", recording_uniform)

    network = LinearClassifierNetwork(1, 2, [(-1000.0, 1000.0), (-0.001, 0.001)])
    network.randomize()

    weight_call_1, weight_call_2, threshold_call = calls[:3]
    assert weight_call_1 == approx((-0.02, 0.02))  # 20 / half_width 1000
    assert weight_call_2 == approx((-20_000.0, 20_000.0))  # 20 / half_width 0.001
    assert threshold_call == (-5, 5)


def test_randomize_produces_reachable_classifiers_at_a_tiny_bounds_scale():

    # with unscaled weights, at small bounds the threshold dominates w.x, nearly every random
    # classifier is one class, and regeneration_attempts runs out
    bounds = square_bounds(0.001)

    _reference, training_data = reachable_reference_and_training_data(1, 2, bounds, 50)

    assert len(training_data) == 50


def test_learn_reduces_to_single_node_update_for_cardinality_one():

    dimension = 2
    bounds = square_bounds(10.0)
    learning_rate = 0.25

    for category in (0, 1):
        network = LinearClassifierNetwork.randomized(1, dimension, bounds)
        node = network.hidden_layer.nodes[0]
        weights_before = list(node.input_node_weights)
        threshold_before = node.threshold

        state = (3.0, -4.0)
        network.learn(learning_rate, state, category)

        expected_network = LinearClassifierNetwork(1, dimension, bounds)
        expected_node = expected_network.hidden_layer.nodes[0]
        expected_node.update_input_weights(weights_before)
        expected_node.threshold = threshold_before
        expected_network.update_state_layer(state)
        expected_node.learn(learning_rate, category)

        assert node.input_node_weights == expected_node.input_node_weights
        assert node.threshold == expected_node.threshold


def test_learn_matches_the_perceptron_update_rule_by_hand():

    # the cardinality-one test compares the rule with itself; this pins
    # w += learning_rate * (reference - actual) * input to hand-derived values

    dimension = 2
    bounds = square_bounds(10.0)
    learning_rate = 0.25

    # false negative: a fresh network (weights=[1, 1], threshold=0) is inactive at this
    # state (z = 1*3 + 1*-4 + 0 = -1 <= 0), but category=1 wants it active, so d = 1 - 0 = 1
    network = LinearClassifierNetwork(1, dimension, bounds)
    network.learn(learning_rate, (3.0, -4.0), 1)
    node = network.hidden_layer.nodes[0]
    assert node.input_node_weights == [1.75, 0.0]
    assert node.threshold == 0.25

    # false positive: a fresh network is active at this state (z = 1*3 + 1*4 + 0 = 7 > 0),
    # but category=0 wants it inactive, so d = 0 - 1 = -1
    network = LinearClassifierNetwork(1, dimension, bounds)
    network.learn(learning_rate, (3.0, 4.0), 0)
    node = network.hidden_layer.nodes[0]
    assert node.input_node_weights == [0.25, 0.0]
    assert node.threshold == -0.25


def test_association_node_activates_strictly_above_zero():

    # z <= 0 is inactive: weights [1, 1], threshold 0 give z = 1*1 + 1*-1 + 0 = 0 here
    network = LinearClassifierNetwork(1, 2, square_bounds(10.0))
    network.update_state_layer((1.0, -1.0))
    node = network.hidden_layer.nodes[0]

    assert node.z() == 0.0
    assert node.value() == 0.0


def test_required_active_defaults_to_cardinality():

    network = LinearClassifierNetwork(3, 2, square_bounds(10.0))

    assert network.required_active == 3


def test_required_active_must_be_between_one_and_cardinality():

    bounds = square_bounds(10.0)

    with pytest.raises(AssertionError):
        LinearClassifierNetwork(3, 2, bounds, required_active=0)

    with pytest.raises(AssertionError):
        LinearClassifierNetwork(3, 2, bounds, required_active=4)


def test_required_active_one_gives_or_semantics():

    bounds = square_bounds(10.0)
    network = network_with_hidden_thresholds(2, bounds, [-5.0, -5.0, -5.0], required_active=1)

    assert network.classify_state((0.0, 0.0)) == 0.0

    network.hidden_layer.nodes[1].threshold = 1.0  # exactly one hidden node active

    assert network.classify_state((0.0, 0.0)) == 1.0


def test_required_active_two_of_three_gives_majority_semantics():

    bounds = square_bounds(10.0)
    network = network_with_hidden_thresholds(2, bounds, [-5.0, -5.0, -5.0], required_active=2)

    network.hidden_layer.nodes[0].threshold = 1.0  # one of three active

    assert network.classify_state((0.0, 0.0)) == 0.0

    network.hidden_layer.nodes[1].threshold = 1.0  # two of three active

    assert network.classify_state((0.0, 0.0)) == 1.0


def test_learn_converges_under_or_combination():

    # minimum-disturbance selection needs only an output monotone in the active count, which
    # OR also is
    random.seed(0)

    cardinality, dimension, l = 2, 2, 10.0
    bounds = square_bounds(l, dimension)

    reference = LinearClassifierNetwork.randomized(cardinality, dimension, bounds, required_active=1)
    training_data = random_alternating_training_data(400, reference)

    student = LinearClassifierNetwork.randomized(cardinality, dimension, bounds, required_active=1)

    disagreement_before = class_balanced_disagreement_rate(reference, student, per_class_sample_count=300)
    train_linear_classifier_network(student, training_data, learning_rate=0.25, epochs=5)
    disagreement_after = class_balanced_disagreement_rate(reference, student, per_class_sample_count=300)

    assert disagreement_after < disagreement_before
    assert disagreement_after < 0.2


def test_learn_updates_only_the_single_closest_to_flipping_node():

    # the convergence tests could pass with a wrong selection: given known z() values, only the
    # responsible node with the smallest |z| may change

    dimension = 2
    bounds = square_bounds(10.0)
    learning_rate = 0.25
    state = (2.0, 3.0)

    # false negative: of two inactive nodes, the one at |z|=0.5 moves, not the one at |z|=5.0;
    # the active third node is untouched
    network = network_with_hidden_thresholds(dimension, bounds, [-5.0, -0.5, 2.0])
    far, near, active = network.hidden_layer.nodes
    network.learn(learning_rate, state, 1)

    assert near.input_node_weights == [0.5, 0.75]
    assert near.threshold == -0.25
    assert (list(far.input_node_weights), far.threshold) == ([0.0, 0.0], -5.0)
    assert (list(active.input_node_weights), active.threshold) == ([0.0, 0.0], 2.0)

    # false positive: all three nodes active; the closest-to-flipping one (threshold 0.4,
    # |z|=0.4) should be updated, not the other two
    network = network_with_hidden_thresholds(dimension, bounds, [3.0, 0.4, 6.0])
    far, near, farther = network.hidden_layer.nodes
    network.learn(learning_rate, state, 0)

    assert near.input_node_weights == [-0.5, -0.75]
    assert near.threshold == approx(0.15)
    assert (list(far.input_node_weights), far.threshold) == ([0.0, 0.0], 3.0)
    assert (list(farther.input_node_weights), farther.threshold) == ([0.0, 0.0], 6.0)

    # already correct: all three active and category=1 means the output already matches -
    # no hidden node should change at all
    network = network_with_hidden_thresholds(dimension, bounds, [3.0, 0.4, 6.0])
    before = network.snapshot()
    network.learn(learning_rate, state, 1)
    assert network.snapshot() == before
