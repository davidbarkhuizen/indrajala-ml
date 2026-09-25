from pathlib import Path

import pytest

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.ensemble_backprop_classifier_network import EnsembleBackpropClassifierNetwork
from tests.helpers import approx, assert_save_and_load_round_trip, assert_snapshot_restore_round_trip


def _fixed_classifier(output_weight: float, output_bias: float) -> BackpropClassifierNetwork:
    # classifiers differ only in the output layer: a_h = sigmoid(0.5*2.0 + 0.1) =
    # 0.7502601055951177 for all of them
    classifier = BackpropClassifierNetwork([1], 1, [(-10.0, 10.0)])
    hidden_node = classifier.hidden_layers[0].nodes[0]
    output_node = classifier.output_layer.nodes[0]
    hidden_node.update_input_weights([0.5])
    hidden_node.bias = 0.1
    output_node.update_input_weights([output_weight])
    output_node.bias = output_bias
    return classifier


def test_ensemble_requires_at_least_two_classifiers():

    with pytest.raises(AssertionError):
        EnsembleBackpropClassifierNetwork([_fixed_classifier(0.8, -0.2)])


def test_predict_probabilities_matches_each_sub_networks_own_output():

    # a_o0 = sigmoid(0.8*a_h - 0.2) = 0.5987376536170401
    # a_o1 = sigmoid(-0.3*a_h + 0.4) = 0.5436193278499907
    # a_o2 = sigmoid(2.0*a_h + 0.0) = 0.8176520510294325
    ensemble = EnsembleBackpropClassifierNetwork(
        [_fixed_classifier(0.8, -0.2), _fixed_classifier(-0.3, 0.4), _fixed_classifier(2.0, 0.0)]
    )

    probabilities = ensemble.predict_probabilities((2.0,))

    assert probabilities[0] == approx(0.5987376536170401)
    assert probabilities[1] == approx(0.5436193278499907)
    assert probabilities[2] == approx(0.8176520510294325)


def test_classify_state_returns_the_argmax_across_sub_networks():

    ensemble = EnsembleBackpropClassifierNetwork(
        [_fixed_classifier(0.8, -0.2), _fixed_classifier(-0.3, 0.4), _fixed_classifier(2.0, 0.0)]
    )

    # classifier 2's output (0.818) is the clear highest of the three
    assert ensemble.classify_state((2.0,)) == 2


def test_snapshot_and_restore_round_trip():

    ensemble = EnsembleBackpropClassifierNetwork(
        [BackpropClassifierNetwork.randomized([3], 2, [(-10.0, 10.0), (-10.0, 10.0)]) for _ in range(3)]
    )

    def step():
        for classifier in ensemble.classifiers:
            classifier.learn(0.1, (1.0, -2.0), 1.0)

    assert_snapshot_restore_round_trip(ensemble, step)


def test_save_and_load_round_trip(tmp_path: Path):

    ensemble = EnsembleBackpropClassifierNetwork(
        [BackpropClassifierNetwork.randomized([3], 2, [(-10.0, 10.0), (-10.0, 10.0)]) for _ in range(3)]
    )
    for classifier in ensemble.classifiers:
        for _ in range(5):
            classifier.learn(0.1, (1.0, -2.0), 1.0)

    loaded = assert_save_and_load_round_trip(
        ensemble,
        EnsembleBackpropClassifierNetwork.load,
        tmp_path,
        "ensemble.json",
        [(1.0, -2.0), (-3.0, 4.0), (0.0, 0.0)],
    )

    assert loaded.class_count == ensemble.class_count
