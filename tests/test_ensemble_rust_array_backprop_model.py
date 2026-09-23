import pytest

import indrajala_math_rust as pa
from indrajala_ml.model.ensemble_rust_array_backprop_classifier_network import (
    EnsembleRustArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork


def _fixed_classifier(output_weight: float, output_bias: float) -> RustArrayBackpropClassifierNetwork:
    # dimension=1, one hidden node - fixed hidden weights shared by every classifier in these
    # tests, only the output layer differs, so each classifier's predict_probability is
    # independently hand-computable: a_h = sigmoid(0.5*2.0 + 0.1) = 0.7502601055951177 - the same
    # fixture values test_ensemble_array_backprop_model.py's own test uses
    classifier = RustArrayBackpropClassifierNetwork([1], 1)
    classifier.layers[0].W = pa.Array([[0.5]])
    classifier.layers[0].b = pa.Array([0.1])
    classifier.output_layer.W = pa.Array([[output_weight]])
    classifier.output_layer.b = pa.Array([output_bias])
    return classifier


def test_ensemble_requires_at_least_two_classifiers():

    with pytest.raises(AssertionError):
        EnsembleRustArrayBackpropClassifierNetwork([_fixed_classifier(0.8, -0.2)])


def test_predict_probabilities_matches_each_sub_networks_own_output():

    # a_o0 = sigmoid(0.8*a_h - 0.2) = 0.5987376536170401
    # a_o1 = sigmoid(-0.3*a_h + 0.4) = 0.5436193278499907
    # a_o2 = sigmoid(2.0*a_h + 0.0) = 0.8176520510294325
    # (independently computed, not re-derived from the implementation under test)
    ensemble = EnsembleRustArrayBackpropClassifierNetwork(
        [_fixed_classifier(0.8, -0.2), _fixed_classifier(-0.3, 0.4), _fixed_classifier(2.0, 0.0)]
    )

    probabilities = ensemble.predict_probabilities((2.0,))

    assert probabilities[0] == pytest.approx(0.5987376536170401)
    assert probabilities[1] == pytest.approx(0.5436193278499907)
    assert probabilities[2] == pytest.approx(0.8176520510294325)


def test_classify_state_returns_the_argmax_across_sub_networks():

    ensemble = EnsembleRustArrayBackpropClassifierNetwork(
        [_fixed_classifier(0.8, -0.2), _fixed_classifier(-0.3, 0.4), _fixed_classifier(2.0, 0.0)]
    )

    # classifier 2's output (0.818) is the clear highest of the three
    assert ensemble.classify_state((2.0,)) == 2


def test_snapshot_and_restore_round_trip():

    ensemble = EnsembleRustArrayBackpropClassifierNetwork(
        [RustArrayBackpropClassifierNetwork.randomized([3], 2) for _ in range(3)]
    )

    before = [[(W.copy(), b.copy()) for W, b in classifier_snapshot] for classifier_snapshot in ensemble.snapshot()]

    for classifier in ensemble.classifiers:
        for _ in range(5):
            classifier.learn(0.1, (1.0, -2.0), 1.0)

    after = ensemble.snapshot()
    assert any(
        W1.tolist() != W2.tolist()
        for classifier_before, classifier_after in zip(before, after)
        for (W1, _b1), (W2, _b2) in zip(classifier_before, classifier_after)
    )

    ensemble.restore(before)

    for classifier_before, classifier_after in zip(before, ensemble.snapshot()):
        for (W1, b1), (W2, b2) in zip(classifier_before, classifier_after):
            assert W1.tolist() == W2.tolist()
            assert b1.tolist() == b2.tolist()


def test_save_and_load_round_trip(tmp_path):

    ensemble = EnsembleRustArrayBackpropClassifierNetwork(
        [RustArrayBackpropClassifierNetwork.randomized([3], 2) for _ in range(3)]
    )
    for classifier in ensemble.classifiers:
        for _ in range(5):
            classifier.learn(0.1, (1.0, -2.0), 1.0)

    path = str(tmp_path / "ensemble_rust_array.json")
    ensemble.save(path)
    loaded = EnsembleRustArrayBackpropClassifierNetwork.load(path)

    assert loaded.class_count == ensemble.class_count
    for state in [(1.0, -2.0), (-3.0, 4.0), (0.0, 0.0)]:
        assert loaded.predict_probabilities(state) == pytest.approx(ensemble.predict_probabilities(state))
        assert loaded.classify_state(state) == ensemble.classify_state(state)
