from pathlib import Path
from typing import Any

import pytest

from indrajala_ml.model.array_backprop_classifier_network import ArrayBackpropClassifierNetwork
from indrajala_ml.model.ensemble_array_backprop_classifier_network import EnsembleArrayBackpropClassifierNetwork
from indrajala_ml.model.ensemble_rust_array_backprop_classifier_network import (
    EnsembleRustArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork
from tests.helpers import Backend, approx

EnsembleCls = type[EnsembleArrayBackpropClassifierNetwork] | type[EnsembleRustArrayBackpropClassifierNetwork]
ENSEMBLE_CLS: dict[str, EnsembleCls] = {
    "numpy": EnsembleArrayBackpropClassifierNetwork,
    "rust": EnsembleRustArrayBackpropClassifierNetwork,
}
# a classifier (class) is Any here: an ensemble takes its own backend's classifiers, which a union can't express
CLASSIFIER_CLS: dict[str, Any] = {"numpy": ArrayBackpropClassifierNetwork, "rust": RustArrayBackpropClassifierNetwork}


@pytest.fixture
def ensemble_cls(backend: Backend) -> EnsembleCls:
    return ENSEMBLE_CLS[backend.name]


@pytest.fixture
def classifier_cls(backend: Backend) -> Any:
    return CLASSIFIER_CLS[backend.name]


def _fixed_classifier(backend: Backend, output_weight: float, output_bias: float) -> Any:
    # one input and one hidden node, the same in every classifier, so each output is
    # hand-computable: a_h = sigmoid(0.5*2.0 + 0.1) = 0.7502601055951177
    classifier = CLASSIFIER_CLS[backend.name]([1], 1)
    classifier.layers[0].W = backend.owned([[0.5]])
    classifier.layers[0].b = backend.owned([0.1])
    classifier.output_layer.W = backend.owned([[output_weight]])
    classifier.output_layer.b = backend.owned([output_bias])
    return classifier


def _fixed_ensemble(
    ensemble_cls: EnsembleCls, backend: Backend
) -> EnsembleArrayBackpropClassifierNetwork | EnsembleRustArrayBackpropClassifierNetwork:
    return ensemble_cls(
        [
            _fixed_classifier(backend, 0.8, -0.2),
            _fixed_classifier(backend, -0.3, 0.4),
            _fixed_classifier(backend, 2.0, 0.0),
        ]
    )


def test_ensemble_requires_at_least_two_classifiers(ensemble_cls: EnsembleCls, backend: Backend):

    with pytest.raises(AssertionError):
        ensemble_cls([_fixed_classifier(backend, 0.8, -0.2)])


def test_predict_probabilities_matches_each_sub_networks_own_output(ensemble_cls: EnsembleCls, backend: Backend):

    # computed independently, as in test_ensemble_backprop_classifier_network.py:
    # a_o0 = sigmoid(0.8*a_h - 0.2) = 0.5987376536170401
    # a_o1 = sigmoid(-0.3*a_h + 0.4) = 0.5436193278499907
    # a_o2 = sigmoid(2.0*a_h + 0.0) = 0.8176520510294325
    ensemble = _fixed_ensemble(ensemble_cls, backend)

    probabilities = ensemble.predict_probabilities((2.0,))

    assert probabilities[0] == approx(0.5987376536170401)
    assert probabilities[1] == approx(0.5436193278499907)
    assert probabilities[2] == approx(0.8176520510294325)


def test_classify_state_returns_the_argmax_across_sub_networks(ensemble_cls: EnsembleCls, backend: Backend):

    ensemble = _fixed_ensemble(ensemble_cls, backend)

    assert ensemble.classify_state((2.0,)) == 2


def test_snapshot_and_restore_round_trip(ensemble_cls: EnsembleCls, classifier_cls: Any):

    ensemble = ensemble_cls([classifier_cls.randomized([3], 2) for _ in range(3)])

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


def test_save_and_load_round_trip(ensemble_cls: EnsembleCls, classifier_cls: Any, tmp_path: Path):

    ensemble = ensemble_cls([classifier_cls.randomized([3], 2) for _ in range(3)])
    for classifier in ensemble.classifiers:
        for _ in range(5):
            classifier.learn(0.1, (1.0, -2.0), 1.0)

    path = str(tmp_path / "ensemble.json")
    ensemble.save(path)
    loaded = ensemble_cls.load(path)

    assert loaded.class_count == ensemble.class_count
    for state in [(1.0, -2.0), (-3.0, 4.0), (0.0, 0.0)]:
        assert loaded.predict_probabilities(state) == approx(ensemble.predict_probabilities(state))
        assert loaded.classify_state(state) == ensemble.classify_state(state)
