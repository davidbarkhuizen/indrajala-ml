from __future__ import annotations

import numpy as np

from indrajala_ml.model.array_backprop_classifier_network import ArrayBackpropClassifierNetwork
from indrajala_ml.model.classification import argmax_first_occurrence
from indrajala_ml.model.model_io import load_json, save_json


class EnsembleArrayBackpropClassifierNetwork:
    """
    The numpy-array-backed sibling of EnsembleBackpropClassifierNetwork: a direct structural
    mirror, substituting ArrayBackpropClassifierNetwork for BackpropClassifierNetwork
    throughout. Same
    "assemble already-constructed classifiers, don't build them" composition as its per-node
    counterpart - see that class's own docstring for why.

    Unlike ArrayBackpropClassifierNetwork itself (no class_count notion at all), this wrapper
    does have a real class_count - the number of assembled sub-networks. save()/load() use the
    bare save_json/load_json primitives (model_io.py), not save_array_model_json: that helper's
    own snapshot handling assumes a flat per-layer (W, b) list (one network's own snapshot), but
    this class's snapshot is nested one level deeper (one such list per classifier) - the same
    "envelope shape doesn't fit the fixed layout" case save_json's own docstring names
    ConvMultiClassBackpropClassifierNetwork.save as an example of.
    """

    def __init__(self, classifiers: list[ArrayBackpropClassifierNetwork]) -> None:
        assert len(classifiers) >= 2, f"an ensemble needs at least 2 classifiers; got {len(classifiers)}"
        self.classifiers = classifiers
        self.class_count = len(classifiers)

    def predict_probabilities(self, state: tuple[float, ...]) -> list[float]:
        return [classifier.predict_probability(state) for classifier in self.classifiers]

    def classify_state(self, state: tuple[float, ...]) -> int:
        return argmax_first_occurrence(self.predict_probabilities(state))

    def snapshot(self) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        return [classifier.snapshot() for classifier in self.classifiers]

    def restore(self, snapshot: list[list[tuple[np.ndarray, np.ndarray]]]) -> None:
        for classifier, classifier_snapshot in zip(self.classifiers, snapshot):
            classifier.restore(classifier_snapshot)

    def save(self, path: str) -> None:
        assert len({classifier.dimension for classifier in self.classifiers}) == 1, "every classifier must share a dimension"
        assert (
            len({tuple(classifier.layer_sizes) for classifier in self.classifiers}) == 1
        ), "every classifier must share layer_sizes"
        first = self.classifiers[0]
        save_json(
            path,
            {
                "layer_sizes": first.layer_sizes,
                "dimension": first.dimension,
                "class_count": self.class_count,
                "snapshot": [
                    [(W.tolist(), b.tolist()) for W, b in classifier_snapshot] for classifier_snapshot in self.snapshot()
                ],
            },
        )

    @classmethod
    def load(cls, path: str) -> "EnsembleArrayBackpropClassifierNetwork":
        state = load_json(path)

        classifiers = [
            ArrayBackpropClassifierNetwork(state["layer_sizes"], state["dimension"])
            for _ in range(state["class_count"])
        ]
        ensemble = cls(classifiers)
        ensemble.restore(
            [[(np.array(W), np.array(b)) for W, b in classifier_snapshot] for classifier_snapshot in state["snapshot"]]
        )
        return ensemble
