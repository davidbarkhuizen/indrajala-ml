from __future__ import annotations

import indrajala_ml_array as pa

from indrajala_ml.model.classification import argmax_first_occurrence
from indrajala_ml.model.model_io import load_json, save_json
from indrajala_ml.model.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork


class EnsembleRustArrayBackpropClassifierNetwork:
    """
    The Rust-array-core-backed sibling of EnsembleArrayBackpropClassifierNetwork, built
    unconditionally alongside its numpy counterpart (see RustArrayBackpropClassifierNetwork's
    own docstring). A direct structural mirror,
    substituting RustArrayBackpropClassifierNetwork throughout - same save()/load() shape (bare
    save_json/load_json, not save_array_model_json) for the same nested-snapshot reason
    EnsembleArrayBackpropClassifierNetwork's own docstring gives.
    """

    def __init__(self, classifiers: list[RustArrayBackpropClassifierNetwork]) -> None:
        assert len(classifiers) >= 2, f"an ensemble needs at least 2 classifiers; got {len(classifiers)}"
        self.classifiers = classifiers
        self.class_count = len(classifiers)

    def predict_probabilities(self, state: tuple[float, ...]) -> list[float]:
        return [classifier.predict_probability(state) for classifier in self.classifiers]

    def classify_state(self, state: tuple[float, ...]) -> int:
        return argmax_first_occurrence(self.predict_probabilities(state))

    def snapshot(self) -> list[list[tuple["pa.Array", "pa.Array"]]]:
        return [classifier.snapshot() for classifier in self.classifiers]

    def restore(self, snapshot: list[list[tuple["pa.Array", "pa.Array"]]]) -> None:
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
    def load(cls, path: str) -> "EnsembleRustArrayBackpropClassifierNetwork":
        state = load_json(path)

        classifiers = [
            RustArrayBackpropClassifierNetwork(state["layer_sizes"], state["dimension"])
            for _ in range(state["class_count"])
        ]
        ensemble = cls(classifiers)
        ensemble.restore(
            [
                [(pa.Array(W), pa.Array(b)) for W, b in classifier_snapshot]
                for classifier_snapshot in state["snapshot"]
            ]
        )
        return ensemble
