from __future__ import annotations

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.classification import argmax_first_occurrence
from indrajala_ml.model.model_io import load_model_json, save_model_json


class EnsembleBackpropClassifierNetwork:
    """
    A multiclass classifier made of class_count independent BackpropClassifierNetworks, one per
    class, each trained on its own "is this class C?" problem with no shared state. Unlike
    MultiClassBackpropClassifierNetwork (one shared hidden layer, jointly trained), nothing needs
    synchronizing, so the sub-networks train in parallel processes (ensemble_train.py).

    __init__ takes already-built classifiers, not layer sizes: an ensemble is assembled from
    separately trained classifiers or rebuilt from a saved one, never trained as a whole.
    """

    def __init__(self, classifiers: list[BackpropClassifierNetwork]) -> None:
        assert len(classifiers) >= 2, f"an ensemble needs at least 2 classifiers; got {len(classifiers)}"
        self.classifiers = classifiers
        self.class_count = len(classifiers)

    def predict_probabilities(self, state: tuple[float, ...]) -> list[float]:
        return [classifier.predict_probability(state) for classifier in self.classifiers]

    def classify_state(self, state: tuple[float, ...]) -> int:
        return argmax_first_occurrence(self.predict_probabilities(state))

    def snapshot(self) -> list[list[list[tuple[list[float], float]]]]:
        return [classifier.snapshot() for classifier in self.classifiers]

    def restore(self, snapshot: list[list[list[tuple[list[float], float]]]]) -> None:
        for classifier, classifier_snapshot in zip(self.classifiers, snapshot):
            classifier.restore(classifier_snapshot)

    def save(self, path: str) -> None:
        assert len({classifier.dimension for classifier in self.classifiers}) == 1, "every classifier must share a dimension"
        first = self.classifiers[0]
        save_model_json(
            path,
            layer_sizes=[layer.size for layer in first.hidden_layers],
            dimension=first.dimension,
            input_bounds=first.input_bounds,
            class_count=self.class_count,
            snapshot=self.snapshot(),
        )

    @classmethod
    def load(cls, path: str) -> "EnsembleBackpropClassifierNetwork":
        state = load_model_json(path)

        classifiers = [
            BackpropClassifierNetwork(state["layer_sizes"], state["dimension"], state["input_bounds"])
            for _ in range(state["class_count"])
        ]
        ensemble = cls(classifiers)
        ensemble.restore(state["snapshot"])
        return ensemble
