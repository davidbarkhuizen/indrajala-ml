from __future__ import annotations

from typing import Any

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.classification import argmax_first_occurrence
from indrajala_ml.model.classifier_protocols import BinaryClassifier
from indrajala_ml.model.format2 import PYTHON, ensemble_classifiers, ensemble_to_json, is_format2
from indrajala_ml.model.model_io import load_json, save_json


# ClassifierT, the sub-networks' class: any single-output network, BackpropClassifierNetwork unless
# the trainer is given another classifier_cls (ensemble_train.py)
class EnsembleBackpropClassifierNetwork[ClassifierT: BinaryClassifier = BackpropClassifierNetwork]:
    """
    A multiclass classifier made of class_count independent BackpropClassifierNetworks, one per
    class, each trained on its own "is this class C?" problem with no shared state. Unlike
    MultiClassBackpropClassifierNetwork (one shared hidden layer, jointly trained), nothing needs
    synchronizing, so the sub-networks train in parallel processes (ensemble_train.py).

    __init__ takes already-built classifiers, not layer sizes: an ensemble is assembled from
    separately trained classifiers or rebuilt from a saved one, never trained as a whole.
    """

    def __init__(self, classifiers: list[ClassifierT]) -> None:
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

    def checkpoint(self) -> list[Any]:
        return [classifier.checkpoint() for classifier in self.classifiers]

    def restore_checkpoint(self, checkpoint: list[Any]) -> None:
        for classifier, classifier_checkpoint in zip(self.classifiers, checkpoint):
            classifier.restore_checkpoint(classifier_checkpoint)

    def save(self, path: str) -> None:
        # format 2 (format2.py), one pure-Python file per sub-network;
        # EnsembleArrayBackpropClassifierNetwork saves array classifiers
        classifiers: list[BackpropClassifierNetwork[Any]] = [
            c for c in self.classifiers if isinstance(c, BackpropClassifierNetwork)
        ]
        assert len(classifiers) == len(self.classifiers), "save needs BackpropClassifierNetwork classifiers"
        save_json(path, ensemble_to_json(PYTHON, list(classifiers)))

    @classmethod
    def load(
        cls: type[EnsembleBackpropClassifierNetwork[BackpropClassifierNetwork]], path: str
    ) -> EnsembleBackpropClassifierNetwork[BackpropClassifierNetwork]:
        # pure-Python sub-networks, as BackpropClassifierNetwork: from format 2, which refuses a
        # sub-network whose specs or rule aren't its own, or from the legacy envelope, with fresh
        # optimizer state
        state = load_json(path)
        if is_format2(state):
            return cls(
                [BackpropClassifierNetwork.from_format2(classifier) for classifier in ensemble_classifiers(state)]
            )

        input_bounds = [tuple(bound) for bound in state["input_bounds"]]
        classifiers = [
            BackpropClassifierNetwork(state["layer_sizes"], state["dimension"], input_bounds)
            for _ in range(state["class_count"])
        ]
        ensemble = cls(classifiers)
        ensemble.restore(state["snapshot"])
        return ensemble
