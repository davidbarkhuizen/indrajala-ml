from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Self, cast

from indrajala_ml.model.array_network_shapes import ArraySingleOutputShape
from indrajala_ml.model.checkpoint import Checkpoint
from indrajala_ml.model.classification import argmax_first_occurrence
from indrajala_ml.model.format2 import ensemble_classifiers, ensemble_to_json, is_format2
from indrajala_ml.model.model_io import load_json, save_json


class ArrayEnsembleBase[ClassifierT: ArraySingleOutputShape[Any]]:
    """
    An ensemble of single-output array networks, for either backend, assembled from already-built
    classifiers as EnsembleBackpropClassifierNetwork is. EnsembleArrayBackpropClassifierNetwork and
    EnsembleRustArrayBackpropClassifierNetwork differ only in classifier_cls, the class load()
    builds.

    class_count is the number of sub-networks. save() writes format 2 (format2.py), one format-2
    file per sub-network nested in the ensemble's. load() reads it or the legacy envelope, whose
    snapshot nests one network's (W, b) list per classifier.
    """

    # the single-output network load() builds, one per class; ClassifierT's class (a ClassVar
    # can't name a type variable)
    classifier_cls: ClassVar[type[ArraySingleOutputShape[Any]]]

    def __init__(self, classifiers: list[ClassifierT]) -> None:
        assert len(classifiers) >= 2, f"an ensemble needs at least 2 classifiers; got {len(classifiers)}"
        self.classifiers = classifiers
        self.class_count = len(classifiers)

    def predict_probabilities(self, state: tuple[float, ...]) -> list[float]:
        return [classifier.predict_probability(state) for classifier in self.classifiers]

    def classify_state(self, state: tuple[float, ...]) -> int:
        return argmax_first_occurrence(self.predict_probabilities(state))

    def snapshot(self) -> list[list[tuple[Any, ...]]]:
        return [classifier.snapshot() for classifier in self.classifiers]

    def restore(self, snapshot: Sequence[Sequence[tuple[Any, ...]]]) -> None:
        for classifier, classifier_snapshot in zip(self.classifiers, snapshot):
            classifier.restore(classifier_snapshot)

    def checkpoint(self) -> list[Checkpoint[Any, Any]]:
        return [classifier.checkpoint() for classifier in self.classifiers]

    def restore_checkpoint(self, checkpoint: Sequence[Checkpoint[Any, Any]]) -> None:
        for classifier, classifier_checkpoint in zip(self.classifiers, checkpoint):
            classifier.restore_checkpoint(classifier_checkpoint)

    def save(self, path: str) -> None:
        save_json(path, ensemble_to_json(self.classifiers[0].implementation, self.classifiers))

    @classmethod
    def load(cls, path: str) -> Self:
        state = load_json(path)
        if is_format2(state):
            # each sub-network as classifier_cls, which refuses one whose specs or rule aren't its
            # own; numpy and Rust files load into either
            classifiers = [cls.classifier_cls.from_format2(classifier) for classifier in ensemble_classifiers(state)]
            return cls(cast("list[ClassifierT]", classifiers))

        # the legacy envelope, whose classifiers load with fresh optimizer state
        classifiers = [
            cls.classifier_cls(state["layer_sizes"], state["dimension"]) for _ in range(state["class_count"])
        ]
        # every subclass sets classifier_cls to its ClassifierT
        ensemble = cls(cast("list[ClassifierT]", classifiers))
        # each classifier's restore converts the file's nested lists through its backend
        ensemble.restore(state["snapshot"])
        return ensemble
