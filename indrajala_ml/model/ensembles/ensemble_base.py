"""
What every ensemble shares, whatever its sub-networks: building from already-built classifiers,
classifying by the most probable class, and gathering each sub-network's snapshot and checkpoint.
EnsembleBackpropClassifierNetwork (pure Python) and ArrayEnsembleBase (numpy and Rust) add their
own save and load.

Backend-free, so the pure-Python ensemble imports it too.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from indrajala_ml.model.protocols.classification import argmax_first_occurrence
from indrajala_ml.model.protocols.classifier_protocols import BinaryClassifier


class EnsembleBase[ClassifierT: BinaryClassifier]:
    """
    class_count sub-networks, one per class, each answering "is this class C?". The ensemble
    classifies a state as the class whose sub-network gives it the highest probability, the
    first such class on a tie.
    """

    def __init__(self, classifiers: list[ClassifierT]) -> None:
        assert len(classifiers) >= 2, f"an ensemble needs at least 2 classifiers; got {len(classifiers)}"
        self.classifiers = classifiers
        self.class_count = len(classifiers)

    def predict_probabilities(self, state: tuple[float, ...]) -> list[float]:
        return [classifier.predict_probability(state) for classifier in self.classifiers]

    def classify_state(self, state: tuple[float, ...]) -> int:
        return argmax_first_occurrence(self.predict_probabilities(state))

    # each sub-network's own snapshot and checkpoint, in class order
    def snapshot(self) -> list[Any]:
        return [classifier.snapshot() for classifier in self.classifiers]

    def restore(self, snapshot: Sequence[Any]) -> None:
        for classifier, classifier_snapshot in zip(self.classifiers, snapshot):
            classifier.restore(classifier_snapshot)

    def checkpoint(self) -> list[Any]:
        return [classifier.checkpoint() for classifier in self.classifiers]

    def restore_checkpoint(self, checkpoint: Sequence[Any]) -> None:
        for classifier, classifier_checkpoint in zip(self.classifiers, checkpoint):
            classifier.restore_checkpoint(classifier_checkpoint)
