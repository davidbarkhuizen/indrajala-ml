from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Generic, TypeVar, cast

from typing_extensions import Self

from indrajala_ml.model.array_network_shapes import ArraySingleOutputShape
from indrajala_ml.model.classification import argmax_first_occurrence
from indrajala_ml.model.model_io import load_json, save_json

# the sub-networks' class: a single-output array network on either backend
ClassifierT = TypeVar("ClassifierT", bound=ArraySingleOutputShape[Any])


class ArrayEnsembleBase(Generic[ClassifierT]):
    """
    An ensemble of single-output array networks, for either backend, assembled from already-built
    classifiers as EnsembleBackpropClassifierNetwork is. EnsembleArrayBackpropClassifierNetwork and
    EnsembleRustArrayBackpropClassifierNetwork differ only in classifier_cls, the class load()
    builds.

    class_count is the number of sub-networks. save()/load() use save_json/load_json directly: the
    snapshot nests one network's (W, b) list per classifier, which save_array_model_json's flat
    layout can't hold.
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

    def save(self, path: str) -> None:
        assert len({classifier.dimension for classifier in self.classifiers}) == 1, (
            "every classifier must share a dimension"
        )
        assert len({tuple(classifier.layer_sizes) for classifier in self.classifiers}) == 1, (
            "every classifier must share layer_sizes"
        )
        first = self.classifiers[0]
        save_json(
            path,
            {
                "layer_sizes": first.layer_sizes,
                "dimension": first.dimension,
                "class_count": self.class_count,
                "snapshot": [
                    [(W.tolist(), b.tolist()) for W, b in classifier_snapshot]
                    for classifier_snapshot in self.snapshot()
                ],
            },
        )

    @classmethod
    def load(cls, path: str) -> Self:
        state = load_json(path)

        classifiers = [
            cls.classifier_cls(state["layer_sizes"], state["dimension"]) for _ in range(state["class_count"])
        ]
        # every subclass sets classifier_cls to its ClassifierT
        ensemble = cls(cast("list[ClassifierT]", classifiers))
        # each classifier's restore converts the file's nested lists through its backend
        ensemble.restore(state["snapshot"])
        return ensemble
