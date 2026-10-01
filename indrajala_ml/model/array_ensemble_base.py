from __future__ import annotations

from typing import Any, ClassVar, Self, cast

from indrajala_ml.model.array_network_shapes import ArraySingleOutputShape
from indrajala_ml.model.ensemble_base import EnsembleBase
from indrajala_ml.model.format2 import ensemble_classifiers, ensemble_to_json, is_format2
from indrajala_ml.model.model_io import load_json, save_json


class ArrayEnsembleBase[ClassifierT: ArraySingleOutputShape[Any]](EnsembleBase[ClassifierT]):
    """
    An ensemble of single-output array networks, for either backend, assembled from already-built
    classifiers as EnsembleBackpropClassifierNetwork is; both share EnsembleBase.
    EnsembleArrayBackpropClassifierNetwork and EnsembleRustArrayBackpropClassifierNetwork differ
    only in classifier_cls, the class load() builds.

    class_count is the number of sub-networks. save() writes format 2 (format2.py), one format-2
    file per sub-network nested in the ensemble's. load() reads it or the legacy envelope, whose
    snapshot nests one network's (W, b) list per classifier.
    """

    # the single-output network load() builds, one per class; ClassifierT's class (a ClassVar
    # can't name a type variable)
    classifier_cls: ClassVar[type[ArraySingleOutputShape[Any]]]

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
