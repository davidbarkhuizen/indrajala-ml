from __future__ import annotations

from indrajala_ml.model.array_backprop_classifier_network import ArrayBackpropClassifierNetwork
from indrajala_ml.model.array_ensemble_base import ArrayEnsembleBase


class EnsembleArrayBackpropClassifierNetwork(ArrayEnsembleBase):
    """
    ArrayEnsembleBase over ArrayBackpropClassifierNetwork: the numpy form of
    EnsembleBackpropClassifierNetwork.
    """

    classifier_cls = ArrayBackpropClassifierNetwork
