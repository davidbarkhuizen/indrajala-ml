from __future__ import annotations

from indrajala_ml.model.ensembles.array_ensemble_base import ArrayEnsembleBase
from indrajala_ml.model.networks.numpy.array_backprop_classifier_network import ArrayBackpropClassifierNetwork


class EnsembleArrayBackpropClassifierNetwork(ArrayEnsembleBase[ArrayBackpropClassifierNetwork]):
    """
    ArrayEnsembleBase over ArrayBackpropClassifierNetwork: the numpy form of
    EnsembleBackpropClassifierNetwork.
    """

    classifier_cls = ArrayBackpropClassifierNetwork
