from __future__ import annotations

from indrajala_ml.model.ensembles.array_ensemble_base import ArrayEnsembleBase
from indrajala_ml.model.networks.rust.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork


class EnsembleRustArrayBackpropClassifierNetwork(ArrayEnsembleBase[RustArrayBackpropClassifierNetwork]):
    """
    ArrayEnsembleBase over RustArrayBackpropClassifierNetwork.
    """

    classifier_cls = RustArrayBackpropClassifierNetwork
