from __future__ import annotations

from indrajala_ml.model.array_ensemble_base import ArrayEnsembleBase
from indrajala_ml.model.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork


class EnsembleRustArrayBackpropClassifierNetwork(ArrayEnsembleBase[RustArrayBackpropClassifierNetwork]):
    """
    ArrayEnsembleBase over RustArrayBackpropClassifierNetwork.
    """

    classifier_cls = RustArrayBackpropClassifierNetwork
