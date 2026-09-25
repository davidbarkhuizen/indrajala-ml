from __future__ import annotations

from indrajala_ml.model.array_ensemble_base import ArrayEnsembleBase
from indrajala_ml.model.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork


class EnsembleRustArrayBackpropClassifierNetwork(ArrayEnsembleBase):
    """
    The Rust-array-core-backed sibling of EnsembleArrayBackpropClassifierNetwork:
    ArrayEnsembleBase over RustArrayBackpropClassifierNetwork, built unconditionally alongside its
    numpy counterpart (see RustArrayBackpropClassifierNetwork's docstring).
    """

    classifier_cls = RustArrayBackpropClassifierNetwork
