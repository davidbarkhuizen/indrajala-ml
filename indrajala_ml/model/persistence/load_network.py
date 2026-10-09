"""
load_network(path): whatever a format-2 file (format2.py) describes, built from its specs and rule,
never from a class named in it (the composable-layers workplan, Format 2). A network is the
Sequential network of the file's implementation and shape, bit-identical to the preset that may have
saved it. An ensemble is its implementation's ensemble class, of such sub-networks.

A module of its own, beside the networks: it imports all three implementations, which the
pure-Python networks mustn't.
"""

from __future__ import annotations

from typing import Any, Literal

from indrajala_ml.model.ensembles.ensemble_array_backprop_classifier_network import (
    EnsembleArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.ensembles.ensemble_backprop_classifier_network import EnsembleBackpropClassifierNetwork
from indrajala_ml.model.ensembles.ensemble_rust_array_backprop_classifier_network import (
    EnsembleRustArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.networks.python.sequential_backprop_network import (
    SequentialBackpropClassifierNetwork,
    SequentialMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.persistence.format2 import (
    ENSEMBLE,
    PYTHON,
    NetworkFile,
    ensemble_classifiers,
    is_format2,
    network_from_json,
    restore_file,
)
from indrajala_ml.model.persistence.model_io import load_json

_ENSEMBLES: dict[str, Any] = {
    PYTHON: EnsembleBackpropClassifierNetwork,
    NUMPY.name: EnsembleArrayBackpropClassifierNetwork,
    RUST.name: EnsembleRustArrayBackpropClassifierNetwork,
}


_ARRAY_SHAPES: dict[str, Literal["multiclass", "single_output", "sequence"]] = {
    "multiclass": "multiclass",
    "single_output": "single_output",
    "sequence": "sequence",
}


def load_network(path: str) -> Any:
    """The network or ensemble a format-2 file describes, with its weights and optimizer state."""
    state = load_json(path)
    if not is_format2(state):
        raise ValueError(f"{path} isn't a format-2 file; a legacy file loads with its class's load()")
    if state["shape"] == ENSEMBLE:
        classifiers = [_sequential(network_from_json(classifier)) for classifier in ensemble_classifiers(state)]
        return _ENSEMBLES[state["implementation"]](classifiers)
    return _sequential(network_from_json(state))


def _sequential(file: NetworkFile) -> Any:
    network: Any
    if file.implementation == PYTHON and file.shape == "sequence":
        raise NotImplementedError("a sequence network in pure Python: not yet (the sequence task workplan, stage 6)")
    if file.implementation == PYTHON:
        cls = (
            SequentialMultiClassBackpropClassifierNetwork
            if file.shape == "multiclass"
            else SequentialBackpropClassifierNetwork
        )
        network = cls(file.input_shape, file.layers, file.update_rule, file.input_bounds)
    else:
        backend = NUMPY if file.implementation == NUMPY.name else RUST
        network = SequentialArrayNetwork(
            file.input_shape, file.layers, file.update_rule, shape=_ARRAY_SHAPES[file.shape], backend=backend
        )
    restore_file(network, file)
    return network
