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
from indrajala_ml.model.networks.numpy.window_recurrent_array_network import WindowRecurrentArrayNetwork
from indrajala_ml.model.networks.python.sequential_backprop_network import (
    SequentialBackpropClassifierNetwork,
    SequentialMultiClassBackpropClassifierNetwork,
    SequentialSequenceBackpropNetwork,
)
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.persistence.format2 import (
    ENSEMBLE,
    PYTHON,
    WINDOW_RECURRENT,
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


_PYTHON_SHAPES: dict[str, Any] = {
    "multiclass": SequentialMultiClassBackpropClassifierNetwork,
    "single_output": SequentialBackpropClassifierNetwork,
    "sequence": SequentialSequenceBackpropNetwork,
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
    if file.shape == WINDOW_RECURRENT:
        if file.implementation != NUMPY.name:
            raise ValueError(
                f"a window-recurrent network isn't built in {file.implementation} yet (the window-recurrent adder "
                "workplan: Rust at stage 5, pure Python at stage 6)"
            )
        assert file.recurrence is not None
        network = WindowRecurrentArrayNetwork(file.input_shape, file.layers, file.recurrence, file.update_rule)
        restore_file(network, file)
        return network
    if file.implementation == PYTHON:
        cls = _PYTHON_SHAPES[file.shape]
        network = cls(file.input_shape, file.layers, file.update_rule, file.input_bounds)
    else:
        backend = NUMPY if file.implementation == NUMPY.name else RUST
        network = SequentialArrayNetwork(
            file.input_shape, file.layers, file.update_rule, shape=_ARRAY_SHAPES[file.shape], backend=backend
        )
    restore_file(network, file)
    return network
