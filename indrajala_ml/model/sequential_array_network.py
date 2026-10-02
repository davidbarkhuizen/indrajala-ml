"""
Array networks of any accepted layer specs and update rule (the composable-layers workplan, The
design): SequentialArrayNetwork builds one for a shape and backend, as one of the four classes
below, which the registry walks cover as they do the presets.

    network = SequentialArrayNetwork(
        (28, 28, 1),
        [Conv(5, 8), Pool(2), Dense(30), Dense(10, output=True)],
        Momentum(0.9),
        shape="multiclass",
        backend=RUST,
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import indrajala_math_rust as pa

from indrajala_ml.model.array_backend import NUMPY, RUST
from indrajala_ml.model.array_backprop_classifier_network import ArrayBackpropClassifierNetwork
from indrajala_ml.model.array_layer import FloatArray
from indrajala_ml.model.protocols.array_protocols import ArrayBackend
from indrajala_ml.model.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork
from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.array_network_shapes import SequentialMultiClassShape, SequentialSingleOutputShape
from indrajala_ml.model.specs.layer_specs import LayerSpec
from indrajala_ml.model.specs.spec_shapes import InputShape
from indrajala_ml.model.specs.update_rules import UpdateRule
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)


class SequentialVectorizedMultiClassBackpropClassifierNetwork(
    SequentialMultiClassShape[FloatArray], VectorizedMultiClassBackpropClassifierNetwork
):
    """SequentialMultiClassShape on the numpy backend."""


class SequentialRustArrayMultiClassBackpropClassifierNetwork(
    SequentialMultiClassShape[pa.Array], RustArrayMultiClassBackpropClassifierNetwork
):
    """SequentialMultiClassShape on the Rust backend."""


class SequentialArrayBackpropClassifierNetwork(SequentialSingleOutputShape[FloatArray], ArrayBackpropClassifierNetwork):
    """SequentialSingleOutputShape on the numpy backend."""


class SequentialRustArrayBackpropClassifierNetwork(
    SequentialSingleOutputShape[pa.Array], RustArrayBackpropClassifierNetwork
):
    """SequentialSingleOutputShape on the Rust backend."""


def SequentialArrayNetwork(
    input_shape: InputShape,
    layers: Sequence[LayerSpec],
    update_rule: UpdateRule,
    shape: Literal["multiclass", "single_output"] = "multiclass",
    backend: ArrayBackend[Any] = NUMPY,
) -> Any:
    """The sequential network class for shape and backend, built from the rest of the arguments."""
    assert backend.name in (NUMPY.name, RUST.name), f"unknown backend {backend.name!r}"
    rust = backend.name == RUST.name
    if shape == "multiclass":
        cls: Any = (
            SequentialRustArrayMultiClassBackpropClassifierNetwork
            if rust
            else SequentialVectorizedMultiClassBackpropClassifierNetwork
        )
    else:
        assert shape == "single_output", f"shape is 'multiclass' or 'single_output'; got {shape!r}"
        cls = SequentialRustArrayBackpropClassifierNetwork if rust else SequentialArrayBackpropClassifierNetwork
    return cls(input_shape, layers, update_rule)
