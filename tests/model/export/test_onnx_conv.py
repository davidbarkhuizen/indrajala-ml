"""
The ONNX exporter's images (model/export/onnx/layers.py, the export-and-hub workplan, stage 4):
every numpy and Rust fixture over an image whose kinds stage 4 covers, and conv networks of every
geometry (ReLU and linear conv, strides, several channels, batch norm per channel, max pool,
patches and the token mean) export in float32 and float64 and give our numpy forward's outputs
under both evaluators (D13). The input is NCHW, our row's own channel-major order, and conv is
im2col and a MatMul (D10).
"""

from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.export.onnx.check import TOLERANCE
from indrajala_ml.model.export.onnx.exporter import export_onnx
from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    BatchNorm,
    Conv,
    Dense,
    Embedding,
    GatedDense,
    LayerNorm,
    LayerSpec,
    Patches,
    Pool,
    Position,
    Residual,
    TokenMean,
    expand_specs,
)
from indrajala_ml.model.specs.spec_shapes import InputShape
from indrajala_ml.model.specs.update_rules import Adam
from tests.model.persistence.test_format2 import ENSEMBLES, _trained
from tests.saved_model_fixtures import FIXTURES

DTYPES = ["float32", "float64"]
# the kinds stage 5 adds
_LATER = (Attention, LayerNorm, Position, Embedding, GatedDense)


def _image(name: str) -> bool:
    if name in ENSEMBLES:
        return False
    network = FIXTURES[name].build()
    return (
        network.implementation != "python"
        and len(network.input_shape) == 3
        and not any(isinstance(spec, _LATER) for spec in expand_specs(network.layer_specs))
    )


IMAGES = sorted(name for name in FIXTURES if _image(name))


def test_there_are_image_fixtures():
    assert any("Conv" in name for name in IMAGES)
    assert any(name.startswith("BatchNorm") for name in IMAGES)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", IMAGES)
def test_an_image_fixture_exports_as_its_forward(name: str, dtype: str):
    export = export_onnx(_trained(name), dtype=dtype)  # pyright: ignore[reportArgumentType]

    assert export.report is not None
    assert max(export.report.max_difference.values()) <= TOLERANCE[export.dtype]


ARCHITECTURES: dict[str, tuple[InputShape, list[LayerSpec]]] = {
    "relu conv, pool": ((8, 8, 1), [Conv(3, 3), Pool(2), Dense(3, output=True, loss="squared")]),
    "three channels, stride 2": (
        (9, 9, 3),
        [
            Conv(4, 3, stride=2),
            Dense(5, activation="relu"),
            Dense(3, output=True, activation="softmax", loss="cross_entropy"),
        ],
    ),
    "linear conv, batch norm per channel": (
        (8, 8, 2),
        [Conv(3, 3, activation="linear"), BatchNorm("relu"), Pool(2, stride=1), Dense(3, output=True, loss="squared")],
    ),
    "two convs, a pool of stride 2 over 3": (
        (10, 10, 2),
        [Conv(3, 3), Conv(4, 2, stride=1), Pool(3, stride=2), Dense(2, output=True, loss="cross_entropy")],
    ),
    "a 1x1 kernel": ((5, 5, 3), [Conv(2, 1), Dense(3, output=True, loss="squared")]),
    "patches, token mean": (
        (8, 8, 3),
        [
            Patches(4),
            TokenMean(),
            Dense(4, activation="gelu"),
            Dense(3, output=True, activation="softmax", loss="cross_entropy"),
        ],
    ),
    "a residual block over a flattened conv": (
        (6, 6, 1),
        [
            Conv(2, 3),
            Dense(5, activation="relu"),
            Residual((Dense(6, activation="relu"), Dense(5, activation="linear", bias=True))),
            Dense(3, output=True, activation="softmax", loss="cross_entropy"),
        ],
    ),
}


def _architecture(name: str, backend: Any) -> Any:
    input_shape, layers = ARCHITECTURES[name]
    network = SequentialArrayNetwork(input_shape, layers, Adam(), backend=backend)
    network.rng = backend.default_rng(5)
    network.randomize()
    for layer in network.layers:
        if hasattr(layer, "running_var"):
            layer.running_mean = backend.owned(np.linspace(-0.5, 0.5, len(layer.running_mean.tolist())))
            layer.running_var = backend.owned(np.linspace(0.5, 2.0, len(layer.running_var.tolist())))
    return network


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", [NUMPY, RUST], ids=["numpy", "rust"])
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_every_image_architecture_exports_as_its_forward(architecture: str, backend: Any, dtype: str):
    export = export_onnx(_architecture(architecture, backend), dtype=dtype)  # pyright: ignore[reportArgumentType]

    assert export.report is not None
    assert max(export.report.max_difference.values()) <= TOLERANCE[export.dtype]


def test_an_image_input_is_nchw():
    model = export_onnx(_architecture("three channels, stride 2", NUMPY)).model
    dims = [dim.dim_value or dim.dim_param for dim in model.graph.input[0].type.tensor_type.shape.dim]
    assert dims == ["batch", 3, 9, 9]


def test_a_float64_conv_runs_without_a_conv_op():
    # onnxruntime has no float64 Conv (D10): the graph is im2col and a MatMul in both dtypes
    model = export_onnx(_architecture("relu conv, pool", NUMPY), dtype="float64").model
    ops = {node.op_type for node in model.graph.node}
    assert "Conv" not in ops
    assert {"Slice", "MatMul", "MaxPool"} <= ops
