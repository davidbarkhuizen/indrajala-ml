"""
The ONNX exporter's core (model/export/onnx/, the export-and-hub workplan, stage 3): every numpy
and Rust fixture of a flat input, and dense networks of every activation, batch norm, residual
blocks and a linear output, export in float32 and float64 and give our numpy forward's outputs
under both evaluators (D13); a file, a directory and the live network export alike; the erf
decomposition is fdlibm's within an ulp; the model is opset 23 at IR 10 (D11); and what a later
stage exports is refused naming its stage.
"""

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime  # pyright: ignore[reportMissingTypeStubs]
import pytest
from onnx.reference import ReferenceEvaluator

from indrajala_ml.model.export.onnx.check import TOLERANCE, ExportCheckError, check_export, inputs, numpy_forward
from indrajala_ml.model.export.onnx.exporter import document, export_onnx
from indrajala_ml.model.export.onnx.functions import erf
from indrajala_ml.model.export.onnx.graph import IR_VERSION, OPSET, Graph
from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.layers.numpy import erf as numpy_erf
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.persistence.format3 import save_directory
from indrajala_ml.model.specs.layer_specs import BatchNorm, Dense, LayerSpec, Residual
from indrajala_ml.model.specs.update_rules import Adam
from tests.model.persistence.test_format2 import ENSEMBLES, _trained
from tests.saved_model_fixtures import FIXTURES

DTYPES = ["float32", "float64"]


def _flat(name: str) -> bool:
    network = FIXTURES[name].build()
    first = network.classifiers[0] if name in ENSEMBLES else network
    return (
        first.implementation != "python"
        and len(first.input_shape) == 1
        and first.format2_shape
        in (
            "multiclass",
            "single_output",
        )
    )


FLAT = sorted(name for name in FIXTURES if _flat(name))


def _network(name: str) -> Any:
    if name in ENSEMBLES:
        ensemble = FIXTURES[name].build()
        for classifier in ensemble.classifiers:
            classifier.randomize()
        return ensemble
    return _trained(name)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", FLAT)
def test_a_flat_fixture_exports_as_its_forward(name: str, dtype: str):
    export = export_onnx(_network(name), dtype=dtype)  # pyright: ignore[reportArgumentType]

    assert export.report is not None
    assert set(export.report.max_difference) == {"reference", "onnxruntime"}
    assert max(export.report.max_difference.values()) <= TOLERANCE[export.dtype]


# dense networks of every activation (D12), on both backends

ARCHITECTURES: dict[str, list[LayerSpec]] = {
    "sigmoid, softmax": [Dense(6), Dense(3, output=True, activation="softmax", loss="cross_entropy")],
    "relu, sigmoid cross-entropy": [Dense(6, activation="relu"), Dense(3, output=True, loss="cross_entropy")],
    "gelu": [Dense(6, activation="gelu"), Dense(3, output=True, activation="softmax", loss="cross_entropy")],
    "gelu_tanh": [Dense(6, activation="gelu_tanh"), Dense(3, output=True, loss="squared")],
    "dropout": [Dense(6, dropout=0.5), Dense(3, output=True, loss="squared")],
    "batch norm": [
        Dense(6, activation="linear"),
        BatchNorm(),
        Dense(6, activation="linear"),
        BatchNorm("relu", epsilon=1e-3),
        Dense(3, output=True, activation="softmax", loss="cross_entropy"),
    ],
    "residual": [
        Dense(6, activation="relu"),
        Residual((Dense(8, activation="gelu"), Dense(6, activation="linear", bias=True))),
        Dense(3, output=True, activation="softmax", loss="cross_entropy"),
    ],
    "linear output": [Dense(6, activation="relu"), Dense(2, output=True, activation="linear", loss="given")],
}


def _architecture(name: str, backend: Any) -> Any:
    layers = ARCHITECTURES[name]
    output = layers[-1]
    assert isinstance(output, Dense)
    shape = "single_output" if output.size == 1 else "multiclass"
    network = SequentialArrayNetwork((5,), layers, Adam(), shape=shape, backend=backend)
    network.rng = backend.default_rng(3)
    network.randomize()
    if any(isinstance(spec, BatchNorm) for spec in layers):
        # running averages off their starting point, so the check reads them
        for layer in network.layers:
            if hasattr(layer, "running_var"):
                layer.running_mean = backend.owned(np.linspace(-0.5, 0.5, layer.size))
                layer.running_var = backend.owned(np.linspace(0.5, 2.0, layer.size))
    return network


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", [NUMPY, RUST], ids=["numpy", "rust"])
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_every_dense_architecture_exports_as_its_forward(architecture: str, backend: Any, dtype: str):
    export = export_onnx(_architecture(architecture, backend), dtype=dtype)  # pyright: ignore[reportArgumentType]

    assert export.report is not None
    assert max(export.report.max_difference.values()) <= TOLERANCE[export.dtype]


def test_a_linear_output_is_named_outputs():
    model = export_onnx(_architecture("linear output", NUMPY)).model
    assert [output.name for output in model.graph.output] == ["outputs"]
    assert [i.name for i in model.graph.input] == ["x"]


def test_the_model_is_opset_23_at_ir_10_with_format_3s_names(tmp_path: Path):
    path = tmp_path / "model.onnx"
    export_onnx(_architecture("batch norm", NUMPY), path, dtype="float64")

    model: Any = onnx.load(str(path))  # pyright: ignore[reportUnknownMemberType]
    assert model.ir_version == IR_VERSION
    assert [(o.domain, o.version) for o in model.opset_import] == [("", OPSET)]
    assert {i.name for i in model.graph.initializer} >= {
        "layers.0.W",
        "layers.1.gamma",
        "layers.1.running_var",
        "layers.4.b",
    }
    assert {p.key: p.value for p in model.metadata_props}["dtype"] == "float64"


def test_a_file_a_directory_and_the_network_export_alike(tmp_path: Path):
    network = _architecture("residual", RUST)
    network.save(str(tmp_path / "model.json"))
    save_directory(network, tmp_path / "model")

    models = [
        export_onnx(source, dtype="float64").model.SerializeToString()
        for source in (network, str(tmp_path / "model.json"), tmp_path / "model")
    ]

    assert models[0] == models[1] == models[2]


def test_the_check_refuses_an_export_that_differs():
    network = _architecture("sigmoid, softmax", NUMPY)
    model = export_onnx(network, dtype="float64").model
    state = document(network)
    state["weights"][0][1] = state["weights"][0][1] + 1e-6

    with pytest.raises(ExportCheckError, match="differ from our numpy forward"):
        check_export(model, state, "float64")


def test_the_reference_forward_is_the_networks_own():
    network = _architecture("gelu", RUST)
    x = inputs(5)
    expected = np.array(network.forward_batch(network.backend.matrix(x.tolist())).tolist())
    assert np.allclose(numpy_forward(document(network), x), expected, rtol=0, atol=1e-12)


# erf (D10: no float64 Erf in onnxruntime, so fdlibm's ranges)


@pytest.mark.parametrize("dtype", DTYPES)
def test_erf_is_fdlibms_within_an_ulp(dtype: str):
    x = np.concatenate([np.linspace(-7.0, 7.0, 4001), [0.0, -0.0, 1e-30, 0.84375, 1.25, 1 / 0.35, 6.0, -6.0, 50.0]])
    g = Graph(dtype)  # pyright: ignore[reportArgumentType]
    g.rename(erf(g, "x"), "y")
    model = g.model([g.value_info("x", [len(x)])], [g.value_info("y", [len(x)])], "erf", {})
    feed = {"x": x.astype(dtype)}
    expected = numpy_erf.erf(feed["x"].astype(np.float64))
    session: Any = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    reference: Any = ReferenceEvaluator(model)

    for actual in (reference.run(None, feed)[0], session.run(None, feed)[0]):
        tolerance = 2.3e-16 if dtype == "float64" else 1.2e-7
        np.testing.assert_allclose(np.asarray(actual, dtype=np.float64), expected, rtol=0, atol=tolerance)


# refusals


def test_a_pure_python_network_is_refused(tmp_path: Path):
    network = FIXTURES["SequentialBackpropClassifierNetwork"].build()
    network.save(str(tmp_path / "model.json"))
    with pytest.raises(ValueError, match="pure-Python"):
        export_onnx(str(tmp_path / "model.json"))


def test_the_dtype_is_float32_or_float64():
    with pytest.raises(ValueError, match="float32 or float64"):
        export_onnx(_architecture("gelu", NUMPY), dtype="float16")  # pyright: ignore[reportArgumentType]
