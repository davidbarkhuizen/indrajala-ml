"""
The ONNX exporter's sequence layers (model/export/onnx/sequence.py, the export-and-hub workplan,
stage 5): every numpy and Rust sequence and attention fixture, and sequence models of every
attention option (heads, grouped heads, causal, rotary, segments, document_mask off, separator
opens), GELU and SwiGLU FFNs and a token-mean classifier, export in float32 and float64 and give our
numpy forward's outputs under both evaluators (D13), onnxruntime at its default optimisations. The
graph derives a batch's segments and positions from its ids as segment_context does, and avoids
onnxruntime's float32 scale fusion.
"""

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime  # pyright: ignore[reportMissingTypeStubs]
import pytest

from indrajala_ml.model.export.onnx.check import TOLERANCE
from indrajala_ml.model.export.onnx.exporter import export_onnx
from indrajala_ml.model.export.onnx.graph import Graph
from indrajala_ml.model.export.onnx.sequence import segments
from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.layers.numpy.segments import segment_context
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    Dense,
    Dropout,
    Embedding,
    GatedDense,
    LayerNorm,
    LayerSpec,
    Position,
    Residual,
    StateInput,
    TokenMean,
    expand_specs,
)
from indrajala_ml.model.specs.update_rules import Adam
from tests.model.persistence.test_format2 import ENSEMBLES, _trained
from tests.saved_model_fixtures import FIXTURES

DTYPES = ["float32", "float64"]


def _sequence_or_attention(name: str) -> bool:
    if name in ENSEMBLES:
        return False
    network = FIXTURES[name].build()
    kinds = (Attention, LayerNorm, Position, Embedding, GatedDense)
    return (
        network.implementation != "python"
        and network.format2_shape in ("multiclass", "single_output", "sequence")
        and any(isinstance(spec, kinds) for spec in expand_specs(network.layer_specs))
        and not any(isinstance(spec, StateInput) for spec in expand_specs(network.layer_specs))
    )


FIXTURE_NAMES = sorted(name for name in FIXTURES if _sequence_or_attention(name))


def test_the_fixtures_cover_sequences_and_patch_attention():
    assert any(FIXTURES[name].build().format2_shape == "sequence" for name in FIXTURE_NAMES)
    assert any(name.startswith("MultiHeadAttention") for name in FIXTURE_NAMES)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_a_sequence_fixture_exports_as_its_forward(name: str, dtype: str):
    export = export_onnx(_trained(name), dtype=dtype)  # pyright: ignore[reportArgumentType]

    assert export.report is not None
    assert set(export.report.max_difference) == {"reference", "onnxruntime"}
    assert max(export.report.max_difference.values()) <= TOLERANCE[export.dtype]


TOKENS = 6
VOCABULARY = 9


def _model(attention: Attention, ffn: tuple[LayerSpec, ...] | None = None, segmented: bool = True) -> list[LayerSpec]:
    body = ffn or (Dense(10, activation="relu"), Dense(8, activation="linear", bias=True))
    return [
        Embedding(VOCABULARY, 8, segments=segmented),
        # a token dropout stands right after a Position
        *([] if attention.positions == "rotary" else [Position(), Dropout(0.1)]),
        Residual((LayerNorm(), attention)),
        Residual((LayerNorm(), *body)),
        LayerNorm(epsilon=1e-4),
        Dense(VOCABULARY, output=True, activation="softmax", loss="cross_entropy"),
    ]


ARCHITECTURES: dict[str, list[LayerSpec]] = {
    "one head, unmasked, no segments": _model(Attention(), segmented=False),
    "two heads, causal, segments": _model(Attention(heads=2, causal=True)),
    "four heads sharing two, causal": _model(Attention(heads=4, key_size=2, causal=True, key_value_heads=2)),
    "one shared key/value head": _model(Attention(heads=4, key_size=2, key_value_heads=1)),
    "rotary": _model(Attention(heads=2, causal=True, positions="rotary", rotary_base=500.0)),
    "across documents": _model(Attention(heads=2, causal=True, document_mask=False)),
    "gelu FFN": _model(
        Attention(heads=2, causal=True), (Dense(10, activation="gelu"), Dense(8, activation="linear", bias=True))
    ),
    "SwiGLU": _model(Attention(heads=2, causal=True), (GatedDense(6), Dense(8, activation="linear", bias=True))),
    "GEGLU, tanh": _model(
        Attention(heads=2), (GatedDense(6, activation="gelu_tanh"), Dense(8, activation="linear", bias=True))
    ),
}


def _architecture(name: str, backend: Any) -> Any:
    network = SequentialArrayNetwork((TOKENS,), ARCHITECTURES[name], Adam(), shape="sequence", backend=backend)
    network.rng = backend.default_rng(7)
    network.randomize()
    return network


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", [NUMPY, RUST], ids=["numpy", "rust"])
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_every_sequence_architecture_exports_as_its_forward(architecture: str, backend: Any, dtype: str):
    export = export_onnx(_architecture(architecture, backend), dtype=dtype)  # pyright: ignore[reportArgumentType]

    assert export.report is not None
    assert max(export.report.max_difference.values()) <= TOLERANCE[export.dtype]


def test_a_token_mean_classifier_reads_ids():
    layers: list[LayerSpec] = [
        Embedding(VOCABULARY, 8),
        Position(),
        Residual((LayerNorm(), Attention(heads=2))),
        TokenMean(),
        Dense(3, output=True, activation="softmax", loss="cross_entropy"),
    ]
    network = SequentialArrayNetwork((TOKENS,), layers, Adam(), backend=NUMPY)
    network.randomize()

    export = export_onnx(network, dtype="float64")

    assert [i.name for i in export.model.graph.input] == ["ids"]
    assert export.report is not None
    assert max(export.report.max_difference.values()) <= TOLERANCE["float64"]


def test_a_sequence_graph_reads_ids_and_gives_each_tokens_probabilities():
    model = export_onnx(_architecture("two heads, causal, segments", NUMPY)).model
    (graph_input,) = model.graph.input
    (graph_output,) = model.graph.output

    assert graph_input.name == "ids"
    assert graph_input.type.tensor_type.elem_type == onnx.TensorProto.INT64
    assert [d.dim_value or d.dim_param for d in graph_output.type.tensor_type.shape.dim] == [
        "batch",
        TOKENS,
        VOCABULARY,
    ]


@pytest.mark.parametrize("separator_opens", [False, True], ids=["separator ends", "separator opens"])
def test_the_graphs_segments_are_segment_contexts(separator_opens: bool):
    # pads and separators anywhere in a row, not only where packing puts them
    spec = Embedding(VOCABULARY, 4, segments=True, separator_opens=separator_opens)
    ids = np.random.default_rng(3).integers(VOCABULARY - 4, VOCABULARY, (64, TOKENS))
    g = Graph("float64")
    context = segments(g, "ids", spec, TOKENS)
    g.rename(context.segments, "segment_ids")
    g.rename(context.positions, "segment_positions")
    int64 = onnx.TensorProto.INT64
    model = g.model(
        [g.value_info("ids", ["batch", TOKENS], int64)],
        [
            g.value_info("segment_ids", ["batch", TOKENS], int64),
            g.value_info("segment_positions", ["batch", TOKENS], int64),
        ],
        "segments",
        {},
    )
    session: Any = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

    actual_segments, actual_positions = session.run(None, {"ids": ids})
    expected = segment_context(ids.astype(np.float64), spec.separator, spec.pad, separator_opens)

    np.testing.assert_array_equal(actual_segments, expected.segments)
    np.testing.assert_array_equal(actual_positions, expected.positions)


def test_onnxruntime_fuses_no_scale_into_a_float32_alpha(tmp_path: Path):
    # MatMulScaleFusion folds a scalar divisor into FusedMatMul's alpha, a float32 attribute
    model = export_onnx(_architecture("two heads, causal, segments", NUMPY), dtype="float64").model
    runtime: Any = onnxruntime
    options = runtime.SessionOptions()
    options.optimized_model_filepath = str(tmp_path / "optimised.onnx")
    onnxruntime.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])

    optimised: Any = onnx.load(str(tmp_path / "optimised.onnx"))  # pyright: ignore[reportUnknownMemberType]
    assert "FusedMatMul" not in {node.op_type for node in optimised.graph.node}
