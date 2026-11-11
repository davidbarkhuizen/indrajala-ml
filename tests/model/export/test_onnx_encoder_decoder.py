"""
The encoder-decoder's ONNX export (model/export/onnx/encoder_decoder.py, the export-and-hub
workplan, stage 7): the full forward, the encoder and the decode step with its cache. For the
numpy and Rust fixtures and for decoders of every option (rotary self-attention, learned positions,
grouped heads in both attentions, a wider memory), in both dtypes under both evaluators: the full
forward gives our network's probabilities, and decoding every target token through the encoder
and decode step from an empty cache, teacher-forced, gives the full forward's at each token (D13).
"""

from pathlib import Path
from typing import Any

import onnx
import pytest

from indrajala_ml.model.export.onnx.check import TOLERANCE
from indrajala_ml.model.export.onnx.exporter import export_onnx
from indrajala_ml.model.networks.numpy.encoder_decoder_array_network import EncoderDecoderArrayNetwork
from indrajala_ml.model.networks.rust.encoder_decoder_rust_array_network import EncoderDecoderRustArrayNetwork
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    CrossAttention,
    Dense,
    Embedding,
    EncoderDecoder,
    LayerNorm,
    Position,
    Residual,
)
from indrajala_ml.model.specs.update_rules import Adam
from tests.model.persistence.test_format2 import _trained

DTYPES = ["float32", "float64"]
BACKENDS: dict[str, Any] = {"numpy": EncoderDecoderArrayNetwork, "rust": EncoderDecoderRustArrayNetwork}
VOCABULARY = 7


def _ffn(width: int) -> Residual:
    return Residual((LayerNorm(), Dense(2 * width, "gelu"), Dense(width, "linear", bias=True)))


def _spec(self_attention: Attention, cross: CrossAttention, positions: bool, memory_width: int = 6) -> EncoderDecoder:
    return EncoderDecoder(
        (
            Embedding(VOCABULARY, memory_width, segments=True),
            *([Position()] if positions else []),
            Residual((LayerNorm(), Attention(heads=2))),
            _ffn(memory_width),
            LayerNorm(),
        ),
        (
            Embedding(VOCABULARY, 4, segments=True, separator_opens=True),
            *([Position()] if positions else []),
            Residual((LayerNorm(), self_attention)),
            Residual((LayerNorm(), cross)),
            _ffn(4),
            LayerNorm(),
            Dense(VOCABULARY - 1, output=True, activation="softmax", loss="cross_entropy"),
        ),
    )


ARCHITECTURES = {
    "learned positions": _spec(Attention(heads=2, causal=True), CrossAttention(heads=2), positions=True),
    "rotary": _spec(
        Attention(heads=2, causal=True, positions="rotary", rotary_base=100.0), CrossAttention(heads=2), positions=False
    ),
    "grouped heads, a wider memory": _spec(
        Attention(heads=4, key_size=2, causal=True, key_value_heads=2),
        CrossAttention(heads=4, key_size=2, key_value_heads=1),
        positions=True,
        memory_width=8,
    ),
}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", ["EncoderDecoderArrayNetwork", "EncoderDecoderRustArrayNetwork"])
def test_an_encoder_decoder_fixture_exports_and_decodes_as_its_forward(name: str, dtype: str):
    export = export_onnx(_trained(name), dtype=dtype)  # pyright: ignore[reportArgumentType]

    assert export.report is not None
    assert set(export.report.max_difference) == {"reference", "onnxruntime"}
    assert max(export.report.max_difference.values()) <= TOLERANCE[export.dtype]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_every_decoder_exports_and_decodes_as_its_forward(architecture: str, backend: str, dtype: str):
    network = BACKENDS[backend].randomized(ARCHITECTURES[architecture], 5, 6, Adam(), seed=3)

    export = export_onnx(network, dtype=dtype)  # pyright: ignore[reportArgumentType]

    assert export.report is not None
    assert max(export.report.max_difference.values()) <= TOLERANCE[export.dtype]


def test_the_three_models_are_saved_side_by_side(tmp_path: Path):
    export = export_onnx(_trained("EncoderDecoderArrayNetwork"), tmp_path / "translator.onnx")

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "translator.onnx",
        "translator_decoder_step.onnx",
        "translator_encoder.onnx",
    ]
    step: Any = onnx.load(str(tmp_path / "translator_decoder_step.onnx"))  # pyright: ignore[reportUnknownMemberType]
    names = {value.name for value in step.graph.input}
    assert {"ids", "segments", "positions", "column", "past_segments", "memory", "memory_segments"} <= names
    assert {"past_keys.0", "past_values.0"} <= names
    assert set(export.parts) == {"encoder", "decoder_step"}
