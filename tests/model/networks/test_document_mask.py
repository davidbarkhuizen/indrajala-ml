"""
The unmasked arm's switch (the context-length workplan, D8): an Attention with document_mask False
attends across paragraphs, every pad still masked, so on a batch's segments it is the masked layer
on segments that tell only the pads apart, by bits, in all three implementations; the spec saves it
only when not the default.
"""

from __future__ import annotations

from typing import Any

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model.layers.forward_context import ForwardContext
from indrajala_ml.model.layers.numpy.attention_array_layer import AttentionArrayLayer
from indrajala_ml.model.layers.python.attention_layer import AttentionLayer
from indrajala_ml.model.layers.rust.attention_rust_array_layer import AttentionRustArrayLayer
from indrajala_ml.model.persistence.format2_json import layer_from_json, layer_to_json
from indrajala_ml.model.specs.layer_specs import Attention
from tests.helpers import bits
from tests.model.networks.test_attention_python_network import _input

N, T, D, H = 2, 40, 4, 2
# a row of pads, then paragraphs; one paragraph, then another
SEGMENTS = np.array([[0] * 3 + [1] * 17 + [2] * 20, [1] * 25 + [2] * 15])
PADS_ONLY = (SEGMENTS != 0).astype(np.intp)


def _weights() -> list[np.ndarray]:
    # the four projections' weights and biases, numpy's shapes
    rng = np.random.default_rng(1)
    return [rng.uniform(-0.6, 0.6, p.shape) for p in AttentionArrayLayer(T, D, H).parameters()]


def _X() -> np.ndarray:
    return np.random.default_rng(2).uniform(-1.0, 1.0, (N, T * D))


def _numpy(document_mask: bool, segments: np.ndarray) -> np.ndarray:
    layer = AttentionArrayLayer(T, D, H, causal=True, positions="rotary", document_mask=document_mask)
    layer.set_parameters(_weights())
    return layer.forward_batch(_X(), context=ForwardContext(segments=segments, examples=N))


def _rust(document_mask: bool, segments: np.ndarray) -> np.ndarray:
    layer = AttentionRustArrayLayer(T, D, H, causal=True, positions="rotary", document_mask=document_mask)
    layer.set_parameters([pa.Array(w.tolist()) for w in _weights()])
    context = ForwardContext(segments=pa.Array(segments.astype(np.float64).tolist()), examples=N)
    return np.array(layer.forward_batch(pa.Array(_X().tolist()), context=context).tolist())


def _python(document_mask: bool, segments: np.ndarray) -> list[float]:
    out: list[float] = []
    weights = _weights()
    rows = [
        (list(w), b) for W, bias in zip(weights[::2], weights[1::2], strict=True) for w, b in zip(W, bias, strict=True)
    ]
    for n in range(N):
        layer = AttentionLayer(
            _input(_X()[n].tolist()), T, D, H, causal=True, positions="rotary", document_mask=document_mask
        )
        layer.restore_state(rows)
        layer.forward(context=ForwardContext(segments=[[int(s) for s in segments[n]]], examples=1))
        out += [node.value() for node in layer.nodes]
    return out


@pytest.mark.parametrize("implementation", [_numpy, _rust, _python], ids=["numpy", "rust", "python"])
def test_unmasked_is_the_masked_layer_on_segments_that_tell_only_the_pads_apart(implementation: Any):
    unmasked = implementation(False, SEGMENTS)
    assert bits(unmasked) == bits(implementation(True, PADS_ONLY))
    assert bits(unmasked) != bits(implementation(True, SEGMENTS))


def test_document_mask_is_saved_only_when_off():
    assert "document_mask" not in layer_to_json(Attention(heads=2))
    entry = layer_to_json(Attention(heads=2, document_mask=False))
    assert entry["document_mask"] is False and layer_from_json(entry) == Attention(heads=2, document_mask=False)
