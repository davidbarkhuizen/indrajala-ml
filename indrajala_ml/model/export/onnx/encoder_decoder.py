"""
An encoder-decoder network's graphs (the export-and-hub workplan, stage 7, D12): three models, as
optimum's encoder, decoder and decoder_with_past.

- **model**, the full forward: source (N, S) and target (N, T) int64 ids to each target token's
  probabilities (N, T, V), as the network's own pass.
- **encoder**: source (N, S) to memory (N, S, d_m), the encoder's output, and memory_segments
  (N, S) int64, its segments, which the decoder's cross-attention masks by.
- **decoder_step**: one token a row, AttentionArrayLayer.step's and the Decoder's: ids (N, 1) with
  its segment and its position within it (N, 1) int64 (the host's, as the Decoder's context), its
  absolute column (1,) int64 (rotary's), the past tokens' segments (N, L) and each self-attention
  layer's past keys and values (N, g, L, d_k), and the memory; to its probabilities (N, 1, V),
  each layer's present keys and values and present_segments, the last T kept (the cache's window),
  the next step's past. Cross-attention's keys and values are computed from the memory each step.

The check (D13) runs the full model against our numpy forward, then decodes every target token
by token through the encoder and decoder_step, teacher-forced, from an empty cache: each step's
probabilities are the full forward's at that token.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from indrajala_ml.model.export.onnx.graph import Graph
from indrajala_ml.model.export.onnx.layers import Memory, Tensor, walk_all
from indrajala_ml.model.export.onnx.sequence import StepCache, StepContext
from indrajala_ml.model.persistence.format3 import tensor_names
from indrajala_ml.model.specs.layer_specs import Attention, EncoderDecoder, expand_specs
from indrajala_ml.model.specs.spec_shapes import encoder_decoder_shapes

SOURCE = "source"
TARGET = "target"
PROBABILITIES = "probabilities"
MEMORY = "memory"
MEMORY_SEGMENTS = "memory_segments"
IDS = "ids"
SEGMENTS = "segments"
POSITIONS = "positions"
COLUMN = "column"
PAST_SEGMENTS = "past_segments"
PRESENT_SEGMENTS = "present_segments"


def _split(spec: EncoderDecoder, weights: Sequence[Sequence[Any]]) -> tuple[Any, Any, Any, Any]:
    # the encoder's names and weights, then the decoder's (format 2's one index space, D14)
    names = tensor_names([*spec.encoder, *spec.decoder], None)
    count = len(expand_specs(spec.encoder))
    return names[:count], weights[:count], names[count:], weights[count:]


def _encode(g: Graph, spec: EncoderDecoder, source_tokens: int, weights: Sequence[Sequence[Any]]) -> Memory:
    encoder_names, encoder_weights, _, _ = _split(spec, weights)
    walked = walk_all(
        g, spec.encoder, (source_tokens,), Tensor(SOURCE, (source_tokens,)), encoder_names, encoder_weights
    )
    assert walked.context is not None, "an encoder starts with an Embedding with segments (D14)"
    return Memory(walked.output, walked.context.segments)


def full(
    g: Graph, spec: EncoderDecoder, source_tokens: int, target_tokens: int, weights: Sequence[Sequence[Any]]
) -> tuple[list[Any], list[Any]]:
    """The full forward's graph: its inputs' and outputs' value infos."""
    from indrajala_ml.model.export.onnx.dependencies import onnx

    memory = _encode(g, spec, source_tokens, weights)
    _, _, decoder_names, decoder_weights = _split(spec, weights)
    _, decoder_shapes = encoder_decoder_shapes(spec, source_tokens, target_tokens)
    walked = walk_all(
        g,
        spec.decoder,
        (target_tokens,),
        Tensor(TARGET, (target_tokens,)),
        decoder_names,
        decoder_weights,
        shapes=decoder_shapes,
        memory=memory,
    )
    g.rename(walked.output.name, PROBABILITIES)
    int64 = onnx().TensorProto.INT64
    return (
        [g.value_info(SOURCE, ["batch", source_tokens], int64), g.value_info(TARGET, ["batch", target_tokens], int64)],
        [g.value_info(PROBABILITIES, ["batch", *walked.output.dims])],
    )


def encoder(
    g: Graph, spec: EncoderDecoder, source_tokens: int, target_tokens: int, weights: Sequence[Sequence[Any]]
) -> tuple[list[Any], list[Any]]:
    """The encoder's graph: source to memory and memory_segments (target_tokens unread: the builders' one signature)."""
    del target_tokens
    from indrajala_ml.model.export.onnx.dependencies import onnx

    memory = _encode(g, spec, source_tokens, weights)
    g.rename(memory.tensor.name, MEMORY)
    assert memory.segments is not None
    g.rename(memory.segments, MEMORY_SEGMENTS)
    int64 = onnx().TensorProto.INT64
    return (
        [g.value_info(SOURCE, ["batch", source_tokens], int64)],
        [
            g.value_info(MEMORY, ["batch", *memory.tensor.dims]),
            g.value_info(MEMORY_SEGMENTS, ["batch", source_tokens], int64),
        ],
    )


def self_attention_count(spec: EncoderDecoder) -> int:
    """The decoder's self-attention layers, each with a cache in decoder_step."""
    return sum(isinstance(layer, Attention) for layer in expand_specs(spec.decoder))


def decoder_step(
    g: Graph, spec: EncoderDecoder, source_tokens: int, target_tokens: int, weights: Sequence[Sequence[Any]]
) -> tuple[list[Any], list[Any]]:
    """The decode step's graph (the module's docstring)."""
    from indrajala_ml.model.export.onnx.dependencies import onnx

    int64 = onnx().TensorProto.INT64
    encoder_shapes, decoder_shapes = encoder_decoder_shapes(spec, source_tokens, 1)
    memory_dims = encoder_shapes[-1].output_shape
    memory = Memory(Tensor(MEMORY, tuple(memory_dims)), MEMORY_SEGMENTS)
    step = StepContext(SEGMENTS, POSITIONS, COLUMN, PAST_SEGMENTS, target_tokens)
    # each self-attention and its tokens' width
    attentions = [
        (layer, shape.input_shape[-1])
        for layer, shape in zip(expand_specs(spec.decoder), decoder_shapes, strict=True)
        if isinstance(layer, Attention)
    ]
    caches = [
        StepCache(f"past_keys.{i}", f"past_values.{i}", f"present_keys.{i}", f"present_values.{i}")
        for i in range(len(attentions))
    ]
    _, _, decoder_names, decoder_weights = _split(spec, weights)
    walked = walk_all(
        g,
        spec.decoder,
        (1,),
        Tensor(IDS, (1,)),
        decoder_names,
        decoder_weights,
        shapes=decoder_shapes,
        memory=memory,
        step=step,
        caches=caches,
    )
    g.rename(walked.output.name, PROBABILITIES)
    window = [g.int64([-target_tokens]), g.int64([2**62]), g.int64([1])]
    g.rename(g.op("Slice", [g.op("Concat", [PAST_SEGMENTS, SEGMENTS], axis=1), *window]), PRESENT_SEGMENTS)
    inputs = [
        g.value_info(IDS, ["batch", 1], int64),
        g.value_info(SEGMENTS, ["batch", 1], int64),
        g.value_info(POSITIONS, ["batch", 1], int64),
        g.value_info(COLUMN, [1], int64),
        g.value_info(PAST_SEGMENTS, ["batch", "past"], int64),
        g.value_info(MEMORY, ["batch", *memory_dims]),
        g.value_info(MEMORY_SEGMENTS, ["batch", source_tokens], int64),
    ]
    outputs = [g.value_info(PROBABILITIES, ["batch", *walked.output.dims])]
    for (attention, features), cache in zip(attentions, caches, strict=True):
        kv_heads = attention.key_value_heads or attention.heads
        key_size = attention.head_size(features)
        inputs += [
            g.value_info(cache.past_keys, ["batch", kv_heads, "past", key_size]),
            g.value_info(cache.past_values, ["batch", kv_heads, "past", key_size]),
        ]
        outputs += [
            g.value_info(cache.present_keys, ["batch", kv_heads, "present", key_size]),
            g.value_info(cache.present_values, ["batch", kv_heads, "present", key_size]),
        ]
    outputs.append(g.value_info(PRESENT_SEGMENTS, ["batch", "present"], int64))
    return inputs, outputs
