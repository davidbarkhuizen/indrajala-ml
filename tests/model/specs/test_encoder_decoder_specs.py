"""An encoder-decoder's specs, validation and shapes (the cross-attention workplan, stage 3, D13, D14)."""

import pytest

from indrajala_ml.model.specs.layer_specs import (
    Attention,
    CrossAttention,
    Dense,
    Dropout,
    Embedding,
    EncoderDecoder,
    LayerNorm,
    LayerSpec,
    Position,
    Residual,
    StateInput,
    expand_specs,
)
from indrajala_ml.model.specs.spec_shapes import encoder_decoder_shapes, spec_shapes
from indrajala_ml.model.specs.spec_validation import (
    validate_encoder_decoder,
    validate_layer_specs,
    validate_window_recurrent,
)
from indrajala_ml.model.specs.window_recurrence import Codes, Factor, WindowRecurrence

SOURCE, TARGET = 10, 8


def ffn(width: int) -> Residual:
    return Residual((LayerNorm(), Dense(2 * width, "relu"), Dense(width, "linear", bias=True)))


def encoder(width: int = 16) -> tuple[LayerSpec, ...]:
    return (
        Embedding(SOURCE, width, segments=True),
        Residual((LayerNorm(), Attention(heads=2))),
        ffn(width),
        LayerNorm(),
    )


def decoder(width: int = 12, cross: CrossAttention | None = None) -> tuple[LayerSpec, ...]:
    return (
        Embedding(TARGET, width, segments=True, separator_opens=True),
        Residual((LayerNorm(), Attention(heads=2, causal=True))),
        Residual((LayerNorm(), cross or CrossAttention(heads=2))),
        ffn(width),
        LayerNorm(),
        Dense(TARGET, "softmax", output=True, loss="cross_entropy"),
    )


def test_an_encoder_and_a_decoder_with_cross_attention_validate():
    validate_encoder_decoder(EncoderDecoder(encoder(), decoder()))
    # the residual dropout after a CrossAttention, and grouped key/value heads, as an Attention's
    validate_encoder_decoder(
        EncoderDecoder(
            encoder(),
            (
                Embedding(TARGET, 12, segments=True, separator_opens=True),
                Position(),
                Residual((LayerNorm(), Attention(heads=4, causal=True, key_value_heads=2))),
                Residual((LayerNorm(), CrossAttention(heads=4, key_value_heads=1, dropout=0.1), Dropout(0.1))),
                LayerNorm(),
                Dense(TARGET, "softmax", output=True, loss="cross_entropy"),
            ),
        )
    )


def test_the_layers_are_the_encoder_s_then_the_decoder_s():
    spec = EncoderDecoder(list(encoder()), list(decoder()))  # pyright: ignore[reportArgumentType]: lists, kept as tuples
    assert spec == EncoderDecoder(encoder(), decoder())  # lists kept as tuples
    assert spec.layers() == [*encoder(), *decoder()]
    assert len(expand_specs(spec.layers())) == len(expand_specs(encoder())) + len(expand_specs(decoder()))


def test_a_cross_attention_stands_only_in_a_decoder():
    with pytest.raises(AssertionError, match="only in an EncoderDecoder's decoder"):
        validate_layer_specs(list(decoder()))
    window = [
        Embedding(27, 16),
        StateInput(),
        Residual((LayerNorm(), CrossAttention(heads=2))),
        Dense(10, "softmax", output=True, loss="cross_entropy"),
    ]
    with pytest.raises(AssertionError, match="only in an EncoderDecoder's decoder"):
        validate_window_recurrent(window, WindowRecurrence((Factor(4, Codes(8)),)), (3,))
    with pytest.raises(AssertionError, match="CrossAttention stands in the decoder"):
        validate_encoder_decoder(
            EncoderDecoder((*encoder()[:-1], Residual((LayerNorm(), CrossAttention())), LayerNorm()), decoder())
        )


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (EncoderDecoder((Embedding(SOURCE, 16), *encoder()[1:]), decoder()), "an encoder starts with an Embedding"),
        (
            EncoderDecoder((Embedding(SOURCE, 16, segments=True, separator_opens=True), *encoder()[1:]), decoder()),
            "an encoder starts with an Embedding",
        ),
        (EncoderDecoder(encoder()[:-1], decoder()), "an encoder ends in a LayerNorm"),
        (
            EncoderDecoder(
                (*encoder()[:-1], Dense(16, "softmax", output=True, loss="cross_entropy"), LayerNorm()), decoder()
            ),
            "an encoder has no output layer",
        ),
        (
            EncoderDecoder((encoder()[0], Residual((LayerNorm(), Attention(causal=True))), LayerNorm()), decoder()),
            "an encoder reads its whole source",
        ),
        (
            EncoderDecoder(encoder(), (Embedding(TARGET, 12, segments=True), *decoder()[1:])),
            "a decoder starts with an Embedding with segments and separator_opens",
        ),
        (EncoderDecoder(encoder(), decoder()[:-1]), "the last layer must be the output layer"),
        (
            EncoderDecoder(encoder(), (decoder()[0], Residual((LayerNorm(), Attention(heads=2))), *decoder()[2:])),
            "a decoder's self-attention is causal",
        ),
        (
            EncoderDecoder(encoder(), (decoder()[0], *decoder()[3:])),
            "at least one CrossAttention",
        ),
        (EncoderDecoder(encoder(), decoder(cross=CrossAttention(heads=0))), "at least one head"),
        (EncoderDecoder(encoder(), decoder(cross=CrossAttention(heads=4, key_value_heads=3))), "divides its heads"),
        (EncoderDecoder(encoder(), decoder(cross=CrossAttention(heads=2, key_value_heads=2))), "key_value_heads=None"),
        (EncoderDecoder(encoder(), decoder(cross=CrossAttention(dropout=1.0))), r"dropout is in \[0.0, 1.0\)"),
    ],
)
def test_validation_refuses(spec: EncoderDecoder, message: str):
    with pytest.raises(AssertionError, match=message):
        validate_encoder_decoder(spec)


def test_separator_opens_is_a_rule_for_segments():
    with pytest.raises(ValueError, match="needs segments=True"):
        Embedding(8, 4, separator_opens=True)


def test_the_shape_walk_records_the_memory_s_width_on_each_cross_attention():
    # the memory (16 features) wider than the decoder's tokens (12)
    spec = EncoderDecoder(encoder(16), decoder(12))
    encoder_shapes, decoder_shapes = encoder_decoder_shapes(spec, source_tokens=20, target_tokens=21)
    assert encoder_shapes[-1].output_shape == (20, 16)
    crosses = [
        shape
        for layer, shape in zip(expand_specs(spec.decoder), decoder_shapes, strict=True)
        if isinstance(layer, CrossAttention)
    ]
    assert [(s.input_shape, s.output_shape, s.memory_shape) for s in crosses] == [((21, 12), (21, 12), (20, 16))]
    assert all(s.memory_shape is None for s in decoder_shapes if s not in crosses)
    assert decoder_shapes[-1].output_shape == (21, TARGET)
    # alone, the decoder's walk knows no memory
    assert all(s.memory_shape is None for s in spec_shapes(spec.decoder, (21,)))


def test_a_cross_attention_without_a_key_size_divides_the_token_width_among_its_heads():
    with pytest.raises(AssertionError, match="its heads divide it"):
        encoder_decoder_shapes(EncoderDecoder(encoder(), decoder(12, CrossAttention(heads=5))), 20, 21)
    # a key_size of its own needs no division
    encoder_decoder_shapes(EncoderDecoder(encoder(), decoder(12, CrossAttention(heads=5, key_size=8))), 20, 21)
    assert CrossAttention(heads=3).head_size(12) == 4 and CrossAttention(heads=3, key_size=7).head_size(12) == 7
    assert CrossAttention(heads=4).kv_heads() == 4 and CrossAttention(heads=4, key_value_heads=2).kv_heads() == 2
