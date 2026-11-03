"""An encoder-decoder's format-2 entries (the cross-attention workplan, stage 3, D13, D14)."""

import json

from indrajala_ml.model.persistence.format2 import ENCODER_DECODER
from indrajala_ml.model.persistence.format2_json import (
    _layer_parameter_names,  # pyright: ignore[reportPrivateUsage]
    encoder_decoder_from_json,
    encoder_decoder_input_from_json,
    encoder_decoder_input_to_json,
    encoder_decoder_to_json,
    layer_from_json,
    layer_to_json,
)
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    CrossAttention,
    Dense,
    Embedding,
    EncoderDecoder,
    LayerNorm,
    Residual,
)

SPEC = EncoderDecoder(
    (Embedding(10, 16, segments=True), Residual((LayerNorm(), Attention(heads=2))), LayerNorm()),
    (
        Embedding(8, 12, segments=True, separator_opens=True),
        Residual((LayerNorm(), Attention(heads=2, causal=True))),
        Residual((LayerNorm(), CrossAttention(heads=2, key_value_heads=1, dropout=0.1))),
        LayerNorm(),
        Dense(8, "softmax", output=True, loss="cross_entropy"),
    ),
)


def test_a_cross_attention_entry_holds_its_fields_when_not_the_default():
    assert layer_to_json(CrossAttention()) == {"kind": "cross_attention"}
    entry = layer_to_json(CrossAttention(heads=4, key_size=8, key_value_heads=2, dropout=0.1))
    assert entry == {"kind": "cross_attention", "heads": 4, "key_size": 8, "key_value_heads": 2, "dropout": 0.1}
    assert layer_from_json(entry) == CrossAttention(heads=4, key_size=8, key_value_heads=2, dropout=0.1)


def test_an_embedding_entry_says_separator_opens_only_when_set():
    # as before the field, so earlier files don't change
    assert layer_to_json(Embedding(8, 12, segments=True)) == {
        "kind": "embedding",
        "vocabulary": 8,
        "size": 12,
        "segments": True,
    }
    entry = layer_to_json(Embedding(8, 12, segments=True, separator_opens=True))
    assert entry["separator_opens"] is True
    assert layer_from_json(entry) == Embedding(8, 12, segments=True, separator_opens=True)


def test_an_encoder_decoder_s_layers_and_input_round_trip_through_json():
    document = {"format": 2, "shape": ENCODER_DECODER, **encoder_decoder_to_json(SPEC)}
    assert [entry["kind"] for entry in document["encoder"]] == ["embedding", "residual", "layer_norm"]
    assert document["layers"][2]["body"][1]["kind"] == "cross_attention"
    assert encoder_decoder_from_json(json.loads(json.dumps(document))) == SPEC
    entry = encoder_decoder_input_to_json(1024, 1025)
    assert entry == {"source": {"dimension": 1024}, "target": {"dimension": 1025}}
    assert encoder_decoder_input_from_json(entry) == (1024, 1025)


def test_a_cross_attention_s_parameters_are_an_attention_s_in_the_one_index_space():
    names = _layer_parameter_names(SPEC.layers(), None)
    attention = ("Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo")
    # the encoder's expanded layers (embedding, fork, norm, attention, add, norm), then the decoder's
    assert names[:6] == [("E",), (), ("gamma", "beta"), attention, (), ("gamma", "beta")]
    assert names[6] == ("E",) and names[13] == attention  # the decoder's CrossAttention
