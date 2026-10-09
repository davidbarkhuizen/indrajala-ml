"""
Format 2's layer entries (model/persistence/format2_json.py): an entry added to a spec after files
were saved without it is left out at its default, so those files are what such a network saves now.
The sequence task workplan's (stage 2): an embedding's entry, its table E, and a causal attention's.
"""

from indrajala_ml.model.persistence.checkpoint import OptimizerState
from indrajala_ml.model.persistence.format2_json import (
    layer_from_json,
    layer_to_json,
    optimizer_state_from_json,
    optimizer_state_to_json,
)
from indrajala_ml.model.specs.layer_specs import Attention, BatchNorm, Conv, Dense, Embedding
from indrajala_ml.model.specs.update_rules import Adam


def test_a_batch_norm_entry_holds_a_group_size_only_when_it_has_one():
    # no "group_size" without ghost groups: files saved before BatchNorm had one are what such a
    # network saves now, and they load with the default
    plain = {"kind": "batch_norm", "activation": "relu", "epsilon": 1e-5, "running_rate": 0.1}
    assert layer_to_json(BatchNorm("relu")) == plain
    assert layer_to_json(BatchNorm("relu", group_size=32)) == {**plain, "group_size": 32}
    assert layer_from_json(plain) == BatchNorm("relu")
    assert layer_from_json({**plain, "group_size": 32}) == BatchNorm("relu", group_size=32)


def test_a_relu_conv_entry_is_unchanged():
    # no "activation": files saved before ConvSpec had one are what a ReLU conv network saves now
    assert layer_to_json(Conv(3, 8, stride=2)) == {"kind": "conv", "kernel_size": 3, "channel_count": 8, "stride": 2}


def test_an_attention_entry_holds_causal_only_when_masked():
    # no "causal" unmasked: the attention fixtures are what such a network saves now, and older
    # checkouts load them (the sequence task workplan, D7)
    assert layer_to_json(Attention()) == {"kind": "attention"}
    assert layer_to_json(Attention(heads=4, causal=True)) == {"kind": "attention", "heads": 4, "causal": True}
    assert layer_from_json({"kind": "attention", "causal": True}) == Attention(causal=True)


def test_an_embedding_has_its_entry():
    assert layer_to_json(Embedding(65, 64)) == {"kind": "embedding", "vocabulary": 65, "size": 64}
    assert layer_from_json({"kind": "embedding", "vocabulary": 65, "size": 64}) == Embedding(65, 64)


def test_an_embeddings_optimizer_state_is_its_tables():
    # E, the (vocabulary, size) table, on numpy and Rust; a token-wise output layer's W and b
    specs = [Embedding(3, 2), Dense(3, output=True, activation="softmax", loss="cross_entropy")]
    state = OptimizerState(4, {0: [[[0.5, 0.25]] * 3, [[0.125, 1.0]] * 3], 1: [[1.0], [2.0], [3.0], [4.0]]})
    entry = optimizer_state_to_json(Adam(), False, state, specs)
    assert entry["layers"][0] == {"m_E": [[0.5, 0.25]] * 3, "v_E": [[0.125, 1.0]] * 3}
    assert list(entry["layers"][1]) == ["m_W", "v_W", "m_b", "v_b"]
    assert optimizer_state_from_json(Adam(), False, entry, specs) == state
