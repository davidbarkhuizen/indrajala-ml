"""
Format 2's layer entries (model/persistence/format2_json.py): an entry added to a spec after files
were saved without it is left out at its default, so those files are what such a network saves now.
The sequence task workplan's (stage 2): an embedding's entry, its table E, and a causal attention's.
The attention-dropout workplan's (stage 2, D7): an attention's dropout, and a token dropout's entry.
The rotary-positions workplan's (stage 2, D2): an attention's positions and rotary base.
The audit-remedies workplan's (stage 2, D2): a dense entry's loss, and every format-2 fixture re-saved.
"""

import json
from pathlib import Path

import pytest

from indrajala_ml.model.persistence.checkpoint import OptimizerState
from indrajala_ml.model.persistence.format2_json import (
    layer_from_json,
    layer_to_json,
    optimizer_state_from_json,
    optimizer_state_to_json,
)
from indrajala_ml.model.specs.layer_specs import Attention, BatchNorm, Conv, Dense, Dropout, Embedding, Position
from indrajala_ml.model.specs.update_rules import Adam
from tests.saved_model_fixtures import FIXTURE_DIR, FIXTURES, fixture_class


def test_a_batch_norm_entry_holds_a_group_size_only_when_it_has_one():
    # no "group_size" without ghost groups: files saved before BatchNorm had one are what such a
    # network saves now, and they load with the default
    plain = {"kind": "batch_norm", "activation": "relu", "epsilon": 1e-5, "running_rate": 0.1}
    assert layer_to_json(BatchNorm("relu")) == plain
    assert layer_to_json(BatchNorm("relu", group_size=32)) == {**plain, "group_size": 32}
    assert layer_from_json(plain) == BatchNorm("relu")
    assert layer_from_json({**plain, "group_size": 32}) == BatchNorm("relu", group_size=32)


def test_a_dense_entry_holds_its_loss_as_before_the_output_layer_named_it():
    # a hidden layer has no loss (the audit-remedies workplan, D2) and its entry holds "squared", as
    # every file saved before it; the reader reads that back as no loss, and an output layer's as named
    hidden = {"kind": "dense", "size": 5, "activation": "relu", "dropout": None, "output": False, "loss": "squared"}
    assert layer_to_json(Dense(5, activation="relu")) == hidden
    assert layer_from_json(hidden) == Dense(5, activation="relu")
    for loss in ("squared", "cross_entropy"):
        output = Dense(3, output=True, loss=loss)
        assert layer_to_json(output) == {**hidden, "size": 3, "activation": "sigmoid", "output": True, "loss": loss}
        assert layer_from_json(layer_to_json(output)) == output


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


def test_an_attention_entry_holds_dropout_only_when_it_drops():
    # no "dropout" without it: the attention fixtures are what such a network saves now, and older
    # checkouts load them (the attention-dropout workplan, D7)
    assert layer_to_json(Attention(dropout=0.0)) == {"kind": "attention"}
    assert layer_to_json(Attention(causal=True, dropout=0.1)) == {"kind": "attention", "causal": True, "dropout": 0.1}
    assert layer_from_json({"kind": "attention", "dropout": 0.1}) == Attention(dropout=0.1)


def test_a_dropout_has_its_entry_and_no_optimizer_state():
    assert layer_to_json(Dropout(0.1)) == {"kind": "dropout", "p": 0.1}
    assert layer_from_json({"kind": "dropout", "p": 0.1}) == Dropout(0.1)
    # no weights, so no state: the optimizer steps the layers around it only
    specs = [
        Embedding(3, 2),
        Position(),
        Dropout(0.1),
        Dense(3, output=True, activation="softmax", loss="cross_entropy"),
    ]
    state = OptimizerState(1, {0: [[[0.5, 0.25]] * 3, [[0.125, 1.0]] * 3], 1: [[[1.0, 2.0]], [[3.0, 4.0]]]})
    entry = optimizer_state_to_json(Adam(), False, state, specs)
    assert entry["layers"][2] is None
    assert optimizer_state_from_json(Adam(), False, entry, specs) == state


def test_an_attention_entry_holds_its_positions_only_when_it_rotates():
    # no "positions" or "rotary_base" without rotation: older files are what such a network saves now
    # (the rotary-positions workplan, D2)
    assert layer_to_json(Attention(positions="none")) == {"kind": "attention"}
    assert layer_to_json(Attention(positions="rotary")) == {"kind": "attention", "positions": "rotary"}
    entry = {"kind": "attention", "causal": True, "positions": "rotary", "rotary_base": 500.0}
    assert layer_to_json(Attention(causal=True, positions="rotary", rotary_base=500.0)) == entry
    assert layer_from_json(entry) == Attention(causal=True, positions="rotary", rotary_base=500.0)


FORMAT2_FIXTURES = sorted(
    name for name in FIXTURES if json.loads((FIXTURE_DIR / f"{name}.json").read_text()).get("format") == 2
)


@pytest.mark.parametrize("name", FORMAT2_FIXTURES)
def test_every_format_2_fixtures_layer_entries_re_save_unchanged(name: str, tmp_path: Path):
    # the output layer naming its loss (the audit-remedies workplan, D2) moves no file's layers:
    # every format-2 fixture loads and saves them again as they were written (some fixtures predate
    # the saved generator state, which a re-save adds, so the layers are compared, not the bytes)
    path = tmp_path / f"{name}.json"
    fixture_class(name).load(str(FIXTURE_DIR / f"{name}.json")).save(str(path))
    saved = json.loads(path.read_text())["layers"]
    assert saved == json.loads((FIXTURE_DIR / f"{name}.json").read_text())["layers"]
