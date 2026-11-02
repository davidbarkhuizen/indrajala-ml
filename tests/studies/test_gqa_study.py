"""
The grouped-query attention study's configuration against its workplan (stage 8, D7, D13): each arm
is the FFN study's SwiGLU model with only its attention's heads changed (8 query heads of 8 sharing g);
the cache each arm needs; the registered comparisons; a smoke run in which g8 saves its network and
an uptraining run starts from its conversion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from indrajala_ml.model.specs.layer_specs import Attention, GatedDense, Residual
from indrajala_ml.studies import context_length_study as cls
from indrajala_ml.studies import ffn_activation_study as ffn
from indrajala_ml.studies import gqa_study as gqa


def _layers(specs: list[Any], kind: type) -> list[Any]:
    return [layer for spec in specs if isinstance(spec, Residual) for layer in spec.body if isinstance(layer, kind)]


@pytest.mark.parametrize("arm", gqa.ARMS)
def test_only_the_attentions_heads_change_from_the_swiglu_model(arm: str):
    specs, swiglu = gqa.arm_specs(arm, 60), ffn.arm_specs("swiglu", 60)
    g = gqa.key_value_heads(arm)
    attentions = _layers(specs, Attention)
    assert len(attentions) == 2
    for attention, before in zip(attentions, _layers(swiglu, Attention), strict=True):
        assert attention.heads == 8 and attention.head_size(64) == 8
        assert attention.key_value_heads == (None if g == 8 else g)
        assert attention.causal and attention.positions == "rotary" and attention.document_mask
        assert {**vars(attention), "heads": before.heads, "key_value_heads": None} == vars(before)
    # D13: SwiGLU in every FFN, GatedDense(170)
    assert _layers(specs, GatedDense) == [GatedDense(170)] * 2
    assert [s for s in specs if not isinstance(s, Residual)] == [s for s in swiglu if not isinstance(s, Residual)]


def test_the_arms_and_their_key_value_heads():
    assert [gqa.key_value_heads(arm) for arm in gqa.ARMS] == [8, 4, 2, 1, 4, 2, 1]
    assert gqa.CORPORA == cls.LONG_FORM and gqa.SEEDS == 5 and gqa.TOKENS == 256 and gqa.UPTRAIN_EPOCHS == 2
    with pytest.raises(ValueError, match="unknown arm"):
        gqa.key_value_heads("g3")


def test_the_cache_a_token_is_two_layers_keys_and_values_at_g_heads_of_8():
    assert [gqa.cache_bytes_per_token(arm) for arm in gqa.SCRATCH_ARMS] == [2048, 1024, 512, 256]


def test_the_registered_comparisons():
    assert gqa.COMPARISONS == [
        ("G1 MHA < MQA", "g8", "g1", "one"),
        ("G2 GQA-2 vs MHA", "g2", "g8", "two"),
        ("G3 GQA-2 < MQA", "g2", "g1", "one"),
        ("U1 uptrained vs scratch at g 2", "up2", "g2", "two"),
    ]


def test_an_uptraining_run_needs_its_g8_network(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="hasn't ended"):
        gqa.train_run(("euclid-heath", "spread", "up2"), 0, 1, tmp_path, "numpy", limit=4)


def test_a_smoke_g8_run_saves_its_network_and_an_uptraining_run_starts_from_its_conversion(tmp_path: Path):
    scratch = gqa.train_run(("euclid-heath", "spread", "g8"), 0, 2, tmp_path, "numpy", limit=4)
    assert len(scratch["held_out_cross_entropies"]) == 2
    assert len(scratch["position_bits"]) == cls.POSITION_BUCKETS
    saved = json.loads(gqa.model_path(tmp_path, "euclid-heath", "spread", 0).read_text())
    assert "key_value_heads" not in json.dumps(saved["layers"])
    up = gqa.train_run(("euclid-heath", "spread", "up2"), 0, gqa.UPTRAIN_EPOCHS, tmp_path, "numpy", limit=4)
    assert len(up["held_out_cross_entropies"]) == gqa.UPTRAIN_EPOCHS
    # it started from the trained network: below a fresh g2 network's first epoch on the same windows
    fresh = gqa.train_run(("euclid-heath", "spread", "g2"), 0, 2, tmp_path, "numpy", limit=4)
    assert up["held_out_cross_entropies"][0] < fresh["held_out_cross_entropies"][0]


def test_the_report_reads_two_epoch_uptraining_runs():
    run = {
        "held_out_cross_entropies": [1.0, 0.9],
        "train_cross_entropies": [1.0, 0.9],
        "position_bits": [1.0] * cls.POSITION_BUCKETS,
        "epoch_seconds": [1.0, 1.0],
    }
    results: Any = {("euclid-heath", "spread", "up2"): [run] * 5, ("euclid-heath", "spread", "g2"): [run] * 5}
    out = gqa.report(results, 30)
    assert "up2" in out and "g8: 2048 cache bytes a token" in out
