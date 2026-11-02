"""
The FFN activations study's configuration against its workplan (stage 6, D4, D7, D11): each arm is
the context-length study's 256 model with only its FFNs changed, at equal parameters; the ReLU arm
is that model exactly; the control read by sha256; the registered comparisons; a smoke run's record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from indrajala_ml.model.specs.layer_specs import Dense, GatedDense, Residual
from indrajala_ml.studies import context_length_study as cls
from indrajala_ml.studies import ffn_activation_study as fas


def _hidden(specs: list[Any]) -> list[Any]:
    return [
        layer
        for spec in specs
        if isinstance(spec, Residual)
        for layer in spec.body
        if isinstance(layer, GatedDense) or (isinstance(layer, Dense) and not layer.bias)
    ]


def test_the_relu_arm_is_the_context_length_studys_256_model():
    assert fas.arm_specs("relu", 60) == cls.arm_specs("256", 60)
    assert fas.TOKENS == cls.context("256")


@pytest.mark.parametrize("arm", fas.ARMS)
def test_only_the_ffns_change(arm: str):
    specs, control = fas.arm_specs(arm, 60), cls.arm_specs("256", 60)
    assert len(specs) == len(control)
    hidden = _hidden(specs)
    assert len(hidden) == 2  # two blocks, each one FFN
    expected = {"relu": Dense(256, activation="relu"), "gelu": Dense(256, activation="gelu"), "swiglu": GatedDense(170)}
    assert all(layer == expected[arm] for layer in hidden)
    # everything else as the control's
    strip = [s for s in specs if not isinstance(s, Residual)]
    assert strip == [s for s in control if not isinstance(s, Residual)]


def test_the_arms_are_at_equal_parameters_d4():
    counts = {arm: fas.ffn_parameters(fas.arm_specs(arm, 60), 64) for arm in fas.ARMS}
    assert counts == {"relu": [33_088, 33_088], "gelu": [33_088, 33_088], "swiglu": [33_044, 33_044]}


def test_the_registered_comparisons():
    assert fas.COMPARISONS == [
        ("F1 SwiGLU < ReLU", "swiglu", "relu", "one"),
        ("F2 GELU < ReLU", "gelu", "relu", "one"),
        ("F3 SwiGLU vs GELU", "swiglu", "gelu", "two"),
    ]
    assert fas.NEW_ARMS == ["gelu", "swiglu"] and fas.SEEDS == 5


def test_the_claims_are_computed_over_the_long_form_corpora():
    high, low = {"held_out_cross_entropies": [1.0]}, {"held_out_cross_entropies": [0.9]}
    results: Any = {}
    for corpus in fas.CORPORA:
        for split in fas.SPLITS:
            results[(corpus, split, "relu")] = [high] * fas.SEEDS
            results[(corpus, split, "swiglu")] = [low] * fas.SEEDS
    claims = cls.claims(results, fas.COMPARISONS)
    assert "F1 SwiGLU < ReLU (one-sided): 30 of 30 pairs lower" in claims and "holds" in claims
    assert "F2 GELU < ReLU: no pairs" in claims


def test_the_control_is_refused_unless_its_hash_is_the_recorded_one(tmp_path: Path):
    path = tmp_path / "sweep-herodotus-rawlinson-muqaddimah.json"
    path.write_text(json.dumps({"provenance": {}, "results": []}))
    with pytest.raises(ValueError, match="sha256"):
        fas.read_control([path])


@pytest.mark.parametrize("arm", fas.NEW_ARMS)
def test_a_smoke_run_records_every_epoch_and_the_position_bits(arm: str):
    result = fas.train_run(("euclid-heath", "spread", arm), 0, 2, "numpy", limit=4)
    assert len(result["held_out_cross_entropies"]) == 2 and len(result["epoch_seconds"]) == 2
    assert len(result["position_bits"]) == cls.POSITION_BUCKETS
