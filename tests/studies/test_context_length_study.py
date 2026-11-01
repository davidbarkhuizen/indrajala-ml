"""
The context-length study's configuration against its workplan (stage 9, D2, D5, D6, D8): the arms'
windows and masks, the model (segments, the end-of-paragraph class), the token budget, the recipe's
schedule; a smoke run's record; and the registered claims computed from results as registered.
"""

from __future__ import annotations

from typing import Any

import pytest

from indrajala_ml.data import text_data
from indrajala_ml.model.specs.layer_specs import Attention, Dense, Embedding, Residual
from indrajala_ml.studies import context_length_study as cls


def _attentions(specs: list[Any]) -> list[Attention]:
    return [
        layer for spec in specs if isinstance(spec, Residual) for layer in spec.body if isinstance(layer, Attention)
    ]


@pytest.mark.parametrize("arm", cls.ARMS)
def test_each_arm_is_the_recipe_model_at_its_window_and_mask(arm: str):
    specs = cls.arm_specs(arm, 60)
    embedding, output = specs[0], specs[-1]
    assert isinstance(embedding, Embedding) and embedding.segments and embedding.vocabulary == 62
    assert isinstance(output, Dense) and output.size == 61  # the end-of-paragraph token predicted
    attentions = _attentions(specs)
    assert len(attentions) == 2 and all(a.positions == "rotary" and a.causal and a.heads == 4 for a in attentions)
    assert all(a.document_mask == (not arm.endswith("unmasked")) for a in attentions)
    assert cls.context(arm) in (64, 256, 1024)
    # the token budget: every arm's steps hold 2,048 tokens (D5)
    assert text_data.budget_batch_size(cls.context(arm)) * cls.context(arm) == 2048


def test_the_schedule_warms_up_over_the_first_epoch_and_ends_at_a_tenth():
    schedule = cls.schedule(0.002, 10, 30)
    assert schedule(0) < schedule(9) <= 0.002
    assert schedule(10) == pytest.approx(0.002) and schedule(299) == pytest.approx(0.0002, rel=1e-2)


def test_the_claims_cover_the_long_form_corpora_and_are_holm_corrected():
    # every long-form pair lower at 1,024 than at 64: C1 holds; nothing else present
    run = {"held_out_cross_entropies": [1.0]}
    low = {"held_out_cross_entropies": [0.9]}
    results: Any = {}
    for corpus in cls.CORPORA:
        for split in cls.SPLITS:
            results[(corpus, split, "64")] = [run] * cls.SEEDS
            results[(corpus, split, "1024")] = [low] * cls.SEEDS
    claims = cls.claims(results)
    assert "C1 1024 < 64 (one-sided): 30 of 30 pairs lower" in claims and "holds" in claims
    assert "C2 256 < 64: no pairs" in claims


def test_a_smoke_run_records_every_epoch_and_the_position_bits():
    result = cls.train_run(("euclid-heath", "spread", "256"), 0, 2, "numpy", limit=4)
    assert len(result["held_out_cross_entropies"]) == 2 and len(result["epoch_seconds"]) == 2
    assert result["batch"] == 8 and len(result["position_bits"]) == cls.POSITION_BUCKETS
