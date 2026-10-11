"""The attention-dropout study's arms, runs and report (indrajala_ml/studies/attention_dropout_study.py)."""

import random

import pytest

from indrajala_ml.data.text_data import Split
from indrajala_ml.model.specs.layer_specs import Attention, Dropout, Residual, expand_specs
from indrajala_ml.model.specs.spec_validation import validate_layer_specs
from indrajala_ml.studies import attention_dropout_study as study
from indrajala_ml.studies import sequence_study


def _dropouts(arm: str) -> tuple[list[float], list[float]]:
    expanded = expand_specs(study.arm_specs(arm, 65))
    attention = [spec.dropout for spec in expanded if isinstance(spec, Attention)]
    return attention, [spec.p for spec in expanded if isinstance(spec, Dropout)]


def test_the_control_is_the_sequence_studys_2_layer_model():
    assert study.arm_specs("control", 65) == sequence_study.arm_specs("2-layer", 65)


@pytest.mark.parametrize(
    ("arm", "attention", "residual"),
    [
        ("attention 0.1", [0.1, 0.1], []),
        # after the position, then ending each of the four blocks' bodies
        ("residual 0.1", [0.0, 0.0], [0.1] * 5),
        ("both 0.1", [0.1, 0.1], [0.1] * 5),
        ("both 0.2", [0.2, 0.2], [0.2] * 5),
    ],
)
def test_each_arm_drops_out_in_gpts_places(arm: str, attention: list[float], residual: list[float]):
    specs = study.arm_specs(arm, 65)
    validate_layer_specs(specs)
    assert _dropouts(arm) == (attention, residual)
    blocks = [spec for spec in specs if isinstance(spec, Residual)]
    assert all(isinstance(block.body[-1], Dropout) for block in blocks) == bool(residual)


def test_an_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="unknown arm"):
        study.arm_specs("both 0.3", 65)


def _probe(split_by: Split) -> None:
    rng = random.Random(0)

    def window() -> tuple[tuple[float, ...], tuple[int, ...]]:
        ids = [rng.randrange(5) for _ in range(sequence_study.CONTEXT + 1)]
        return tuple(float(i) for i in ids[:-1]), tuple(ids[1:])

    sequence_study._datasets.clear()  # pyright: ignore[reportPrivateUsage]
    sequence_study._datasets[("probe", 1, split_by)] = (  # pyright: ignore[reportPrivateUsage]
        [window() for _ in range(40)],
        [window() for _ in range(8)],
        5,
    )


def test_a_run_records_every_epoch_repeats_from_its_seed_and_reports_its_gap():
    _probe("spread")
    context = {"limit": 1, "epochs": 2}
    config: study.Config = ("probe", "spread", "both 0.2")
    first, second = study.run_config(context, config, 0), study.run_config(context, config, 0)
    assert len(first["held_out_cross_entropies"]) == len(first["train_cross_entropies"]) == 2
    assert first["held_out_cross_entropies"] == second["held_out_cross_entropies"]
    control = study.run_config(context, ("probe", "spread", "control"), 0)
    # dropout changes the training; the evaluation is in inference either way
    assert first["held_out_cross_entropies"] != control["held_out_cross_entropies"]

    results: study.Results = {("probe", "spread", "control"): [control], config: [first]}
    text = study.report(list(results), results, 2)
    assert text.startswith("probe, spread") and "both 0.2" in text and "against control" in text
    expected = sequence_study.bits(first["held_out_cross_entropies"][-1] - first["train_cross_entropies"][-1])
    assert study.gap(first) == expected
