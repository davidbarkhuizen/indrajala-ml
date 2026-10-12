"""The rotary-positions study's arms, runs and report (indrajala_ml/studies/rotary_study.py)."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from indrajala_ml.model.specs.layer_specs import Attention, Position, expand_specs
from indrajala_ml.model.specs.spec_validation import validate_layer_specs
from indrajala_ml.studies import rotary_study as study
from indrajala_ml.studies import sequence_study
from indrajala_ml.studies.common import read_results
from tests.studies.test_attention_dropout_study import _probe  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("arm", study.ARMS)
def test_an_arm_is_the_sequence_studys_without_its_position_and_every_attention_rotary(arm: str):
    learned = sequence_study.arm_specs(arm, 65)
    rotary = study.arm_specs(arm, 65)
    validate_layer_specs(rotary)
    assert study.arm_specs(arm, 65, "learned") == learned
    assert not any(isinstance(spec, Position) for spec in rotary)
    assert len(rotary) == len(learned) - 1
    attentions = [spec for spec in expand_specs(rotary) if isinstance(spec, Attention)]
    assert attentions and all(a.positions == "rotary" and a.rotary_base == 10000.0 for a in attentions)
    assert [a.heads for a in attentions] == [sequence_study.HEADS] * len(attentions)


def test_every_corpus_and_arm_has_a_rate_and_a_recorded_baseline():
    keys = {(corpus, arm) for corpus in study.CORPORA for arm in study.ARMS}
    assert set(study.RATES) == keys == set(study.RECORDED)


def test_an_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="unknown arm"):
        study.arm_specs("ffn", 65)


def test_a_run_records_every_epoch_repeats_from_its_seed_and_differs_from_learned():
    _probe("contiguous")
    study.RATES[("probe", "1-layer")] = 0.004
    try:
        context = {"limit": 1, "epochs": 2}
        config: study.Config = ("probe", "1-layer", "rotary")
        first, second = study.run_config(context, config, 0), study.run_config(context, config, 0)
        assert len(first["held_out_cross_entropies"]) == len(first["train_cross_entropies"]) == 2
        assert first["held_out_cross_entropies"] == second["held_out_cross_entropies"]
        learned = study.run_config(context, ("probe", "1-layer", "learned"), 0)
        assert first["held_out_cross_entropies"] != learned["held_out_cross_entropies"]
        # the position table, 64 rows of 64 features
        assert learned["parameter_count"] - first["parameter_count"] == 64 * sequence_study.TOKEN_SIZE

        results: study.Results = {("probe", "1-layer", "learned"): [learned], config: [first]}
        text = study.report(list(results), results, 2)
        assert text.startswith("probe") and "rotary" in text and "learned" in text
    finally:
        del study.RATES[("probe", "1-layer")]


def _run(held_out: float, epochs: int = 10) -> dict[str, Any]:
    return {
        "held_out_cross_entropies": [held_out] * epochs,
        "held_out_accuracies": [0.5] * epochs,
        "train_cross_entropies": [1.0] * epochs,
        "epoch_seconds": [2.0] * epochs,
        "parameter_count": 100,
    }


def test_a_report_sets_rotary_against_learned_seed_by_seed():
    learned, rotary = ("euclid-heath", "2-layer", "learned"), ("euclid-heath", "2-layer", "rotary")
    # nats: rotary lower by 0.1 on the first seed and by 0.3 on the second
    results: study.Results = {learned: [_run(1.0), _run(1.2)], rotary: [_run(0.9), _run(0.9)]}
    text = study.report(list(results), results, 10)
    # per seed, in bits: -0.144 and -0.433, mean -0.289 ± 0.204
    assert "-0.289 ± +0.204" in text  # mean_sd signs both, as the attention-dropout report


def _control_file(path: Path, corpora: list[str]) -> str:
    entries = [{"config": [c, "contiguous", "control"], "runs": [_run(1.0)]} for c in corpora]
    entries += [{"config": [c, "spread", "control"], "runs": [_run(2.0)]} for c in corpora]
    entries += [{"config": [c, "contiguous", "both 0.1"], "runs": [_run(3.0)]} for c in corpora]
    path.write_text(json.dumps(entries))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_control_file_gives_the_2_layer_learned_runs_of_its_contiguous_control(tmp_path: Path):
    path = tmp_path / "sweep.json"
    digest = _control_file(path, study.CONTROL_CORPORA)
    control = study.read_control(path, digest)
    assert set(control) == {(c, "2-layer", "learned") for c in study.CONTROL_CORPORA}
    assert all(runs == [_run(1.0)] for runs in control.values())


def test_a_control_file_with_another_hash_or_a_corpus_missing_is_refused(tmp_path: Path):
    path = tmp_path / "sweep.json"
    _control_file(path, study.CONTROL_CORPORA)
    with pytest.raises(ValueError, match="sha256"):
        study.read_control(path)
    digest = _control_file(path, study.CONTROL_CORPORA[:2])
    with pytest.raises(ValueError, match="contiguous controls"):
        study.read_control(path, digest)


def test_a_learned_row_off_the_recorded_means_is_named():
    def recorded(corpus: str, arm: str) -> list[dict[str, Any]]:
        # held-out nats whose bits are the recorded means at epochs 1, 5 and 10
        means = [mean for mean, _sd in study.RECORDED[(corpus, arm)][:3]]  # type: ignore[misc]
        curve = [means[0]] * 4 + [means[1]] * 5 + [means[2]]
        return [{**_run(0.0), "held_out_cross_entropies": [b * math.log(2) for b in curve]}]

    results: study.Results = {
        ("euclid-heath", "2-layer", "learned"): recorded("euclid-heath", "2-layer"),
        ("euclid-heath", "1-layer", "learned"): [_run(1.0)],
        ("euclid-heath", "1-layer", "rotary"): [_run(1.0)],
    }
    assert study.unreproduced(results) == [
        "euclid-heath 1-layer: [1.443, 1.443, 1.443], recorded [2.018, 1.682, 1.622]"
    ]


def test_the_sweep_reruns_only_the_learned_runs_the_control_lacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AB_RUNS_ROOT", str(tmp_path / "ab-runs"))
    control = {(c, "2-layer", "learned"): [_run(1.0)] for c in study.CONTROL_CORPORA}

    def read_control(_path: str) -> study.Results:
        return control

    def unreproduced(_results: study.Results) -> list[str]:
        return []

    monkeypatch.setattr(study, "read_control", read_control)
    monkeypatch.setattr(study, "unreproduced", unreproduced)
    ran: list[study.Config] = []

    def sweep(configs: list[study.Config], *_args: Any, **_kwargs: Any) -> study.Results:
        ran.extend(configs)
        return {config: [_run(1.0)] for config in configs}

    monkeypatch.setattr(study, "run_parameter_sweep", sweep)
    out = tmp_path / "sweep.json"
    study.main(["sweep", "--control", "control.json", "--seeds", "1", "--out", str(out)])
    learned = [config for config in ran if config[2] == "learned"]
    assert sorted(learned) == sorted(
        [(c, "1-layer", "learned") for c in study.CORPORA] + [("muqaddimah", "2-layer", "learned")]
    )
    assert len([config for config in ran if config[2] == "rotary"]) == 8
    written = read_results(out)
    assert len(written) == 16 and {tuple(e["config"]) for e in written} >= set(control)
