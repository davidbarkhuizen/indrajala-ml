"""The training-recipe study's arms, runs, resumption, report and claims (indrajala_ml/studies/lr_schedule_study.py)."""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from indrajala_ml.studies import lr_schedule_study as study
from indrajala_ml.studies import rotary_study
from indrajala_ml.training.lr_schedule import Constant, CosineDecay, Warmup
from tests.studies.test_attention_dropout_study import _probe  # pyright: ignore[reportPrivateUsage]


def test_the_cosine_arms_warm_up_for_an_epoch_then_decay_to_a_tenth_at_the_last_step():
    assert study.schedule("constant", 0.002, 100, 30) == Constant(0.002)
    cosine = study.schedule("cosine", 0.002, 100, 30)
    assert cosine == study.schedule("recipe", 0.002, 100, 30) == Warmup(100, CosineDecay(0.002, 2900, 0.0002))
    assert cosine(99) == pytest.approx(0.002)
    assert cosine(2999) == pytest.approx(0.0002, rel=1e-5)
    assert cosine(3000) == 0.0002


def test_only_the_recipe_decays_and_clips():
    assert study.rule("constant").weight_decay == study.rule("cosine").weight_decay == 0.0
    assert study.rule("recipe").weight_decay == 0.1
    assert study.clip_norm("recipe") == 1.0 and study.clip_norm("cosine") is None


def test_every_corpus_has_its_tuned_2_layer_rate():
    assert all((corpus, "2-layer") in rotary_study.RATES for corpus in study.CORPORA)


def test_a_run_records_every_epoch_repeats_from_its_seed_and_is_written_and_resumed(tmp_path: Path):
    _probe("contiguous")
    rotary_study.RATES[("probe", "2-layer")] = 0.004
    try:
        context = {"limit": 1, "epochs": 2, "out_dir": str(tmp_path)}
        runs = {arm: study.run_config(context, ("probe", "contiguous", arm), 1) for arm in study.ARMS}
        for run in runs.values():
            assert len(run["held_out_cross_entropies"]) == len(run["train_cross_entropies"]) == 2
            assert run["steps_per_epoch"] == math.ceil(40 / 32)
        assert len({tuple(run["held_out_cross_entropies"]) for run in runs.values()}) == 3  # the arms differ

        written = tmp_path / "runs" / "probe-contiguous-recipe-seed1.json"
        assert json.loads(written.read_text()) == runs["recipe"]
        # a sweep started again reads the run back rather than training it
        written.write_text(json.dumps({**runs["recipe"], "resumed": True}))
        assert study.run_config(context, ("probe", "contiguous", "recipe"), 1)["resumed"] is True

        fresh = study.train_run(("probe", "contiguous", "recipe"), 1, 2, 1, None)
        assert fresh["held_out_cross_entropies"] == runs["recipe"]["held_out_cross_entropies"]
        # time's one epoch, of the study's 30-epoch schedule: the first epoch of a 2-epoch run's
        # schedule differs from it only after warmup, which a one-epoch run never leaves
        one = study.train_run(("probe", "contiguous", "cosine"), 1, 1, 1, None, schedule_epochs=study.EPOCHS)
        assert one["held_out_cross_entropies"] == runs["cosine"]["held_out_cross_entropies"][:1]
    finally:
        del rotary_study.RATES[("probe", "2-layer")]


def _run(held_out: float, epochs: int = 4) -> dict[str, Any]:
    return {
        "held_out_cross_entropies": [held_out + 0.1 * (epochs - e) for e in range(epochs)],
        "held_out_accuracies": [0.5] * epochs,
        "train_cross_entropies": [1.0] * epochs,
        "epoch_seconds": [2.0] * epochs,
    }


def _results(cosine_lower: int, recipe_lower: int) -> study.Results:
    # 8 cells of 5 seeds: cosine below constant on cosine_lower of the 40 pairs, recipe below cosine
    # on recipe_lower
    results: study.Results = {}
    pair = 0
    for corpus in study.CORPORA:
        for split in study.SPLITS:
            constant: list[dict[str, Any]] = []
            cosine: list[dict[str, Any]] = []
            recipe: list[dict[str, Any]] = []
            for _seed in range(5):
                constant.append(_run(2.0))
                cosine.append(_run(1.9 if pair < cosine_lower else 2.1))
                recipe.append(_run(1.8 if pair < recipe_lower else 2.2) if pair < cosine_lower else _run(2.0))
                pair += 1
            results[(corpus, split, "constant")] = constant
            results[(corpus, split, "cosine")] = cosine
            results[(corpus, split, "recipe")] = recipe
    return results


def test_the_claims_are_the_registered_sign_tests_after_holm():
    assert study.sign_test(26, 40, "one") < 0.05 < study.sign_test(25, 40, "one")
    assert study.sign_test(27, 40, "two") < 0.05 < study.sign_test(26, 40, "two")
    assert study.sign_test(13, 40, "two") < 0.05 < study.sign_test(14, 40, "two")
    assert study.holm([0.01, 0.04]) == [0.02, 0.04]
    assert study.holm([0.04, 0.03]) == [0.06, 0.06]

    text = study.claims(_results(40, 20))
    assert "40 of 40 pairs lower" in text and "holds" in text and "undecided" in text
    assert "refuted" in study.claims(_results(20, 20))


def test_a_report_sets_each_arm_against_the_one_before_it(tmp_path: Path):
    results = _results(40, 40)
    text = study.report(results, 4)
    assert all(corpus in text for corpus in study.CORPORA)
    assert study.report({("probe", "contiguous", "recipe"): [_run(2.0, epochs=1)]}, 1).startswith("probe")
    assert "-0.144 ± +0.000 vs constant" in text and "-0.144 ± +0.000 vs cosine" in text

    halves = [
        {k: v for k, v in results.items() if k[0] in corpora} for corpora in (study.CORPORA[:2], study.CORPORA[2:])
    ]
    paths: list[str] = []
    for i, half in enumerate(halves):
        path = tmp_path / f"half{i}.json"
        path.write_text(
            json.dumps({"provenance": {}, "results": [{"config": [*c], "runs": r} for c, r in half.items()]})
        )
        paths.append(str(path))
    assert study._merged(paths) == results  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ValueError, match="two result files"):
        study._merged([paths[0], paths[0]])  # pyright: ignore[reportPrivateUsage]
