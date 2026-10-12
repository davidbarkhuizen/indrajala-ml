"""
The window-recurrent adder's harness (indrajala_ml/studies/window_recurrent_study.py): the encoding
at an alignment through the final window, the curriculum's examples and batches, the checks and the
break length against stand-in networks that add (or carry only inside a window), and a real
network's run, its steps saved and resumed.
"""

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from indrajala_ml.data import addition_data as ad
from indrajala_ml.studies import window_recurrent_study as wr

# small catalogue sizes, and test lengths past A3's and M1's exhaustive spaces, so a check is quick
SMALL = wr.Settings(
    epoch_examples=64,
    max_epochs=3,
    patience=1,
    screen=20,
    steps=2,
    test_lengths=(8, 14, 16),
    sampled=200,
    relation_sampled=20,
    per_stratum=10,
    codes_cases=20,
    seeds=1,
)


class _Backend:
    def matrix(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        return [list(row) for row in rows]


class AddingNetwork:
    """
    A stand-in that reads the column tokens and adds them, the carry passed along every column, or,
    with window set, reset at each window's start (a network whose state carries nothing).
    """

    backend = _Backend()

    def __init__(self, window: int | None = None) -> None:
        self.window = window
        self.batches: list[list[Any]] = []

    def prepare_dataset(self, rows: Sequence[tuple[tuple[float, ...], object]]) -> list[tuple[float, ...]]:
        return [tokens for tokens, _labels in rows]

    def classify_rows(self, prepared: Sequence[tuple[float, ...]]) -> list[tuple[int, ...]]:
        predictions: list[tuple[int, ...]] = []
        for tokens in prepared:
            carry, digits = 0, list[int]()
            for t, token in enumerate(int(x) for x in tokens):
                if self.window is not None and t % self.window == 0:
                    carry = 0
                total = token // 9 + token // 3 % 3 + token % 3 + carry
                digits.append(total % 3)
                carry = total // 3
            predictions.append(tuple(digits))
        return predictions

    def state_codes(self, X: Sequence[Sequence[float]]) -> list[list[int]]:
        return [[0, len(row) % 5] for row in X]

    def learn_batch(self, rate: float, batch: list[Any]) -> None:
        self.batches.append(batch)


# the encoding


def test_an_example_is_its_columns_at_the_alignment_through_one_more_window():
    triple = (5, 1, 0)  # 12 + 01 + 00 in base 3, least significant first: (2, 1), (1, 0)
    tokens, labels = wr.example(triple, 2, 3, 2)
    # 2 zero columns, the 2 columns, 2 more to fill 2 windows, then the final window
    assert wr.data_windows(2, 2, 3) == 2
    assert tokens == (0.0, 0.0, 9 * 2 + 3 * 1 + 0, 9 * 1 + 0 + 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert labels == (0, 0, *ad.digits(6, 7))


@pytest.mark.parametrize("k", [1, 2, 4])
def test_lengths_fill_exactly_their_window_count(k: int):
    for windows in (1, 2, 3):
        for a in range(k):
            lengths = wr.lengths_for(windows, a, k)
            assert len(lengths) > 0
            assert all(wr.data_windows(n, a, k) == windows for n in lengths)
            assert wr.data_windows(lengths[0] - 1, a, k) < windows or lengths[0] == 1


def test_a_cases_alignment_is_its_own_whatever_the_order():
    rng = random.Random(3)
    triples = [ad.uniform_triple(8, rng) for _ in range(400)]
    assert {wr.alignment(t, 8, 1) for t in triples} == {0}
    assert {wr.alignment(t, 8, 4) for t in triples} == {0, 1, 2, 3}
    for x, y, z in triples[:20]:
        assert wr.alignment((x, y, z), 8, 4) == wr.alignment((z, x, y), 8, 4)
    # independent of D8's tenth: the held-out triples take every alignment too
    held = [t for t in (ad.uniform_triple(8, rng) for _ in range(4000)) if ad.held_out(t, 8)]
    assert {wr.alignment(t, 8, 4) for t in held} == {0, 1, 2, 3}


def test_an_epoch_shares_its_examples_equally_among_the_window_counts():
    examples = wr.training_examples([1, 2, 3], 2, 300, random.Random(5))
    widths = [len(tokens) for tokens, _labels in examples]
    # data windows w, then the final window: (w + 1) * k columns
    assert {width: widths.count(width) for width in set(widths)} == {4: 100, 6: 100, 8: 100}
    adder = AddingNetwork()
    for (tokens, labels), digits in zip(examples, adder.classify_rows([t for t, _ in examples]), strict=True):
        assert labels == digits
        assert len(labels) == len(tokens)


def test_an_epochs_batches_each_hold_one_window_count():
    network = AddingNetwork()
    examples = wr.training_examples([1, 2, 3], 2, 300, random.Random(5))
    wr.train_epoch(network, examples, 16, 0.01, random.Random(6))
    assert all(len({len(tokens) for tokens, _ in batch}) == 1 for batch in network.batches)
    assert sorted(map(str, (e for b in network.batches for e in b))) == sorted(map(str, examples))


def test_short_lengths_train_on_every_triple():
    # below n = 3 nothing is held out, so B1 there can pass whole (W24)
    examples = wr.training_examples([1], 1, 3000, random.Random(1))
    seen = {tokens for tokens, _labels in examples}
    assert len(seen) == 27


# the checks


@pytest.mark.parametrize("k", [1, 3])
def test_an_adding_network_passes_the_catalogue_at_any_alignment(k: int):
    model = wr.WindowAdder(AddingNetwork(), k)
    for n in (2, 8):
        results = ad.evaluate(model.adder(n), n, wr.check_properties(n, SMALL.catalogue()), shrink=False)
        assert all(r.holds for r in results), [r.id for r in results if not r.holds]


def test_a_network_carrying_nothing_across_windows_fails_the_carry_at_distance():
    model = wr.WindowAdder(AddingNetwork(window=2), 2)
    results = {r.id: r for r in ad.evaluate(model.adder(8), 8, wr.test_properties(SMALL.catalogue()), shrink=False)}
    assert not results["M3"].holds and not results["B1"].holds
    assert results["M3"].reach() < 8


def test_b3_is_checked_from_three_columns_and_never_at_a_test_length():
    catalogue = SMALL.catalogue()
    assert "B3" not in {p.id for p in wr.check_properties(2, catalogue)}
    assert "B3" in {p.id for p in wr.check_properties(3, catalogue)}
    assert "B3" not in {p.id for p in wr.test_properties(catalogue)}


def test_the_break_length_is_the_last_of_the_passing_prefix():
    assert wr.break_length({8: True, 10: True, 12: False, 16: True}) == 10
    assert wr.break_length({8: False, 10: True}) == 0
    assert wr.break_length({8: True, 10: True}) == 10


def test_a_step_with_an_adding_network_passes_at_once_and_breaks_at_the_longest_test():
    spec = wr.RunSpec("curriculum", 1, 4, 8, 0)
    result = wr.run_step(AddingNetwork(), spec, 2, [1, 2], SMALL)
    assert result["stop"] == "passed" and result["epochs"] == 1 and result["passes"]
    assert result["lengths"] == [1, 2]
    assert result["break"] == 16 and result["window_ratio"] == 16 / 2
    assert result["examples"] == SMALL.epoch_examples
    assert all(result["tests"][str(n)]["reach"] == n for n in SMALL.test_lengths)
    assert 0 in result["codes_used"] and set(result["codes_used"]) <= set(range(5))


def test_a_step_with_a_network_that_never_learns_stops_on_the_plateau():
    spec = wr.RunSpec("curriculum", 1, 4, 8, 0)
    result = wr.run_step(AddingNetwork(window=1), spec, 2, [1, 2], SMALL)
    assert result["stop"] == "plateau" and result["epochs"] == 2 and not result["passes"]
    assert result["break"] == 0 and result["window_ratio"] == 0


def test_the_arms_steps():
    curriculum, control = wr.RunSpec("curriculum", 2, 4, 8, 0), wr.RunSpec("all-at-once", 2, 4, 8, 0)
    assert curriculum.step_counts(SMALL) == [[1], [1, 2]]
    assert control.step_counts(SMALL) == [[1, 2]]


# a real network's run


def test_a_run_trains_each_step_saves_it_and_resumes_after_the_last_saved(tmp_path: Path):
    settings = wr.Settings(
        epoch_examples=64,
        max_epochs=1,
        patience=1,
        screen=20,
        steps=2,
        test_lengths=(3,),
        sampled=50,
        relation_sampled=10,
        per_stratum=5,
        codes_cases=20,
        seeds=1,
    )
    study = wr.Study(tmp_path, settings, [wr.RunSpec("curriculum", 1, 4, 3, 0)], 0)
    lines: list[str] = []
    results = wr.run(study, log=lines.append)
    key = study.runs[0].key
    result = results[key]
    assert [s["counts"] for s in result["steps"]] == [[1], [1, 2]]
    assert result["parameters"] > 0 and len(result["crate"]) == 12
    assert set(result["steps"][1]["codes_used"]) <= set(range(3))
    assert len(lines) == 1 and lines[0].startswith(f"{key}: ") and "step 2 " in lines[0]
    step_one = (tmp_path / "models" / key / "step-1.json").read_text()
    # a stopped run: its result and last step gone; the rerun keeps step 1 and trains step 2 again
    (tmp_path / "results" / f"{key}.json").unlink()
    (tmp_path / "models" / key / "step-2.json").unlink()
    rerun = wr.run(study, log=lambda line: None)[key]
    assert (tmp_path / "models" / key / "step-1.json").read_text() == step_one
    assert rerun["steps"][1] == {**result["steps"][1], **{k: rerun["steps"][1][k] for k in _TIMES}}
    assert "s/epoch train" in wr.report(study)
    assert wr.status(study).startswith("1 of 1 runs finished")


_TIMES = ("train_seconds", "screen_seconds", "check_seconds", "tests")


def test_a_run_that_raises_is_recorded_and_run_once_more(tmp_path: Path):
    calls: list[str] = []

    def train(spec: wr.RunSpec, settings: wr.Settings, models: Path) -> dict[str, Any]:
        calls.append(spec.key)
        if len(calls) == 1:
            raise RuntimeError("a run that dies")
        return {"key": spec.key, "steps": [], "works": False}

    study = wr.Study(tmp_path, SMALL, [wr.RunSpec("curriculum", 1, 4, 3, 0)], 0, train)
    lines: list[str] = []
    wr.run(study, log=lines.append)
    assert len(calls) == 2 and lines[0].endswith("error (attempt 1)")
    assert json.loads((tmp_path / "results" / f"{calls[0]}.json").read_text())["attempts"] == 2


def test_a_config_names_the_matrix(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "directory": str(tmp_path / "study"),
                "workers": 2,
                "steps": 2,
                "seeds": 2,
                "test_lengths": [8, 16],
                "matrix": [{"k": [1, 2], "s": [4], "m": [4, 8]}, {"arm": "all-at-once", "k": [2], "s": [4], "m": [8]}],
            }
        )
    )
    study = wr.load_config(path)
    assert study.workers == 2 and study.settings.steps == 2 and study.settings.test_lengths == (8, 16)
    assert len(study.runs) == 2 * 2 * 2 + 2
    assert study.runs[-1].key == "all-at-once-k2-s4-m8-seed1"
