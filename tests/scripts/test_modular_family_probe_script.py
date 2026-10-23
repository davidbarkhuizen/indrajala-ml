"""scripts/modular_family_probe.py's run command: which runs a pool trains, and where."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import modular_family_probe as script  # scripts/ isn't a package

TINY = ["--factors", "1", "--budget-epochs", "1", "--workers", "2"]


def test_runs_trains_the_chosen_task_and_seed_pairs_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AB_RUNS_ROOT", str(tmp_path / "locks"))  # the study's machine lock, not the machine's
    out = tmp_path / "out"
    assert script.main(["run", "--runs", "word-problem-s3:1", "move-to-front-3:0", *TINY, "--out", str(out)]) == 0
    trained = sorted(str(log.parent.relative_to(out)) for log in out.glob("*/*/log.jsonl"))
    assert trained == ["move-to-front-3/seed0", "word-problem-s3/seed1"]


def test_runs_names_its_tasks_so_task_is_refused(tmp_path: Path):
    with pytest.raises(SystemExit):
        script.main(["run", "--runs", "word-problem-s3:1", "--task", "word-problem-s3", "--out", str(tmp_path)])


def test_run_needs_an_epoch_budget_and_has_no_wall_clock_one_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("AB_RUNS_ROOT", str(tmp_path / "locks"))
    # epochs are the binding budget (the owner, 2026-10-23); --budget-minutes is a runaway guard alone
    with pytest.raises(SystemExit):
        script.main(["run", "--runs", "word-problem-s3:1", "--factors", "1", "--out", str(tmp_path)])
    out = tmp_path / "out"
    assert script.main(["run", "--runs", "word-problem-s3:1", *TINY, "--out", str(out)]) == 0
    saved = json.loads((out / "seed1" / "checkpoint.json").read_text())
    assert "budget_seconds" not in saved["run"] and saved["run"]["budget_epochs"] == 1  # a guard, not identity
