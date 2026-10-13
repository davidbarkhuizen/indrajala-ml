"""scripts/window_recurrent_study.py's check and certify commands on a saved network."""

import json
import sys
from pathlib import Path

import pytest

from indrajala_ml.studies import window_recurrent_study as wr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import window_recurrent_study as script  # scripts/ isn't a package


@pytest.fixture
def saved(tmp_path: Path) -> Path:
    path = tmp_path / "curriculum-k1-s4-m3-seed0" / "step-1.network.json"
    path.parent.mkdir()
    wr.build(1, 4, 3, seed=0).save(str(path))
    return path


def test_check_runs_the_long_tests_as_asked(saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    out = tmp_path / "check.json"
    argv = ["check", str(saved), "--n", "16", "--properties", "B1", "M3", "--sampled", "30"]
    assert script.main([*argv, "--per-stratum", "2", "--doubling", "--json", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in lines] == ["n 16, B1", "n 16, M3"]
    result = json.loads(out.read_text())["lengths"]["16"]
    assert result["properties"]["B1"]["cases"] == 30
    assert sorted(map(int, result["properties"]["M3"]["strata"])) == [1, 2, 4, 8, 16]


def test_certify_writes_each_networks_certificate(saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    out = tmp_path / "certificates"
    assert script.main(["certify", str(saved), "--out", str(out)]) == 0
    assert capsys.readouterr().out.startswith("curriculum-k1-s4-m3-seed0-step-1: ")
    certificate = json.loads((out / "curriculum-k1-s4-m3-seed0-step-1.json").read_text())
    assert certificate["network"] == str(saved) and len(certificate["sha256"]) == 64
    assert certificate["codes"] == 3 and certificate["evaluations"] == 27 * len(certificate["reachable"])
