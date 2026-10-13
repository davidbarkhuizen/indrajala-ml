"""scripts/window_recurrent_study.py's check and certify commands on a saved network."""

import json
import sys
from pathlib import Path

import pytest

from indrajala_ml.studies import window_recurrent_study as wr
from indrajala_ml.studies import window_tasks as wt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import window_recurrent_study as script  # scripts/ isn't a package


@pytest.fixture
def saved(tmp_path: Path) -> Path:
    path = tmp_path / "curriculum-k1-s4-m3-seed0" / "step-1.network.json"
    path.parent.mkdir()
    wr.build(wt.ADDITION, 1, 4, 3, seed=0).save(str(path))
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


def test_certify_strict_records_its_snap_ratio(saved: Path, tmp_path: Path):
    out = tmp_path / "strict"
    assert script.main(["certify", str(saved), "--out", str(out), "--snap-ratio", "1"]) == 0
    certificate = json.loads((out / "curriculum-k1-s4-m3-seed0-step-1.json").read_text())
    assert certificate["snap_ratio"] == 1.0 and certificate["snap"]["short"] == 0


@pytest.fixture
def saved_division(tmp_path: Path) -> Path:
    path = tmp_path / "modular-division-5-curriculum-k1-s4-m4-seed0" / "step-1.network.json"
    path.parent.mkdir()
    wr.build(wt.ModularDivision(5), 1, 4, 4, seed=0).save(str(path))
    return path


def test_check_and_certify_a_division_network(saved_division: Path, tmp_path: Path):
    out = tmp_path / "check.json"
    argv = ["check", str(saved_division), "--task", "modular-division-5", "--n", "16", "--properties", "U1", "S2"]
    assert script.main([*argv, "--sampled", "30", "--per-stratum", "2", "--doubling", "--json", str(out)]) == 0
    result = json.loads(out.read_text())["lengths"]["16"]["properties"]
    assert result["U1"]["cases"] == 30 and sorted(map(int, result["S2"]["strata"])) == [1, 2, 4, 8, 16]
    certificates = tmp_path / "certificates"
    assert (
        script.main(["certify", str(saved_division), "--task", "modular-division-5", "--out", str(certificates)]) == 0
    )
    certificate = json.loads((certificates / "modular-division-5-curriculum-k1-s4-m4-seed0-step-1.json").read_text())
    assert certificate["task"] == "modular-division-5" and certificate["evaluations"] == 9 * len(
        certificate["reachable"]
    )
