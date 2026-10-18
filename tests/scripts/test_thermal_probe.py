"""scripts/thermal_probe.py: workers under a thermal watch, stopped over the limit."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import thermal_probe as probe  # scripts/ isn't a package


@pytest.fixture
def idle_workers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # sleeping workers in place of the crate's product; the machine lock in a temporary root
    monkeypatch.setattr(probe, "worker_command", lambda: [sys.executable, "-c", "import time; time.sleep(60)"])
    monkeypatch.setattr(probe, "throttled", lambda: 0)
    monkeypatch.setenv("AB_RUNS_ROOT", str(tmp_path / "locks"))


def _readings(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> None:
    readings = iter(values)
    monkeypatch.setattr(probe.ab, "cpu_temp_c", lambda: next(readings, values[-1]))


@pytest.mark.usefixtures("idle_workers")
def test_a_probe_within_its_limit_runs_its_time_and_records_the_peak(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _readings(monkeypatch, [45.0, 52.0, 58.5, 57.0])
    out = tmp_path / "probe.json"
    argv = ["--cores", "2", "--minutes", "0.005", "--every", "0.05", "--out", str(out)]
    assert probe.main(argv) == 0
    result = json.loads(out.read_text())
    assert result["started_at_c"] == 45.0 and result["peak_c"] == 58.5 and not result["stopped"]
    assert result["cores"] == 2 and result["ran_s"] >= 0.3 and result["throttled_before"] == 0


@pytest.mark.usefixtures("idle_workers")
def test_a_probe_over_its_limit_stops_every_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _readings(monkeypatch, [45.0, 60.0, 65.0, 65.4])  # 65.0 runs on (inclusive), 65.4 stops it
    out = tmp_path / "probe.json"
    argv = ["--cores", "2", "--minutes", "1", "--every", "0.01", "--out", str(out)]
    assert probe.main(argv) == 1
    result = json.loads(out.read_text())
    assert result["stopped"] and result["peak_c"] == 65.4 and result["ran_s"] < 10
    assert [temp for _, temp in result["samples"]] == [60.0, 65.0, 65.4]
