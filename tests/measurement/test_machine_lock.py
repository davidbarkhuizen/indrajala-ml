"""The machine lock (indrajala_ml/measurement/machine_lock.py) and the study command lines that take it."""

import json
import multiprocessing
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from indrajala_ml.measurement import machine_lock
from indrajala_ml.studies import attention_dropout_study, sequence_study

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import addition_study  # scripts/ isn't a package
import batch_size_scaling_sweep
import multi_head_attention_study
import patch_attention_study
import patch_geometry_study
import residual_depth_study


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AB_RUNS_ROOT", str(tmp_path))
    return tmp_path


def test_the_holder_writes_one_json_line_naming_itself(root: Path) -> None:
    with machine_lock.hold("sweep x", "study"):
        record = json.loads((root / "run.lock").read_text())
    assert record["what"] == "sweep x" and record["kind"] == "study" and isinstance(record["pid"], int)
    assert "T" in record["since"]


def test_a_second_holder_is_refused_naming_the_first(root: Path) -> None:
    with (
        machine_lock.hold("run --bench cmd", "run"),
        pytest.raises(machine_lock.MachineBusy, match=r"held by run run --bench cmd, pid \d+, since "),
        machine_lock.hold("sweep", "study"),
    ):
        pass
    with machine_lock.hold("sweep", "study"):  # freed when the first ended
        pass


def test_a_killed_holder_frees_the_lock(root: Path) -> None:
    hold = (
        "import sys; from indrajala_ml.measurement import machine_lock\n"
        "with machine_lock.hold('sweep', 'study'):\n"
        "    print('held', flush=True); import time; time.sleep(60)"
    )
    holder = subprocess.Popen([sys.executable, "-c", hold], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout is not None and holder.stdout.readline() == "held\n"
        with pytest.raises(machine_lock.MachineBusy, match="held by study sweep"), machine_lock.hold("run", "run"):
            pass
    finally:
        holder.kill()
        holder.wait()
    with machine_lock.hold("run", "run"):
        pass


def test_an_old_plain_text_record_is_shown_as_it_is() -> None:
    assert machine_lock.describe("pid 1, run x, since 08:00\n") == "pid 1, run x, since 08:00"


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def test_a_forkserver_worker_does_not_keep_the_lock(root: Path) -> None:
    context = multiprocessing.get_context("forkserver")
    with machine_lock.hold("sweep", "study"):
        worker = context.Process(target=_sleep, args=(30,))
        worker.start()
    try:
        with machine_lock.hold("run", "run"):  # the study's process let go; its worker never held it
            pass
    finally:
        worker.kill()
        worker.join()


def test_a_study_exits_before_its_block_when_the_machine_is_taken(root: Path) -> None:
    ran = False
    with (
        machine_lock.hold("run --bench cmd", "run"),
        pytest.raises(SystemExit, match="not started: .* held by run run --bench cmd"),
        machine_lock.study(["scripts/x_study.py", "sweep"]),
    ):
        ran = True
    assert not ran
    with machine_lock.study(["scripts/x_study.py", "sweep", "--epochs", "1"]):
        assert json.loads((root / "run.lock").read_text())["what"] == "x_study.py sweep --epochs 1"


COMMAND_LINES: list[tuple[str, Callable[[list[str]], Any], list[str]]] = [
    ("sequence time", sequence_study.main, ["time"]),
    ("attention dropout time", attention_dropout_study.main, ["time"]),
    ("patch attention time", patch_attention_study.main, ["time"]),
    ("patch geometry time", patch_geometry_study.main, ["time"]),
    ("multi-head attention time", multi_head_attention_study.main, ["time"]),
    ("residual depth tune", residual_depth_study.main, ["tune"]),
    ("batch size baseline", batch_size_scaling_sweep.main, ["baseline"]),
    ("addition run", addition_study.main, ["run", str(REPO / "scripts/addition_sweep.json")]),
]


@pytest.mark.parametrize(("name", "main", "argv"), COMMAND_LINES, ids=[c[0] for c in COMMAND_LINES])
def test_every_study_command_line_that_trains_is_refused_while_the_machine_is_taken(
    root: Path, name: str, main: Callable[[list[str]], Any], argv: list[str]
) -> None:
    with machine_lock.hold("run --bench cmd", "run"), pytest.raises(SystemExit, match="not started: "):
        main(argv)
