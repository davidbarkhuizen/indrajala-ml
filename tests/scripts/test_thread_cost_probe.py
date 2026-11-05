"""
scripts/measurement/thread_cost_probe.py: each subcommand runs end to end on a tiny case and prints the probe
lines ab.py reads, and the prototype pool does a call's work at every thread count and spin. The
readings themselves aren't checked: this is a timing tool.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts" / "measurement"
sys.path.insert(0, str(SCRIPTS))
import thread_cost_probe as probe  # scripts/ isn't a package


def probe_lines(*arguments: str) -> list[dict[str, object]]:
    command = [sys.executable, str(SCRIPTS / "thread_cost_probe.py"), *arguments]
    stdout = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    return [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]


def test_a_sweep_entry_selects_a_product_or_a_layer_op() -> None:
    assert probe.focused_arguments("32x30x784") == ["--matmul", "32x30x784", "--only-matmul"]
    assert probe.focused_arguments("dense 30 x 784|forward_batch|512") == [
        *("--shape", "dense 30 x 784", "--op", "forward_batch"),
        *("--batch", "512", "--batch-sizes", "512"),
    ]


def test_the_table_s_gain_is_over_the_group_s_first_row_per_unit_of_work() -> None:
    table = probe.Table({}, {})
    table.add("g", "1 x 1", 2.0)
    table.add("g", "2 x 1", 2.0, 2)
    table.add("g", "4 x 1", 4.0, 4)
    gains = [line.split()[-1] for line in table.lines()[1:]]
    assert gains == ["1.00", "2.00", "2.00"]


@pytest.mark.skipif(shutil.which("rustup") is None, reason="names a toolchain rustup lists")
def test_a_worktree_s_empty_rust_directory_still_gets_a_toolchain(tmp_path: Path) -> None:
    assert "RUSTUP_TOOLCHAIN" in probe.toolchain_environment(tmp_path)
    (tmp_path / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "stable"\n')
    assert probe.toolchain_environment(tmp_path).get("RUSTUP_TOOLCHAIN") == os.environ.get("RUSTUP_TOOLCHAIN")


@pytest.mark.skipif(shutil.which("rustc") is None, reason="the call cost is built with rustc")
def test_calls_runs_every_mechanism_warm_and_cold() -> None:
    lines = probe_lines(
        "calls",
        *("--threads", "1", "3", "--idle-ms", "0", "0.1", "--work-k", "0", "1", "--spin-us", "0", "5"),
        *("--repeats", "1", "--loops", "1", "--target-ms", "1"),
    )
    cases = [str(line["case"]) for line in lines if line["metric"] == "per call"]
    assert len(cases) == 3 * 2 * 2 * 2
    assert "1k steps, cold (0.1 ms idle), pool, spin 5 µs, 3 threads" in cases


def test_scaling_reads_numpy_as_the_control() -> None:
    lines = probe_lines(
        "scaling",
        *("--threads", "1", "2", "--idle-ms", "0", "1", "--matmul", "8x8x8"),
        "--repeats",
        "1",
        "--loops",
        "1",
    )
    lines = [line for line in lines if line["metric"] == "per call"]  # not the peak temperature
    assert len(lines) == 2 * 2 * 2
    assert {line.get("control", False) for line in lines if str(line["case"]).endswith("/ numpy")} == {True}
    assert {line.get("control", False) for line in lines if str(line["case"]).endswith("/ rust")} == {False}


def test_scaling_times_a_layer_op() -> None:
    lines = probe_lines(
        "scaling",
        *("--threads", "2", "--idle-ms", "0", "--op", "dense 10 x 30|forward_batch|32", "--backends", "rust"),
        *("--repeats", "1", "--loops", "1"),
    )
    per_call = [line["case"] for line in lines if line["metric"] == "per call"]
    assert per_call == ["dense 10 x 30|forward_batch|32, 2 threads, warm / rust"]


def test_epochs_runs_n_processes_at_each_thread_count() -> None:
    lines = probe_lines(
        "epochs",
        *("--backdrop", "window", "--threads", "1", "2", "--threshold", "1", "--processes", "2", "--repeats", "1"),
        *("--", "--cases", "learn-linear-k1-s16-m16-B4-W3", "--loops", "1"),
    )
    learn = [line for line in lines if "learn" in str(line["case"])]
    assert len(learn) == 2 * 2
    assert {line["case"] for line in learn} == {
        "2 x 1 thread, learn linear k1 s16 m16 B4 W3",
        "2 x 2 threads, learn linear k1 s16 m16 B4 W3",
    }
    assert all(line["control"] for line in lines if "matmul 256" in str(line["case"]))


def test_epochs_sweeps_openblas_and_names_it_in_the_case() -> None:
    lines = probe_lines(
        "epochs",
        *("--backdrop", "window", "--threads", "1", "--openblas", "1", "0", "--repeats", "1"),
        *("--", "--cases", "learn-linear-k1-s16-m16-B4-W3", "--loops", "1"),
    )
    learn = {str(line["case"]) for line in lines if "learn" in str(line["case"])}
    assert learn == {
        "1 x 1 thread, OpenBLAS 1, learn linear k1 s16 m16 B4 W3",
        "1 x 1 thread, OpenBLAS default, learn linear k1 s16 m16 B4 W3",
    }


def test_epochs_reads_a_backdrop_s_out_file(tmp_path: Path) -> None:
    out = tmp_path / "runs.json"
    out.write_text(json.dumps({"conv B=32 / rust": [{"epoch": 1.5}]}))
    assert probe.backdrop_rows(probe.BACKDROPS["prepared"], "", out) == [("conv B=32 / rust", "epoch", 1.5, "s", False)]
