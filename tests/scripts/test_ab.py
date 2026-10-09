"""
scripts/ab.py: the report on archived A/Bs (tests/fixtures/ab/, each with the table its PR
published), and run/extend end to end on a toy probe in a temporary repository. Nothing is timed.
"""

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import ab  # scripts/ isn't a package

from indrajala_ml.measurement import benchmark_archive

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures/ab"
RUNS = sorted(d.name for d in FIXTURES.iterdir() if (d / "manifest.json").is_file())


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _section(markdown: str, metric: str) -> list[list[str]]:
    """The data rows of the report's table for metric."""
    lines = markdown.splitlines()
    start = lines.index(f"### {metric}") + 4
    rows: list[list[str]] = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        rows.append(_cells(line))
    return rows


@pytest.mark.parametrize("run", ["pr477-ab2", "pr479", "pr480"])
def test_report_reproduces_the_published_table(run: str) -> None:
    published = [_cells(line) for line in (FIXTURES / run / "published.md").read_text().splitlines()[2:]]
    ours = _section(ab.markdown_report(ab.report_data(FIXTURES / run)), "epoch")
    assert len(published) == 8
    assert [row[: len(published[0])] for row in ours] == published


def test_report_headers_number_the_passes() -> None:
    markdown = ab.markdown_report(ab.report_data(FIXTURES / "pr480"))
    assert "| per-pass medians old (1, 4, 6) / new (2, 3, 5) |" in markdown
    assert markdown.count("### ") == 3  # epoch, epoch, loader, prepare


def test_pr480_marks_passes_1_and_5_shifted_and_balanced() -> None:
    data = ab.report_data(FIXTURES / "pr480")
    assert [(s.number, s.side, s.ratio < 1) for s in data.shifted] == [(1, "old", True), (5, "new", True)]
    assert ab.balancing_order(data.shifted) == ""
    assert any(line.startswith("shifted:") and line.endswith("balanced") for line in ab.brief_report(data))
    assert not any(row.consistent for row in data.rows)


def test_pr480_before_its_balancing_pair_names_the_extend_order(tmp_path: Path) -> None:
    run = tmp_path / "pr480"
    shutil.copytree(FIXTURES / "pr480", run)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["passes"] = manifest["passes"][:4]
    manifest["extends"] = []
    (run / "manifest.json").write_text(json.dumps(manifest))
    data = ab.report_data(run)
    assert [(s.number, s.side) for s in data.shifted] == [(1, "old")]
    assert ab.balancing_order(data.shifted) == "NO"
    assert any(line.endswith("next: ab.py extend --order NO") for line in ab.brief_report(data))


def test_balancing_order_pairs_start_with_the_side_that_has_fewer() -> None:
    fast_old, slow_old, fast_new = (
        ab.ShiftedPass(1, "old", 0.9),
        ab.ShiftedPass(2, "old", 1.1),
        ab.ShiftedPass(3, "new", 0.9),
    )
    assert ab.balancing_order([fast_old]) == "NO"
    assert ab.balancing_order([fast_new]) == "ON"
    assert ab.balancing_order([fast_old, fast_new]) == ""
    assert ab.balancing_order([fast_old, slow_old, fast_new]) == "NO"


def test_pr477_regression_is_consistent_and_the_fix_is_not() -> None:
    before = {ab._row_label(row): row for row in ab.report_data(FIXTURES / "pr477-ab1").rows}
    assert before["epoch / dense single / rust"].consistent
    assert round(before["epoch / dense single / rust"].delta * 100, 1) == 5.4
    # +8.1% but the new side's passes differ by 5%: separated, inside the spread (the rule
    # the owner chose over "separated" alone, which flags a no-change row 1 time in 3 at 2 passes a side)
    assert before["epoch / dense single / numpy"].separated
    assert not before["epoch / dense single / numpy"].consistent
    after = {ab._row_label(row): row for row in ab.report_data(FIXTURES / "pr477-ab2").rows}
    assert not after["epoch / dense single / numpy"].separated
    assert not after["epoch / dense single / rust"].consistent


def test_pr479_flags_its_consistent_rows_and_the_moved_control() -> None:
    data = ab.report_data(FIXTURES / "pr479")
    consistent = {ab._row_label(row) for row in data.rows if row.consistent}
    assert "epoch, loader / conv single / rust" in consistent  # the PR's own "both new passes above both old"
    brief = ab.brief_report(data)
    assert brief[2].startswith("CONTROL MOVED: 1 of 8 control rows consistent, e.g. prepare / dense B=32 / rust +0.8%")


def test_control_backend_adds_that_backends_rows_as_controls(tmp_path: Path) -> None:
    run = tmp_path / "pr480"
    shutil.copytree(FIXTURES / "pr480", run)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["control_backend"] = "numpy"
    (run / "manifest.json").write_text(json.dumps(manifest))
    controls = [row for row in ab.report_data(run).rows if row.control]
    assert len(controls) == 8 + 8  # prepare, plus epoch and epoch, loader on numpy
    assert all(row.metric == "prepare" or row.case.endswith("/ numpy") for row in controls)


@pytest.mark.parametrize("run", RUNS)
def test_brief_stays_within_its_line_limit(run: str) -> None:
    assert len(ab.brief_report(ab.report_data(FIXTURES / run))) <= ab.BRIEF_LINES


def test_brief_folds_consistent_rows_past_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ab, "BRIEF_LINES", 6)
    brief = ab.brief_report(ab.report_data(FIXTURES / "pr477-ab1"))
    assert len(brief) == 6
    assert brief[-2].startswith("... ") and brief[-2].endswith("more consistent rows (report --md)")


def test_too_few_passes_says_so(tmp_path: Path) -> None:
    run = tmp_path / "pr479"
    shutil.copytree(FIXTURES / "pr479", run)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["passes"] = manifest["passes"][:2]
    (run / "manifest.json").write_text(json.dumps(manifest))
    data = ab.report_data(run)
    assert "too few complete passes" in ab.brief_report(data)[2]
    assert not any(row.separated for row in data.rows)


def test_failed_passes_are_left_out(tmp_path: Path) -> None:
    run = tmp_path / "pr480"
    shutil.copytree(FIXTURES / "pr480", run)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["passes"][5]["status"] = "failed"
    (run / "manifest.json").write_text(json.dumps(manifest))
    assert ab.report_data(run).passes == {"old": [1, 4], "new": [2, 3, 5]}


def test_parse_order() -> None:
    assert ab.parse_order("ONNO") == ["old", "new", "new", "old"]
    assert ab.parse_order("no") == ["new", "old"]
    for bad in ("", "ONX", "O N"):
        with pytest.raises(ab.AbError):
            ab.parse_order(bad)


def test_probe_rows_follow_the_contract(tmp_path: Path) -> None:
    stdout = tmp_path / "pass.stdout"
    stdout.write_text(
        "warming up\n"
        '{"case": "a / rust", "metric": "t", "value": 1.5, "unit": "us"}\n'
        '{"case": "b", "metric": "t", "value": 2, "unit": "us", "control": true}\n'
        '{"not": "a row"}\n'
    )
    rows = ab.CommandProbe().rows(tmp_path / "none.json", stdout, "rust")
    assert rows == [ab.Row("a / rust", "t", 1.5, "us", True), ab.Row("b", "t", 2.0, "us", True)]


# ---- run and extend on a toy repository

PROBE = """\
import hashlib
import json
from indrajala_ml import toy
for i in range(3):
    print(json.dumps({"case": "toy", "metric": "value", "value": toy.VALUE + i / 1000, "unit": "s"}))
    print(json.dumps({"case": "steady", "metric": "value", "value": 1 + i / 1000, "unit": "s", "control": True}))
"""


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True, capture_output=True
    )


@pytest.fixture
def toy_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository of two commits: indrajala_ml.toy.VALUE is 1.0, then 2.0."""
    repo = tmp_path / "repo"
    (repo / "indrajala_ml").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "indrajala_ml/train.py").write_text("")
    (repo / "scripts/probe.py").write_text(PROBE)
    _git(repo, "init", "-q", "-b", "main")
    for value in ("1.0", "2.0"):
        (repo / "indrajala_ml/toy.py").write_text(f"VALUE = {value}\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-q", "-m", f"toy {value}")
    monkeypatch.setattr(ab, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(ab, "WORKTREES_ROOT", tmp_path / "worktrees")
    return repo


def _run(repo: Path, *args: str) -> int:
    return ab.main(["--repo", str(repo), *args])


def test_run_and_extend_end_to_end(toy_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_args = [
        "run",
        "--bench",
        "cmd",
        "--old",
        "HEAD~1",
        "--order",
        "ONNO",
        "--skip-profile",
        "--",
        "scripts/probe.py",
    ]
    assert _run(toy_repo, *run_args) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2 and out[0].startswith("started ") and out[1].startswith("done ")

    run_dir = ab.find_run(None)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["state"] == "done"
    assert [(p["number"], p["side"], p["status"]) for p in manifest["passes"]] == [
        (1, "old", "ok"),
        (2, "new", "ok"),
        (3, "new", "ok"),
        (4, "old", "ok"),
    ]
    for side in ("old", "new"):
        tree = Path(manifest[side]["tree"])
        assert tree.parent == ab.WORKTREES_ROOT
        assert all(
            Path(p["provenance"]["train"]).parent.parent == tree for p in manifest["passes"] if p["side"] == side
        )
    assert len((run_dir / "progress.jsonl").read_text().splitlines()) == 8

    assert _run(toy_repo, "extend", "--order", "NO", "--skip-profile") == 0
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert [(p["number"], p["side"]) for p in manifest["passes"]][4:] == [(5, "new"), (6, "old")]
    assert (run_dir / "pass-06-old.stdout").is_file()
    capsys.readouterr()

    assert _run(toy_repo, "report", "--brief") == 0
    brief = capsys.readouterr().out.splitlines()
    assert brief[2] == "controls: 1 rows, all within noise"
    assert brief[3] == "consistent: value / toy +99.9% (1.001 -> 2.001 s)"
    assert _run(toy_repo, "status") == 0
    assert capsys.readouterr().out == f"{run_dir.name}: done, 6 passes done\n"


def test_pooled_report_of_an_aa_pools_every_pass(toy_repo: Path, tmp_path: Path) -> None:
    args = ["run", "--bench", "cmd", "--old", "HEAD", "--order", "ONNO", "--skip-profile", "--", "scripts/probe.py"]
    assert _run(toy_repo, *args) == 0
    out = tmp_path / "pooled.md"
    assert _run(toy_repo, "report", "--pooled", str(out)) == 0
    lines = out.read_text().splitlines()
    assert "- spread of per-pass medians over 2 rows: median 0.0%, 90th percentile 0.0%, max 0.0%" in lines
    assert (
        "- pass shifts (median over rows of pass median / pooled median): 1 +0.0%, 2 +0.0%, 3 +0.0%, 4 +0.0%" in lines
    )
    assert "| case | median (min-max) s | per-pass medians (1, 2, 3, 4) | spread |" in lines
    assert "| toy | 2.001 (2.000-2.002) | 2.001, 2.001, 2.001, 2.001 | 0.0% |" in lines
    assert "| steady (control) | 1.001 (1.000-1.002) | 1.001, 1.001, 1.001, 1.001 | 0.0% |" in lines


def test_pooled_report_refuses_an_ab() -> None:
    with pytest.raises(ab.AbError, match="is for an A/A"):
        ab.pooled_report(ab.report_data(FIXTURES / "pr480"))


def test_a_policy_change_during_a_pass_fails_it_and_stops_the_run(
    toy_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    reference: dict[str, dict[str, object]] = {
        "frequency": {"governor": "powersave", "energy_performance_preference": "balance_performance"},
        "power_limits": {"long_term_w": 65, "short_term_w": 120},
    }
    reset: dict[str, dict[str, object]] = {**reference, "power_limits": {"long_term_w": 95, "short_term_w": 120}}
    readings = iter([reference, reset])  # after pass 1, then after pass 2: thermald put back 95 W

    def preflight(*_: object) -> dict[str, object]:
        return {"profile": "identity matches"}

    def reference_policy(*_: object) -> dict[str, dict[str, object]]:
        return reference

    def power_policy() -> dict[str, dict[str, object]]:
        return next(readings)

    monkeypatch.setattr(ab, "_preflight", preflight)
    monkeypatch.setattr(ab, "reference_policy", reference_policy)
    monkeypatch.setattr(ab, "power_policy", power_policy)

    assert _run(toy_repo, "run", "--bench", "cmd", "--old", "HEAD~1", "--order", "ONNO", "--", "scripts/probe.py") == 1
    err = capsys.readouterr().err
    assert "pass-02-new: the machine's frequency policy or power limits changed during the pass" in err
    assert "identity.power_limits.long_term_w: 65 -> 95" in err
    manifest = json.loads((ab.find_run(None) / "manifest.json").read_text())
    assert manifest["state"] == "failed"
    assert [(p["number"], p["status"]) for p in manifest["passes"]] == [(1, "ok"), (2, "failed")]
    assert manifest["passes"][1]["policy_changes"] == ["identity.power_limits.long_term_w: 65 -> 95"]


def test_a_pass_whose_tree_lacks_the_module_aborts(toy_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _git(toy_repo, "rm", "-q", "indrajala_ml/train.py")
    _git(toy_repo, "commit", "-q", "-m", "no train")
    assert _run(toy_repo, "run", "--bench", "cmd", "--skip-profile", "--old", "HEAD~1", "--", "scripts/probe.py") == 1
    assert "provenance: the trainer" in capsys.readouterr().err
    assert json.loads((ab.find_run(None) / "manifest.json").read_text())["state"] == "failed"


def test_a_run_across_the_source_layout_move_finds_each_sides_trainer(toy_repo: Path) -> None:
    # the old side has indrajala_ml/train.py, the new one indrajala_ml/training/train.py (D4)
    (toy_repo / "indrajala_ml/training").mkdir()
    _git(toy_repo, "mv", "indrajala_ml/train.py", "indrajala_ml/training/train.py")
    _git(toy_repo, "commit", "-q", "-m", "move train")
    assert _run(toy_repo, "run", "--bench", "cmd", "--skip-profile", "--old", "HEAD~1", "--", "scripts/probe.py") == 0
    manifest = json.loads((ab.find_run(None) / "manifest.json").read_text())
    for side, trainer in (("old", "indrajala_ml/train.py"), ("new", "indrajala_ml/training/train.py")):
        tree = Path(manifest[side]["tree"])
        assert all(Path(p["provenance"]["train"]) == tree / trainer for p in manifest["passes"] if p["side"] == side)


def test_run_refuses_uncommitted_changes(toy_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (toy_repo / "indrajala_ml/toy.py").write_text("VALUE = 3.0\n")
    assert _run(toy_repo, "run", "--bench", "cmd", "--skip-profile", "--old", "HEAD~1", "--", "scripts/probe.py") == 1
    assert "uncommitted changes" in capsys.readouterr().err


def test_clean_removes_unreferenced_worktrees(toy_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        _run(
            toy_repo,
            "run",
            "--bench",
            "cmd",
            "--old",
            "HEAD~1",
            "--order",
            "ON",
            "--skip-profile",
            "--",
            "scripts/probe.py",
        )
        == 0
    )
    stray = ab.WORKTREES_ROOT / "stray"
    _git(toy_repo, "worktree", "add", "-q", "--detach", str(stray), "HEAD")
    capsys.readouterr()
    assert _run(toy_repo, "clean", "--worktrees") == 0
    assert capsys.readouterr().out == "removed 1 worktrees (stray), kept 2\n"
    assert not stray.exists()


# ---- stage 2: the other benchmarks, against the pyo3 upgrade's summarize.py tables


def _summary(title: str) -> dict[str, list[str]]:
    """{case: [old min-max, new min-max, new/old]} from one table of pyo3-summary.md."""
    lines = (FIXTURES / "pyo3-summary.md").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"### {title}")) + 4
    table: dict[str, list[str]] = {}
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        cells = _cells(line)
        table[cells[0]] = cells[1:]
    return table


def _as_summary(row: ab.RowStats) -> list[str]:
    def spread(runs: list[float]) -> str:
        return f"{min(runs):.4g}-{max(runs):.4g}"

    return [spread(row.old_runs), spread(row.new_runs), f"{row.new_median / row.old_median:.2f}"]


@pytest.mark.parametrize(
    ("run", "title", "suffix"),
    [
        ("pyo3-probe", "Array methods", ""),
        ("pyo3-focused", "Fused dense ops", " / rust"),
    ],
)
def test_report_reproduces_the_pyo3_summary(run: str, title: str, suffix: str) -> None:
    expected = _summary(title)
    rows = {row.case: row for row in ab.report_data(FIXTURES / run).rows}
    assert len(rows) == len(expected)
    assert {case: _as_summary(rows[case + suffix]) for case in expected} == expected


def test_epoch_op_profile_reproduces_the_pyo3_summary() -> None:
    # summarize.py dropped a run's op under 5 ms, so only ops at or above it in every run compare
    expected = _summary("Conv epoch")
    rows = {row.case: row for row in ab.report_data(FIXTURES / "pyo3-epoch").rows}
    compared = [case for case in expected if min(rows[case].old_runs + rows[case].new_runs) >= 0.005]
    assert len(compared) >= 10
    assert {case: _as_summary(rows[case]) for case in compared} == {case: expected[case] for case in compared}


def test_the_boundary_probe_goes_through_the_probe_contract() -> None:
    data = ab.report_data(FIXTURES / "pyo3-probe")
    assert {row.unit for row in data.rows} == {"ns"} and len(data.rows) == 17
    consistent = {row.case for row in data.rows if row.consistent}
    assert "Array.from_rows(10x30)" in consistent  # new/old 0.59, the largest gain
    assert len(ab.brief_report(data)) <= ab.BRIEF_LINES


def test_a_probe_report_is_the_same_from_any_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = ab.report_data(FIXTURES / "pyo3-probe")
    reports: list[str] = []
    for cwd in (tmp_path, FIXTURES):
        monkeypatch.chdir(cwd)
        reports.append(ab.protocol_paragraph(data))
    assert reports[0] == reports[1]
    assert "ran `python boundary_probe.py`" in reports[0]


def test_rust_only_benchmarks_say_there_is_no_control() -> None:
    brief = ab.brief_report(ab.report_data(FIXTURES / "pyo3-epoch"))
    assert brief[2] == ab.EpochOpProfile.no_control


def _write(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data))
    return path


def test_focused_benchmark_rows(tmp_path: Path) -> None:
    result = {"shape": "dense 10 x 30", "op": "forward_batch", "batch": 32, "backend": "numpy", "median_us": 2.0}
    out = _write(tmp_path / "f.json", {"results": [result | {"malloc": "default"}, result | {"malloc": "raised"}]})
    rows = ab.FocusedBenchmark().rows(out, tmp_path / "none", "numpy")
    assert [(r.case, r.metric, r.unit, r.control) for r in rows] == [
        ("dense 10 x 30 forward_batch b32 / numpy", "per call", "µs", True),
        ("dense 10 x 30 forward_batch b32 (malloc raised) / numpy", "per call", "µs", True),
    ]


def test_accuracy_pass_timing_rows_leave_out_mismatch_counts(tmp_path: Path) -> None:
    out = _write(tmp_path / "a.json", {"conv / rust": [{"per row": 1.0, "batched 32": 0.5, "mismatches 32": 0}]})
    rows = ab.AccuracyPassTiming().rows(out, tmp_path / "none", "numpy")
    assert [(r.case, r.metric, r.control) for r in rows] == [
        ("conv / rust", "per row", False),
        ("conv / rust", "batched 32", False),
    ]


def test_op_call_timing_rows(tmp_path: Path) -> None:
    run = {"run_s": 3.0, "conv_accumulate_gradient_batch": {"(32, 4608) (18432, 72)": {"calls": 63, "median_us": 90.0}}}
    out = _write(tmp_path / "o.json", {"runs": {"conv / mini-batch (32) / 0:0": [run]}})
    rows = ab.OpCallTiming().rows(out, tmp_path / "none", None)
    assert [(r.case, r.metric, r.value, r.unit) for r in rows] == [
        ("conv / mini-batch (32) / 0:0", "whole run", 3.0, "s"),
        (
            "conv / mini-batch (32) / 0:0 / conv_accumulate_gradient_batch (32, 4608) (18432, 72)",
            "per call",
            90.0,
            "µs",
        ),
    ]


def test_batch_size_timing_rows(tmp_path: Path) -> None:
    out = _write(tmp_path / "b.json", {"rust 512": [{"epoch": 1.0, "steps": 0.5}]})
    rows = ab.BatchSizeTiming().rows(out, tmp_path / "none", "rust")
    assert [(r.case, r.metric, r.control) for r in rows] == [
        ("B=512 / rust", "epoch", True),
        ("B=512 / rust", "steps", True),
    ]


def test_every_adapter_names_its_script_and_output(tmp_path: Path) -> None:
    for adapter in ab.ADAPTERS.values():
        if adapter.name == "cmd":
            continue
        command = adapter.command(ab.REPO, list(adapter.default_args), tmp_path / "out.json")
        assert Path(command[0]) == ab.REPO / f"scripts/{adapter.name}.py" and Path(command[0]).is_file()
        assert command[-1] == str(tmp_path / "out.json")


# ---- stage 3: crate A/Bs, on fake extension modules (nothing is built)


def _fake_wheel(directory: Path, extension: bytes) -> Path:
    """A minimal wheel of an indrajala_math_rust package whose "extension" is the given bytes."""
    info = "indrajala_math_rust-0.1.0.dist-info"
    files = {
        "indrajala_math_rust/__init__.py": b"",
        "indrajala_math_rust/indrajala_math_rust.cpython-314-x86_64-linux-gnu.so": extension,
        f"{info}/METADATA": b"Metadata-Version: 2.1\nName: indrajala-math-rust\nVersion: 0.1.0\n",
        f"{info}/WHEEL": b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = [
        f"{name},sha256={base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()},{len(data)}"
        for name, data in files.items()
    ]
    directory.mkdir(parents=True, exist_ok=True)
    wheel = directory / "indrajala_math_rust-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr(f"{info}/RECORD", "\n".join([*record, f"{info}/RECORD,,"]) + "\n")
    return wheel


@pytest.fixture
def toy_crates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, toy_repo: Path) -> list[str]:
    """Two commits of a toy crate repository, each with a cached fake wheel."""
    crate_repo = tmp_path / "crate"
    crate_repo.mkdir()
    _git(crate_repo, "init", "-q", "-b", "main")
    commits: list[str] = []
    for version in ("1", "2"):
        (crate_repo / "src.rs").write_text(f"// {version}\n")
        _git(crate_repo, "add", ".")
        _git(crate_repo, "commit", "-q", "-m", version)
        commits.append(
            subprocess.run(
                ["git", "-C", str(crate_repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    monkeypatch.setattr(ab, "CRATE_REPO", crate_repo)
    for commit in commits:
        _fake_wheel(ab._wheel_dir(commit), f"extension {commit}".encode())
    return commits


def test_ensure_site_installs_the_wheel_outside_the_venv(toy_crates: list[str]) -> None:
    site, sha = ab.ensure_site(toy_crates[0])
    extension = site / "indrajala_math_rust/indrajala_math_rust.cpython-314-x86_64-linux-gnu.so"
    assert extension.read_bytes() == f"extension {toy_crates[0]}".encode()
    assert sha == hashlib.sha256(extension.read_bytes()).hexdigest()
    assert site.is_relative_to(ab.RUNS_ROOT / "wheels")
    assert ab.ensure_site(toy_crates[0]) == (site, sha)  # cached: installed once


def test_provenance_checks_the_sides_extension(toy_repo: Path, toy_crates: list[str], tmp_path: Path) -> None:
    (old_site, old_sha), (_, new_sha) = ab.ensure_site(toy_crates[0]), ab.ensure_site(toy_crates[1])
    found = ab.check_provenance(tmp_path, toy_repo, ab._pass_env(toy_repo, str(old_site)), old_sha)
    assert Path(found["extension"]).is_relative_to(old_site)
    with pytest.raises(ab.AbError, match="not the side's"):
        ab.check_provenance(tmp_path, toy_repo, ab._pass_env(toy_repo, str(old_site)), new_sha)


def test_resolve_crate_takes_short_refs(toy_crates: list[str]) -> None:
    assert ab.resolve_crate(toy_crates[0][:7]) == toy_crates[0]
    with pytest.raises(ab.AbError):
        ab.resolve_crate("0000000")  # not there, and the toy repository has no origin to fetch


def _commit_crate(files: dict[str, str], date: str | None = None) -> str:
    """A new commit of the toy crate repository writing files, optionally dated date."""
    env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date} if date else None
    for name, text in files.items():
        (ab.CRATE_REPO / name).parent.mkdir(parents=True, exist_ok=True)
        (ab.CRATE_REPO / name).write_text(text)
    git = ["git", "-C", str(ab.CRATE_REPO), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "add", "."], check=True, capture_output=True)
    subprocess.run([*git, "commit", "-q", "-m", "change"], check=True, capture_output=True, env=env)
    return subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def test_extract_crate_stamps_files_with_the_extraction_time(toy_crates: list[str], tmp_path: Path) -> None:
    commit = _commit_crate({"src/lib.rs": "// old\n"}, date="2000-01-01T00:00:00")
    before = time.time()
    ab.extract_crate(commit, tmp_path / "source")
    assert (tmp_path / "source/src/lib.rs").read_text() == "// old\n"
    assert (tmp_path / "source/src/lib.rs").stat().st_mtime >= before - 1  # not the commit's 2000


def test_check_crate_builds_refuses_one_extension_for_different_sources(toy_crates: list[str]) -> None:
    base = _commit_crate({"src/lib.rs": "// 1\n"})
    rust = _commit_crate({"src/lib.rs": "// 2\n"})
    tests_only = _commit_crate({"tests/test_ops.py": "# tests\n"})
    same = {"old": "a" * 64, "new": "a" * 64}
    with pytest.raises(ab.AbError, match="differ in src/lib.rs but built the same extension"):
        ab.check_crate_builds({"old": base, "new": rust}, same)
    ab.check_crate_builds({"old": base, "new": rust}, {"old": "a" * 64, "new": "b" * 64})
    ab.check_crate_builds({"old": rust, "new": tests_only}, same)  # nothing it's built from changed


def test_a_stale_crate_build_aborts_the_run(
    toy_repo: Path, toy_crates: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    old, new = _commit_crate({"src/lib.rs": "// 1\n"}), _commit_crate({"src/lib.rs": "// 2\n"})
    for commit in (old, new):
        _fake_wheel(ab._wheel_dir(commit), b"one extension for both")
    args = ["run", "--bench", "cmd", "--old", "HEAD", "--order", "ON", "--skip-profile"]
    assert _run(toy_repo, *args, "--old-crate", old, "--new-crate", new, "--", "scripts/probe.py") == 1
    assert "a stale build" in capsys.readouterr().err
    assert json.loads((ab.find_run(None) / "manifest.json").read_text())["passes"] == []


def test_crate_ab_end_to_end(toy_repo: Path, toy_crates: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    old, new = toy_crates
    args = ["run", "--bench", "cmd", "--old", "HEAD", "--order", "ONNO", "--skip-profile"]
    assert _run(toy_repo, *args, "--old-crate", old[:7], "--new-crate", new[:7], "--", "scripts/probe.py") == 0
    manifest = json.loads((ab.find_run(None) / "manifest.json").read_text())
    assert (manifest["old"]["crate"], manifest["new"]["crate"]) == (old, new)
    for record in manifest["passes"]:
        side = manifest[record["side"]]
        assert Path(record["provenance"]["extension"]).is_relative_to(side["site"])
        assert record["provenance"]["sha256"] == side["extension_sha"]
    assert manifest["old"]["extension_sha"] != manifest["new"]["extension_sha"]
    capsys.readouterr()
    assert _run(toy_repo, "report", "--brief") == 0
    header = capsys.readouterr().out.splitlines()[0]
    assert header.endswith(
        f"; crate old {old[:7]} (.so {manifest['old']['extension_sha'][:12]}) "
        f"vs new {new[:7]} (.so {manifest['new']['extension_sha'][:12]})"
    )


def test_clean_removes_unreferenced_wheels(
    toy_repo: Path, toy_crates: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    old, new = toy_crates
    args = ["run", "--bench", "cmd", "--old", "HEAD", "--order", "ON", "--skip-profile"]
    assert _run(toy_repo, *args, "--old-crate", old, "--new-crate", new, "--", "scripts/probe.py") == 0
    _fake_wheel(ab._wheel_dir("f" * 40), b"stray")
    (ab.RUNS_ROOT / "wheels/target").mkdir()
    capsys.readouterr()
    assert _run(toy_repo, "clean", "--wheels") == 0
    assert capsys.readouterr().out == "removed 1 wheels (fffffff), kept 2\n"
    assert (ab.RUNS_ROOT / "wheels/target").is_dir() and ab._wheel_dir(old).is_dir()


def test_the_machine_line_says_when_load_follows_crate_builds() -> None:
    manifest = {"preflight": {"profile": "identity matches", "load": [2.3, 1.6, 1.7], "after_builds": ["a" * 40]}}
    line = ab._machine_line(manifest | {"extends": [], "passes": []})
    assert line.startswith(
        "profile: identity matches; max 1-min load 2.30 (HIGH) (measured just after this run's crate builds)"
    )


REPO = Path(__file__).resolve().parent.parent.parent


def test_each_host_has_its_profile_and_noise_rules() -> None:
    i7, ryzen = ab.host_profile(REPO, "jebel"), ab.host_profile(REPO, "pyramidon")
    assert i7 is not None and i7.name == "i7-9700k.json"
    assert ryzen is not None and ryzen.name == "ryzen7-3700u.json"
    assert ab.host_profile(REPO, "elsewhere") is None
    assert ab.noise_rules(i7) == {"shifted_pass": 0.02, "high_load": 1.5, "small_consistent": 0.02}
    assert ab.noise_rules(ryzen) == ab.DEFAULT_RULES
    assert ab.noise_rules(None) == ab.DEFAULT_RULES


def _with_rules(tmp_path: Path, run: str, rules: dict[str, float]) -> Path:
    copy = tmp_path / run
    shutil.copytree(FIXTURES / run, copy)
    manifest = json.loads((copy / "manifest.json").read_text())
    manifest["noise_rules"] = rules
    (copy / "manifest.json").write_text(json.dumps(manifest))
    return copy


def test_the_shift_threshold_is_the_runs_recorded_rule(tmp_path: Path) -> None:
    assert len(ab.report_data(FIXTURES / "pr480").shifted) == 2  # no rules recorded: the defaults
    assert ab.report_data(_with_rules(tmp_path, "pr480", {"shifted_pass": 0.5})).shifted == []


def test_consistent_rows_under_the_small_rule_are_marked_small(tmp_path: Path) -> None:
    plain = ab.report_data(FIXTURES / "pr477-ab1")
    consistent = [row for row in plain.rows if row.consistent and not row.control]
    assert consistent and not any(plain.small(row) for row in consistent)
    marked = ab.report_data(_with_rules(tmp_path, "pr477-ab1", {"small_consistent": 1.0}))
    assert all(marked.small(row) for row in marked.rows if row.consistent)
    assert any(line.startswith("consistent, small: ") for line in ab.brief_report(marked))
    assert "| consistent, small |" in ab.markdown_report(marked)


def test_a_host_without_a_profile_is_refused(
    toy_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ab.socket, "gethostname", lambda: "elsewhere")
    assert _run(toy_repo, "run", "--bench", "cmd", "--old", "HEAD~1", "--", "scripts/probe.py") == 1
    assert "machine profile: none in docs/machine_profiles for host elsewhere" in capsys.readouterr().err


# ---- archive (the benchmark archive workplan, stage 2): into a temporary clone with no remote

RYZEN = REPO / "docs/machine_profiles/ryzen7-3700u.json"
I7 = REPO / "docs/machine_profiles/i7-9700k.json"


@pytest.fixture
def archive_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "t@t" if "EMAIL" in var else "t")
    clone = tmp_path / "archive"
    clone.mkdir()
    (clone / "INDEX.md").write_text("# Index\n")
    _git(clone, "init", "-q", "-b", "main")
    _git(clone, "add", ".")
    _git(clone, "commit", "-q", "-m", "empty")
    monkeypatch.setattr(ab, "RUNS_ROOT", tmp_path / "runs")
    return clone


def _fixture_run(name: str, root: Path) -> Path:
    """A fixture run as ab.py left it: without the published table its test compares with."""
    run = root / name
    shutil.copytree(FIXTURES / name, run, ignore=shutil.ignore_patterns("published.md"))
    return run


def _archive(clone: Path, *args: str) -> int:
    return ab.main(["archive", *args, "--archive-repo", str(clone), "--no-pr"])


def _show(clone: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(clone), *args], check=True, capture_output=True, text=True).stdout


def test_archive_copies_a_run_its_reports_and_its_assigned_profile(archive_clone: Path, tmp_path: Path) -> None:
    run = _fixture_run("pr480", tmp_path)
    assert _archive(archive_clone, str(run), "--profile", str(RYZEN), "--reason", "PR #480") == 0
    record_dir = archive_clone / "runs/pyramidon/2026-09-29-pr480"
    record = json.loads((record_dir / "record.json").read_text())
    assert record["profile"] == "machines/pyramidon/20261008T171257Z-profile.json"
    assert (record["profile_source"], record["reason"], record["replaces"]) == ("assigned", "PR #480", None)
    assert record["files"] == sorted(p.name for p in run.iterdir())
    assert record["reports"] == ["brief.txt", "report.md"]  # an A/B: no pooled report
    assert (record_dir / "report.md").read_text() == ab.markdown_report(ab.report_data(run))
    assert (archive_clone / record["profile"]).read_bytes() == RYZEN.read_bytes()
    index = [line for line in (archive_clone / "INDEX.md").read_text().splitlines() if line.startswith("- ")]
    assert index == [
        (
            "- 2026-10-08 · pyramidon · profile · [20261008T171257Z-profile.json]"
            "(machines/pyramidon/20261008T171257Z-profile.json) · cited by a record"
        ),
        "- 2026-09-29 · pyramidon · run · [2026-09-29-pr480](runs/pyramidon/2026-09-29-pr480) · PR #480",
    ]
    assert _show(archive_clone, "status", "--porcelain") == ""
    committed = _show(archive_clone, "show", "--name-only", "--format=%s", "HEAD").splitlines()
    assert committed[0] == "Archive 1 run from pyramidon"
    # subject, blank line, raw files, reports and record, profile, INDEX.md
    assert len(committed) == 2 + len(record["files"]) + 3 + 1 + 1


def test_archive_takes_a_batch_in_one_commit_and_refuses_a_record_already_there(
    archive_clone: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = [str(_fixture_run(name, tmp_path)) for name in ("pr479", "pr480")]
    assert _archive(archive_clone, *runs, "--profile", str(RYZEN)) == 0
    assert _show(archive_clone, "log", "--format=%s", "-1").strip() == "Archive 2 runs from pyramidon"
    branch = _show(archive_clone, "rev-parse", "--abbrev-ref", "HEAD").strip()
    assert _archive(archive_clone, runs[1], "--profile", str(RYZEN)) == 1
    assert "runs/pyramidon/2026-09-29-pr480 is already archived" in capsys.readouterr().err
    # the failed batch's branch is gone and the tree clean
    assert _show(archive_clone, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"
    assert branch in _show(archive_clone, "branch")
    assert _show(archive_clone, "status", "--porcelain") == ""


def test_archive_refuses_files_ab_py_did_not_write(
    archive_clone: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "pr480"
    shutil.copytree(FIXTURES / "pr480", run)
    assert _archive(archive_clone, str(run), "--profile", str(RYZEN)) == 1
    assert "holds files ab.py didn't write; move them out first: published.md" in capsys.readouterr().err
    assert _show(archive_clone, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"


def test_archive_needs_a_profile_for_a_run_that_recorded_none_and_a_reason_for_a_failed_run(
    archive_clone: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _fixture_run("pr480", tmp_path)
    assert _archive(archive_clone, str(run)) == 1
    assert "recorded no matching profile identity: name its profile with --profile" in capsys.readouterr().err
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["state"] = "failed"
    (run / "manifest.json").write_text(json.dumps(manifest))
    assert _archive(archive_clone, str(run), "--profile", str(RYZEN)) == 1
    assert "failed: archive it only with --reason" in capsys.readouterr().err


def _checked_run(tmp_path: Path, identity: str) -> Path:
    """pr480 as if it had run on jebel under the i7's profile, its machine check recording identity."""
    run = _fixture_run("pr480", tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["repo"] = str(REPO)
    check = {
        "hostname": "jebel",
        "profile": "identity matches",
        "profile_path": "docs/machine_profiles/i7-9700k.json",
        "profile_identity": identity,
    }
    manifest["preflight"] = check
    for extend in manifest["extends"]:
        extend["preflight"] = check
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


def test_archive_cites_the_profile_a_run_checked_and_refuses_one_re_recorded_since(
    archive_clone: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    identity = benchmark_archive.identity_sha256(json.loads(I7.read_text()))
    run = _checked_run(tmp_path / "a", identity)
    assert _archive(archive_clone, str(run), "--profile", str(RYZEN)) == 1
    assert "names its own profile: archive it without --profile" in capsys.readouterr().err
    assert _archive(archive_clone, str(run)) == 0
    record = json.loads((archive_clone / "runs/jebel/2026-09-29-pr480/record.json").read_text())
    assert (record["profile"], record["profile_source"]) == ("machines/jebel/20261008T062106Z-profile.json", "checked")

    assert _archive(archive_clone, str(_checked_run(tmp_path / "b", "0" * 64))) == 1
    assert "i7-9700k.json was re-recorded with another identity since the run" in capsys.readouterr().err


def test_a_machine_check_records_the_host_and_its_profiles_identity(toy_repo: Path, tmp_path: Path) -> None:
    (toy_repo / "docs/machine_profiles").mkdir(parents=True)
    shutil.copyfile(I7, toy_repo / "docs/machine_profiles/i7-9700k.json")
    manifest = {"repo": str(toy_repo)}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = ab._preflight(run_dir, manifest, skip_profile=False, allow_change=True)  # pyright: ignore[reportPrivateUsage]
    assert record["hostname"] == ab.socket.gethostname()
    assert record["profile_identity"] == benchmark_archive.identity_sha256(json.loads(I7.read_text()))


def test_report_passes_pools_only_the_passes_named() -> None:
    whole = ab.report_data(FIXTURES / "pr480")
    first = ab.report_data(FIXTURES / "pr480", "1-4")
    assert whole.passes == {"old": [1, 4, 6], "new": [2, 3, 5]}
    assert first.passes == {"old": [1, 4], "new": [2, 3]}
    assert all(len(row.old) == 2 and len(row.new) == 2 for row in first.rows)
    assert "passes ONNO (2 old, 2 new)" in ab.brief_report(first)[0]
    assert "this report pools passes 1-4 only" in ab.protocol_paragraph(first)
    assert "pools passes" not in ab.protocol_paragraph(whole)
    assert ab.report_data(FIXTURES / "pr480", "1,4,2-3").passes == first.passes


@pytest.mark.parametrize(("passes", "message"), [("1-9", "has no pass 7, 8, 9"), ("one", "give pass numbers")])
def test_report_passes_refuses_what_the_run_lacks(passes: str, message: str) -> None:
    with pytest.raises(ab.AbError, match=message):
        ab.report_data(FIXTURES / "pr480", passes)
