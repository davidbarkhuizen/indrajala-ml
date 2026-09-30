"""
scripts/ab.py: the report on archived A/Bs (tests/fixtures/ab/, each with the table its PR
published), and run/extend end to end on a toy probe in a temporary repository. Nothing is timed.
"""

import base64
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import ab  # scripts/ isn't a package

FIXTURES = Path(__file__).resolve().parent / "fixtures/ab"
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


def test_a_pass_whose_tree_lacks_the_module_aborts(toy_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _git(toy_repo, "rm", "-q", "indrajala_ml/train.py")
    _git(toy_repo, "commit", "-q", "-m", "no train")
    assert _run(toy_repo, "run", "--bench", "cmd", "--skip-profile", "--old", "HEAD~1", "--", "scripts/probe.py") == 1
    assert "provenance: indrajala_ml.train" in capsys.readouterr().err
    assert json.loads((ab.find_run(None) / "manifest.json").read_text())["state"] == "failed"


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
