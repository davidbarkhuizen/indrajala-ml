"""
indrajala_ml/measurement/benchmark_archive.py, against a temporary git repository standing in for
the archive's clone (no remote, so a batch branches from HEAD), with no network.
"""

import gzip
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from indrajala_ml.measurement import benchmark_archive

REPO = Path(__file__).resolve().parent.parent.parent
I7 = REPO / "docs/machine_profiles/i7-9700k.json"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An archive clone holding only its index, on main."""
    for var in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(var, "t")
    for var in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(var, "t@t")
    repo = tmp_path / "archive"
    repo.mkdir()
    (repo / "INDEX.md").write_text("# Index\n\nOne line per record, newest first.\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "empty")
    return repo


def test_index_lines_go_newest_first_new_before_old_on_a_date(archive: Path) -> None:
    line = benchmark_archive.index_line
    benchmark_archive.add_index_lines(archive, [line("2026-10-07", "h", "run", "a", "runs/h/a", "first")])
    benchmark_archive.add_index_lines(
        archive,
        [line("2026-10-01", "h", "run", "b", "runs/h/b", "older"), line("2026-10-07", "h", "run", "c", "runs/h/c", "")],
    )
    text = (archive / "INDEX.md").read_text()
    assert text.startswith("# Index\n\nOne line per record, newest first.\n\n- 2026-10-07 · h · run · [c]")
    assert [row.split(" · ")[3] for row in text.splitlines() if row.startswith("- ")] == [
        "[c](runs/h/c)",
        "[a](runs/h/a)",
        "[b](runs/h/b)",
    ]
    assert text.splitlines()[4].endswith("· cited by a record")


def test_a_profile_is_snapshotted_once_and_an_edited_one_takes_the_next_name(archive: Path, tmp_path: Path) -> None:
    assert benchmark_archive.add_profile(archive, I7) == ("machines/jebel/20261008T062106Z-profile.json", True)
    assert benchmark_archive.add_profile(archive, I7) == ("machines/jebel/20261008T062106Z-profile.json", False)
    edited = json.loads(I7.read_text())
    edited["noise_rules"]["shifted_pass"] = 0.03
    (tmp_path / "edited.json").write_text(json.dumps(edited))
    assert benchmark_archive.add_profile(archive, tmp_path / "edited.json") == (
        "machines/jebel/20261008T062106Z-2-profile.json",
        True,
    )


def test_identity_hash_ignores_state_and_noise_rules() -> None:
    profile: dict[str, Any] = json.loads(I7.read_text())
    other: dict[str, Any] = {**profile, "state": {}, "noise_rules": {}}
    assert benchmark_archive.identity_sha256(profile) == benchmark_archive.identity_sha256(other)
    other["identity"] = {**profile["identity"], "memory": {}}
    assert benchmark_archive.identity_sha256(profile) != benchmark_archive.identity_sha256(other)


def _golden(tmp_path: Path, entries: dict[str, object]) -> Path:
    path = tmp_path / f"golden-{len(entries)}-{max(entries)}.json"
    path.write_text(json.dumps(entries, indent=1))
    return path


def _archive_golden(archive: Path, golden: Path, reason: str, date: str) -> tuple[list[str], list[str], str]:
    return benchmark_archive.archive_golden(
        archive,
        golden,
        repo=REPO,
        commit=_git(REPO, "rev-parse", "HEAD"),
        reason=reason,
        note="n",
        profile=I7,
        date=date,
        replaces=None,
    )


def test_golden_versions_name_what_was_added_and_refuse_a_moved_entry_as_new_functionality(
    archive: Path, tmp_path: Path
) -> None:
    commit7 = _git(REPO, "rev-parse", "--short=7", "HEAD")
    paths, lines, summary = _archive_golden(archive, _golden(tmp_path, {"a": ["0x1p+0"]}), "material", "2026-10-07")
    assert paths == [
        f"golden/jebel/2026-10-07-{commit7}.json.gz",
        f"golden/jebel/2026-10-07-{commit7}.md",
        "machines/jebel/20261008T062106Z-profile.json",
    ]
    assert summary.endswith("the first version on jebel, 1 entries")
    assert lines[0].startswith(f"- 2026-10-07 · jebel · golden · [2026-10-07-{commit7}.json.gz]")
    gz = (archive / paths[0]).read_bytes()
    assert json.loads(gzip.decompress(gz)) == {"a": ["0x1p+0"]}
    assert gz[4:8] == b"\0\0\0\0"  # no timestamp: the same file gzips to the same bytes

    added = _golden(tmp_path, {"a": ["0x1p+0"], "b": ["0x1p+1"]})
    paths, _, summary = _archive_golden(archive, added, "new-functionality", "2026-10-08")
    note = (archive / paths[1]).read_text()
    meta = json.loads(note.split("```json\n")[1].split("\n```")[0])
    assert (meta["added"], meta["moved"], meta["removed"]) == (["b"], [], [])
    assert meta["previous"] == f"golden/jebel/2026-10-07-{commit7}.json.gz"
    assert meta["crate"] == _git(REPO, "ls-tree", "HEAD", "rust").split()[2]
    assert "Added: `b`." in note

    moved = _golden(tmp_path, {"a": ["0x1p+2"], "b": ["0x1p+1"]})
    with pytest.raises(benchmark_archive.ArchiveError, match=r"entries moved \(a\)"):
        _archive_golden(archive, moved, "new-functionality", "2026-10-09")
    _, _, summary = _archive_golden(archive, moved, "material", "2026-10-09")
    assert summary.endswith("0 added, 1 moved, 0 removed")
    with pytest.raises(benchmark_archive.ArchiveError, match="already archived"):
        _archive_golden(archive, moved, "material", "2026-10-09")


def test_golden_script_archives_onto_a_new_branch_without_training(archive: Path, tmp_path: Path) -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    import golden_training_run

    golden = _golden(tmp_path, {"a": ["0x1p+0"]})
    golden_training_run.main(
        [
            "archive",
            str(golden),
            "--reason",
            "material",
            "--note",
            "stage 0",
            "--date",
            "2026-10-07",
            "--profile",
            str(I7),
            "--archive-repo",
            str(archive),
            "--no-pr",
        ]
    )
    assert _git(archive, "rev-parse", "--abbrev-ref", "HEAD").startswith("archive-")
    assert _git(archive, "status", "--porcelain") == ""
    changed = _git(archive, "show", "--name-only", "--format=", "HEAD").splitlines()
    assert sorted(changed) == sorted(
        [
            *(
                f"golden/jebel/2026-10-07-{_git(REPO, 'rev-parse', '--short=7', 'HEAD')}{s}"
                for s in (".json.gz", ".md")
            ),
            "machines/jebel/20261008T062106Z-profile.json",
            "INDEX.md",
        ]
    )
