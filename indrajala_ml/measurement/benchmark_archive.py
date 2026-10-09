"""
The benchmark archive (docs/benchmark-archive-workplan.md): a clone of davidbarkhuizen/indrajala-benchmarks
that scripts/ab.py archive and scripts/golden_training_run.py archive add records to, one PR per batch.
Its FORMAT.md describes the records; its CI validates them and re-renders every run's reports.

A batch starts on a fresh branch from the archive's main (`start`), copies its records and the
profile snapshots they cite (`add_profile`), adds their INDEX.md lines (`add_index_lines`), and ends
with one commit, then a PR that is squash-merged once CI is green (`finish`; `no_pr` stops after the
commit). Records are immutable once merged: a destination that already exists is refused.
"""

import datetime
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

ARCHIVE_REPO = Path(os.environ.get("AB_ARCHIVE_REPO", Path.home() / "code/indrajala-benchmarks"))
FORMAT = 1
INDEX_LINE = re.compile(r"^- (\d{4}-\d{2}-\d{2}) · ")
PROFILES_DIR = "docs/machine_profiles"
MERGE_TIMEOUT_S = 900


class ArchiveError(Exception):
    pass


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=False, capture_output=True, text=True)
    if result.returncode:
        raise ArchiveError(f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()[-500:]}")
    return result.stdout.strip()


def _gh(*args: str) -> str:
    result = subprocess.run(["gh", *args], check=False, capture_output=True, text=True)
    if result.returncode:
        raise ArchiveError(f"gh {' '.join(args[:3])} failed: {result.stderr.strip()[-500:]}")
    return result.stdout.strip()


def resolve_commit(repo: Path, ref: str) -> str:
    return _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")


# ---- profiles


def host_profile(repo: Path, hostname: str) -> Path | None:
    """The machine profile under PROFILES_DIR recorded on hostname, if any."""
    for path in sorted((repo / PROFILES_DIR).glob("*.json")):
        if json.loads(path.read_text()).get("state", {}).get("hostname") == hostname:
            return path
    return None


def identity_sha256(profile: dict[str, Any]) -> str:
    """The hash of a profile's identity, which ab.py records with each machine check: a profile
    re-recorded with another identity no longer matches the runs made under the old one."""
    return hashlib.sha256(json.dumps(profile["identity"], sort_keys=True).encode()).hexdigest()


def add_profile(archive: Path, profile_path: Path) -> tuple[str, bool]:
    """The archive path of profile_path's snapshot, copied in if no identical one is there yet: a
    profile edited after capture (its noise_rules) takes the next free name. Returns (path, added)."""
    content = profile_path.read_bytes()
    state = json.loads(content)["state"]
    stamp = re.sub(r"[-:]", "", state["captured_at"])
    host_dir = archive / "machines" / state["hostname"]
    n = 1
    while True:
        name = f"{stamp}-profile.json" if n == 1 else f"{stamp}-{n}-profile.json"
        path = host_dir / name
        if not path.exists():
            host_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            return str(path.relative_to(archive)), True
        if path.read_bytes() == content:
            return str(path.relative_to(archive)), False
        n += 1


def profile_index_line(archive: Path, snapshot: str) -> str:
    state = json.loads((archive / snapshot).read_text())["state"]
    return index_line(state["captured_at"][:10], state["hostname"], "profile", Path(snapshot).name, snapshot, "")


# ---- INDEX.md


def index_line(date: str, host: str, kind: str, name: str, path: str, reason: str) -> str:
    reason = " ".join(reason.split()) or "cited by a record"
    return f"- {date} · {host} · {kind} · [{name}]({path}) · {reason}"


def add_index_lines(archive: Path, lines: list[str]) -> None:
    """Merge lines into INDEX.md, newest first; a new line goes before older ones of its date."""
    index = archive / "INDEX.md"
    text = index.read_text().splitlines()
    head = [line for line in text if not INDEX_LINE.match(line)]
    while head and not head[-1]:
        head.pop()
    old = [line for line in text if INDEX_LINE.match(line)]
    new = sorted(lines, key=_line_date, reverse=True)
    merged = sorted(new + old, key=_line_date, reverse=True)  # stable: new before old on a date
    index.write_text("\n".join([*head, "", *merged]) + "\n")


def _line_date(line: str) -> str:
    match = INDEX_LINE.match(line)
    assert match is not None
    return match[1]


# ---- a batch: branch, commit, PR


def start(archive: Path) -> str:
    """A fresh branch for a batch, from origin/main (fetched) or, without a remote, from HEAD."""
    if not (archive / ".git").exists():
        raise ArchiveError(f"no archive clone at {archive} (git clone it, or set AB_ARCHIVE_REPO)")
    if _git(archive, "status", "--porcelain"):
        raise ArchiveError(f"the archive clone {archive} has uncommitted changes")
    base = f"archive-{datetime.datetime.now(datetime.UTC):%Y%m%d-%H%M%S}"
    existing = set(_git(archive, "branch", "--format=%(refname:short)").splitlines())
    branch, n = base, 1
    while branch in existing:  # two batches within a second
        n += 1
        branch = f"{base}-{n}"
    if _git(archive, "remote"):
        _git(archive, "fetch", "--quiet", "origin")
        _git(archive, "checkout", "--quiet", "-b", branch, "origin/main")
    else:
        _git(archive, "checkout", "--quiet", "-b", branch)
    return branch


def abandon(archive: Path, branch: str) -> None:
    """Back to main, dropping a batch that failed before its commit."""
    _git(archive, "reset", "--quiet", "--hard")
    _git(archive, "clean", "--quiet", "-fd")
    _git(archive, "checkout", "--quiet", "main")
    _git(archive, "branch", "--quiet", "-D", branch)


def finish(archive: Path, branch: str, paths: list[str], title: str, body: str, no_pr: bool) -> str:
    """Commit paths, then (unless no_pr) push, open the PR, wait for CI, squash-merge and return to
    an updated main. Returns the commit, or the merged PR's URL."""
    _git(archive, "add", "--", *paths)
    _git(archive, "commit", "--quiet", "-m", f"{title}\n\n{body}")
    if no_pr:
        return _git(archive, "rev-parse", "HEAD")
    repo = _repo_slug(archive)
    if subprocess.run(
        ["git", "-C", str(archive), "push", "--quiet", "-u", "origin", branch], capture_output=True, check=False
    ).returncode:
        # creating a branch by push has failed on GitHub (an Internal Server Error); create the ref first
        base = _git(archive, "rev-parse", "origin/main")
        _gh("api", "-X", "POST", f"repos/{repo}/git/refs", "-f", f"ref=refs/heads/{branch}", "-f", f"sha={base}")
        _git(archive, "push", "--quiet", "-u", "origin", branch)
    url = _gh("pr", "create", "--repo", repo, "--head", branch, "--title", title, "--body", body)
    _wait_for_checks(repo, url)
    _gh("pr", "merge", url, "--repo", repo, "--squash", "--delete-branch")
    _git(archive, "checkout", "--quiet", "main")
    _git(archive, "pull", "--quiet", "--ff-only")
    _git(archive, "branch", "--quiet", "-D", branch)
    return url


def _repo_slug(archive: Path) -> str:
    url = _git(archive, "remote", "get-url", "origin")
    match = re.search(r"github\.com[:/](.+?)(\.git)?$", url)
    if match is None:
        raise ArchiveError(f"the archive's origin {url} isn't on GitHub")
    return match[1]


def _wait_for_checks(repo: str, url: str) -> None:
    deadline = time.monotonic() + MERGE_TIMEOUT_S
    while time.monotonic() < deadline:
        time.sleep(10)
        checks = json.loads(_gh("pr", "view", url, "--repo", repo, "--json", "statusCheckRollup"))
        states = [c.get("conclusion") or c.get("state") for c in checks["statusCheckRollup"]]
        if states and all(s in ("SUCCESS", "SKIPPED", "NEUTRAL") for s in states):
            return
        if any(s in ("FAILURE", "CANCELLED", "TIMED_OUT", "ERROR", "ACTION_REQUIRED") for s in states):
            raise ArchiveError(f"CI failed on {url}: fix the records on its branch, or close it")
    raise ArchiveError(f"CI didn't finish on {url} within {MERGE_TIMEOUT_S // 60} minutes")


# ---- golden runs


def archive_golden(
    archive: Path,
    golden: Path,
    *,
    repo: Path,
    commit: str,
    reason: str,
    note: str,
    profile: Path,
    date: str,
    replaces: str | None,
) -> tuple[list[str], list[str], str]:
    """Copy a golden file into the archive as golden/<host>/<date>-<commit7>.json.gz with its note
    (.md), comparing it with the host's previous version. Returns (paths written, INDEX.md lines, a
    one-line summary). Refuses a new-functionality version in which an earlier entry moved."""
    if reason not in ("material", "new-functionality"):
        raise ArchiveError(f"reason {reason!r}: material or new-functionality")
    raw = golden.read_bytes()
    entries: dict[str, Any] = json.loads(raw)
    crate_listing = _git(repo, "ls-tree", commit, "rust")
    crate = crate_listing.split()[2] if crate_listing else None
    snapshot, added_profile = add_profile(archive, profile)
    host = Path(snapshot).parent.name
    host_dir = archive / "golden" / host
    stem = f"{date}-{commit[:7]}"
    if (host_dir / f"{stem}.json.gz").exists():
        raise ArchiveError(f"golden/{host}/{stem} is already archived (a correction is a new record: --replaces)")
    versions = sorted(host_dir.glob("*.json.gz"))
    previous = str(versions[-1].relative_to(archive)) if versions else None
    before: dict[str, Any] = json.loads(gzip.decompress(versions[-1].read_bytes())) if versions else {}
    added = sorted(entries.keys() - before.keys())
    removed = sorted(before.keys() - entries.keys())
    moved = sorted(name for name in entries.keys() & before.keys() if entries[name] != before[name])
    if reason == "new-functionality" and (moved or removed):
        raise ArchiveError(
            f"new-functionality, but against {previous} entries moved ({', '.join(moved) or 'none'}) or were "
            f"removed ({', '.join(removed) or 'none'}): a material change needs the owner's approval"
        )
    meta: dict[str, Any] = {
        "format": FORMAT,
        "kind": "golden",
        "host": host,
        "date": date,
        "commit": commit,
        "crate": crate,
        "reason": reason,
        "note": note,
        "previous": previous,
        "added": added,
        "moved": moved,
        "removed": removed,
        "networks": len(entries),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "profile": snapshot,
        "replaces": replaces,
    }
    host_dir.mkdir(parents=True, exist_ok=True)
    (host_dir / f"{stem}.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    (host_dir / f"{stem}.md").write_text(golden_note(meta))
    paths = [f"golden/{host}/{stem}.json.gz", f"golden/{host}/{stem}.md"]
    lines = [index_line(date, host, "golden", f"{stem}.json.gz", paths[0], f"{reason}: {note}")]
    if added_profile:
        paths.append(snapshot)
        lines.append(profile_index_line(archive, snapshot))
    if previous is None:
        summary = f"golden/{host}/{stem}: the first version on {host}, {len(entries)} entries"
    else:
        summary = f"golden/{host}/{stem}: {len(added)} added, {len(moved)} moved, {len(removed)} removed"
    return paths, lines, summary


def golden_note(meta: dict[str, Any]) -> str:
    def names(items: list[str]) -> str:
        return ", ".join(f"`{name}`" for name in items) or "none"

    previous = f"`{meta['previous']}`" if meta["previous"] else "none: the first version on this host"
    lines = [
        f"# Golden run: {meta['host']}, {meta['date']}, {meta['commit'][:7]}",
        "",
        "| | |",
        "| --- | --- |",
        f"| indrajala-ml | `{meta['commit']}` |",
        f"| crate (`rust/`) | `{meta['crate']}` |",
        f"| reason | {meta['reason']} |",
        f"| previous | {previous} |",
        f"| entries | {meta['networks']} |",
        f"| profile | `{meta['profile']}` |",
        "",
        meta["note"],
        "",
    ]
    if meta["previous"]:
        lines += [f"Added: {names(meta['added'])}.", "", f"Moved: {names(meta['moved'])}.", ""]
        lines += [f"Removed: {names(meta['removed'])}.", ""]
    if meta["replaces"]:
        lines += [f"Replaces `{meta['replaces']}`.", ""]
    lines += ["```json", json.dumps(meta, indent=1), "```"]
    return "\n".join(lines) + "\n"


def copy_files(source: Path, destination: Path, names: list[str]) -> None:
    destination.mkdir(parents=True)
    for name in names:
        shutil.copyfile(source / name, destination / name)
