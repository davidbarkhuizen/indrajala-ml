import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "datapack.py"
spec = importlib.util.spec_from_file_location("datapack", SCRIPT)
assert spec is not None and spec.loader is not None
script = importlib.util.module_from_spec(spec)
sys.modules["datapack"] = script
spec.loader.exec_module(script)


def write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def no_networks(data: Path, listed: dict[str, tuple[str, str]]) -> set[str]:
    return set()


def toy(data: Path) -> None:
    """Two parts: `a` whose manifest lists paths from its stem's directory, `b` from its own."""
    one = write(data / "a" / "runs" / "one.json", "1")
    two = write(data / "a" / "runs" / "two.json", "2")
    (data / "a.sha256").write_text(f"{one}  runs/one.json\n{two}  runs/two.json\n")
    copy = write(data / "b" / "copy.json", "1")  # the same content as a/runs/one.json
    (data / "b" / "b.sha256").write_text(f"{copy}  copy.json\n")


def test_a_manifest_base_is_where_every_listed_file_is(tmp_path: Path):
    toy(tmp_path)
    assert script.manifest_base(tmp_path, "a.sha256") == Path("a")
    assert script.manifest_base(tmp_path, "b/b.sha256") == Path("b")
    (tmp_path / "a" / "runs" / "two.json").unlink()
    with pytest.raises(ValueError):
        script.manifest_base(tmp_path, "a.sha256")


def test_the_verify_pack_keeps_the_paths_read_and_the_cited_hashes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data, out = tmp_path / "data", tmp_path / "out"
    toy(data)
    extra = write(data / "unlisted.json", "x")
    monkeypatch.setattr(script, "MANIFESTS", {"a": ["a.sha256"], "b": ["b/b.sha256"]})
    monkeypatch.setattr(script, "certified_networks", no_networks)
    reads = {"a/runs/two.json", "b/copy.json"}

    def check_reads(data: Path, tex: Path) -> set[str]:
        return set(reads)

    monkeypatch.setattr(script, "check_reads", check_reads)
    script.build(data, Path("paper.tex"), out, ["a", "b"])
    lines = (out / "verify.sha256").read_text().splitlines()
    assert sorted(line.split("  ")[1] for line in lines) == sorted(reads)
    assert script.check(out)
    reads.add("unlisted.json")  # read by the check but in no manifest: no pack
    with pytest.raises(SystemExit):
        script.build(data, Path("paper.tex"), tmp_path / "again", [])
    assert extra not in (out / "MANIFEST.sha256").read_text()


def test_a_changed_file_fails_the_check(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data, out = tmp_path / "data", tmp_path / "out"
    toy(data)
    monkeypatch.setattr(script, "MANIFESTS", {"a": ["a.sha256"]})
    monkeypatch.setattr(script, "certified_networks", no_networks)

    def check_reads(data: Path, tex: Path) -> set[str]:
        return {"a/runs/one.json"}

    monkeypatch.setattr(script, "check_reads", check_reads)
    (data / "a" / "runs" / "two.json").write_text("changed")
    script.build(data, Path("paper.tex"), out, ["a"])
    assert not script.check(out)
