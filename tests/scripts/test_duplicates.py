"""
scripts/measurement/duplicates.py: a copy is found whatever its literals, an import list or a
repeated line alone is not one, a run of shared windows is one clone, and a tree's share counts a
line once.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "measurement"))
import duplicates  # scripts/ isn't a package

BODY = [f"    value_{i} = compute('{i}', {i})" for i in range(10)]


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


def test_a_copy_with_other_literals_is_one_clone_of_its_length(tmp_path: Path):
    a = _write(tmp_path / "a.py", ["def f():", *BODY])
    b = _write(tmp_path / "b.py", ["# a comment", "", "def f():", *(line.replace("'", '"') for line in BODY)])
    found = duplicates.clones([a, b], lines=8)
    assert found == [duplicates.Clone(11, (a, 1, 11), (b, 3, 13))]


def test_an_import_list_and_brackets_alone_are_not_a_clone(tmp_path: Path):
    imports = ["from x import (", *(f"    Name{i}," for i in range(12)), ")", "import os", "import sys"]
    a = _write(tmp_path / "a.py", imports)
    b = _write(tmp_path / "b.py", imports)
    assert duplicates.clones([a, b], lines=8) == []


def test_a_repeated_line_is_not_a_copy_of_itself_but_two_blocks_in_one_file_are(tmp_path: Path):
    repeated = _write(tmp_path / "r.py", ["x = f()"] * 20)
    assert duplicates.clones([repeated], lines=8) == []
    twice = _write(tmp_path / "t.py", ["def f():", *BODY, "def g():", *BODY])
    assert duplicates.clones([twice], lines=8) == [duplicates.Clone(10, (twice, 2, 11), (twice, 13, 22))]


def test_a_shorter_copy_than_the_window_is_not_reported(tmp_path: Path):
    a = _write(tmp_path / "a.py", BODY[:7])
    b = _write(tmp_path / "b.py", BODY[:7])
    assert duplicates.clones([a, b], lines=8) == []


def test_a_trees_share_counts_each_line_once(tmp_path: Path):
    tree = tmp_path / "tree"
    tree.mkdir()
    a = _write(tree / "a.py", ["def f():", *BODY])
    b = _write(tree / "b.py", ["def f():", *BODY])
    c = _write(tree / "c.py", ["def f():", *BODY])
    files = duplicates.python_files([tree])
    assert files == [a, b, c]
    found = duplicates.clones(files, lines=8)
    assert len(found) == 3  # a-b, a-c, b-c
    assert duplicates.shares(found, files, [tree]) == {tree: (33, 33)}


def test_main_lists_the_clones_and_the_shares(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _write(tmp_path / "a.py", ["def f():", *BODY])
    _write(tmp_path / "b.py", ["def f():", *BODY, "", "x = 1"])
    assert duplicates.main([str(tmp_path), "--lines", "8"]) == 0
    out = capsys.readouterr().out
    assert "1 clones of 8 lines or more" in out
    assert "22 of     24 lines in a clone (91.7%)" in out
