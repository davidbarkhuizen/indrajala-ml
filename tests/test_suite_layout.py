"""
`./cli test` runs tests/ and rust/tests/ in one pytest session, and neither suite is a package, so
pytest imports each test module by its basename: two modules with one basename stop collection.
CI runs each suite as its own job and can't see it (two test_window_program.py did, from #801 to
the fix).
"""

from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_no_two_test_modules_share_a_basename() -> None:
    by_name: defaultdict[str, list[str]] = defaultdict(list)
    for suite in (REPO / "tests", REPO / "rust" / "tests"):
        for module in suite.rglob("test_*.py"):
            by_name[module.name].append(str(module.relative_to(REPO)))
    assert {name: paths for name, paths in by_name.items() if len(paths) > 1} == {}
