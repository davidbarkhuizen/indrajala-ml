"""
The DRY audit's clone scan (docs/next-steps.md, "From the DRY audit"; the DRY third pass workplan,
stage 1): copies of `--lines` or more statement lines between or within Python files, and each
tree's share of lines inside one.

    python scripts/measurement/duplicates.py [--lines 8] [--top 40] [ROOT ...]

A line is compared normalised: strings and numbers masked, so two copies with other literals still
match; blank, comment, import and bracket-only lines dropped, so an import list or a call's closing
brackets never make a clone alone. Every window of `--lines` normalised lines is hashed; windows
shared by two places that run on together merge into one clone. A window shared by more than
COMMON places (a stock idiom) is skipped. Each tree's share counts a line once however many clones
it is in, both sides of a clone counted. The default roots are the package, the scripts, both test
suites and `cli`.

Standard library only, so it runs before the venv is built.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOTS = ("indrajala_ml", "scripts", "tests", "rust/tests", "cli")
COMMON = 40  # a window in more places than this is an idiom, not a copy

_SKIPPED = {'"""', "'''", "(", ")", "]", "}", "),", "],", "else:", "try:", "pass"}
_IMPORT = re.compile(r"^(from \S+ import|import )")
_NAME_IN_A_LIST = re.compile(r"^[A-Za-z_][\w.]*( as \w+)?,$")
_STRING = re.compile(r"\"[^\"]*\"|'[^']*'")
_NUMBER = re.compile(r"\b\d+(\.\d+)?\b")


def normalised(line: str) -> str | None:
    """The line as compared, or None for one dropped."""
    s = line.strip()
    if not s or s.startswith("#") or s in _SKIPPED or _IMPORT.match(s) or _NAME_IN_A_LIST.match(s):
        return None
    return _NUMBER.sub("N", _STRING.sub("S", s))


@dataclass(frozen=True)
class Clone:
    """`lines` normalised lines at first[1]..first[2] of first[0], the same at second's."""

    lines: int
    first: tuple[Path, int, int]
    second: tuple[Path, int, int]


def python_files(roots: Iterable[Path]) -> list[Path]:
    """Every .py file under the roots (a root may be a file), caches skipped, sorted."""
    found: set[Path] = set()
    for root in roots:
        if root.is_file():
            found.add(root)
        else:
            found.update(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(found)


def clones(files: Sequence[Path], lines: int) -> list[Clone]:
    """Every clone of `lines` or more normalised lines among the files, the longest first."""
    kept: dict[Path, list[tuple[int, str]]] = {}
    for f in files:
        text = f.read_text(errors="replace").splitlines()
        kept[f] = [(i + 1, n) for i, line in enumerate(text) if (n := normalised(line)) is not None]

    places: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for f, seq in kept.items():
        for k in range(len(seq) - lines + 1):
            if len({n for _, n in seq[k : k + lines]}) == 1:
                continue  # one line repeated: a table's rows, not a copy
            window = "\n".join(n for _, n in seq[k : k + lines])
            places[hashlib.blake2b(window.encode(), digest_size=16).hexdigest()].append((f, k))

    # the windows two places share, by the pair and their offset, so a run of them is one clone
    shared: dict[tuple[Path, Path, int | None], set[tuple[int, int]]] = defaultdict(set)
    for occurrences in places.values():
        if len(occurrences) > COMMON:
            continue
        for a in range(len(occurrences)):
            for b in range(a + 1, len(occurrences)):
                (fa, ka), (fb, kb) = sorted((occurrences[a], occurrences[b]), key=lambda o: (str(o[0]), o[1]))
                if fa == fb and kb - ka < lines:
                    continue  # a window overlapping itself: a repeated line, not a copy
                shared[(fa, fb, kb - ka if fa == fb else None)].add((ka, kb))

    found: list[Clone] = []
    for (fa, fb, _), starts in shared.items():
        ordered = sorted(starts)
        run_start = previous = ordered[0]
        for k in [*ordered[1:], None]:
            if k is not None and k == (previous[0] + 1, previous[1] + 1):
                previous = k
                continue
            first = (fa, kept[fa][run_start[0]][0], kept[fa][previous[0] + lines - 1][0])
            second = (fb, kept[fb][run_start[1]][0], kept[fb][previous[1] + lines - 1][0])
            found.append(Clone(previous[0] - run_start[0] + lines, first, second))
            if k is not None:
                run_start = previous = k
    return sorted(found, key=lambda c: (-c.lines, str(c.first[0]), c.first[1], str(c.second[0]), c.second[1]))


def shares(found: Iterable[Clone], files: Sequence[Path], roots: Sequence[Path]) -> dict[Path, tuple[int, int]]:
    """Each root's (lines inside a clone, lines in all), a line counted once."""
    inside: dict[Path, set[int]] = defaultdict(set)
    for clone in found:
        for f, start, end in (clone.first, clone.second):
            inside[f].update(range(start, end + 1))
    result: dict[Path, tuple[int, int]] = {}
    for root in roots:
        mine = [f for f in files if f == root or root in f.parents]
        total = sum(len(f.read_text(errors="replace").splitlines()) for f in mine)
        result[root] = (sum(len(inside[f]) for f in mine), total)
    return result


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="*", type=Path, help=f"default: {' '.join(ROOTS)}")
    parser.add_argument("--lines", type=int, default=8, help="the shortest clone, in normalised lines")
    parser.add_argument("--top", type=int, default=40, help="the clones listed, the longest first (0: all)")
    args = parser.parse_args(argv)
    roots = args.roots or [Path(r) for r in ROOTS if Path(r).exists()]
    files = python_files(roots)
    found = clones(files, args.lines)
    for clone in found[: args.top or None]:
        (fa, a0, a1), (fb, b0, b1) = clone.first, clone.second
        print(f"{clone.lines:4d}  {fa}:{a0}-{a1}  <->  {fb}:{b0}-{b1}")
    print(f"\n{len(found)} clones of {args.lines} lines or more")
    for root, (inside, total) in shares(found, files, roots).items():
        print(f"  {root!s:<14} {inside:6d} of {total:6d} lines in a clone ({100 * inside / max(total, 1):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
