#!/usr/bin/env python3
"""
The paper's tables regenerated from the result files (the paper workplan's T7): every number in a
table comes from the runs' logs and recertifications, never typed by hand.

`paper_tables.py show NAME...` prints each table's LaTeX rows (the lines between `\\midrule` and
`\\bottomrule`). `paper_tables.py check TEX...` regenerates every table whose label a manuscript
holds and compares its rows with the manuscript's, a line per table; it exits 1 on any difference.

Each table is a function of the data directory (`--data`, default data/window-recurrent), named by
its LaTeX label. Added as the manuscript's tables are written (stage 6); begun with P2's.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path

from indrajala_ml.studies import inference
from indrajala_ml.studies import modular_family as mf

Table = Callable[[Path], list[str]]


def span(values: list[int]) -> str:
    """A range as the manuscripts write it: "lo--hi", one value alone, "--" for none."""
    if not values:
        return "--"
    low, high = min(values), max(values)
    return str(low) if low == high else f"{low}--{high}"


def arm_row(label: str, runs: list[mf.ArmRun]) -> str:
    """An arm's row: certified of runs, the Wilson 95% interval, and the successes' steps and epochs."""
    certified = [r.outcome for r in runs if r.certified]
    low, high = inference.wilson(len(certified), len(runs))
    successes = f"{span([o.step for o in certified])}, {span([o.epochs for o in certified])}" if certified else "--"
    return f"{label} & {len(certified)} of {len(runs)} & {low:.2f}--{high:.2f} & {successes} \\\\"


def approximation(data: Path) -> list[str]:
    """P2's approximation claim at p 5 (tab:approximation): four arms, 20 seeds each, both verdicts."""
    root = data / "approximation-claim"
    verdicts = mf.recertifications(root / "certify")
    arms = [
        ("family", "sum, difference (cover)"),
        ("sum", "sum (an approximation)"),
        ("quotient", "none"),
        ("product", "product (not an approximation)"),
    ]
    return [arm_row(label, mf.arm_runs(root, arm, verdicts)) for arm, label in arms]


TABLES: dict[str, Table] = {"tab:approximation": approximation}


def manuscript_rows(tex: str, label: str) -> list[str] | None:
    """The rows of the table labelled `label` in a manuscript: its lines between \\midrule and \\bottomrule."""
    for table in re.findall(r"\\begin\{table\}.*?\\end\{table\}", tex, re.DOTALL):
        if f"\\label{{{label}}}" in table:
            body = table.split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
            return [line.strip() for line in body.splitlines() if line.strip()]
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/window-recurrent"))
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show", help="tables' rows, regenerated")
    show.add_argument("names", nargs="+", choices=sorted(TABLES))
    check = commands.add_parser("check", help="each table a manuscript holds against its regeneration")
    check.add_argument("manuscripts", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.command == "show":
        for name in args.names:
            print(f"% {name}")
            print("\n".join(TABLES[name](args.data)))
        return 0
    differs = False
    for tex in args.manuscripts:
        text = tex.read_text()
        for name, table in TABLES.items():
            held = manuscript_rows(text, name)
            if held is None:
                continue
            made = table(args.data)
            if held == made:
                print(f"{tex}: {name}: {len(made)} rows, the same")
                continue
            differs = True
            print(f"{tex}: {name}: DIFFERS")
            for old, new in zip(held, made, strict=False):
                if old != new:
                    print(f"  manuscript: {old}\n  regenerated: {new}")
            if len(held) != len(made):
                print(f"  {len(held)} rows in the manuscript, {len(made)} regenerated")
    return 1 if differs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
