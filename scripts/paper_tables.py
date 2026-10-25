#!/usr/bin/env python3
"""
The paper's tables regenerated from the result files (the paper workplan's T7): every number in a
table comes from the runs' logs and recertifications, never typed by hand.

`paper_tables.py show NAME...` prints each table's LaTeX rows (the lines between `\\midrule` and
`\\bottomrule`). `paper_tables.py check TEX...` regenerates every table whose label a manuscript
holds and compares its rows with the manuscript's, a line per table; it exits 1 on any difference.

Each table is a function of the data directory (`--data`, default data/window-recurrent), named by
its LaTeX label. Begun with P2's; T1's, T3's, T2's and T5's rates from their batches (`paper/<test>/`:
runs, recertifications, long tests), a run counted only when it succeeded (certified in both
verdicts and every long-test case passed), as their findings count; T4's decomposition readings (`approximations`' JSON or text) and T6's
answers-proof results.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def median(values: list[int]) -> str:
    """The median as the findings write it: whole, or to a half."""
    middle = statistics.median(values)
    return str(int(middle)) if middle == int(middle) else f"{middle:.1f}"


def failures(runs: list[mf.ArmRun]) -> tuple[int, int, int]:
    """An arm's failures (runs not succeeded), how many of the uncertified came near a solution, and collapsed (readout's)."""
    stopped = [r.outcome for r in runs if r.outcome.end != "certified"]
    return (
        sum(not r.succeeded for r in runs),
        sum(o.peak >= mf.NEAR for o in stopped),
        sum(o.collapses > 0 for o in stopped),
    )


def succeeded(runs: list[mf.ArmRun]) -> tuple[list[mf.ArmRun], str]:
    """An arm's successes (certified in both verdicts and long-tested, ArmRun.succeeded) and its rate's cells."""
    won = [r for r in runs if r.succeeded]
    low, high = inference.wilson(len(won), len(runs))
    return won, f"{len(won)} of {len(runs)} & {low:.2f}--{high:.2f}"


def batch_runs(data: Path, test: str, arm: str) -> list[mf.ArmRun]:
    """A paper test's arm (its batch's runs/<arm>), each run read with its recertifications and long tests."""
    root = data / "paper" / test
    return mf.arm_runs(root / "runs", arm, mf.recertifications(root / "certify"), mf.long_tested(root / "long-tests"))


def arms_table(test: str, arms: list[tuple[str, str]]) -> Table:
    """A comparison's arms (T1, T3): succeeded of runs, the Wilson 95% interval, failures near a solution, collapsed."""

    def table(data: Path) -> list[str]:
        lines: list[str] = []
        for arm, label in arms:
            runs = batch_runs(data, test, arm)
            _, rate = succeeded(runs)
            failed, near, collapsed = failures(runs)
            lines.append(f"{label} & {rate} & {near} of {failed} & {collapsed} \\\\")
        return lines

    return table


def rates_table(backdrops: list[tuple[str, int]]) -> Table:
    """T2's and T5's rates (one batch, D17): q, succeeded of runs, Wilson 95%, the successes' epochs, failures near."""

    def table(data: Path) -> list[str]:
        lines: list[str] = []
        for backdrop, q in backdrops:
            runs = batch_runs(data, "t2-t5", backdrop)
            won, rate = succeeded(runs)
            epochs = [r.outcome.epochs for r in won]
            course = f"{span(epochs)}, {median(epochs)}" if len(epochs) > 1 else span(epochs)
            failed, near, _ = failures(runs)
            lines.append(f"\\texttt{{{backdrop}}} & {q} & {rate} & {course} & {near} of {failed} \\\\")
        return lines

    return table


@dataclass(frozen=True)
class Reading:
    """One network's decomposition reading (`approximations`): its target's states, whether its factors
    cover the target, and each factor's approximation's blocks with the congruences it determines."""

    states: int
    cover: bool
    factors: list[tuple[int, list[str]]]


NET = re.compile(r": (\d+) states, meet \d+, cover (True|False)")
FACTOR = re.compile(r"factor \d+: \d+ codes, approximation (\d+) blocks \([^)]*\), determines ([^;]*)")


def readings(path: Path) -> list[Reading]:
    """A reading's networks: a directory of JSON (T4's first reading) or a text file, a line a network (T2's and T5's)."""
    if path.is_dir():
        found: list[Reading] = []
        for file in sorted(path.glob("*.json")):
            net = json.loads(file.read_text())
            factors = [(len(f["approximation"]), list(f["determines"])) for f in net["factors"]]
            found.append(Reading(net["states"], net["cover"], factors))
        return found
    found = []
    for line in path.read_text().splitlines():
        net = NET.search(line)
        assert net is not None, line
        factors = [
            (int(blocks), [] if names.strip() == "-" else [n.strip() for n in re.split(r", (?=[A-Za-z])", names)])
            for blocks, names in FACTOR.findall(line)
        ]
        found.append(Reading(int(net[1]), net[2] == "True", factors))
    return found


def tex(name: str) -> str:
    """A congruence's name in LaTeX: its braces escaped (`places {1}`)."""
    return name.replace("{", "\\{").replace("}", "\\}")


def natural(name: str) -> list[int | str]:
    """A sort key reading a name's numbers as numbers (`d7-family` before `d11-family`)."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def decomposition_row(label: str, nets: list[Reading]) -> str:
    """A set's row (T4, D10's rules): networks; factors nontrivial (1 < blocks < states); networks naming a
    congruence in a nontrivial factor (in both factors); covers; the congruences determined, most first."""
    held = [[1 < blocks < n.states and bool(names) for blocks, names in n.factors] for n in nets]
    nontrivial = sum(1 < blocks < n.states for n in nets for blocks, _ in n.factors)
    factors = sum(len(n.factors) for n in nets)
    named, both = sum(any(h) for h in held), sum(all(h) for h in held)
    counts = Counter(name for n in nets for _, names in n.factors for name in names)
    determines = ", ".join(
        f"\\texttt{{{tex(k)}}} {v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    return (
        f"{label} & {len(nets)} & {nontrivial}/{factors} & {named} ({both}) & {sum(n.cover for n in nets)} & "
        f"{determines or '--'} \\\\"
    )


def decomposition_table(sets: list[tuple[str, list[str]]]) -> Table:
    """T4's sets, each the readings under paper/ named (directories or text files) pooled into one row."""

    def table(data: Path) -> list[str]:
        return [
            decomposition_row(label, [n for path in paths for n in readings(data / "paper" / path)])
            for label, paths in sets
        ]

    return table


def proofs(data: Path) -> list[dict[str, Any]]:
    """T6's readings (D11): every answers-proof result under paper/<test>/proof, a bound each."""
    return [
        bound
        for path in sorted((data / "paper").glob("*/proof/*/*.answers-proof.json"))
        for bound in json.loads(path.read_text())["bounds"]
    ]


def proof_outcome(bound: dict[str, Any]) -> str:
    """A bound's outcome: no stop (silent), or the stop (kept, false-stop) with Chow's verdict."""
    if bound["verdict"] == "silent":
        return "silent"
    chow = bound["chow"]
    verdict = "out of reach" if "refused" in chow else ("proved" if chow["equivalent"] else "refuted")
    return f"{bound['verdict']} {verdict}"


PROOF_ROWS = [
    ("silent", "no stop (the run never certified)"),
    ("kept proved", "kept, Chow proved"),
    ("kept out of reach", "kept, out of Chow's reach"),
    ("false-stop proved", "false stop, Chow proved the machine equal to the target"),
    ("false-stop refuted", "false stop, Chow refuted"),
    ("false-stop out of reach", "false stop, out of Chow's reach"),
]


def proof(data: Path) -> list[str]:
    """T6's outcomes (tab:proof): each outcome's count at the exact bound and at 1.1 times it."""
    counts = Counter((b["slack"], proof_outcome(b)) for b in proofs(data))
    assert {o for _, o in counts} <= {o for o, _ in PROOF_ROWS}, counts
    return [f"{label} & {counts[(1.0, o)]} & {counts[(1.1, o)]} \\\\" for o, label in PROOF_ROWS]


def refuted(data: Path) -> list[str]:
    """T6b's refuted stops (tab:refuted): the run, the bound, the stop and certified steps, the counterexample's length."""
    lines: list[str] = []
    for path in sorted((data / "paper").glob("*/proof/*/*.answers-proof.json"), key=lambda p: natural(p.name)):
        for b in json.loads(path.read_text())["bounds"]:
            if proof_outcome(b) == "false-stop refuted":
                run = path.name.removesuffix(".answers-proof.json")
                lines.append(
                    f"\\texttt{{{run}}} & {b['bound']} & {b['stop']}, {b['certified']} & {len(b['chow']['counterexample'])} \\\\"
                )
    return lines


TABLES: dict[str, Table] = {
    "tab:approximation": approximation,
    "tab:extra-outputs": arms_table(
        "t1",
        [
            ("sum", "sum (an approximation)"),
            ("suffix", "suffix (state-carrying, no approximation)"),
            ("token", "token (stateless)"),
        ],
    ),
    "tab:baselines": arms_table(
        "t3",
        [
            ("approximation", "snapped codes, transformer window (control)"),
            ("approximation-continuous", "continuous state (no codes)"),
            ("approximation-linear", "linear window (no blocks)"),
        ],
    ),
    "tab:scale": rates_table([("d7-family", 49), ("d11-family", 121), ("d13-family", 169), ("d11-quotient", 121)]),
    "tab:breadth": rates_table([("s3", 6), ("s4", 24), ("mtf4", 24), ("rq12", 12), ("rq16", 16)]),
    "tab:decomposition": decomposition_table(
        [
            ("$p$ 5 division claim, family", ["t4/p5-division-claim-family"]),
            ("$p$ 5 division claim, quotient", ["t4/p5-division-claim-quotient"]),
            ("$p$ 5 approximation claim, family", ["t4/p5-approximation-claim-family"]),
            ("$p$ 5 approximation claim, sum", ["t4/p5-approximation-claim-sum"]),
            ("$p$ 5 approximation claim, quotient", ["t4/p5-approximation-claim-quotient"]),
            ("$p$ 5 probe 2a (apart)", ["t4/p5-probe-2a"]),
            ("$p$ 7 probe 4 and fork probe", ["t4/p7-probe-4", "t4/p7-fork"]),
            ("$p$ 11 ladder", ["t4/p11-ladder"]),
            ("$p$ 13 W1", ["t4/p13-w1"]),
            ("word-problem-s3", ["t4/s3-round1"]),
            ("word-problem-s4", ["t4/s4-round1"]),
            ("move-to-front-3", ["t4/mtf3-round1"]),
        ]
    ),
    "tab:decomposition-fresh": decomposition_table(
        [
            (f"\\texttt{{{b}}}", [f"t2-t5/approximations/{b}.txt"])
            for b in ("d7-family", "d11-family", "d13-family", "s3", "s4", "mtf4", "rq12")
        ]
    ),
    "tab:proof": proof,
    "tab:refuted": refuted,
}


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
