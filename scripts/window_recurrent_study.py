"""
The window-recurrent adder (docs/window-recurrent-adder-workplan.md): whether a network that reads a
window of k columns and hands a snapped state from window to window learns, through a curriculum of
window counts, to add base-3 numbers of any length, under indrajala_ml/studies/window_recurrent_study.py's
harness and indrajala_ml/data/addition_data.py's catalogue.

    python scripts/window_recurrent_study.py run CONFIG.json
    python scripts/window_recurrent_study.py status CONFIG.json
    python scripts/window_recurrent_study.py report CONFIG.json [--md FILE]
    python scripts/window_recurrent_study.py check NETWORK.json --n N [N ...]

A config names the study's directory (results/, one JSON file per run, and models/, each run's steps
and their networks in format 2), its workers, its settings and its matrix (load_config's
docstring). `run` resumes from what is on disk, a run from its last saved step, and prints a line
per finished run, then the report; it holds the machine lock, as every study. `check` loads a saved
step's network and runs the catalogue (without B3) at each length.

OPENBLAS_NUM_THREADS=1, as every study.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from indrajala_ml.data import addition_data as ad
from indrajala_ml.measurement import machine_lock
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.studies import window_recurrent_study as wr


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "report"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
        if name == "report":
            command.add_argument("--md", type=Path, help="write the report to this file too")
    check = commands.add_parser("check")
    check.add_argument("network", type=Path)
    check.add_argument("--n", type=int, nargs="+", required=True, help="the lengths to check")
    args = parser.parse_args(argv)

    if args.command == "check":
        network = load_network(str(args.network))
        model = wr.WindowAdder(network, network.columns)
        for n in args.n:
            for result in ad.evaluate(model.adder(n), n, wr.test_properties(ad.CATALOGUE)):
                print(
                    f"n {n}, {result.id}: {result.passed}/{result.cases}"
                    + ("" if result.holds else f", e.g. {result.failures[0]}")
                )
        return 0
    study = wr.load_config(args.config)
    if args.command == "run":
        with machine_lock.study():
            wr.run(study, log=lambda line: print(line, flush=True))
        print(wr.report(study))
    elif args.command == "status":
        print(wr.status(study))
    else:
        text = wr.report(study)
        print(text)
        if args.md is not None:
            args.md.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
