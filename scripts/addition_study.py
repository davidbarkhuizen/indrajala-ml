"""
The addition study (docs/addition-study-workplan.md): the smallest network, of eight candidates
built from the package's layers, that adds base-3 numbers with every property of
indrajala_ml/data/addition_data.py's catalogue, under indrajala_ml/studies/addition_study.py's
harness.

    python scripts/addition_study.py run CONFIG.json
    python scripts/addition_study.py status CONFIG.json
    python scripts/addition_study.py report CONFIG.json [--md FILE]
    python scripts/addition_study.py check MODEL_DIRECTORY
    python scripts/addition_study.py candidates

A config names the sweep's directory (results/, one JSON file per run, and models/, each run's
trained network(s) in format 2 beside a manifest), its workers, its training settings and its
studies (load_config's docstring). `run` resumes from the results on disk and prints a line per
finished run, then the report; on jebel, under nohup in the background, it needs nobody watching.
`check` loads a saved model and runs the whole catalogue against it.

OPENBLAS_NUM_THREADS=1, as every study.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from indrajala_ml.data import addition_data as ad
from indrajala_ml.studies import addition_study as st


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "report"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
        if name == "report":
            command.add_argument("--md", type=Path, help="write the report to this file too")
    commands.add_parser("check").add_argument("model", type=Path)
    commands.add_parser("candidates")
    args = parser.parse_args(argv)

    if args.command == "candidates":
        for candidate in st.CANDIDATES.values():
            print(
                f"{candidate.name}: {candidate.description}; shapes {list(candidate.shapes)}, {candidate.rung} {candidate.rungs}"
            )
        return 0
    if args.command == "check":
        model, n = st.load_model(args.model)
        for result in ad.evaluate(model.add, n):
            print(
                f"{result.id}: {result.passed}/{result.cases}"
                + ("" if result.holds else f", e.g. {result.failures[0]}")
            )
        return 0
    sweep = st.load_config(args.config)
    if args.command == "run":
        st.run(sweep, log=lambda line: print(line, flush=True))
        print(st.report(sweep))
    elif args.command == "status":
        print(st.status(sweep))
    else:
        text = st.report(sweep)
        print(text)
        if args.md is not None:
            args.md.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
