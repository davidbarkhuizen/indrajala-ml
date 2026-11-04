"""
Which crate kernels a benchmark runs, at which shapes, and for how long (the crate performance
workplan, K25): a profile names a function, not its operands. It runs the command with a census
build of the crate first on the path, adds up what each process wrote, and prints the sites by time.

    (in rust/) maturin build --release --features census --out DIR
    pip install --no-deps --no-index --target SITE DIR/*.whl
    python scripts/measurement/shape_census.py --site SITE [--top 25] [--json out.json] -- COMMAND ...

    python scripts/measurement/shape_census.py --site SITE -- python scripts/measurement/attention_timing.py \
        --no-memory --backends rust --tokens 1024

A site's time holds whatever it calls: `conv_forward`'s holds its transpose. So a share is of its own
kernel's total, for comparing shapes within a kernel, not of the run. The clock and a lock at every
site slow the short ones, so never time a census build.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

Key = tuple[str, int, int, int]


def add_up(directory: Path) -> dict[Key, list[int]]:
    """Each (kernel, m, k, n)'s calls and nanoseconds, over every process's file in directory."""
    totals: dict[Key, list[int]] = defaultdict(lambda: [0, 0])
    for path in sorted(directory.glob("census-*.json")):
        for kernel, m, k, n, calls, nanos in json.loads(path.read_text()):
            totals[(kernel, m, k, n)][0] += calls
            totals[(kernel, m, k, n)][1] += nanos
    return dict(totals)


def table(totals: dict[Key, list[int]], top: int) -> list[str]:
    """The kernels by time, then the sites by time, each with its share of its kernel."""
    per_kernel: dict[str, int] = defaultdict(int)
    for (kernel, *_), (_, nanos) in totals.items():
        per_kernel[kernel] += nanos
    lines = ["kernel           seconds"]
    lines += [f"{kernel:16} {nanos / 1e9:8.3f}" for kernel, nanos in sorted(per_kernel.items(), key=lambda i: -i[1])]
    lines += ["", "kernel                     m      k      n      calls   seconds  of kernel  us a call"]
    for (kernel, m, k, n), (calls, nanos) in sorted(totals.items(), key=lambda i: -i[1][1])[:top]:
        share = nanos / per_kernel[kernel]
        lines.append(
            f"{kernel:16} {m:10} {k:6} {n:6} {calls:10} {nanos / 1e9:9.3f} {share:10.0%} {nanos / calls / 1e3:10.1f}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", required=True, help="the directory a census build is installed in")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--json", help="also write [kernel, m, k, n, calls, nanoseconds] rows here")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command to run")
    with tempfile.TemporaryDirectory() as directory:
        path = [args.site, os.environ.get("PYTHONPATH", "")]
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, path)), "INDRAJALA_CENSUS_DIR": directory}
        code = subprocess.run(command, env=env, check=False, stdout=subprocess.DEVNULL).returncode
        totals = add_up(Path(directory))
    if not totals:
        sys.exit(f"no census written (the command's exit status {code}): is {args.site} a census build?")
    print("\n".join(table(totals, args.top)))
    if args.json:
        rows = [[*key, *value] for key, value in sorted(totals.items())]
        Path(args.json).write_text(json.dumps(rows) + "\n")
    sys.exit(code)


if __name__ == "__main__":
    main()
