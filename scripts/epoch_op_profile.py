"""
Rust time per crate op inside real training epochs: the conv demo's `rust_op_breakdown` (a
cProfile of one Rust training run on its 2000-row MNIST subset), one process per (architecture,
trainer, repeat), with the order rotated each repeat. It reports each op's min-max seconds and
call count over the repeats, and the profiled total.

    python scripts/epoch_op_profile.py [--architectures conv ...] [--trainers ...] [--op NAME ...]
                                       [--repeats 3] [--label old] [--out runs.json]

This measures an op's share of an epoch directly. A change worth about 5% of an epoch can be
lost in whole-epoch timing: the per-example conv forward's epoch A/B read -5.6% for Rust while
numpy's control moved -5.1% in the same runs, and this profile, builds alternated, separated them
cleanly. For an old/new crate comparison, run it once per build with
`--label` and alternate the builds (old, new, new, old, ...), as the protocols in
docs/optimizations/measurement.md describe; cProfile's overhead inflates the Python side, so
compare ops across builds, not against timed epochs.
"""

import argparse
import json
import sys

from indrajala_ml.demos import demo_conv_rust_vs_vectorized_digit_recognition as demo

from process_runs import run_json_worker


def worker(architecture: str, trainer: str) -> dict:
    datasets = demo.load_datasets()
    side, train_data, _test_data, epochs = datasets[f"MNIST 28x28 ({demo.MNIST_TRAIN_LIMIT} train)"]
    total, ops = demo.rust_op_breakdown(side, demo.ARCHITECTURES[architecture], trainer, train_data, epochs)
    return {"epochs": epochs, "total": total, "ops": {name: [seconds, calls] for name, seconds, calls in ops}}


def run_in_process(architecture: str, trainer: str) -> dict:
    command = [sys.executable, "-B", __file__, "--worker", architecture, trainer]
    return run_json_worker(command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--architectures", nargs="+", choices=list(demo.ARCHITECTURES), default=list(demo.ARCHITECTURES))
    parser.add_argument("--trainers", nargs="+", choices=demo.TRAINERS, default=demo.TRAINERS)
    parser.add_argument("--op", action="append", default=[], help="only report ops whose name contains this, repeatable")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--label", default="", help="a name for this build, printed and saved with the runs")
    parser.add_argument("--out", help="write every run here as JSON")
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        print(json.dumps(worker(*args.worker)))
        return

    cells = [(architecture, trainer) for architecture in args.architectures for trainer in args.trainers]
    runs = {cell: [] for cell in cells}
    for repeat in range(args.repeats):
        order = cells[repeat % len(cells) :] + cells[: repeat % len(cells)]
        for cell in order:
            runs[cell].append(run_in_process(*cell))
        print(f"repeat {repeat + 1}/{args.repeats} done", file=sys.stderr, flush=True)

    label = f" [{args.label}]" if args.label else ""
    print(f"Rust op seconds in one profiled training run{label}, min-max over {args.repeats} processes\n")
    print("| architecture | trainer | op | seconds | calls |")
    print("|---|---|---|---|---|")
    for (architecture, trainer), cell_runs in runs.items():
        totals = [run["total"] for run in cell_runs]
        print(f"| {architecture} | {trainer} | (profiled total) | {min(totals):.3f}-{max(totals):.3f} | |")
        names = {name for run in cell_runs for name in run["ops"]}
        names = [n for n in names if not args.op or any(o in n for o in args.op)]
        largest = sorted(names, key=lambda n: -max(run["ops"].get(n, [0, 0])[0] for run in cell_runs))
        for name in largest:
            seconds = [run["ops"].get(name, [0.0, 0])[0] for run in cell_runs]
            calls = max(run["ops"].get(name, [0.0, 0])[1] for run in cell_runs)
            print(f"| {architecture} | {trainer} | {name} | {min(seconds):.4f}-{max(seconds):.4f} | {calls} |")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"label": args.label, "runs": {f"{a} / {t}": r for (a, t), r in runs.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
