"""
The accuracy sweeps of the batch-size-scaling study (findings in indrajala_ml/batch_size_scaling.py),
on full MNIST with the Rust backend.

    python scripts/batch_size_scaling_sweep.py baseline --out baseline.json
    python scripts/batch_size_scaling_sweep.py scaling --lr32 0.0=3.0 --lr32 0.9=0.5 --out scaling.json
    python scripts/batch_size_scaling_sweep.py baseline --architecture conv --epochs 2 --seeds 3

`baseline` (stage 1) sweeps the batch-32 rate at momentum 0.0 and 0.9 (conv: at 0.0 only, since
no conv network has momentum). `scaling` (stage 2) runs every batch size x rate (scaled,
unscaled) x warmup x momentum cell, with each momentum's own batch-32 rate from stage 1. Both
print per-epoch test accuracy (mean ± sd over seeds) and write every run's raw result to --out as
JSON.

Each worker loads the dataset itself, once per process (about 0.5 GB each), rather than having
it pickled through the pool.
"""

import argparse
import json
import statistics
import sys

from indrajala_ml import batch_size_scaling as bss
from indrajala_ml.benchmark_sweep import run_parameter_sweep
from indrajala_ml.mnist_data import load_mnist_dataset

BASELINE_RATES = {
    "dense": [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    "conv": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
}
MOMENTA = {"dense": [0.0, 0.9], "conv": [0.0]}
BATCH_SIZES = [32, 128, 512, 1024]
WARMUP_EPOCHS = [0.0, 0.25, 1.0]
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 5
WORKERS = 4  # each worker holds its own copy of the dataset; memory, not cores, is the limit
STABLE_ACCURACY = 0.20  # chance (0.10) plus a margin: every seed must end at least here

_datasets: dict = {}


def _load(context: dict) -> tuple[list, list]:
    key = (context["train_path"], context["test_path"], context["limit"])
    if key not in _datasets:
        _datasets.clear()
        _datasets[key] = (
            load_mnist_dataset(context["train_path"], limit=context["limit"]),
            load_mnist_dataset(context["test_path"], limit=context["limit"]),
        )
    return _datasets[key]


def run_config(context: dict, config: tuple, seed: int) -> dict:
    # config: (batch_size, rate, warmup_epochs, momentum)
    batch_size, rate, warmup_epochs, momentum = config
    train_data, test_data = _load(context)
    return bss.train_and_evaluate(
        "rust",
        train_data,
        test_data,
        batch_size,
        rate,
        warmup_epochs,
        momentum,
        context["epochs"],
        seed,
        context["architecture"],
    )


def _mean_sd(values: list[float]) -> str:
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.2%} ± {sd:.2%}"


def per_epoch_row(runs: list[dict]) -> list[str]:
    epochs = len(runs[0]["test_accuracies"])
    return [_mean_sd([run["test_accuracies"][e] for run in runs]) for e in range(epochs)]


def final_accuracies(runs: list[dict]) -> list[float]:
    return [run["test_accuracies"][-1] for run in runs]


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def baseline(context: dict, seeds: list[int], rates: list[float], momenta: list[float]) -> dict:
    configs = [(bss.BASE_BATCH_SIZE, rate, 0.0, momentum) for momentum in momenta for rate in rates]
    results = run_parameter_sweep(configs, seeds, run_config, context, worker_count=WORKERS, report_progress=True)

    epochs = context["epochs"]
    for momentum in momenta:
        print(f"\n### momentum {momentum}\n")
        rows = []
        for rate in rates:
            runs = results[(bss.BASE_BATCH_SIZE, rate, 0.0, momentum)]
            finals = final_accuracies(runs)
            stable = min(finals) >= STABLE_ACCURACY
            rows.append([f"{rate:g}"] + per_epoch_row(runs) + [f"{min(finals):.2%}", "yes" if stable else "no"])
        print(_table(["rate"] + [f"epoch {e + 1}" for e in range(epochs)] + ["worst seed", "stable"], rows))

        stable_rates = [
            rate
            for rate in rates
            if min(final_accuracies(results[(bss.BASE_BATCH_SIZE, rate, 0.0, momentum)])) >= STABLE_ACCURACY
        ]
        if stable_rates:
            best = max(
                stable_rates,
                key=lambda rate: statistics.mean(final_accuracies(results[(bss.BASE_BATCH_SIZE, rate, 0.0, momentum)])),
            )
            finals = final_accuracies(results[(bss.BASE_BATCH_SIZE, best, 0.0, momentum)])
            print(f"\nlr_32 = {best:g}: batch-32 band {_mean_sd(finals)} (min {min(finals):.2%}, max {max(finals):.2%})")
    return results


def scaling(context: dict, seeds: list[int], lr32: dict[float, float], batch_sizes: list[int], warmups: list[float]) -> dict:
    configs = []
    labels = {}
    for momentum, base_rate in lr32.items():
        for batch_size in batch_sizes:
            for warmup in warmups:
                for rate_kind, rate in (("scaled", bss.scaled_learning_rate(base_rate, batch_size)), ("unscaled", base_rate)):
                    # at batch 32 the two rates coincide; run the cell once
                    if batch_size == bss.BASE_BATCH_SIZE and rate_kind == "unscaled":
                        continue
                    config = (batch_size, rate, warmup, momentum)
                    configs.append(config)
                    labels[config] = rate_kind
    results = run_parameter_sweep(configs, seeds, run_config, context, worker_count=WORKERS, report_progress=True)

    epochs = context["epochs"]
    train_size = context["train_size"]
    for momentum, base_rate in lr32.items():
        band_runs = results[(bss.BASE_BATCH_SIZE, base_rate, 0.0, momentum)]
        band = final_accuracies(band_runs)
        low, high = min(band), max(band)
        print(f"\n### momentum {momentum}, lr_32 = {base_rate:g}")
        print(f"batch-32 band (no warmup, final epoch): {_mean_sd(band)}, seeds span {low:.2%} - {high:.2%}\n")
        rows = []
        for config in configs:
            batch_size, rate, warmup, config_momentum = config
            if config_momentum != momentum:
                continue
            runs = results[config]
            mean = statistics.mean(final_accuracies(runs))
            steps = bss.warmup_steps(warmup, train_size, batch_size)
            rows.append(
                [str(batch_size), labels[config], f"{rate:g}", f"{warmup:g} ({steps})", str(runs[0]["steps"])]
                + per_epoch_row(runs)
                + ["yes" if low <= mean <= high else ("above" if mean > high else "no")]
            )
        print(
            _table(
                ["B", "rate", "value", "warmup epochs (steps)", "total steps"]
                + [f"epoch {e + 1}" for e in range(epochs)]
                + ["in band"],
                rows,
            )
        )
    return results


def _parse_lr32(values: list[str]) -> dict[float, float]:
    pairs = [value.split("=") for value in values]
    return {float(momentum): float(rate) for momentum, rate in pairs}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["baseline", "scaling"])
    parser.add_argument("--architecture", choices=bss.ARCHITECTURES, default="dense")
    parser.add_argument("--lr32", action="append", default=[], help="momentum=rate, once per momentum (scaling)")
    parser.add_argument("--out", help="write every run's raw result here as JSON")
    parser.add_argument("--limit", type=int, help="use only the first LIMIT train and test rows (smoke runs)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    args = parser.parse_args(argv)

    context = {
        "train_path": bss.TRAIN_PATH,
        "test_path": bss.TEST_PATH,
        "limit": args.limit,
        "epochs": args.epochs,
        "train_size": args.limit or 60000,
        "architecture": args.architecture,
    }
    seeds = SEEDS[: args.seeds]

    if args.stage == "baseline":
        results = baseline(context, seeds, BASELINE_RATES[args.architecture], MOMENTA[args.architecture])
    else:
        lr32 = _parse_lr32(args.lr32)
        if not lr32:
            sys.exit("scaling needs --lr32 momentum=rate for each momentum")
        results = scaling(context, seeds, lr32, BATCH_SIZES, WARMUP_EPOCHS)

    if args.out:
        with open(args.out, "w") as f:
            json.dump([{"config": list(config), "runs": runs} for config, runs in results.items()], f, indent=1)


if __name__ == "__main__":
    main()
