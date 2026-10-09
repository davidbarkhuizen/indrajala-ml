"""
The patch-model studies' protocol (scripts/patch_attention_study.py, the layer-norm and attention
workplan's D11; scripts/multi_head_attention_study.py, the multi-head attention workplan's D8): full
MNIST, numpy, Adam at batch 32, one learning rate per arm, the trainer's epoch loop.

A study names its arms and a function from an arm to its layers; everything else is shared:

- the patch models' parts: 16 patches of 7 x 7 embedded as d = 32 features with learned
  positions, FFN blocks 64 wide, the mean over the tokens, a layer norm and a softmax output;
- a run (run_config): fan-in-aware initialization from the seed, the epochs, test accuracy after
  every epoch, the parameter count, and the seconds per epoch spent in learn_batch;
- the stages: `time` (one epoch of one arm, which sizes the grid), `tune` (each arm over the rates,
  the best mean final test accuracy) and `sweep` (each arm at its rate).

OPENBLAS_NUM_THREADS=1 and WORKERS jobs at once, so seconds per epoch compare arms within one
study, not machines.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from indrajala_ml.data.mnist_data import load_mnist_dataset
from indrajala_ml.measurement.benchmark_sweep import run_parameter_sweep
from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.protocols.classifier_protocols import Example
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    Dense,
    LayerNorm,
    LayerSpec,
    Patches,
    Position,
    Residual,
    TokenMean,
)
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.studies import batch_size_scaling as bss
from indrajala_ml.training.multiclass_evaluate import accuracy

INPUT_SHAPE = (bss.SIDE, bss.SIDE, 1)
PATCH_SIZE = 7  # 16 tokens of 49
TOKEN_SIZE = 32
FFN_SIZE = 64
BATCH_SIZE = 32
TUNE_EPOCHS = 2
TUNE_RATES = [0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008]
WORKERS = 4  # each worker holds its own copy of the dataset; memory, not cores, is the limit

# (arm, rate)
Config = tuple[str, float]
Datasets = tuple[list[Example[int]], list[Example[int]]]
# a study's arm to its layers; module-level, since run_parameter_sweep pickles the context
ArmSpecs = Callable[[str], list[LayerSpec]]

_datasets: dict[tuple[str, str, int | None], Datasets] = {}


def output() -> Dense:
    return Dense(bss.CLASS_COUNT, output=True, activation="softmax", loss="cross_entropy")


def ffn_block() -> Residual:
    return Residual(
        (LayerNorm(), Dense(FFN_SIZE, activation="relu"), Dense(TOKEN_SIZE, activation="linear", bias=True))
    )


def attention_block(attention: Attention | None = None) -> Residual:
    return Residual((LayerNorm(), attention or Attention()))


def patch_model(blocks: Sequence[LayerSpec]) -> list[LayerSpec]:
    """The patches, embedding and positions, then blocks, then the mean, a layer norm and the output."""
    return [
        Patches(PATCH_SIZE),
        Dense(TOKEN_SIZE, activation="linear", bias=True),
        Position(),
        *blocks,
        TokenMean(),
        LayerNorm(),
        output(),
    ]


def initial_network(specs: list[LayerSpec], seed: int) -> Any:
    # a network starting with a dense layer (a dense control) reads the flat row
    input_shape = (bss.DIMENSION,) if isinstance(specs[0], Dense) else INPUT_SHAPE
    network = SequentialArrayNetwork(input_shape, specs, Adam(), backend=NUMPY)
    network.rng = NUMPY.default_rng(seed)
    network.randomize()
    return network


def parameter_count(network: Any) -> int:
    """Every learned value: the arrays of the network's snapshot (no arm has batch norm's running
    statistics)."""

    def count(state: Any) -> int:
        if isinstance(state, np.ndarray):
            return state.size
        return sum(count(part) for part in state)

    return count(network.snapshot())


def _load(context: dict[str, Any]) -> Datasets:
    key = (context["train_path"], context["test_path"], context["limit"])
    if key not in _datasets:
        _datasets.clear()
        _datasets[key] = (
            load_mnist_dataset(context["train_path"], limit=context["limit"]),
            load_mnist_dataset(context["test_path"], limit=context["limit"]),
        )
    return _datasets[key]


def run_config(context: dict[str, Any], config: Config, seed: int) -> dict[str, Any]:
    arm, rate = config
    train_data, test_data = _load(context)
    arm_specs: ArmSpecs = context["arm_specs"]
    network = initial_network(arm_specs(arm), seed)

    shuffle_rng = random.Random(seed)
    test_accuracies: list[float] = []
    epoch_seconds: list[float] = []
    step = 0
    for _ in range(context["epochs"]):
        steps, seconds = bss.train_epoch(network, train_data, BATCH_SIZE, rate, step, shuffle_rng)
        step += steps
        epoch_seconds.append(seconds)
        test_accuracies.append(accuracy(network, test_data))
    return {
        "test_accuracies": test_accuracies,
        "epoch_seconds": epoch_seconds,
        "parameter_count": parameter_count(network),
    }


def _mean_sd(values: list[float], fmt: str = ".2%") -> str:
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{format(statistics.mean(values), fmt)} ± {format(sd, fmt)}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def _finals(runs: list[dict[str, Any]]) -> list[float]:
    return [run["test_accuracies"][-1] for run in runs]


def _rows(configs: list[Config], results: dict[Config, list[dict[str, Any]]], epochs: int) -> list[list[str]]:
    return [
        [arm, f"{rate:g}", str(results[(arm, rate)][0]["parameter_count"])]
        + [_mean_sd([run["test_accuracies"][e] for run in results[(arm, rate)]]) for e in range(epochs)]
        + [_mean_sd([s for run in results[(arm, rate)] for s in run["epoch_seconds"]], ".1f")]
        for arm, rate in configs
    ]


def _header(epochs: int) -> list[str]:
    return ["arm", "rate", "parameters"] + [f"epoch {e + 1}" for e in range(epochs)] + ["seconds per epoch"]


def time_one(context: dict[str, Any], arm: str, rate: float) -> dict[Config, list[dict[str, Any]]]:
    """One epoch of one arm in this process: the cost that sizes the grid."""
    result = run_config({**context, "epochs": 1}, (arm, rate), 0)
    print(_table(_header(1), _rows([(arm, rate)], {(arm, rate): [result]}, 1)))
    return {(arm, rate): [result]}


def tune(
    context: dict[str, Any], seeds: list[int], arms: list[str], rates: list[float]
) -> dict[Config, list[dict[str, Any]]]:
    configs: list[Config] = [(arm, rate) for arm in arms for rate in rates]
    results = run_parameter_sweep(configs, seeds, run_config, context, worker_count=WORKERS)
    print(_table(_header(context["epochs"]), _rows(configs, results, context["epochs"])))
    print()
    for arm in arms:
        finite = [c for c in configs if c[0] == arm and all(math.isfinite(a) for a in _finals(results[c]))]
        best = max(finite, key=lambda config: statistics.mean(_finals(results[config])))
        print(f"best rate for {arm}: {best[1]:g} ({_mean_sd(_finals(results[best]))})")
    return results


def sweep(
    context: dict[str, Any], seeds: list[int], arms: list[str], rates: list[float]
) -> dict[Config, list[dict[str, Any]]]:
    configs: list[Config] = list(zip(arms, rates, strict=True))
    results = run_parameter_sweep(configs, seeds, run_config, context, worker_count=WORKERS)
    print(_table(_header(context["epochs"]), _rows(configs, results, context["epochs"])))
    return results


def main(
    argv: list[str] | None,
    description: str | None,
    arms: list[str],
    arm_specs: ArmSpecs,
    seed_count: int,
    epochs: int,
    time_arm: str,
) -> None:
    """A study's command line: time, tune or sweep its arms."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["time", "tune", "sweep"])
    parser.add_argument("--arm", default=time_arm, help="time: the arm to time")
    parser.add_argument("--arms", nargs="+", default=arms)
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        help="tune: the rates to try (default TUNE_RATES); sweep: each arm's rate, in --arms' order",
    )
    parser.add_argument("--seeds", type=int, help=f"tune and sweep: default {seed_count}")
    parser.add_argument("--epochs", type=int, help=f"tune: default {TUNE_EPOCHS}; sweep: default {epochs}")
    parser.add_argument("--limit", type=int, help="the first LIMIT rows of each dataset (a smoke run)")
    parser.add_argument("--out", help="every run's raw result, as JSON")
    args = parser.parse_args(argv)

    context = {
        "train_path": bss.TRAIN_PATH,
        "test_path": bss.TEST_PATH,
        "limit": args.limit,
        "epochs": args.epochs or (TUNE_EPOCHS if args.stage == "tune" else epochs),
        "arm_specs": arm_specs,
    }
    seeds = list(range(args.seeds or seed_count))
    if args.stage == "time":
        results = time_one(context, args.arm, (args.rates or [0.001])[0])
    elif args.stage == "tune":
        results = tune(context, seeds, args.arms, args.rates or TUNE_RATES)
    else:
        if args.rates is None or len(args.rates) != len(args.arms):
            sys.exit("sweep needs --rates, one per arm, the rates tune chose")
        results = sweep(context, seeds, args.arms, args.rates)
    if args.out:
        with open(args.out, "w") as f:
            json.dump([{"config": list(config), "runs": runs} for config, runs in results.items()], f, indent=1)
