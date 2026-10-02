"""
The patch-attention study (the layer-norm and attention workplan, stage 6, D11): small patch
models, a conv network and a dense network on full MNIST, numpy, under one protocol.

    python scripts/patch_attention_study.py time --arm attention-ffn
    python scripts/patch_attention_study.py tune --out tune.json
    python scripts/patch_attention_study.py sweep --rates 0.001 0.001 0.001 0.001 0.001 --out sweep.json

The question (Dosovitskiy et al. 2020, "An Image is Worth 16x16 Words"): does attention, tokens
seeing each other before the mean, help a patch model, and how do patch models fare against a conv
network at this data size?

Every patch model cuts the (28, 28, 1) image into 16 patches of 7 x 7, embeds each as d = 32
features (a token-wise Dense(32, "linear", bias=True)), adds learned positions, runs its blocks,
averages over the tokens, layer-norms the mean and classifies:

- ffn: one FFN block, Residual((LayerNorm(), Dense(64, "relu"), Dense(32, "linear", bias=True))),
  so the tokens never see each other until the mean;
- attention: one attention block, Residual((LayerNorm(), Attention()));
- attention-ffn: the attention block, then the FFN block (D1).

And two controls:

- conv: the conv preset's layers (bss.CONV_SPECS, ConvSpec(3, 8) ReLU, then Dense(32) sigmoid) as a
  Sequential network, with the patch models' output layer in place of the preset's sigmoid one;
- dense: Dense(14, "relu"), about attention-ffn's parameter count.

Every network ends in Dense(10, "softmax", cross-entropy). Training: Adam at batch 32, one learning
rate per arm (`tune`: each arm over TUNE_RATES, the best mean final test accuracy), the trainer's
epoch loop (batch_size_scaling.train_epoch), fan-in-aware initialization from each seed. Recorded
per cell: test accuracy after every epoch, the parameter count, and the seconds per epoch spent in
learn_batch (OPENBLAS_NUM_THREADS=1, WORKERS jobs at once, so a comparison between arms, not an
absolute speed).

Findings, numpy, OPENBLAS_NUM_THREADS=1, 4 workers. One attention-ffn epoch alone took 10.9 s
(`time`), so the grid was sized up from the plan's short sweep: `tune` ran every arm at every rate
for 2 epochs and 3 seeds, and `sweep` ran 5 epochs and 3 seeds at each arm's best rate. Its first
two epochs repeat tune's to the digit (same seeds, same rates).

- tune: the best rates were 0.004 (ffn, attention-ffn, dense) and 0.002 (attention, conv). Every
  arm trained at every rate, and none diverged. The neighbouring rates were within about a point.
- Attention then FFN beats FFN alone, but only just: 95.82% +- 0.21% against 95.41% +- 0.45% at
  epoch 5, a gap of less than the FFN arm's standard deviation, with 63% more parameters
  (11050 against 6762) at about twice the time per epoch. Per seed: 95.58, 95.94, 95.94 against
  95.22, 95.92, 95.09. The FFN arm led at epochs 1 and 2, and attention-ffn led from epoch 3 on.
- The attention block alone is the weakest patch model, 92.43% +- 0.99%, below FFN alone at the
  same parameter count (6794 against 6762) and below the dense control. Without a token-wise
  nonlinearity, a patch's 49 values reach the mean only through affine maps and the softmax
  weights.
- The dense control, Dense(14, "relu") at attention-ffn's parameter count, reaches 93.95% +-
  0.21%, 1.9 points under attention-ffn, and is the fastest by 2x.
- The conv network wins, 98.00% +- 0.06%, 2.2 points over the best patch model, with 16x
  attention-ffn's parameters (nearly all in its Dense(32)) and 2-4x its time per epoch (sweep, tune).

So the expected result holds, weakly: attention then FFN beats FFN alone, by less than a standard
deviation over 3 seeds, and the conv network beats every patch model at this data size (ViTs need
more data; Dosovitskiy et al. 2020).

| arm | rate | parameters | epoch 1 | epoch 3 | epoch 5 | seconds per epoch |
|---|---|---|---|---|---|---|
| ffn | 0.004 | 6762 | 92.78% +- 0.39% | 94.22% +- 0.82% | 95.41% +- 0.45% | 11.0 |
| attention | 0.002 | 6794 | 86.99% +- 0.69% | 91.15% +- 0.84% | 92.43% +- 0.99% | 12.8 |
| attention-ffn | 0.004 | 11050 | 90.68% +- 0.46% | 94.78% +- 0.69% | 95.82% +- 0.21% | 25.3 |
| conv | 0.002 | 173498 | 96.89% +- 0.43% | 97.87% +- 0.06% | 98.00% +- 0.06% | 56.4 |
| dense | 0.004 | 11140 | 92.64% +- 0.27% | 93.56% +- 0.08% | 93.95% +- 0.21% | 5.9 |

Test accuracy, mean +- standard deviation over the seeds; seconds per epoch are the mean over the
seeds and epochs, and they vary with which jobs share the machine (attention-ffn's spread is 9.0
s). The per-epoch tables are in the stage's PR.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
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
DENSE_CONTROL_SIZE = 14  # 795 * 14 + 10 = 11140 parameters, attention-ffn has 11050
ARMS = ["ffn", "attention", "attention-ffn", "conv", "dense"]
BATCH_SIZE = 32
SEEDS = [0, 1, 2]
EPOCHS = 5
TUNE_EPOCHS = 2
TUNE_RATES = [0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008]
WORKERS = 4  # each worker holds its own copy of the dataset; memory, not cores, is the limit

# (arm, rate)
Config = tuple[str, float]
Datasets = tuple[list[Example[int]], list[Example[int]]]

_datasets: dict[tuple[str, str, int | None], Datasets] = {}


def _output() -> Dense:
    return Dense(bss.CLASS_COUNT, output=True, activation="softmax", loss="cross_entropy")


def _ffn_block() -> Residual:
    return Residual(
        (LayerNorm(), Dense(FFN_SIZE, activation="relu"), Dense(TOKEN_SIZE, activation="linear", bias=True))
    )


def _attention_block() -> Residual:
    return Residual((LayerNorm(), Attention()))


def arm_specs(arm: str) -> list[LayerSpec]:
    """The layers of one arm of the study."""
    if arm == "conv":
        (dense_size,) = bss.CONV_DENSE_LAYER_SIZES
        return [*bss.CONV_SPECS, Dense(dense_size), _output()]
    if arm == "dense":
        return [Dense(DENSE_CONTROL_SIZE, activation="relu"), _output()]
    blocks = {
        "ffn": [_ffn_block()],
        "attention": [_attention_block()],
        "attention-ffn": [_attention_block(), _ffn_block()],
    }
    if arm not in blocks:
        raise ValueError(f"unknown arm {arm!r}")
    return [
        Patches(PATCH_SIZE),
        Dense(TOKEN_SIZE, activation="linear", bias=True),
        Position(),
        *blocks[arm],
        TokenMean(),
        LayerNorm(),
        _output(),
    ]


def initial_network(arm: str, seed: int) -> Any:
    input_shape = (bss.DIMENSION,) if arm == "dense" else INPUT_SHAPE
    network = SequentialArrayNetwork(input_shape, arm_specs(arm), Adam(), backend=NUMPY)
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
    network = initial_network(arm, seed)

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["time", "tune", "sweep"])
    parser.add_argument("--arm", default="attention-ffn", help="time: the arm to time")
    parser.add_argument("--arms", nargs="+", default=ARMS)
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        help="tune: the rates to try (default TUNE_RATES); sweep: each arm's rate, in --arms' order",
    )
    parser.add_argument("--seeds", type=int, help="tune and sweep: default len(SEEDS)")
    parser.add_argument("--epochs", type=int, help="tune: default TUNE_EPOCHS; sweep: default EPOCHS")
    parser.add_argument("--limit", type=int, help="the first LIMIT rows of each dataset (a smoke run)")
    parser.add_argument("--out", help="every run's raw result, as JSON")
    args = parser.parse_args(argv)

    context = {
        "train_path": bss.TRAIN_PATH,
        "test_path": bss.TEST_PATH,
        "limit": args.limit,
        "epochs": args.epochs or (TUNE_EPOCHS if args.stage == "tune" else EPOCHS),
    }
    seeds = list(range(args.seeds or len(SEEDS)))
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


if __name__ == "__main__":
    main()
