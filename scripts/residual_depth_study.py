"""
The residual-connections depth study (the residual-connections workplan, stage 6, D9): plain and
residual networks of increasing depth on full MNIST, numpy, with and without batch norm.

    python scripts/residual_depth_study.py tune --out tune.json
    python scripts/residual_depth_study.py sweep --rate 0.05 --out sweep.json

The question (He et al. 2016, "Identity Mappings in Deep Residual Networks"): does a plain network
get worse with depth where a residual one doesn't?

Every network is an input projection Dense(64, "relu"), then `depth` hidden layers of 64, then a
softmax output with the cross-entropy loss:

- plain: `depth` layers Dense(64, "relu");
- residual: depth / 2 blocks, each Residual((Dense(64, "relu"), Dense(64, "linear", bias=True)));
- with batch norm, each ReLU layer above is Dense(64, "linear"), BatchNorm("relu") instead, the
  projection's too.

So a plain and a residual network of one depth have the same number of weighted layers (depth + 2,
the output's included) and differ in the identity paths and in the affine layers that end the
blocks (the plain network's are ReLU). Training: SGD with momentum 0.9 at batch 32, one learning
rate tuned (`tune`) on the shallowest plain network without batch norm, the trainer's epoch loop
(batch_size_scaling.train_epoch), fan-in-aware initialization from each seed. Recorded per cell:
test accuracy after every epoch, and the norm of the first layer's mean gradient (its W's) at
initialization, over the first 256 training examples.

Findings, 3 epochs, 3 seeds, numpy, rate 0.025. tune ran with the default OpenBLAS threads and
sweep with OPENBLAS_NUM_THREADS=1 (4 workers on 4 cores); the depth-2 plain cell, in both, has the
same accuracies to the digit:

- tune: 0.025 is best (96.91% +- 0.05% at epoch 3), 0.0125 within the band (96.82% +- 0.28%); 0.05
  and 0.1 lose 1 and 3 points and 0.2 diverges.
- Without batch norm the plain network fails from depth 8: every seed ends predicting one class
  (11.35%, 10.10% or 10.28%, the test shares of 1, 3 and 7), the same at 16. The first layer's
  gradient at initialization shrinks about 6x per two layers, as the initialization predicts:
  uniform(+-1/sqrt(n)) has Var(W) = 1/(3n), so a ReLU layer scales the backward signal's variance
  by n Var(W) / 2 = 1/6 (measured: 6.2x from depth 2 to 4, 42x from 4 to 8 against 36, 1230x from
  8 to 16 against 1296). The residual network holds 96.8-96.9% at every depth, its gradient norm
  near 0.5-0.6.
- With batch norm the plain network trains at every depth, but loses accuracy as it deepens
  (97.52% at 2, 95.49% at 16; at 16 the seeds' standard deviation is 3 points in the first two
  epochs), and its first layer's gradient grows tenfold. The residual network loses 0.3 points
  from 2 to 16 and its gradient grows 1.8x.
- At depth 2 the arms tie, within a standard deviation, with or without batch norm.

So the expected result holds: plain networks get worse with depth and residual ones don't.

| depth | without BN: plain | residual | gradient at init: plain | residual |
|---|---|---|---|---|
| 2 | 96.91% +- 0.05% | 96.87% +- 0.06% | 0.0728 | 0.508 |
| 4 | 96.36% +- 0.42% | 96.92% +- 0.18% | 0.0118 | 0.538 |
| 8 | 10.99% +- 0.62% | 96.80% +- 0.30% | 0.000284 | 0.577 |
| 16 | 10.99% +- 0.62% | 96.89% +- 0.13% | 2.31e-07 | 0.598 |

| depth | with BN: plain | residual | gradient at init: plain | residual |
|---|---|---|---|---|
| 2 | 97.52% +- 0.05% | 97.39% +- 0.22% | 2.76 | 3.17 |
| 4 | 97.17% +- 0.28% | 97.43% +- 0.09% | 3.23 | 3.59 |
| 8 | 96.78% +- 0.37% | 97.36% +- 0.21% | 6.45 | 4.38 |
| 16 | 95.49% +- 0.24% | 97.10% +- 0.28% | 27.6 | 5.62 |

Test accuracy at epoch 3, mean +- standard deviation over the seeds; the gradient norm's mean. The
per-epoch tables are in the stage's PR.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import sys
from typing import Any

import numpy as np

from indrajala_ml import batch_size_scaling as bss
from indrajala_ml.benchmark_sweep import run_parameter_sweep
from indrajala_ml.mnist_data import load_mnist_dataset
from indrajala_ml.model.array_backend import NUMPY
from indrajala_ml.model.classifier_protocols import Example
from indrajala_ml.model.layer_specs import BatchNorm, Dense, LayerSpec, Residual
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.update_rules import Momentum
from indrajala_ml.multiclass_evaluate import accuracy

WIDTH = 64
DEPTHS = [2, 4, 8, 16]
ARMS = ["plain", "residual"]
MOMENTUM = 0.9
BATCH_SIZE = 32
SEEDS = [0, 1, 2]
EPOCHS = 3
TUNE_RATES = [0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2]
GRADIENT_EXAMPLES = 256
WORKERS = 4  # each worker holds its own copy of the dataset; memory, not cores, is the limit

# (arm, depth, batch norm, rate)
Config = tuple[str, int, bool, float]
Datasets = tuple[list[Example[int]], list[Example[int]]]

_datasets: dict[tuple[str, str, int | None], Datasets] = {}


def _relu_layer(batch_norm: bool) -> list[LayerSpec]:
    return [Dense(WIDTH, activation="linear"), BatchNorm("relu")] if batch_norm else [Dense(WIDTH, activation="relu")]


def depth_specs(arm: str, depth: int, batch_norm: bool) -> list[LayerSpec]:
    """A network of the study: the input projection, `depth` hidden layers of WIDTH, the output."""
    assert depth >= 2 and depth % 2 == 0, f"a depth is a whole number of blocks of two layers; got {depth}"
    layers = _relu_layer(batch_norm)
    if arm == "plain":
        for _ in range(depth):
            layers += _relu_layer(batch_norm)
    elif arm == "residual":
        for _ in range(depth // 2):
            layers.append(Residual((*_relu_layer(batch_norm), Dense(WIDTH, activation="linear", bias=True))))
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return [*layers, Dense(bss.CLASS_COUNT, output=True, activation="softmax", loss="cross_entropy")]


def initial_network(arm: str, depth: int, batch_norm: bool, seed: int) -> Any:
    network = SequentialArrayNetwork(
        (bss.DIMENSION,), depth_specs(arm, depth, batch_norm), Momentum(MOMENTUM), backend=NUMPY
    )
    network.rng = NUMPY.default_rng(seed)
    network.randomize()
    return network


class _FirstLayerGradient:
    """Stands in for the network's optimizer for one learn_batch: keeps the first layer's grad_W."""

    def __init__(self) -> None:
        self.grad_W: Any = None

    def begin_step(self) -> None:
        pass

    def apply(self, index: int, layer: Any, _learning_rate: float, _batch_size: int) -> None:
        if index == 0:
            self.grad_W = layer.grad_W.copy()
        if hasattr(layer, "reset_gradient_accum"):
            layer.reset_gradient_accum()


def initial_gradient_norm(network: Any, examples: list[Example[int]]) -> float:
    """The L2 norm of the first layer's mean gradient over examples, on a copy of the network, so
    batch norm's running statistics and the gradient accumulators are as they were."""
    probe = copy.deepcopy(network)
    recorder = _FirstLayerGradient()
    probe.optimizer = recorder
    probe.learn_batch(1.0, examples)
    gradient = np.asarray(recorder.grad_W, dtype=np.float64) / len(examples)
    return float(np.linalg.norm(gradient))


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
    arm, depth, batch_norm, rate = config
    train_data, test_data = _load(context)
    network = initial_network(arm, depth, batch_norm, seed)
    gradient_norm = initial_gradient_norm(network, train_data[:GRADIENT_EXAMPLES])

    shuffle_rng = random.Random(seed)
    test_accuracies: list[float] = []
    step = 0
    for _ in range(context["epochs"]):
        steps, _seconds = bss.train_epoch(network, train_data, BATCH_SIZE, rate, step, shuffle_rng)
        step += steps
        test_accuracies.append(accuracy(network, test_data))
    return {"test_accuracies": test_accuracies, "initial_gradient_norm": gradient_norm}


def _mean_sd(values: list[float], fmt: str = ".2%") -> str:
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{format(statistics.mean(values), fmt)} ± {format(sd, fmt)}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def _finals(runs: list[dict[str, Any]]) -> list[float]:
    return [run["test_accuracies"][-1] for run in runs]


def tune(context: dict[str, Any], seeds: list[int], rates: list[float]) -> dict[Config, list[dict[str, Any]]]:
    configs: list[Config] = [("plain", min(DEPTHS), False, rate) for rate in rates]
    results = run_parameter_sweep(configs, seeds, run_config, context, worker_count=WORKERS)
    rows = [
        [f"{config[3]:g}"]
        + [_mean_sd([run["test_accuracies"][e] for run in results[config]]) for e in range(context["epochs"])]
        for config in configs
    ]
    print(_table(["rate"] + [f"epoch {e + 1}" for e in range(context["epochs"])], rows))
    finite = [config for config in configs if all(math.isfinite(a) for a in _finals(results[config]))]
    best = max(finite, key=lambda config: statistics.mean(_finals(results[config])))
    print(f"\nbest rate: {best[3]:g} ({_mean_sd(_finals(results[best]))})")
    return results


def sweep(
    context: dict[str, Any], seeds: list[int], rate: float, depths: list[int]
) -> dict[Config, list[dict[str, Any]]]:
    configs: list[Config] = [
        (arm, depth, batch_norm, rate) for batch_norm in (False, True) for depth in depths for arm in ARMS
    ]
    results = run_parameter_sweep(configs, seeds, run_config, context, worker_count=WORKERS)
    for batch_norm in (False, True):
        print(f"\n### {'with' if batch_norm else 'without'} batch norm, rate {rate:g}\n")
        rows: list[list[str]] = []
        for depth in depths:
            for arm in ARMS:
                runs = results[(arm, depth, batch_norm, rate)]
                rows.append(
                    [str(depth), arm]
                    + [_mean_sd([run["test_accuracies"][e] for run in runs]) for e in range(context["epochs"])]
                    + [_mean_sd([run["initial_gradient_norm"] for run in runs], ".3g")]
                )
        print(
            _table(
                ["depth", "arm"]
                + [f"epoch {e + 1}" for e in range(context["epochs"])]
                + ["first layer's gradient norm at init"],
                rows,
            )
        )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["tune", "sweep"])
    parser.add_argument("--rate", type=float, help="sweep: the rate tune chose")
    parser.add_argument("--rates", type=float, nargs="+", default=TUNE_RATES, help="tune: the rates to try")
    parser.add_argument("--depths", type=int, nargs="+", default=DEPTHS)
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--limit", type=int, help="the first LIMIT rows of each dataset (a smoke run)")
    parser.add_argument("--out", help="every run's raw result, as JSON")
    args = parser.parse_args(argv)

    context = {
        "train_path": bss.TRAIN_PATH,
        "test_path": bss.TEST_PATH,
        "limit": args.limit,
        "epochs": args.epochs,
    }
    seeds = list(range(args.seeds))
    if args.stage == "tune":
        results = tune(context, seeds, args.rates)
    else:
        if args.rate is None:
            sys.exit("sweep needs --rate, the rate tune chose")
        results = sweep(context, seeds, args.rate, args.depths)
    if args.out:
        with open(args.out, "w") as f:
            json.dump([{"config": list(config), "runs": runs} for config, runs in results.items()], f, indent=1)


if __name__ == "__main__":
    main()
