"""
The sequence task's study (the sequence task workplan, stage 8, D9; scripts/sequence_study.py):
next-character prediction on the four corpora (D2), numpy, Adam at batch 32, one learning rate per
corpus and arm, the trainer's epoch loop, and the held-out windows' mean per-token cross-entropy in
bits per character and per-token accuracy after every epoch (D8).

The arms (D9):

- unigram and bigram: each character's frequency, and each character's frequency after the one
  before it, counted over the training windows' (input, label) pairs with add-one smoothing (a
  character of the vocabulary can be missing from the training text): the floors, not trained;
- ffn: the embedding, positions and one FFN block, so each token sees only itself and its
  position;
- 1-layer and 2-layer: causal transformers, a causal attention block of HEADS heads then an FFN
  block per layer;
- 2-layer-unmasked: 2-layer without the mask, the leak: position t reads the token it predicts.

Every trained arm embeds the CONTEXT token ids as TOKEN_SIZE features with learned positions, and
ends in a layer norm and a token-wise softmax over the corpus's characters. Its windows are
text_data's: non-overlapping CONTEXT + 1 characters, the first 90% of the text for training.

A run (run_config) initializes fan-in-aware from its seed and records, after every epoch, the
held-out cross-entropy and accuracy, the cross-entropy on the first training windows (as many as
held out, so the gap between them shows), and the seconds spent in learn_batch. The stages: `time`
(one epoch of one corpus and arm, which sizes the grid), `tune` (each corpus and arm over the
rates, the lowest mean final held-out cross-entropy) and `sweep` (each corpus and arm at its tuned
rate), with the floors counted beside them.

OPENBLAS_NUM_THREADS=1 and --workers jobs at once, so seconds per epoch compare arms within one
study, not machines.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

from indrajala_ml.data import text_data
from indrajala_ml.measurement.benchmark_sweep import run_parameter_sweep
from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.protocols.classifier_protocols import Example
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    Dense,
    Embedding,
    LayerNorm,
    LayerSpec,
    Position,
    Residual,
)
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.studies import batch_size_scaling as bss
from indrajala_ml.studies.patch_study import parameter_count
from indrajala_ml.training.sequence_evaluate import SequenceEvaluation, sequence_evaluate

CORPORA = list(text_data.CORPORA)
CONTEXT = text_data.CONTEXT
TOKEN_SIZE = 64
HEADS = 4
FFN_SIZE = 4 * TOKEN_SIZE
BATCH_SIZE = 32
TUNE_EPOCHS = 2
TUNE_RATES = [0.00025, 0.0005, 0.001, 0.002, 0.004]
WORKERS = 4
COUNTED_ARMS = ["unigram", "bigram"]
TRAINED_ARMS = ["ffn", "1-layer", "2-layer", "2-layer-unmasked"]

Window = Example[tuple[int, ...]]
Datasets = tuple[list[Window], list[Window], int]
# (corpus, arm, rate)
Config = tuple[str, str, float]
Results = dict[Config, list[dict[str, Any]]]

_datasets: dict[tuple[str, int | None], Datasets] = {}


def attention_block(causal: bool = True) -> Residual:
    return Residual((LayerNorm(), Attention(heads=HEADS, causal=causal)))


def ffn_block() -> Residual:
    return Residual(
        (LayerNorm(), Dense(FFN_SIZE, activation="relu"), Dense(TOKEN_SIZE, activation="linear", bias=True))
    )


def arm_specs(arm: str, vocabulary: int) -> list[LayerSpec]:
    """The layers of one trained arm over a corpus of vocabulary characters."""
    blocks: dict[str, list[LayerSpec]] = {
        "ffn": [ffn_block()],
        "1-layer": [attention_block(), ffn_block()],
        "2-layer": [attention_block(), ffn_block()] * 2,
        "2-layer-unmasked": [attention_block(causal=False), ffn_block()] * 2,
    }
    if arm not in blocks:
        raise ValueError(f"unknown trained arm {arm!r}")
    return [
        Embedding(vocabulary, TOKEN_SIZE),
        Position(),
        *blocks[arm],
        LayerNorm(),
        Dense(vocabulary, output=True, activation="softmax", loss="cross_entropy"),
    ]


def initial_network(specs: list[LayerSpec], seed: int) -> Any:
    network = SequentialArrayNetwork((CONTEXT,), specs, Adam(), "sequence", NUMPY)
    network.rng = NUMPY.default_rng(seed)
    network.randomize()
    return network


def load(corpus: str, limit: int | None = None) -> Datasets:
    """The corpus's training and held-out windows (the first limit of each, for a smoke run) and its
    vocabulary's size; one corpus held per process."""
    key = (corpus, limit)
    if key not in _datasets:
        _datasets.clear()
        train, held_out, vocabulary = text_data.load_text_dataset(text_data.CORPORA[corpus])
        _datasets[key] = (train[:limit], held_out[:limit], len(vocabulary))
    return _datasets[key]


def _pairs(windows: Sequence[Window]) -> tuple[np.ndarray, np.ndarray]:
    # every (input id, label) pair: the label is the character after the input at each position
    inputs = np.array([state for state, _labels in windows], dtype=np.int64).ravel()
    labels = np.array([labels for _state, labels in windows], dtype=np.int64).ravel()
    return inputs, labels


def counted(arm: str, train: Sequence[Window], held_out: Sequence[Window], vocabulary: int) -> SequenceEvaluation:
    """A floor's held-out cross-entropy and accuracy: unigram or bigram frequencies counted over
    train with add-one smoothing."""
    train_inputs, train_labels = _pairs(train)
    inputs, labels = _pairs(held_out)
    if arm == "unigram":
        counts = np.bincount(train_labels, minlength=vocabulary) + 1.0
        probabilities = np.broadcast_to(counts / counts.sum(), (labels.size, vocabulary))
    elif arm == "bigram":
        counts = np.ones((vocabulary, vocabulary))
        np.add.at(counts, (train_inputs, train_labels), 1.0)
        probabilities = (counts / counts.sum(axis=1, keepdims=True))[inputs]
    else:
        raise ValueError(f"unknown counted arm {arm!r}")
    chosen = probabilities[np.arange(labels.size), labels]
    return SequenceEvaluation(
        float(-np.log(chosen).mean()), float((probabilities.argmax(axis=1) == labels).mean()), int(labels.size)
    )


def run_config(context: dict[str, Any], config: Config, seed: int) -> dict[str, Any]:
    corpus, arm, rate = config
    train, held_out, vocabulary = load(corpus, context["limit"])
    network = initial_network(arm_specs(arm, vocabulary), seed)
    train_sample = train[: len(held_out)]

    shuffle_rng = random.Random(seed)
    result: dict[str, Any] = {
        "held_out_cross_entropies": [],
        "held_out_accuracies": [],
        "train_cross_entropies": [],
        "epoch_seconds": [],
        "parameter_count": parameter_count(network),
    }
    step = 0
    for _ in range(context["epochs"]):
        steps, seconds = bss.train_epoch(network, train, BATCH_SIZE, rate, step, shuffle_rng)
        step += steps
        evaluation = sequence_evaluate(network, held_out)
        result["held_out_cross_entropies"].append(evaluation.cross_entropy)
        result["held_out_accuracies"].append(evaluation.accuracy)
        result["train_cross_entropies"].append(sequence_evaluate(network, train_sample).cross_entropy)
        result["epoch_seconds"].append(seconds)
    return result


def bits(nats: float) -> float:
    return nats / math.log(2.0)


def _mean_sd(values: list[float], fmt: str) -> str:
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{format(statistics.mean(values), fmt)} ± {format(sd, fmt)}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def _final(runs: list[dict[str, Any]], key: str = "held_out_cross_entropies") -> list[float]:
    return [run[key][-1] for run in runs]


def _epochs_shown(epochs: int) -> list[int]:
    # the first, middle and last epochs, as indices
    return sorted({0, (epochs - 1) // 2, epochs - 1})


def _header(epochs: int) -> list[str]:
    shown = [f"epoch {e + 1}" for e in _epochs_shown(epochs)]
    return ["arm", "rate", "parameters", *shown, "accuracy", "train", "seconds per epoch"]


def _rows(configs: list[Config], results: Results, epochs: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for config in configs:
        runs = results[config]
        _corpus, arm, rate = config
        rows.append(
            [arm, f"{rate:g}", str(runs[0]["parameter_count"])]
            + [
                _mean_sd([bits(run["held_out_cross_entropies"][e]) for run in runs], ".3f")
                for e in _epochs_shown(epochs)
            ]
            + [_mean_sd(_final(runs, "held_out_accuracies"), ".2%")]
            + [_mean_sd([bits(value) for value in _final(runs, "train_cross_entropies")], ".3f")]
            + [_mean_sd([s for run in runs for s in run["epoch_seconds"]], ".1f")]
        )
    return rows


def _floor_rows(corpus: str, limit: int | None, epochs: int) -> list[list[str]]:
    train, held_out, vocabulary = load(corpus, limit)
    rows: list[list[str]] = []
    for arm in COUNTED_ARMS:
        evaluation = counted(arm, train, held_out, vocabulary)
        train_evaluation = counted(arm, train, train[: len(held_out)], vocabulary)
        held = f"{evaluation.bits_per_token:.3f}"
        rows.append(
            [arm, "counted", str(vocabulary if arm == "unigram" else vocabulary**2)]
            + [held] * len(_epochs_shown(epochs))
            + [f"{evaluation.accuracy:.2%}", f"{train_evaluation.bits_per_token:.3f}", ""]
        )
    return rows


def report(configs: list[Config], results: Results, epochs: int, limit: int | None) -> str:
    """Per corpus, the floors and then each trained arm: held-out bits per character (mean ± sd over
    the seeds) at the first, middle and last epochs, the final per-token accuracy and training bits
    per character, and the seconds per epoch."""
    sections: list[str] = []
    for corpus in dict.fromkeys(corpus for corpus, _arm, _rate in configs):
        rows = _floor_rows(corpus, limit, epochs) + _rows([c for c in configs if c[0] == corpus], results, epochs)
        sections.append(f"{corpus}\n\n{_table(_header(epochs), rows)}")
    return "\n\n".join(sections)


def best_rates(results: Results) -> dict[tuple[str, str], float]:
    """Each corpus and arm's rate with the lowest mean final held-out cross-entropy, of those whose
    every run stayed finite."""
    best: dict[tuple[str, str], tuple[float, float]] = {}
    for (corpus, arm, rate), runs in results.items():
        finals = _final(runs)
        if not all(math.isfinite(value) for value in finals):
            continue
        mean = statistics.mean(finals)
        if (corpus, arm) not in best or mean < best[(corpus, arm)][1]:
            best[(corpus, arm)] = (rate, mean)
    return {key: rate for key, (rate, _mean) in best.items()}


def read_results(path: str) -> Results:
    with open(path) as f:
        return {
            (entry["config"][0], entry["config"][1], float(entry["config"][2])): entry["runs"] for entry in json.load(f)
        }


def main(argv: list[str] | None = None, description: str | None = None) -> None:
    """The study's command line: time, tune or sweep."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["time", "tune", "sweep"])
    parser.add_argument("--corpus", default="herodotus-rawlinson", help="time: the corpus (default: the largest)")
    parser.add_argument("--arm", default="2-layer", help="time: the arm")
    parser.add_argument("--corpora", nargs="+", default=CORPORA, choices=CORPORA)
    parser.add_argument("--arms", nargs="+", default=TRAINED_ARMS, choices=TRAINED_ARMS)
    parser.add_argument("--rates", type=float, nargs="+", help="time: its rate; tune: default TUNE_RATES")
    parser.add_argument("--tuned", help="sweep: tune's --out, from which each corpus and arm takes its best rate")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, help=f"tune: default {TUNE_EPOCHS}; sweep: required")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--limit", type=int, help="the first LIMIT windows of each split (a smoke run)")
    parser.add_argument("--out", help="every run's raw result, as JSON")
    args = parser.parse_args(argv)

    seeds = list(range(args.seeds))
    if args.stage == "time":
        epochs = 1
        configs: list[Config] = [(args.corpus, args.arm, (args.rates or [0.001])[0])]
        results: Results = {configs[0]: [run_config({"limit": args.limit, "epochs": 1}, configs[0], 0)]}
    elif args.stage == "tune":
        epochs = args.epochs or TUNE_EPOCHS
        rates = args.rates or TUNE_RATES
        configs = [(corpus, arm, rate) for corpus in args.corpora for arm in args.arms for rate in rates]
        context = {"limit": args.limit, "epochs": epochs}
        results = run_parameter_sweep(configs, seeds, run_config, context, worker_count=args.workers)
    else:
        if args.tuned is None or args.epochs is None:
            sys.exit("sweep needs --tuned, tune's --out, and --epochs")
        tuned = best_rates(read_results(args.tuned))
        missing = [(corpus, arm) for corpus in args.corpora for arm in args.arms if (corpus, arm) not in tuned]
        if missing:
            sys.exit(f"no finite tuned rate for {missing}")
        epochs = args.epochs
        configs = [(corpus, arm, tuned[(corpus, arm)]) for corpus in args.corpora for arm in args.arms]
        context = {"limit": args.limit, "epochs": epochs}
        results = run_parameter_sweep(configs, seeds, run_config, context, worker_count=args.workers)

    print(report(configs, results, epochs, args.limit))
    if args.stage == "tune":
        print()
        for (corpus, arm), rate in best_rates(results).items():
            mean = statistics.mean(_final(results[(corpus, arm, rate)]))
            print(f"best rate for {corpus} {arm}: {rate:g} ({bits(mean):.3f} bits per character)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump([{"config": list(config), "runs": runs} for config, runs in results.items()], f, indent=1)
