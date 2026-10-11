"""
The attention-dropout study (the attention-dropout workplan, stage 8, D8;
scripts/attention_dropout_study.py): does dropout close the sequence study's train/held-out gaps,
and how much of each gap is overfitting rather than the held-out text being different?

The model and protocol are the sequence study's 2-layer arm (indrajala_ml/studies/sequence_study.py):
numpy, Adam at batch 32, the trainer's epoch loop, held-out bits per character and accuracy after
every epoch, and the training windows' bits per character (the first as many as held out),
measured in inference, so without dropout. Its rate is that study's tuned 2-layer rate, 0.002 on
each of these corpora, for every arm: an arm differs from the control by its dropout alone.

The arms (D8), GPT's dropouts by place:

- control: no dropout, the sequence study's 2-layer model;
- attention 0.1: attn_pdrop, the attention weights only (Attention(dropout));
- residual 0.1: resid_pdrop and embd_pdrop, a Dropout after the Position and ending every block's
  body;
- both 0.1, both 0.2: all three.

The corpora: Euclid (the widest gap, 0.49 bits), Tiny Shakespeare (0.3) and Herodotus (none: the
control, where dropout should only cost). The splits (text_data): contiguous, the last 10% held
out as before, and spread, each of 100 blocks' last tenth held out. A gap that the contiguous split
shows and the spread one doesn't is the text's shift from its start to its end, not overfitting.

The stages: `time` (one epoch of one cell, which sizes the grid) and `sweep` (every corpus, split
and arm). OPENBLAS_NUM_THREADS=1 and --workers jobs at once, as the sequence study.
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Any

from indrajala_ml.data import text_data
from indrajala_ml.measurement.benchmark_sweep import run_parameter_sweep
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    Dense,
    Dropout,
    Embedding,
    LayerNorm,
    LayerSpec,
    Position,
    Residual,
)
from indrajala_ml.studies import sequence_study as ss
from indrajala_ml.studies.common import mean_sd, table

CORPORA = ["euclid-heath", "tinyshakespeare", "herodotus-rawlinson"]
SPLITS: list[text_data.Split] = ["contiguous", "spread"]
# each arm's (attention dropout, residual and embedding dropout)
ARMS: dict[str, tuple[float, float]] = {
    "control": (0.0, 0.0),
    "attention 0.1": (0.1, 0.0),
    "residual 0.1": (0.0, 0.1),
    "both 0.1": (0.1, 0.1),
    "both 0.2": (0.2, 0.2),
}
# the sequence study's tuned 2-layer rate on every corpus here
RATE = 0.002
WORKERS = 6

# (corpus, split, arm)
Config = tuple[str, text_data.Split, str]
Results = dict[Config, list[dict[str, Any]]]


def arm_specs(arm: str, vocabulary: int) -> list[LayerSpec]:
    """The sequence study's 2-layer model over vocabulary characters, with arm's dropouts."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    attention, residual = ARMS[arm]
    dropped: list[LayerSpec] = [Dropout(residual)] if residual else []
    attention_block = Residual((LayerNorm(), Attention(heads=ss.HEADS, causal=True, dropout=attention), *dropped))
    ffn_block = Residual(
        (
            LayerNorm(),
            Dense(ss.FFN_SIZE, activation="relu"),
            Dense(ss.TOKEN_SIZE, activation="linear", bias=True),
            *dropped,
        )
    )
    return [
        Embedding(vocabulary, ss.TOKEN_SIZE),
        Position(),
        *dropped,
        *[attention_block, ffn_block] * 2,
        LayerNorm(),
        Dense(vocabulary, output=True, activation="softmax", loss="cross_entropy"),
    ]


def run_config(context: dict[str, Any], config: Config, seed: int) -> dict[str, Any]:
    corpus, split_by, arm = config
    train, held_out, vocabulary = ss.load(corpus, context["limit"], split_by)
    return ss.train_run(arm_specs(arm, vocabulary), train, held_out, RATE, context["epochs"], seed)


def gap(run: dict[str, Any]) -> float:
    """A run's final held-out minus training bits per character: how far it fits its own text better."""
    return ss.bits(run["held_out_cross_entropies"][-1] - run["train_cross_entropies"][-1])


def _header(epochs: int) -> list[str]:
    shown = [f"epoch {e + 1}" for e in ss.epochs_shown(epochs)]
    return ["arm", *shown, "accuracy", "train", "gap", "against control", "seconds per epoch"]


def _rows(configs: list[Config], results: Results, epochs: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for config in configs:
        corpus, split_by, arm = config
        runs, control = results[config], results.get((corpus, split_by, "control"))
        # the mean of the per-seed differences in final held-out bits: the arms share seeds
        against = (
            mean_sd(
                [
                    ss.bits(r["held_out_cross_entropies"][-1] - c["held_out_cross_entropies"][-1])
                    for r, c in zip(runs, control)
                ],
                "+.3f",
            )
            if control is not None and arm != "control"
            else ""
        )
        rows.append(
            [arm]
            + [
                mean_sd([ss.bits(run["held_out_cross_entropies"][e]) for run in runs], ".3f")
                for e in ss.epochs_shown(epochs)
            ]
            + [mean_sd([run["held_out_accuracies"][-1] for run in runs], ".2%")]
            + [mean_sd([ss.bits(run["train_cross_entropies"][-1]) for run in runs], ".3f")]
            + [mean_sd([gap(run) for run in runs], ".3f"), against]
            + [f"{statistics.mean(s for run in runs for s in run['epoch_seconds']):.1f}"]
        )
    return rows


def report(configs: list[Config], results: Results, epochs: int) -> str:
    """
    Per corpus and split, each arm: held-out bits per character (mean ± sd over the seeds) at the
    first, middle and last epochs, the final accuracy and training bits per character, the gap
    between them, the held-out difference from the control (per seed, then mean ± sd) and the
    seconds per epoch.
    """
    sections: list[str] = []
    for corpus, split_by in dict.fromkeys((corpus, split_by) for corpus, split_by, _arm in configs):
        chosen = [config for config in configs if config[:2] == (corpus, split_by)]
        sections.append(f"{corpus}, {split_by}\n\n{table(_header(epochs), _rows(chosen, results, epochs))}")
    return "\n\n".join(sections)


def main(argv: list[str] | None = None, description: str | None = None) -> None:
    """The study's command line: time or sweep."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["time", "sweep"])
    parser.add_argument("--corpus", default="herodotus-rawlinson", help="time: the corpus (default: the largest)")
    parser.add_argument("--split", default="contiguous", choices=SPLITS, help="time: the split")
    parser.add_argument("--arm", default="both 0.2", choices=list(ARMS), help="time: the arm")
    parser.add_argument("--corpora", nargs="+", default=CORPORA, choices=CORPORA)
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10, help="sweep: the epochs per run")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--limit", type=int, help="the first LIMIT windows of each split (a smoke run)")
    parser.add_argument("--out", help="every run's raw result, as JSON")
    args = parser.parse_args(argv)

    if args.stage == "time":
        epochs = 1
        configs: list[Config] = [(args.corpus, args.split, args.arm)]
        results: Results = {configs[0]: [run_config({"limit": args.limit, "epochs": 1}, configs[0], 0)]}
    else:
        epochs = args.epochs
        configs = [(c, s, a) for c in args.corpora for s in args.splits for a in args.arms]
        context = {"limit": args.limit, "epochs": epochs}
        results = run_parameter_sweep(configs, list(range(args.seeds)), run_config, context, worker_count=args.workers)

    print(report(configs, results, epochs))
    if args.out:
        with open(args.out, "w") as f:
            json.dump([{"config": list(config), "runs": runs} for config, runs in results.items()], f, indent=1)
