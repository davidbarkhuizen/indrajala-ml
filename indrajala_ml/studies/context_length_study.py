"""
The context-length study (the context-length workplan, stage 9, D2, D4-D6, D8;
scripts/studies/context_length_study.py): does a longer window lower the held-out bits per character
on the long-form corpora, and does the document mask help or harm at longer windows?

The model is the training-recipe study's default (D6): the sequence study's 2-layer arm with rotary
attention and no Position (rotary_study.arm_specs), d = 64, 4 heads, no dropout, under the recipe:
warmup over the first epoch, then cosine to a tenth of the corpus's tuned rate (rotary_study.RATES),
AdamW 0.1 and clipping at 1.0. On the packed corpora (the segments workplan): Embedding(V + 2,
segments=True), the output V + 1 classes (the end-of-paragraph token predicted, D11), its target
counted. Only the window and the mask change between arms:

- "64": T = 64, the document mask on (the training-recipe study's model, packed);
- "256" and "1024": T = 256 and 1,024, the mask on;
- "256-unmasked" and "1024-unmasked": the same, attention across paragraphs (Attention's
  document_mask False), pads still masked, positions and counted targets as the masked arms'.

Every arm trains at the token budget (D5): B * T = 2,048, so a step's tokens and an epoch's steps are
the same in every arm. Each epoch's training windows are cut anew (D4, text_data.PackedEpochs) from
the paragraphs shuffled by the run's generator; the held-out windows are fixed. Every corpus
(text_data.CORPORA: Herodotus, Euclid and the Muqaddimah the long-form ones, Tiny Shakespeare a
reference outside the claims) and both held-out splits, 5 seeds; the seed fixes the initial
weights and the shuffles.

Each run records after every epoch the held-out cross-entropy and accuracy over the counted
targets, the cross-entropy of as many training windows as are held out (the first epoch's cut), and
the seconds in learn_batch; after the last epoch, the held-out bits per character by position in
the window (POSITION_BUCKETS equal buckets of positions), whether later positions, with more
context, predict better. Each run is written to its own file as it ends; a sweep started again skips
the runs it finds, so a stopped sweep resumes by whole runs.

The registered claims and refutations are the workplan's (stage 9, written before any data);
report() computes them as registered.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from indrajala_ml.data import text_data
from indrajala_ml.measurement import machine_lock
from indrajala_ml.measurement.benchmark_sweep import run_parameter_sweep
from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.layer_specs import Attention, Dense, Embedding, LayerSpec, Residual
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.studies import batch_size_scaling as bss
from indrajala_ml.studies import rotary_study
from indrajala_ml.studies import sequence_study as ss
from indrajala_ml.studies.common import STUDIES_DIR, mean_sd, provenance, read_results, table, write_results
from indrajala_ml.studies.lr_schedule_study import holm, sign_test
from indrajala_ml.training.lr_schedule import CosineDecay, Schedule, Warmup
from indrajala_ml.training.sequence_evaluate import sequence_evaluate

STUDY = "context-length"
CORPORA = list(text_data.CORPORA)
LONG_FORM = ["herodotus-rawlinson", "euclid-heath", "muqaddimah"]
SPLITS = ["contiguous", "spread"]
ARMS = ["64", "256", "256-unmasked", "1024", "1024-unmasked"]
SEEDS = 5
WARMUP_EPOCHS = 1
FLOOR = 0.1
WEIGHT_DECAY = 0.1
CLIP_NORM = 1.0
POSITION_BUCKETS = 16
BACKENDS = {"numpy": NUMPY, "rust": RUST}

# (corpus, split, arm)
Config = tuple[str, str, str]
Results = dict[Config, list[dict[str, Any]]]


def context(arm: str) -> int:
    return int(arm.split("-")[0])


def masked(arm: str) -> bool:
    return not arm.endswith("-unmasked")


def _attention(spec: LayerSpec, document_mask: bool) -> LayerSpec:
    if isinstance(spec, Residual):
        return Residual(tuple(_attention(layer, document_mask) for layer in spec.body))
    if isinstance(spec, Attention):
        return Attention(
            spec.heads, spec.key_size, spec.causal, spec.dropout, spec.positions, spec.rotary_base, document_mask
        )
    return spec


def arm_specs(arm: str, symbols: int) -> list[LayerSpec]:
    """The 2-layer rotary model on a packed corpus of symbols characters, masked as arm says."""
    specs = rotary_study.arm_specs("2-layer", symbols)
    embedding, *body, output = specs
    assert isinstance(embedding, Embedding) and isinstance(output, Dense)
    return [
        Embedding(symbols + 2, embedding.size, segments=True),
        *(_attention(spec, masked(arm)) for spec in body),
        Dense(symbols + 1, output=True, activation="softmax", loss="cross_entropy"),
    ]


def schedule(rate: float, epoch_steps: int, epochs: int) -> Schedule:
    warmup = WARMUP_EPOCHS * epoch_steps
    return Warmup(warmup, CosineDecay(rate, epochs * epoch_steps - warmup, rate * FLOOR))


def position_bits(network: Any, held_out: Sequence[Any], tokens: int, buckets: int = POSITION_BUCKETS) -> list[float]:
    """Held-out bits per character by position in the window, in buckets equal runs of positions, counted targets only."""
    classes = network.class_count
    pad = network._segments.pad  # pyright: ignore[reportPrivateUsage]
    totals = np.zeros(tokens)
    counts = np.zeros(tokens)
    start = 0
    inputs = np.array([state for state, _ in held_out], dtype=np.float64)
    labels = np.array([window for _, window in held_out], dtype=np.int64)
    for output in network.forward_rows(network.prepare_dataset(list(held_out))):
        probabilities = np.asarray(output.tolist()).reshape(-1, tokens, classes)
        stop = start + probabilities.shape[0]
        chosen = np.take_along_axis(probabilities, labels[start:stop, :, None], axis=2)[..., 0]
        counted = inputs[start:stop] != pad
        totals += np.where(counted, -np.log(chosen), 0.0).sum(axis=0)
        counts += counted.sum(axis=0)
        start = stop
    size = tokens // buckets
    return [
        ss.bits(totals[b * size : (b + 1) * size].sum() / counts[b * size : (b + 1) * size].sum())
        for b in range(buckets)
    ]


def _run_path(out_dir: Path, config: Config, seed: int) -> Path:
    corpus, split, arm = config
    return out_dir / "runs" / f"{corpus}-{split}-{arm}-seed{seed}.json"


def run_config(settings: dict[str, Any], config: Config, seed: int) -> dict[str, Any]:
    """One run, or the result a stopped sweep already wrote for it."""
    out_dir = Path(settings["out_dir"])
    path = _run_path(out_dir, config, seed)
    if path.exists():
        return json.loads(path.read_text())
    result = train_run(config, seed, settings["epochs"], settings["backend"], settings["limit"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(result))
    temporary.rename(path)
    return result


def train_run(
    config: Config,
    seed: int,
    epochs: int,
    backend: str = "numpy",
    limit: int | None = None,
    schedule_epochs: int | None = None,
) -> dict[str, Any]:
    """
    One run of epochs epochs, its schedule laid over schedule_epochs (epochs by default; `time` runs
    the first epoch of the registered schedule).
    """
    corpus, split, arm = config
    tokens = context(arm)
    epochs_data, held_out, vocabulary = text_data.load_packed_text_epochs(
        text_data.CORPORA[corpus], tokens, split_by=cast(text_data.Split, split)
    )
    held_out = held_out[:limit]
    array_backend = BACKENDS[backend]
    network: Any = SequentialArrayNetwork(
        (tokens,), arm_specs(arm, len(vocabulary)), Adam(weight_decay=WEIGHT_DECAY), "sequence", array_backend
    )
    network.rng = array_backend.default_rng(seed)
    network.randomize()
    batch = text_data.budget_batch_size(tokens)
    epoch_steps = math.ceil((len(epochs_data) if limit is None else min(limit, len(epochs_data))) / batch)
    rate = rotary_study.RATES[(corpus, "2-layer")]
    learning_rate = schedule(rate, epoch_steps, epochs if schedule_epochs is None else schedule_epochs)
    shuffle_rng = random.Random(seed)
    result: dict[str, Any] = {
        "rate": rate,
        "batch": batch,
        "steps_per_epoch": epoch_steps,
        "backend": backend,
        "held_out_cross_entropies": [],
        "held_out_accuracies": [],
        "train_cross_entropies": [],
        "epoch_seconds": [],
    }
    step = 0
    train_sample: list[Any] = []
    for _ in range(epochs):
        windows = epochs_data.epoch(shuffle_rng)[:limit]
        if not train_sample:
            train_sample = windows[: len(held_out)]
        steps, seconds = bss.train_epoch(network, windows, batch, learning_rate, step, shuffle_rng, clip_norm=CLIP_NORM)
        step += steps
        evaluation = sequence_evaluate(network, held_out)
        result["held_out_cross_entropies"].append(evaluation.cross_entropy)
        result["held_out_accuracies"].append(evaluation.accuracy)
        result["train_cross_entropies"].append(sequence_evaluate(network, train_sample).cross_entropy)
        result["epoch_seconds"].append(seconds)
    result["position_bits"] = position_bits(network, held_out, tokens)
    return result


def _final_bits(run: dict[str, Any]) -> float:
    return ss.bits(run["held_out_cross_entropies"][-1])


def differences(results: Results, arm: str, against: str, corpora: Sequence[str] = LONG_FORM) -> list[float]:
    """arm's final held-out bits minus against's, per (corpus, split, seed) of corpora present in both."""
    out: list[float] = []
    for (corpus, split, a), runs in results.items():
        if a != arm or corpus not in corpora or (corpus, split, against) not in results:
            continue
        out += [_final_bits(r) - _final_bits(c) for r, c in zip(runs, results[(corpus, split, against)], strict=True)]
    return out


# the registered comparisons (the workplan's stage 9): (name, arm, against, sided)
COMPARISONS = [
    ("C1 1024 < 64", "1024", "64", "one"),
    ("C2 256 < 64", "256", "64", "one"),
    ("C3 1024 < 256", "1024", "256", "one"),
    ("C4 masked vs unmasked at 256", "256", "256-unmasked", "two"),
    ("C5 masked vs unmasked at 1024", "1024", "1024-unmasked", "two"),
]


def claims(results: Results) -> str:
    rows: list[tuple[str, list[float], int, float, str]] = []
    for name, arm, against, sided in COMPARISONS:
        d = differences(results, arm, against)
        lower = sum(x < 0 for x in d)
        rows.append((name, d, lower, sign_test(lower, len(d), sided) if d else 1.0, sided))
    adjusted = holm([p for *_, p, _ in rows])
    lines: list[str] = []
    for (name, d, lower, p, sided), h in zip(rows, adjusted, strict=True):
        if not d:
            lines.append(f"{name}: no pairs in these results")
            continue
        verdict = ("holds" if h < 0.05 else "refuted") if sided == "one" else ("differs" if h < 0.05 else "undecided")
        lines.append(
            f"{name} ({sided}-sided): {lower} of {len(d)} pairs lower, mean {mean_sd(d, '+.3f')} bits;"
            f" p {p:.2g}, Holm {h:.2g}: {verdict}"
        )
    return "\n".join(lines)


def report(results: Results, epochs: int) -> str:
    """Per corpus and split, each arm's final held-out bits, its last three epochs' fall, training bits,
    the first and last position buckets' bits, and seconds per epoch; then the claims."""
    header = [
        "corpus",
        "split",
        "arm",
        "held out",
        "last 3 fell",
        "train",
        "first positions",
        "last positions",
        "s/epoch",
    ]
    rows: list[list[str]] = []
    for corpus in CORPORA:
        for split in SPLITS:
            for arm in ARMS:
                runs = results.get((corpus, split, arm))
                if not runs:
                    continue
                final = [_final_bits(r) for r in runs]
                fell = (
                    [ss.bits(r["held_out_cross_entropies"][-4]) - _final_bits(r) for r in runs] if epochs >= 4 else []
                )
                train = [ss.bits(r["train_cross_entropies"][-1]) for r in runs]
                first = [r["position_bits"][0] for r in runs]
                last = [r["position_bits"][-1] for r in runs]
                seconds = [sum(r["epoch_seconds"]) / len(r["epoch_seconds"]) for r in runs]
                rows.append(
                    [
                        corpus,
                        split,
                        arm,
                        mean_sd(final, ".3f"),
                        mean_sd(fell, "+.3f") if fell else "-",
                        mean_sd(train, ".3f"),
                    ]
                    + [mean_sd(first, ".3f"), mean_sd(last, ".3f"), mean_sd(seconds, ".0f")]
                )
    return table(header, rows) + "\n\n" + claims(results)


def _merged(paths: Sequence[str]) -> Results:
    results: Results = {}
    for path in paths:
        for entry in read_results(path):
            config = tuple(entry["config"])
            if config in results:
                raise ValueError(f"{path}: {config} is in two result files")
            results[config] = entry["runs"]  # type: ignore[index]
    return results


def main(argv: list[str] | None = None, description: str | None = None) -> None:
    """The study's command line: time, sweep or report."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["time", "sweep", "report"])
    parser.add_argument("--corpus", default="herodotus-rawlinson", choices=CORPORA, help="time: the corpus")
    parser.add_argument("--split", default="contiguous", choices=SPLITS, help="time: the split")
    parser.add_argument("--arm", default="1024", choices=ARMS, help="time: the arm")
    parser.add_argument("--corpora", nargs="+", default=CORPORA, choices=CORPORA)
    parser.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS, help="sweep: the arms")
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, required=True, help="the registered epochs (time: the schedule's)")
    parser.add_argument("--backend", default="numpy", choices=sorted(BACKENDS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, help="the first LIMIT windows of each split (a smoke run)")
    parser.add_argument(
        "--out-dir", help="sweep: the results' directory and its runs/ (default data/studies/context-length/sweep)"
    )
    parser.add_argument("--results", nargs="+", help="report: the sweeps' result files")
    args = parser.parse_args(argv)

    if args.stage == "report":
        results = _merged(args.results)
        print(report(results, args.epochs))
        return

    with machine_lock.study():
        made_by = provenance(argv)
        out_dir = Path(args.out_dir) if args.out_dir else STUDIES_DIR / STUDY / args.stage
        if args.stage == "time":
            config: Config = (args.corpus, args.split, args.arm)
            results: Results = {
                config: [train_run(config, 0, 1, args.backend, args.limit, schedule_epochs=args.epochs)]
            }
            print(report(results, 1))
        else:
            settings = {"limit": args.limit, "epochs": args.epochs, "backend": args.backend, "out_dir": out_dir}
            configs = [(c, s, a) for c in args.corpora for s in SPLITS for a in args.arms]
            results = run_parameter_sweep(
                configs, list(range(args.seeds)), run_config, settings, worker_count=args.workers
            )
            print(report(results, args.epochs))
        hosts = "-".join(args.corpora) if args.stage == "sweep" and args.corpora != CORPORA else "all"
        write_results(out_dir / f"{args.stage}-{hosts}.json", made_by, results)
