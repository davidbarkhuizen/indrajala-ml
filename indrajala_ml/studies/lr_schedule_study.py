"""
The training-recipe study (the training-recipe workplan, stage 7, D7, D8, D11;
scripts/studies/lr_schedule_study.py): does the production recipe's learning-rate schedule, warmup
then cosine decay to a tenth of the peak, end the sequence models lower than the constant rate they
have trained at, and what do AdamW's decoupled weight decay and clipping by the global norm add?

The model is the 2-layer transformer steps 7 to 9 settle on: the sequence study's 2-layer arm with
rotary attention and no Position (the rotary-positions workplan's D9; rotary_study.arm_specs), and no
dropout (the attention-dropout study: at this size dropout raised the held-out loss everywhere but
across Euclid's shift). numpy, Adam at batch 32, every corpus (text_data.CORPORA) and both held-out
splits (contiguous: the last 10%; spread: each of 100 blocks' last tenth), 30 epochs, 5 seeds (the
same seeds, so the same initial weights and shuffles, in every arm).

The three arms (D7), each at the sequence study's tuned 2-layer rate for the corpus as its peak
(D8; rotary_study.RATES):

- constant: the rate at every step, as every earlier study trained;
- cosine: Warmup(one epoch, CosineDecay(rate, the other 29 epochs, rate / 10)), linear warmup then
  half a cosine to a tenth (nanoGPT's min_lr = lr / 10), ending at the last step;
- recipe: the cosine schedule with Adam(weight_decay=0.1) and clip_norm=1.0, GPT-3's recipe (Brown
  et al. 2020, appendix B).

Each run records after every epoch the held-out cross-entropy and accuracy, the cross-entropy of
the first as many training windows as are held out, and the seconds in learn_batch; seed 0's final
network is saved with its tokenizer (scripts/generation/sample_text.py samples it). Each run's
result is written to its own file as it finishes (runs/ under the results' directory), and a sweep
started again skips the runs it finds there, so a stopped sweep resumes by whole runs.

The registered claims (D11, written before the data), on the final-epoch held-out bits per
character, paired by (corpus, split, seed): 40 pairs per comparison, a screen at 5 seeds a cell.

- C1, cosine against constant, one-sided (cosine lower: a decaying rate ends nearer a minimum than
  the constant rate's noise allows, the workplan's Why): holds when at least 26 of the 40 pairs are
  lower under cosine (a sign test, one-sided p < 0.05 after Holm over C1 and C2); refuted otherwise.
  Under D8 the cosine's peak is the constant arm's tuned rate, so a loss is reported as possibly an
  under-tuned peak, not as the schedule failing.
- C2, recipe against cosine, two-sided (no direction registered: weight decay and clipping are
  regularization and stability, whose effect at this size and length isn't predicted): differs when
  the two-sided sign test over the 40 pairs is below 0.05 after Holm (at most 13 or at least 27 of
  40 lower); undecided otherwise.
- Descriptive, not tested: per cell, the mean ± sd of each arm's final held-out bits and their
  paired differences; how far each arm's held-out loss fell over its last three epochs (whether it
  converged); the training bits (the gap); the seconds per epoch.

The stages: `time` (one epoch of one corpus, split and arm), `sweep` (every cell, or --corpora's),
and `report` (the tables and claims from one or more sweeps' result files, e.g. two hosts' halves).
OPENBLAS_NUM_THREADS=1 and --workers jobs at once.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from indrajala_ml.data import text_data
from indrajala_ml.generation.tokenizer import CharacterTokenizer, tokenizer_path
from indrajala_ml.measurement import machine_lock
from indrajala_ml.measurement.benchmark_sweep import run_parameter_sweep
from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.studies import batch_size_scaling as bss
from indrajala_ml.studies import rotary_study
from indrajala_ml.studies import sequence_study as ss
from indrajala_ml.studies.common import STUDIES_DIR, mean_sd, provenance, read_results, table, write_results
from indrajala_ml.training.lr_schedule import Constant, CosineDecay, Schedule, Warmup
from indrajala_ml.training.sequence_evaluate import sequence_evaluate

STUDY = "lr-schedule"
CORPORA = ss.CORPORA
SPLITS: list[text_data.Split] = ["contiguous", "spread"]
ARMS = ["constant", "cosine", "recipe"]
ARM = "2-layer"
EPOCHS = 30
SEEDS = 5
WARMUP_EPOCHS = 1
FLOOR = 0.1  # of the peak
WEIGHT_DECAY = 0.1
CLIP_NORM = 1.0
WORKERS = 6

# (corpus, split, arm)
Config = tuple[str, str, str]
Results = dict[Config, list[dict[str, Any]]]


def steps_per_epoch(train_size: int) -> int:
    return math.ceil(train_size / ss.BATCH_SIZE)


def schedule(arm: str, rate: float, epoch_steps: int, epochs: int) -> Schedule:
    """The arm's learning rate: constant, or warmup over the first epoch then cosine to rate * FLOOR at the end."""
    if arm == "constant":
        return Constant(rate)
    warmup = WARMUP_EPOCHS * epoch_steps
    return Warmup(warmup, CosineDecay(rate, epochs * epoch_steps - warmup, rate * FLOOR))


def rule(arm: str) -> Adam:
    return Adam(weight_decay=WEIGHT_DECAY) if arm == "recipe" else Adam()


def clip_norm(arm: str) -> float | None:
    return CLIP_NORM if arm == "recipe" else None


def _run_path(out_dir: Path, config: Config, seed: int) -> Path:
    corpus, split, arm = config
    return out_dir / "runs" / f"{corpus}-{split}-{arm}-seed{seed}.json"


def run_config(context: dict[str, Any], config: Config, seed: int) -> dict[str, Any]:
    """One run, or the result a stopped sweep already wrote for it."""
    out_dir = Path(context["out_dir"])
    path = _run_path(out_dir, config, seed)
    if path.exists():
        return json.loads(path.read_text())
    result = train_run(config, seed, context["epochs"], context["limit"], out_dir if seed == 0 else None)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(result))
    temporary.rename(path)
    return result


def train_run(
    config: Config,
    seed: int,
    epochs: int,
    limit: int | None,
    save_dir: Path | None,
    schedule_epochs: int | None = None,
) -> dict[str, Any]:
    """
    One run of epochs epochs, its schedule laid over schedule_epochs (epochs by default; `time` runs
    the first epoch of a 30-epoch schedule).
    """
    corpus, split, arm = config
    train, held_out, vocabulary = ss.load(corpus, limit, cast(text_data.Split, split))
    network: Any = SequentialArrayNetwork(
        (ss.CONTEXT,), rotary_study.arm_specs(ARM, vocabulary), rule(arm), "sequence", NUMPY
    )
    network.rng = NUMPY.default_rng(seed)
    network.randomize()
    epoch_steps = steps_per_epoch(len(train))
    rate = rotary_study.RATES[(corpus, ARM)]
    learning_rate = schedule(arm, rate, epoch_steps, epochs if schedule_epochs is None else schedule_epochs)
    train_sample = train[: len(held_out)]
    shuffle_rng = random.Random(seed)
    result: dict[str, Any] = {
        "rate": rate,
        "steps_per_epoch": epoch_steps,
        "held_out_cross_entropies": [],
        "held_out_accuracies": [],
        "train_cross_entropies": [],
        "epoch_seconds": [],
    }
    step = 0
    for _ in range(epochs):
        steps, seconds = bss.train_epoch(
            network, train, ss.BATCH_SIZE, learning_rate, step, shuffle_rng, clip_norm=clip_norm(arm)
        )
        step += steps
        evaluation = sequence_evaluate(network, held_out)
        result["held_out_cross_entropies"].append(evaluation.cross_entropy)
        result["held_out_accuracies"].append(evaluation.accuracy)
        result["train_cross_entropies"].append(sequence_evaluate(network, train_sample).cross_entropy)
        result["epoch_seconds"].append(seconds)
    if save_dir is not None:
        model = save_dir / "models" / f"{corpus}-{split}-{arm}.json"
        model.parent.mkdir(parents=True, exist_ok=True)
        network.save(str(model))
        CharacterTokenizer.of(corpus, text_data.load_text(text_data.CORPORA[corpus])).save(tokenizer_path(model))
        result["model"] = str(model)
    return result


def _final_bits(run: dict[str, Any]) -> float:
    return ss.bits(run["held_out_cross_entropies"][-1])


def sign_test(lower: int, pairs: int, sided: str) -> float:
    """The sign test's p for lower of pairs below 0 (ties none: the losses are continuous)."""
    upper_tail = sum(math.comb(pairs, k) for k in range(lower, pairs + 1)) / 2**pairs
    if sided == "one":
        return upper_tail
    lower_tail = sum(math.comb(pairs, k) for k in range(lower + 1)) / 2**pairs
    return min(1.0, 2 * min(upper_tail, lower_tail))


def holm(ps: Sequence[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adjusted = [0.0] * len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


def differences(results: Results, arm: str, against: str) -> list[float]:
    """arm's final held-out bits minus against's, per (corpus, split, seed) present in both."""
    out: list[float] = []
    for (corpus, split, a), runs in results.items():
        if a != arm or (corpus, split, against) not in results:
            continue
        out += [_final_bits(r) - _final_bits(c) for r, c in zip(runs, results[(corpus, split, against)], strict=True)]
    return out


def claims(results: Results) -> str:
    c1, c2 = differences(results, "cosine", "constant"), differences(results, "recipe", "cosine")
    lower1, lower2 = sum(d < 0 for d in c1), sum(d < 0 for d in c2)
    p1, p2 = sign_test(lower1, len(c1), "one"), sign_test(lower2, len(c2), "two")
    h1, h2 = holm([p1, p2])
    return "\n".join(
        [
            (
                f"C1 cosine < constant (one-sided): {lower1} of {len(c1)} pairs lower, mean {mean_sd(c1, '+.3f')} bits;"
                f" p {p1:.2g}, Holm {h1:.2g}: {'holds' if h1 < 0.05 else 'refuted'}"
            ),
            (
                f"C2 recipe vs cosine (two-sided): {lower2} of {len(c2)} pairs lower, mean {mean_sd(c2, '+.3f')} bits;"
                f" p {p2:.2g}, Holm {h2:.2g}: {'differs' if h2 < 0.05 else 'undecided'}"
            ),
        ]
    )


def report(results: Results, epochs: int) -> str:
    """Per corpus and split, each arm's held-out bits over the epochs, its last three epochs' fall, the
    final accuracy and training bits, its difference to the arm before it per seed, and seconds."""
    header = ["split", "arm", *[f"epoch {e + 1}" for e in ss.epochs_shown(epochs)]]
    header += ["last 3 fell", "accuracy", "train", "against", "seconds per epoch"]
    sections: list[str] = []
    for corpus in dict.fromkeys(c for c, _s, _a in results):
        rows: list[list[str]] = []
        for split in SPLITS:
            for arm in ARMS:
                runs = results.get((corpus, split, arm))
                if runs is None:
                    continue
                previous = {"cosine": "constant", "recipe": "cosine"}.get(arm)
                against = ""
                if previous and (corpus, split, previous) in results:
                    paired = [
                        _final_bits(r) - _final_bits(c)
                        for r, c in zip(runs, results[(corpus, split, previous)], strict=True)
                    ]
                    against = f"{mean_sd(paired, '+.3f')} vs {previous}"
                fell = [
                    ss.bits(run["held_out_cross_entropies"][-4] - run["held_out_cross_entropies"][-1])
                    for run in runs
                    if len(run["held_out_cross_entropies"]) >= 4
                ]
                rows.append(
                    [split, arm]
                    + [
                        mean_sd([ss.bits(run["held_out_cross_entropies"][e]) for run in runs], ".3f")
                        for e in ss.epochs_shown(epochs)
                    ]
                    + [mean_sd(fell, "+.3f") if fell else ""]
                    + [mean_sd([run["held_out_accuracies"][-1] for run in runs], ".2%")]
                    + [mean_sd([ss.bits(run["train_cross_entropies"][-1]) for run in runs], ".3f")]
                    + [against]
                    + [f"{statistics.mean(s for run in runs for s in run['epoch_seconds']):.1f}"]
                )
        sections.append(f"{corpus}\n\n{table(header, rows)}")
    return "\n\n".join(sections)


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
    parser.add_argument("--arm", default="recipe", choices=ARMS, help="time: the arm")
    parser.add_argument("--corpora", nargs="+", default=CORPORA, choices=CORPORA)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--limit", type=int, help="the first LIMIT windows of each split (a smoke run)")
    parser.add_argument(
        "--out-dir",
        help="sweep: the results' directory, its runs/ and models/ (default data/studies/lr-schedule/sweep)",
    )
    parser.add_argument("--results", nargs="+", help="report: the sweeps' result files")
    args = parser.parse_args(argv)

    if args.stage == "report":
        results = _merged(args.results)
        epochs = len(next(iter(results.values()))[0]["held_out_cross_entropies"])
        print(report(results, epochs))
        print()
        print(claims(results))
        return

    with machine_lock.study():
        made_by = provenance(argv)
        out_dir = Path(args.out_dir) if args.out_dir else STUDIES_DIR / STUDY / args.stage
        epochs: int = args.epochs if args.stage == "sweep" else 1
        context = {"limit": args.limit, "epochs": epochs, "out_dir": out_dir}
        if args.stage == "time":
            config: Config = (args.corpus, args.split, args.arm)
            results: Results = {config: [train_run(config, 0, 1, args.limit, None, schedule_epochs=args.epochs)]}
        else:
            configs = [(c, s, a) for c in args.corpora for s in SPLITS for a in ARMS]
            results = run_parameter_sweep(
                configs, list(range(args.seeds)), run_config, context, worker_count=args.workers
            )
        print(report(results, epochs))
        hosts = "-".join(args.corpora) if args.stage == "sweep" and args.corpora != CORPORA else "all"
        write_results(out_dir / f"{args.stage}-{hosts}.json", made_by, results)
