"""
The grouped-query attention study (the grouped-query attention workplan, stage 8, D7, D13;
scripts/studies/gqa_study.py): what do fewer key/value heads cost in held-out bits per character,
trained from scratch and uptrained from a multi-head model, against the cache they save?

The model is the FFN study's SwiGLU arm (ffn_activation_study.arm_specs("swiglu")): the context-length
study's "256" model, the 2-layer rotary model, d = 64, no dropout, the document mask on, at windows
of 256 on the packed corpora, under the recipe (warmup over the first epoch, then cosine to a tenth
of the corpus's tuned rate, AdamW 0.1, clipping at 1.0) at the token budget (batches of 8), each
epoch's training windows cut anew, with GatedDense(170) SwiGLU FFNs (D13). Its attention has 8 query
heads of d_k = 8, sharing g key/value heads (D7):

- "g8", "g4", "g2", "g1": trained from scratch for the registered epochs at g = 8 (multi-head), 4, 2
  and 1 (multi-query); g8's final network is saved (models/), the uptraining's start;
- "up4", "up2", "up1": the same seed's g8 network converted to g key/value heads (to_grouped: each
  group's key and value projections its heads' mean, D5; a fresh optimizer state), then trained for
  UPTRAIN_EPOCHS under the same recipe, its schedule over those epochs (Ainslie et al. 2023, §2.1).

The scratch stage ("sweep") must end before the uptraining stage ("uptrain") of the same corpora
starts: each uptraining run reads its g8 network. Runs write their results as they end, so a stopped
stage resumes by whole runs. The registered claims are the workplan's (stage 8, written before any
data); report() computes them as registered.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

from indrajala_ml.measurement import machine_lock
from indrajala_ml.measurement.benchmark_sweep import run_parameter_sweep
from indrajala_ml.model.persistence.grouping import to_grouped
from indrajala_ml.model.specs.layer_specs import Attention, LayerSpec, Residual
from indrajala_ml.studies import context_length_study as cls
from indrajala_ml.studies import ffn_activation_study as ffn
from indrajala_ml.studies.common import STUDIES_DIR, provenance, write_results

STUDY = "grouped-query-attention"
# the long-form corpora only (D7): Tiny Shakespeare is outside this study
CORPORA = cls.LONG_FORM
SPLITS = cls.SPLITS
SEEDS = cls.SEEDS
TOKENS = 256
HEADS = 8
SCRATCH_ARMS = ["g8", "g4", "g2", "g1"]
UPTRAIN_ARMS = ["up4", "up2", "up1"]
ARMS = SCRATCH_ARMS + UPTRAIN_ARMS
# the uptraining's epochs: Ainslie et al.'s 5% of the original steps is 1.5 of 30 epochs (D7)
UPTRAIN_EPOCHS = 2

# (corpus, split, arm)
Config = tuple[str, str, str]
Results = dict[Config, list[dict[str, Any]]]

# the registered comparisons (the workplan's stage 8): (name, arm, against, sided); "one" is arm lower
COMPARISONS = [
    ("G1 MHA < MQA", "g8", "g1", "one"),
    ("G2 GQA-2 vs MHA", "g2", "g8", "two"),
    ("G3 GQA-2 < MQA", "g2", "g1", "one"),
    ("U1 uptrained vs scratch at g 2", "up2", "g2", "two"),
]


def key_value_heads(arm: str) -> int:
    """The arm's key/value heads g: g8 to g1, up4 to up1."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    return int(arm.removeprefix("up").removeprefix("g"))


def arm_specs(arm: str, symbols: int) -> list[LayerSpec]:
    """The SwiGLU 256 model on a packed corpus of symbols characters, its attention 8 heads sharing g."""
    g = key_value_heads(arm)

    def attention(spec: LayerSpec) -> LayerSpec:
        if isinstance(spec, Residual):
            return Residual(tuple(attention(layer) for layer in spec.body))
        if isinstance(spec, Attention):
            return replace(spec, heads=HEADS, key_size=None, key_value_heads=None if g == HEADS else g)
        return spec

    return [attention(spec) for spec in ffn.arm_specs("swiglu", symbols)]


def cache_bytes_per_token(arm: str, features: int = 64, layers: int = 2) -> int:
    """The decode cache's bytes per token: every layer's keys and values, g d_k float64s each."""
    return layers * 2 * key_value_heads(arm) * (features // HEADS) * 8


def _run_path(out_dir: Path, config: Config, seed: int) -> Path:
    corpus, split, arm = config
    return out_dir / "runs" / f"{corpus}-{split}-{arm}-seed{seed}.json"


def model_path(models_dir: Path, corpus: str, split: str, seed: int) -> Path:
    """Where the g8 run of (corpus, split, seed) saves its network, the uptraining's start."""
    return models_dir / f"{corpus}-{split}-g8-seed{seed}.json"


def train_run(
    config: Config,
    seed: int,
    epochs: int,
    models_dir: Path,
    backend: str = "rust",
    limit: int | None = None,
    schedule_epochs: int | None = None,
) -> dict[str, Any]:
    """One run of config: scratch for epochs (g8's network saved), or uptraining from g8's for UPTRAIN_EPOCHS."""
    corpus, split, arm = config
    specs = partial(arm_specs, arm)
    if arm in UPTRAIN_ARMS:
        source = model_path(models_dir, corpus, split, seed)
        if not source.exists():
            raise FileNotFoundError(f"{source}: the g8 run of {corpus} {split} seed {seed} hasn't ended")
        start = to_grouped(json.loads(source.read_text()), key_value_heads(arm))
        return cls.train_packed(
            corpus, split, TOKENS, specs, seed, epochs, backend, limit, schedule_epochs, start=start
        )
    save = model_path(models_dir, corpus, split, seed) if arm == "g8" else None
    return cls.train_packed(corpus, split, TOKENS, specs, seed, epochs, backend, limit, schedule_epochs, save=save)


def run_config(settings: dict[str, Any], config: Config, seed: int) -> dict[str, Any]:
    """One run, or the result a stopped stage already wrote for it."""
    path = _run_path(Path(settings["out_dir"]), config, seed)
    if path.exists():
        return json.loads(path.read_text())
    result = train_run(
        config, seed, settings["epochs"], Path(settings["models_dir"]), settings["backend"], settings["limit"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(result))
    temporary.rename(path)
    return result


def report(results: Results, epochs: int) -> str:
    """context_length_study.report's tables and claims over this study's arms, then each arm's cache."""
    cache = "\n".join(f"{arm}: {cache_bytes_per_token(arm)} cache bytes a token" for arm in SCRATCH_ARMS)
    return cls.report(results, epochs, ARMS, COMPARISONS) + "\n\n" + cache


def main(argv: list[str] | None = None, description: str | None = None) -> None:
    """The study's command line: time, sweep (scratch), uptrain or report."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["time", "sweep", "uptrain", "report"])
    parser.add_argument("--corpus", default="herodotus-rawlinson", choices=CORPORA, help="time: the corpus")
    parser.add_argument("--split", default="contiguous", choices=SPLITS, help="time: the split")
    parser.add_argument("--arm", default="g8", choices=SCRATCH_ARMS, help="time: the arm")
    parser.add_argument("--corpora", nargs="+", default=CORPORA, choices=CORPORA)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, required=True, help="the registered epochs (time: the schedule's)")
    parser.add_argument("--backend", default="rust", choices=sorted(cls.BACKENDS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, help="the first LIMIT windows of each split (a smoke run)")
    parser.add_argument("--out-dir", help="the stage's results and its runs/ (default data/studies/STUDY/STAGE)")
    parser.add_argument("--models-dir", help="g8's saved networks (default data/studies/STUDY/models)")
    parser.add_argument("--results", nargs="+", help="report: the stages' result files")
    args = parser.parse_args(argv)

    if args.stage == "report":
        print(report(cls._merged(args.results), args.epochs))  # pyright: ignore[reportPrivateUsage]
        return

    with machine_lock.study():
        made_by = provenance(argv)
        out_dir = Path(args.out_dir) if args.out_dir else STUDIES_DIR / STUDY / args.stage
        models_dir = Path(args.models_dir) if args.models_dir else STUDIES_DIR / STUDY / "models"
        if args.stage == "time":
            config: Config = (args.corpus, args.split, args.arm)
            run = cls.train_packed(
                args.corpus,
                args.split,
                TOKENS,
                partial(arm_specs, args.arm),
                0,
                1,
                args.backend,
                args.limit,
                schedule_epochs=args.epochs,
            )
            results: Results = {config: [run]}
            print(report(results, 1))
        else:
            arms = SCRATCH_ARMS if args.stage == "sweep" else UPTRAIN_ARMS
            epochs = args.epochs if args.stage == "sweep" else UPTRAIN_EPOCHS
            settings = {
                "limit": args.limit,
                "epochs": epochs,
                "backend": args.backend,
                "out_dir": out_dir,
                "models_dir": models_dir,
            }
            # g8 first, so its networks are saved early
            configs = [(c, s, a) for a in arms for c in args.corpora for s in SPLITS]
            results = run_parameter_sweep(
                configs, list(range(args.seeds)), run_config, settings, worker_count=args.workers
            )
            print(report(results, epochs))
        hosts = "-".join(args.corpora) if args.stage != "time" and args.corpora != CORPORA else "all"
        write_results(out_dir / f"{args.stage}-{args.arm if args.stage == 'time' else hosts}.json", made_by, results)
