"""
The FFN activations study (the FFN activations workplan, stage 6, D4, D7, D11;
scripts/studies/ffn_activation_study.py): does SwiGLU, or GELU, lower the held-out bits per
character against ReLU at equal parameters?

The model is the context-length study's "256" arm (context_length_study.arm_specs): the 2-layer
rotary model, d = 64, 4 heads, no dropout, the document mask on, at windows of 256 on the packed
corpora, under the recipe (warmup over the first epoch, then cosine to a tenth of the corpus's
tuned rate, AdamW 0.1, clipping at 1.0) at the token budget (batches of 8), 30 epochs, each epoch's
training windows cut anew. Only each block's FFN changes between arms (D4, D7):

- "relu": Dense(256, "relu"), the context-length study's model itself;
- "gelu": Dense(256, "gelu"), the exact GELU, the same parameters;
- "swiglu": GatedDense(170), SwiGLU at equal parameters (33,044 to ReLU's 33,088 a block).

The ReLU arm is the context-length study's "256" runs, read from its two result files and checked
by sha256 (read_control; D11): the same model, seeds, recipe and epochs, and the ReLU arithmetic
unchanged since (the golden runs, bit-identical across stages 3 to 5). `time --arm relu --control`
reruns its first epoch and checks the held-out cross-entropy against the file's by bits. The new
arms train with context_length_study.train_packed, so a run records what that study's do: per epoch
the held-out and training cross-entropies, the held-out accuracy and the seconds in learn_batch, and
at the end the held-out bits by position.

The registered claims and refutations are the workplan's (stage 6, written before any data);
report() computes them as registered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

from indrajala_ml.measurement import machine_lock
from indrajala_ml.measurement.benchmark_sweep import run_parameter_sweep
from indrajala_ml.model.specs.layer_specs import Dense, GatedDense, LayerSpec, Residual
from indrajala_ml.studies import context_length_study as cls
from indrajala_ml.studies.common import STUDIES_DIR, provenance, read_results, write_results

STUDY = "ffn-activations"
CORPORA = cls.CORPORA
LONG_FORM = cls.LONG_FORM
SPLITS = cls.SPLITS
ARMS = ["relu", "gelu", "swiglu"]
# the arms this study trains; ReLU's runs are the context-length study's (D11)
NEW_ARMS = ["gelu", "swiglu"]
SEEDS = cls.SEEDS
TOKENS = 256
CONTROL_ARM = "256"
# the context-length study's result files, as its findings record them (its script's docstring)
CONTROL_SHA256 = {
    "sweep-herodotus-rawlinson-muqaddimah.json": "b9dbf3ee0b95ff3d485e456593c69e699de25e27ddd550b2efa19d838d450cac",
    "sweep-euclid-heath-tinyshakespeare.json": "99097142c9f26437c6e030c8e30a06fad6362c1e53c895dccd51f6016611bfe4",
}

# (corpus, split, arm)
Config = tuple[str, str, str]
Results = dict[Config, list[dict[str, Any]]]

# the registered comparisons (the workplan's stage 6): (name, arm, against, sided)
COMPARISONS = [
    ("F1 SwiGLU < ReLU", "swiglu", "relu", "one"),
    ("F2 GELU vs ReLU", "gelu", "relu", "two"),
    ("F3 SwiGLU vs GELU", "swiglu", "gelu", "two"),
]


def _hidden(arm: str, relu: Dense) -> LayerSpec:
    # the FFN's hidden layer in arm, from the ReLU one it replaces
    if arm == "relu":
        return relu
    if arm == "gelu":
        return Dense(relu.size, activation="gelu")
    if arm == "swiglu":
        return GatedDense(relu.size * 2 // 3)  # 170 at 256 (D4)
    raise ValueError(f"unknown arm {arm!r}")


def arm_specs(arm: str, symbols: int) -> list[LayerSpec]:
    """The context-length study's 256 model on a packed corpus of symbols characters, its FFNs arm's."""
    specs = cls.arm_specs(CONTROL_ARM, symbols)
    out: list[LayerSpec] = []
    for spec in specs:
        if isinstance(spec, Residual):
            body = tuple(
                _hidden(arm, layer) if isinstance(layer, Dense) and layer.activation == "relu" else layer
                for layer in spec.body
            )
            spec = Residual(body)
        out.append(spec)
    return out


def ffn_parameters(specs: Sequence[LayerSpec], features: int) -> list[int]:
    """Each FFN block's parameters, its hidden layer's and its affine layer's (D4)."""
    counts: list[int] = []
    for spec in specs:
        if not isinstance(spec, Residual):
            continue
        hidden = next(
            (
                layer
                for layer in spec.body
                if isinstance(layer, GatedDense) or (isinstance(layer, Dense) and not layer.bias)
            ),
            None,
        )
        if hidden is None:
            continue
        width = 2 * hidden.size if isinstance(hidden, GatedDense) else hidden.size
        counts.append(features * width + width + hidden.size * features + features)
    return counts


def _run_path(out_dir: Path, config: Config, seed: int) -> Path:
    corpus, split, arm = config
    return out_dir / "runs" / f"{corpus}-{split}-{arm}-seed{seed}.json"


def train_run(
    config: Config,
    seed: int,
    epochs: int,
    backend: str = "rust",
    limit: int | None = None,
    schedule_epochs: int | None = None,
) -> dict[str, Any]:
    corpus, split, arm = config
    return cls.train_packed(
        corpus, split, TOKENS, partial(arm_specs, arm), seed, epochs, backend, limit, schedule_epochs
    )


def run_config(settings: dict[str, Any], config: Config, seed: int) -> dict[str, Any]:
    """One run, or the result a stopped sweep already wrote for it."""
    path = _run_path(Path(settings["out_dir"]), config, seed)
    if path.exists():
        return json.loads(path.read_text())
    result = train_run(config, seed, settings["epochs"], settings["backend"], settings["limit"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(result))
    temporary.rename(path)
    return result


def read_control(paths: Sequence[str | Path]) -> Results:
    """The ReLU arm: the context-length study's 256 runs, from its two result files checked by sha256 (D11)."""
    control: Results = {}
    for path in paths:
        name = Path(path).name
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if CONTROL_SHA256.get(name) != digest:
            raise ValueError(f"{path}: sha256 {digest}, not the context-length study's {CONTROL_SHA256.get(name)}")
        for entry in read_results(path):
            corpus, split, arm = entry["config"]
            if arm == CONTROL_ARM:
                control[(corpus, split, "relu")] = entry["runs"]
    missing = [(c, s) for c in CORPORA for s in SPLITS if (c, s, "relu") not in control]
    if missing:
        raise ValueError(f"the control files lack the 256 runs of {missing}")
    return control


def report(results: Results, epochs: int) -> str:
    """context_length_study.report's tables and claims, over this study's arms and comparisons."""
    return cls.report(results, epochs, ARMS, COMPARISONS)


def main(argv: list[str] | None = None, description: str | None = None) -> None:
    """The study's command line: time, sweep or report."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["time", "sweep", "report"])
    parser.add_argument("--corpus", default="herodotus-rawlinson", choices=CORPORA, help="time: the corpus")
    parser.add_argument("--split", default="contiguous", choices=SPLITS, help="time: the split")
    parser.add_argument("--arm", default="swiglu", choices=ARMS, help="time: the arm")
    parser.add_argument("--corpora", nargs="+", default=CORPORA, choices=CORPORA)
    parser.add_argument("--arms", nargs="+", default=NEW_ARMS, choices=NEW_ARMS, help="sweep: the arms")
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, required=True, help="the registered epochs (time: the schedule's)")
    parser.add_argument("--backend", default="rust", choices=sorted(cls.BACKENDS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, help="the first LIMIT windows of each split (a smoke run)")
    parser.add_argument(
        "--out-dir", help="the results' directory and its runs/ (default data/studies/ffn-activations/STAGE)"
    )
    parser.add_argument("--results", nargs="+", help="report: the sweeps' result files")
    parser.add_argument(
        "--control", nargs="+", help="report, and time --arm relu: the context-length study's two result files"
    )
    args = parser.parse_args(argv)

    if args.stage == "report":
        results = cls._merged(args.results)  # pyright: ignore[reportPrivateUsage]
        results.update(read_control(args.control))
        print(report(results, args.epochs))
        return

    with machine_lock.study():
        made_by = provenance(argv)
        out_dir = Path(args.out_dir) if args.out_dir else STUDIES_DIR / STUDY / args.stage
        if args.stage == "time":
            config: Config = (args.corpus, args.split, args.arm)
            run = train_run(config, 0, 1, args.backend, args.limit, schedule_epochs=args.epochs)
            results: Results = {config: [run]}
            print(report(results, 1))
            if args.arm == "relu" and args.control:
                recorded = read_control(args.control)[config][0]["held_out_cross_entropies"][0]
                got = run["held_out_cross_entropies"][0]
                same = got.hex() == recorded.hex()
                print(
                    f"control's first epoch {'reproduced by bits' if same else 'NOT reproduced'}: {got!r} against {recorded!r}"
                )
        else:
            settings = {"limit": args.limit, "epochs": args.epochs, "backend": args.backend, "out_dir": out_dir}
            configs = [(c, s, a) for c in args.corpora for s in SPLITS for a in args.arms]
            results = run_parameter_sweep(
                configs, list(range(args.seeds)), run_config, settings, worker_count=args.workers
            )
            print(report(results, args.epochs))
        hosts = "-".join(args.corpora) if args.stage == "sweep" and args.corpora != CORPORA else "all"
        write_results(out_dir / f"{args.stage}-{args.arm if args.stage == 'time' else hosts}.json", made_by, results)
