"""
The rotary-positions study (the rotary-positions workplan, stage 8, D8; scripts/studies/rotary_study.py):
do rotary positions inside attention match the learned Position the sequence study's transformers
add to their tokens, on every corpus, and what does the rotation cost per epoch?

The models are the sequence study's 1-layer and 2-layer arms (indrajala_ml/studies/sequence_study.py)
with their Position out and every attention rotary, Attention(positions="rotary") at the default
base: the same blocks, heads and widths, so 4,096 fewer parameters (the 64 x 64 position table).
The protocol is that study's sweep: numpy, Adam at batch 32, each corpus and arm at the sequence
study's tuned rate (RATES), 10 epochs, 5 seeds (the same seeds, so the same shuffles), held-out bits
per character and accuracy after every epoch, and the training windows' bits per character.

The baseline is the learned-position arms per seed (D8 as amended by the study-runs workplan's D9
(c)): rotary and learned share seeds, so shuffles, and their paired differences are tighter than
either arm's spread. The 2-layer arm's runs on Tiny Shakespeare, Herodotus and Euclid come from the
attention-dropout sweep's contiguous control (--control, the same model, rate, seeds and epochs),
checked against CONTROL_SHA256; the sweep reruns the learned runs that file lacks (the 1-layer arm
everywhere, the 2-layer on the Muqaddimah), or every learned run without --control. Every learned
row at 10 epochs must reproduce RECORDED's means (the sequence study's findings table), or the
sweep says which don't and exits non-zero after writing its results. D9 then decides the default:
rotary if it at least matches learned positions within the seeds' spread.

The stages: `time` (one epoch of one corpus and arm, rotary and learned in turn in one process, so
the rotation's cost is measured alone) and `sweep` (every corpus and arm). OPENBLAS_NUM_THREADS=1
and --workers jobs at once, as the sequence study.
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
from pathlib import Path
from typing import Any

from indrajala_ml.measurement import machine_lock
from indrajala_ml.measurement.benchmark_sweep import run_parameter_sweep
from indrajala_ml.model.specs.layer_specs import Attention, LayerSpec, Position, Residual
from indrajala_ml.studies import sequence_study as ss
from indrajala_ml.studies.common import default_out, mean_sd, provenance, read_results, table, write_results

CORPORA = ss.CORPORA
ARMS = ["1-layer", "2-layer"]
WORKERS = 6
# the sequence study's tuned rate per corpus and arm (its findings table)
RATES: dict[tuple[str, str], float] = {
    ("tinyshakespeare", "1-layer"): 0.004,
    ("tinyshakespeare", "2-layer"): 0.002,
    ("herodotus-rawlinson", "1-layer"): 0.004,
    ("herodotus-rawlinson", "2-layer"): 0.002,
    ("muqaddimah", "1-layer"): 0.004,
    ("muqaddimah", "2-layer"): 0.004,
    ("euclid-heath", "1-layer"): 0.002,
    ("euclid-heath", "2-layer"): 0.002,
}
# the learned-position arms' recorded sweep (the sequence study's findings table), which every learned
# row here must reproduce (unreproduced): held-out bits per character at epochs 1, 5 and 10 as
# (mean, sd) over 5 seeds, then the final accuracy, the training bits per character and the seconds
# per epoch (6 jobs at once)
RECORDED: dict[tuple[str, str], tuple[tuple[float, float], tuple[float, float], tuple[float, float], str, str, str]] = {
    ("tinyshakespeare", "1-layer"): ((2.976, 0.013), (2.666, 0.010), (2.591, 0.008), "47.24% ± 0.15%", "2.336 ± 0.010", "53.4"),
    ("tinyshakespeare", "2-layer"): ((2.977, 0.009), (2.555, 0.013), (2.476, 0.007), "49.63% ± 0.21%", "2.172 ± 0.008", "97.3"),
    ("herodotus-rawlinson", "1-layer"): ((2.490, 0.021), (2.191, 0.008), (2.119, 0.009), "55.85% ± 0.16%", "2.128 ± 0.010", "72.2"),
    ("herodotus-rawlinson", "2-layer"): ((2.442, 0.017), (2.037, 0.006), (1.950, 0.007), "59.14% ± 0.12%", "1.935 ± 0.007", "131.7"),
    ("muqaddimah", "1-layer"): ((3.199, 0.014), (2.948, 0.023), (2.887, 0.010), "44.76% ± 0.15%", "2.647 ± 0.016", "47.4"),
    ("muqaddimah", "2-layer"): ((3.203, 0.026), (2.866, 0.012), (2.794, 0.004), "46.59% ± 0.22%", "2.508 ± 0.004", "87.6"),
    ("euclid-heath", "1-layer"): ((2.018, 0.016), (1.682, 0.015), (1.622, 0.010), "70.85% ± 0.18%", "1.181 ± 0.010", "39.3"),
    ("euclid-heath", "2-layer"): ((1.959, 0.021), (1.567, 0.018), (1.503, 0.015), "72.81% ± 0.41%", "1.014 ± 0.013", "72.2"),
}  # fmt: skip

# the attention-dropout sweep's results (on jebel, data/studies/attention-dropout/sweep-2026-10-10.json):
# its contiguous control is the 2-layer learned arm on these corpora
CONTROL_SHA256 = "54f86f23da639049e2e325631cf87589fdf7aa81fcc8077bf5530de8c2a51291"
CONTROL_CORPORA = ["euclid-heath", "tinyshakespeare", "herodotus-rawlinson"]

# (corpus, arm, positions): "rotary" or "learned"
Config = tuple[str, str, str]
Results = dict[Config, list[dict[str, Any]]]


def _rotary(spec: LayerSpec) -> LayerSpec:
    if isinstance(spec, Residual):
        return Residual(tuple(_rotary(layer) for layer in spec.body))
    if isinstance(spec, Attention):
        return Attention(spec.heads, spec.key_size, spec.causal, spec.dropout, positions="rotary")
    return spec


def arm_specs(arm: str, vocabulary: int, positions: str = "rotary") -> list[LayerSpec]:
    """The sequence study's arm, its Position out and its attention rotary; "learned" is the arm as it was."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    specs = ss.arm_specs(arm, vocabulary)
    if positions == "learned":
        return specs
    return [_rotary(spec) for spec in specs if not isinstance(spec, Position)]


def run_config(context: dict[str, Any], config: Config, seed: int) -> dict[str, Any]:
    corpus, arm, positions = config
    train, held_out, vocabulary = ss.load(corpus, context["limit"])
    return ss.train_run(
        arm_specs(arm, vocabulary, positions), train, held_out, RATES[(corpus, arm)], context["epochs"], seed
    )


def read_control(path: str | Path, expected_sha256: str = CONTROL_SHA256) -> Results:
    """The 2-layer learned runs, per seed, from the attention-dropout sweep's file, checked by hash."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"{path}: sha256 {digest}, not the attention-dropout sweep's {expected_sha256}")
    control: Results = {}
    for entry in read_results(path):
        corpus, split_by, arm = entry["config"]
        if corpus in CONTROL_CORPORA and split_by == "contiguous" and arm == "control":
            control[(corpus, "2-layer", "learned")] = entry["runs"]
    if len(control) != len(CONTROL_CORPORA):
        raise ValueError(f"{path}: contiguous controls for {sorted(c for c, _a, _p in control)}, not {CONTROL_CORPORA}")
    return control


def unreproduced(results: Results) -> list[str]:
    """Each learned row at 10 epochs whose held-out means at epochs 1, 5 and 10 aren't RECORDED's."""
    wrong: list[str] = []
    for (corpus, arm, positions), runs in results.items():
        if positions != "learned" or len(runs[0]["held_out_cross_entropies"]) != 10:
            continue
        got = [
            round(statistics.mean(ss.bits(run["held_out_cross_entropies"][e]) for run in runs), 3) for e in (0, 4, 9)
        ]
        recorded = [mean for mean, _sd in RECORDED[(corpus, arm)][:3]]  # type: ignore[misc]
        if got != recorded:
            wrong.append(f"{corpus} {arm}: {got}, recorded {recorded}")
    return wrong


def _shown(values: list[float]) -> str:
    return mean_sd(values, ".3f")


def report(configs: list[Config], results: Results, epochs: int) -> str:
    """
    Per corpus, each arm learned then rotary: held-out bits per character (mean ± sd over the seeds)
    at the first, middle and last epochs, the final accuracy and training bits per character, and
    the seconds per epoch; for rotary, its final held-out bits minus learned's, per seed, then
    mean ± sd (the arms share seeds). A learned row read from --control shows that sweep's seconds.
    """
    header = ["arm", "positions", "rate", "parameters"]
    header += [f"epoch {e + 1}" for e in ss.epochs_shown(epochs)]
    header += ["accuracy", "train", "against learned", "seconds per epoch"]
    sections: list[str] = []
    for corpus in dict.fromkeys(corpus for corpus, _arm, _positions in configs):
        rows: list[list[str]] = []
        for config in [c for c in configs if c[0] == corpus]:
            _corpus, arm, positions = config
            runs, learned = results[config], results.get((corpus, arm, "learned"))
            against = ""
            if positions == "rotary" and learned is not None:
                against = mean_sd(
                    [
                        ss.bits(r["held_out_cross_entropies"][-1] - c["held_out_cross_entropies"][-1])
                        for r, c in zip(runs, learned, strict=True)
                    ],
                    "+.3f",
                )
            rows.append(
                [arm, positions, f"{RATES[(corpus, arm)]:g}", str(runs[0]["parameter_count"])]
                + [
                    _shown([ss.bits(run["held_out_cross_entropies"][e]) for run in runs])
                    for e in ss.epochs_shown(epochs)
                ]
                + [mean_sd([run["held_out_accuracies"][-1] for run in runs], ".2%")]
                + [_shown([ss.bits(run["train_cross_entropies"][-1]) for run in runs])]
                + [against]
                + [f"{statistics.mean(s for run in runs for s in run['epoch_seconds']):.1f}"]
            )
        sections.append(f"{corpus}\n\n{table(header, rows)}")
    return "\n\n".join(sections)


def main(argv: list[str] | None = None, description: str | None = None) -> None:
    """The study's command line: time or sweep."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["time", "sweep"])
    parser.add_argument("--corpus", default="herodotus-rawlinson", choices=CORPORA, help="time: the corpus")
    parser.add_argument("--arm", default="2-layer", choices=ARMS, help="time: the arm")
    parser.add_argument("--corpora", nargs="+", default=CORPORA, choices=CORPORA)
    parser.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10, help="sweep: the epochs per run")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--limit", type=int, help="the first LIMIT windows of each split (a smoke run)")
    parser.add_argument(
        "--control", help="sweep: the attention-dropout sweep's results, its 2-layer learned runs (CONTROL_SHA256)"
    )
    parser.add_argument(
        "--out", help="every run's raw result, as JSON (default: data/studies/<study>/<stage>-<time>.json)"
    )
    args = parser.parse_args(argv)

    with machine_lock.study():
        made_by = provenance(argv)
        if args.stage == "time":
            epochs = 1
            configs: list[Config] = [(args.corpus, args.arm, positions) for positions in ("learned", "rotary")]
            context = {"limit": args.limit, "epochs": 1}
            results: Results = {config: [run_config(context, config, 0)] for config in configs}
        else:
            epochs = args.epochs
            configs = [(corpus, arm, p) for corpus in args.corpora for arm in args.arms for p in ("learned", "rotary")]
            control = (
                {c: runs for c, runs in read_control(args.control).items() if c in configs} if args.control else {}
            )
            to_run = [config for config in configs if config not in control]
            context = {"limit": args.limit, "epochs": epochs}
            results = {
                **control,
                **run_parameter_sweep(to_run, list(range(args.seeds)), run_config, context, worker_count=args.workers),
            }

        print(report(configs, results, epochs))
        write_results(Path(args.out) if args.out else default_out("rotary", args.stage), made_by, results)
        if wrong := unreproduced(results):
            sys.exit("learned rows that don't reproduce the sequence study's recorded means:\n" + "\n".join(wrong))
