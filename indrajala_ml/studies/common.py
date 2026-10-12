"""
The studies' shared protocol helpers: a per-process cache of the MNIST train and test sets, the
mean ± sd cells and Markdown tables their reports print, and their result files: every run's raw
result with the provenance of the sweep that made it, by default under data/studies/<study>/
(docs/study-runs-workplan.md, D8). Used by indrajala_ml/studies/patch_study.py,
indrajala_ml/studies/sequence_study.py, scripts/residual_depth_study.py and
scripts/batch_size_scaling_sweep.py.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib
import json
import socket
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from indrajala_ml.data.mnist_data import load_mnist_dataset
from indrajala_ml.model.protocols.classifier_protocols import Example

MnistDatasets = tuple[list[Example[int]], list[Example[int]]]

# one entry: a sweep worker keeps the last (train path, test path, limit) it loaded
_mnist_datasets: dict[tuple[str, str, int | None], MnistDatasets] = {}


def load_mnist(context: dict[str, Any]) -> MnistDatasets:
    key = (context["train_path"], context["test_path"], context["limit"])
    if key not in _mnist_datasets:
        _mnist_datasets.clear()
        _mnist_datasets[key] = (
            load_mnist_dataset(context["train_path"], limit=context["limit"]),
            load_mnist_dataset(context["test_path"], limit=context["limit"]),
        )
    return _mnist_datasets[key]


def mean_sd(values: list[float], fmt: str = ".2%") -> str:
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{format(statistics.mean(values), fmt)} ± {format(sd, fmt)}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


STUDIES_DIR = Path("data/studies")


def default_out(study: str, stage: str) -> Path:
    """data/studies/<study>/<stage>-<YYYY-MM-DD-HHMM>.json, in local time."""
    stamp = datetime.datetime.now(datetime.UTC).astimezone().strftime("%Y-%m-%d-%H%M")
    return STUDIES_DIR / study / f"{stage}-{stamp}.json"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()


def crate_hash() -> str:
    """
    The first 12 hex digits of the crate's extension module's sha256: the build, which the
    package's __init__.py (the same in every build) never identified.
    """
    extension = importlib.import_module("indrajala_math_rust.indrajala_math_rust")
    return hashlib.sha256(Path(str(extension.__file__)).read_bytes()).hexdigest()[:12]


def provenance(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    What made a sweep's results, taken when it starts: the commit (and whether the tracked tree
    differed from it), the crate extension's hash, the machine, the command line and the time.
    """
    words = sys.argv if argv is None else [sys.argv[0], *argv]
    return {
        "commit": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "crate": crate_hash(),
        "machine": socket.gethostname(),
        "command": " ".join([Path(words[0]).name, *words[1:]]),
        "started": datetime.datetime.now(datetime.UTC).astimezone().isoformat(timespec="seconds"),
    }


def write_results(path: Path, made_by: dict[str, Any], results: Mapping[tuple[Any, ...], Sequence[Any]]) -> Path:
    """{"provenance": ..., "results": [{"config": [...], "runs": [...]}, ...]} at path; returns it."""
    entries = [{"config": [*config], "runs": [*runs]} for config, runs in results.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"provenance": made_by, "results": entries}, f, indent=1)
    print(f"results: {path}", file=sys.stderr)
    return path


def read_results(path: str | Path) -> list[dict[str, Any]]:
    """A result file's entries: the provenance object's, or a file from before it, a bare list."""
    with open(path) as f:
        data: Any = json.load(f)
    if isinstance(data, list):
        return cast("list[dict[str, Any]]", data)
    return cast("list[dict[str, Any]]", cast("dict[str, Any]", data)["results"])
