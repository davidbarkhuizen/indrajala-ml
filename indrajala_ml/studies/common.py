"""
The studies' shared protocol helpers: a per-process cache of the MNIST train and test sets, and the
mean ± sd cells and Markdown tables their reports print. Used by indrajala_ml/studies/patch_study.py,
indrajala_ml/studies/sequence_study.py, scripts/residual_depth_study.py and
scripts/batch_size_scaling_sweep.py.
"""

from __future__ import annotations

import statistics
from typing import Any

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
