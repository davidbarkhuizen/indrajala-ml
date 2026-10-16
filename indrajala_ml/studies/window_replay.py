"""
A curriculum step replayed from a saved network (docs/snap-wall-workplan.md): what further training
does to it, batch by batch, and whether its proof survives. From a saved step's network (optionally
converted to numpy, its weights, Adam's moments and generator kept; optionally with the snap wall
on), some epochs of a step's data drawn by a named generator, as a run's step draws them:

- **per batch**: the batch's loss before and after its update, and the update's norm (every
  parameter's change, the codebook's included), so a jump of the loss on an ordinary step (a state
  crossing a code boundary) and a run of large updates after it (Adam amplifying it) show;
- **per epoch**: the screen at the step's lengths, the certificate in both verdicts (near ties
  branched, and strict), and the least snap ratio over the screen's first cases' states (every
  other code's distance over the snapped code's, per state; the certificate asks 1.05, the wall
  keeps 1.5).

E2b' seed 6's replay from step 7 found the first: an ordinary step at batch 977 of step 8 took its
loss from 0.001 to 2.2, and the updates after it ran 6-8 times their size (the workplan's Why).
P1 (the workplan's stage 2) replayed 2 epochs of step 8 from that network's step 7, 10 times with
the snap wall off and 10 on: the proof was lost in 9 and in 8 (C1 refuted), the wall cutting the
loss's jumps about 3-fold but not the loss of the proof.
"""

from __future__ import annotations

import math
import random
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from indrajala_ml.model.layers.array.array_backend import NUMPY
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.studies import window_recurrent_certificate as wc
from indrajala_ml.studies import window_recurrent_study as wr
from indrajala_ml.studies import window_tasks as wt

RATIO_CASES = 200  # the screen's first cases a length, for the least snap ratio


def load(path: Path, numpy: bool = False) -> Any:
    """A saved network, as saved or converted to numpy (the same weights, optimizer state and generator)."""
    if not numpy:
        return load_network(str(path))
    text = path.read_text()
    marker = '"implementation": "'
    start = text.index(marker) + len(marker)
    converted = text[:start] + NUMPY.name + text[text.index('"', start) :]
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / path.name
        target.write_text(converted)
        return load_network(str(target))


def set_wall(network: Any, wall: float, ratio: float) -> None:
    """Every snapped factor's wall (numpy's codes only until the crate has it, the workplan's D8)."""
    for factor in network.recurrence_layer.factors:
        if factor.codes is not None:
            assert hasattr(factor.codes, "wall"), "the snap wall is numpy's only: convert the network (numpy=True)"
            factor.codes.wall, factor.codes.wall_ratio = wall, ratio


def _parameters(network: Any) -> npt.NDArray[np.float64]:
    # every parameter, flattened, the codebook's included (numpy's or the crate's arrays)
    parts: list[Any] = []
    for part in network.snapshot():
        for value in part:
            values: list[Any] = value.tolist()  # numpy's arrays and the crate's alike
            parts.append(np.ravel(np.asarray(values, dtype=float)))
    return np.concatenate(parts)


def least_snap_ratio(network: Any, task: wt.Task, lengths: Sequence[int], cases: int = RATIO_CASES) -> float | None:
    """
    The least ratio, over the states of each length's codes-property's first cases, of the next
    nearest code's distance to the snapped code's, over every snapped factor; None if none is snapped.
    """
    factors = network.recurrence_layer.factors
    sizes = [f.state_size for f in network.recurrence.factors]
    offsets = np.cumsum([0, *sizes])
    codes = [None if f.codes is None else np.asarray(f.codes.E.tolist(), dtype=float) for f in factors]
    if all(E is None for E in codes):
        return None
    k: int = network.columns
    property_ = next(p for p in task.catalogue(cases, cases, 1) if p.id == task.codes_id)
    least = math.inf
    for n in lengths:
        drawn = task.property_cases(property_, n)[:cases]
        rows = [wr.example(task, case, n, k, wr.alignment(task, case, n, k)) for case in drawn]
        for width in sorted({len(tokens) for tokens, _ in rows}):
            X = network.backend.matrix([tokens for tokens, _ in rows if len(tokens) == width])
            _, _, cells = network.window_states(X)
            for cell in cells:
                states = np.asarray(cell.tolist(), dtype=float)
                for g, E in enumerate(codes):
                    if E is None or len(E) < 2:
                        continue
                    block = states[:, offsets[g] : offsets[g + 1]]
                    q = ((block[:, np.newaxis, :] - E[np.newaxis, :, :]) ** 2).sum(axis=2)
                    q.sort(axis=1)
                    least = min(least, float(np.min(np.sqrt(q[:, 1] / np.maximum(q[:, 0], 1e-300)))))
    return least


def replay(
    network: Any,
    task: wt.Task,
    counts: Sequence[int],
    settings: wr.Settings,
    epochs: int,
    generator: str,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    epochs of the step whose data window counts are counts, from network (trained in place), each
    epoch's data and order drawn by random.Random(f"{generator}-{epoch}"); the batches' and epochs'
    figures (see the module's docstring).
    """
    k: int = network.columns
    model = wr.WindowModel(network, k, task)
    catalogue = settings.catalogue(task)
    lengths = [w * k for w in counts]
    checks = {n: wr.check_properties(task, n, catalogue) for n in lengths}
    result: dict[str, Any] = {"generator": generator, "counts": list(counts), "epochs": []}
    for epoch in range(epochs):
        start = time.perf_counter()
        rng = random.Random(f"{generator}-{epoch}")
        batches = wr.epoch_batches(
            wr.training_examples(task, counts, k, settings.epoch_examples, rng), settings.batch_size, rng
        )
        rows: list[dict[str, float]] = []
        for batch in batches:
            X = network.backend.matrix([tokens for tokens, _ in batch])
            labels = [labels for _, labels in batch]
            before: float = network.loss_batch(X, labels)
            kept = _parameters(network)
            network.learn_batch(settings.rate, batch)
            after: float = network.loss_batch(X, labels)
            rows.append(
                {
                    "width": len(batch[0][0]),
                    "loss_before": before,
                    "loss_after": after,
                    "update": float(np.linalg.norm(_parameters(network) - kept)),
                }
            )
        screen = [r for n in lengths for r in model.evaluate(n, checks[n], settings.screen)]
        branched = wc.certify(wc.NetworkMachine(network), task)
        strict = wc.certify(wc.NetworkMachine(network), task, snap_ratio=1.0)
        updates = sorted(row["update"] for row in rows)
        summary = {
            "epoch": epoch,
            "screen": sum(r.passed for r in screen) / sum(r.cases for r in screen),
            "certified": branched.certified,
            "certified_strict": strict.certified,
            "minimised": len(branched.minimised),
            "least_snap_ratio": least_snap_ratio(network, task, lengths),
            "jumps": sum(row["loss_after"] > 1.0 >= row["loss_before"] for row in rows),
            "median_update": updates[len(updates) // 2],
            "max_update": updates[-1],
            "seconds": time.perf_counter() - start,
            "batches": rows,
        }
        result["epochs"].append(summary)
        if log is not None:
            log(line(generator, summary))
    return result


def line(name: str, epoch: dict[str, Any]) -> str:
    """An epoch's summary as a line."""
    ratio = epoch["least_snap_ratio"]
    return (
        f"{name} epoch {epoch['epoch']}: screen {epoch['screen']:.3f}, "
        f"certified {epoch['certified']} (strict {epoch['certified_strict']}, {epoch['minimised']} states), "
        f"least snap ratio {'-' if ratio is None else f'{ratio:.3f}'}, {epoch['jumps']} jumps past loss 1, "
        f"update median {epoch['median_update']:.3g} max {epoch['max_update']:.3g}; {epoch['seconds']:.0f} s"
    )
