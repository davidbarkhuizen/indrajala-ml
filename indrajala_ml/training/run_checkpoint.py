"""
A training run's checkpoint (the RNG generators workplan, D10): what
train.train_backprop_network_mini_batch needs to train on from an epoch boundary as though it had
never stopped. That is the network's checkpoint at the last epoch, before the pocket restores the
best one; the pocket's own checkpoint, accuracy and epoch; the per-epoch accuracies and the
convergence series so far; the shuffle generator's state; the epoch and batch counters, which a
learning-rate schedule reads; and the schedule itself (the training-recipe workplan, D5), which a
resumed run must be given again: the trainer refuses another, naming both.

The trainer returns one as result.run_checkpoint and takes one as resume_from=. save_run writes it
as one JSON file holding both networks as format-2 network entries (model/persistence/format2.py); the model
file beside it stays the pocketed model. Single networks only: an ensemble's sub-networks train
separately (ensemble_train.py), each with its own run.

Backend-free: the networks' checkpoints convert through format2.py, as the model files do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from indrajala_ml.model.persistence.checkpoint import Checkpoint
from indrajala_ml.model.persistence.format2 import (
    ENSEMBLE,
    Format2Network,
    checkpoint_to_json,
    file_checkpoint,
    network_from_json,
)
from indrajala_ml.model.persistence.model_io import load_json, save_json
from indrajala_ml.training.lr_schedule import Schedule, schedule_from_json, schedule_to_json

# 2 adds the schedule (the training-recipe workplan, D5); a version 1 file loads with none recorded
RUN_CHECKPOINT = 2
_READABLE = (1, RUN_CHECKPOINT)


@dataclass(frozen=True)
class RunCheckpoint:
    """
    A run stopped after epochs epochs and iterations batches. network is the last epoch's
    checkpoint and best the pocket's: the best epoch's, or the starting point's if best_epoch_index
    is -1 (training_diagnostics.TrainingDiagnostic). shuffle_state is the shuffle generator's random.Random
    getstate(). schedule is the run's learning rate as a Schedule, or None when it wasn't recorded:
    a bare callable's (which save_run refuses) or a version 1 file's.
    """

    network: Checkpoint[Any, Any]
    best: Checkpoint[Any, Any]
    epochs: int
    iterations: int
    shuffle_state: tuple[Any, ...]
    best_epoch_index: int
    best_training_accuracy: float
    epoch_training_accuracies: tuple[float, ...]
    convergence: tuple[tuple[int, float], ...]
    schedule: Schedule | None = None


def _network(student: object) -> Format2Network:
    # a single format-2 network; an ensemble's checkpoint is its sub-networks' list
    if getattr(student, "format2_shape", ENSEMBLE) == ENSEMBLE:
        raise ValueError(f"a run file holds one network's run, not a {type(student).__name__}'s")
    return cast("Format2Network", student)


def run_to_json(run: RunCheckpoint, student: object) -> dict[str, Any]:
    """run's file contents: student is the network the run trains, for its specs and rule."""
    network = _network(student)
    if run.schedule is None:
        raise ValueError(
            "a run file records its learning-rate schedule, and this run's is a bare callable: "
            "train with a float or an lr_schedule.py schedule to save the run"
        )
    version, internal, gauss_next = run.shuffle_state
    return {
        "run_checkpoint": RUN_CHECKPOINT,
        "epochs": run.epochs,
        "iterations": run.iterations,
        "shuffle": {"version": version, "internal": list(internal), "gauss_next": gauss_next},
        "best_epoch_index": run.best_epoch_index,
        "best_training_accuracy": run.best_training_accuracy,
        "epoch_training_accuracies": list(run.epoch_training_accuracies),
        "convergence": [list(pair) for pair in run.convergence],
        "learning_rate": schedule_to_json(run.schedule),
        "network": checkpoint_to_json(network, run.network),
        "best": checkpoint_to_json(network, run.best),
    }


def run_from_json(state: dict[str, Any], student: object) -> RunCheckpoint:
    """
    A run file's checkpoint, for student, the network that resumes it: each network entry must be
    what student is (format2.check_loadable) and hold a generator state.
    """
    if state.get("run_checkpoint") not in _READABLE:
        raise ValueError(f"not a run file: run_checkpoint is {state.get('run_checkpoint')!r}")
    network = _network(student)
    shuffle = state["shuffle"]
    return RunCheckpoint(
        network=file_checkpoint(network, network_from_json(state["network"])),
        best=file_checkpoint(network, network_from_json(state["best"])),
        epochs=state["epochs"],
        iterations=state["iterations"],
        shuffle_state=(shuffle["version"], tuple(shuffle["internal"]), shuffle["gauss_next"]),
        best_epoch_index=state["best_epoch_index"],
        best_training_accuracy=state["best_training_accuracy"],
        epoch_training_accuracies=tuple(state["epoch_training_accuracies"]),
        convergence=tuple((iteration, rate) for iteration, rate in state["convergence"]),
        schedule=schedule_from_json(state["learning_rate"]) if "learning_rate" in state else None,
    )


def save_run(path: str, run: RunCheckpoint, student: object) -> None:
    save_json(path, run_to_json(run, student))


def load_run(path: str, student: object) -> RunCheckpoint:
    return run_from_json(load_json(path), student)
