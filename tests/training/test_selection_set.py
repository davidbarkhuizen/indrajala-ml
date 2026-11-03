"""
A selection set for the trainer's pocket (train.train_backprop_network_mini_batch's selection; the
audit-remedies workplan, stage 7, D7): the pocket keeps the epoch whose accuracy on the selection
set is strictly best, the training accuracy still recorded; the run checkpoint records what the
pocket selected on and the selection set's accuracies, in its file only for such a run; a run
resumes by bits as it was selected, and the other way is refused. Without one, nothing changes.
"""

from random import Random
from typing import Any

import pytest

from indrajala_ml.geometry import square_bounds
from indrajala_ml.pcg64 import default_rng
from indrajala_ml.training.run_checkpoint import run_from_json, run_to_json
from indrajala_ml.training.train import train_backprop_network_mini_batch
from indrajala_ml.training.training_data import reachable_reference_and_training_data
from indrajala_ml.training.training_diagnostics import ConvergenceSeries
from tests.helpers import bits
from tests.training.test_run_checkpoint import BATCH_SIZE, EPOCHS, IMPLEMENTATIONS, SCHEDULE, STOPPED_AFTER, _student


def _data() -> tuple[list[Any], list[Any]]:
    # training rows and a held-out selection set of the same task
    _, training = reachable_reference_and_training_data(
        1, 2, square_bounds(10.0), 24, rng=default_rng(1), data_rng=Random(1)
    )
    _, held_out = reachable_reference_and_training_data(
        1, 2, square_bounds(10.0), 16, rng=default_rng(1), data_rng=Random(7)
    )
    return training, held_out


def _train(
    student: Any, epochs: int, selection: Any, resume_from: Any = None, rng: Random | None = None
) -> ConvergenceSeries:
    training, _ = _data()
    return train_backprop_network_mini_batch(
        student,
        training,
        BATCH_SIZE,
        learning_rate=SCHEDULE,
        epochs=epochs,
        rng=Random(3) if rng is None else rng,
        resume_from=resume_from,
        selection=selection,
    )


def _outcome(student: Any, result: ConvergenceSeries) -> Any:
    run = result.run_checkpoint
    assert run is not None and result.diagnostic is not None
    return bits(
        {
            "weights": student.snapshot(),
            "epoch_training": result.diagnostic.epoch_training_accuracies,
            "best": [result.diagnostic.best_epoch_index, result.diagnostic.best_training_accuracy],
            "selection": [list(run.epoch_selection_accuracies), run.best_selection_accuracy, run.selected_on],
        }
    )


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_the_pocket_keeps_the_strictly_best_selection_epoch(implementation: str):
    _, held_out = _data()
    student = _student(implementation)
    start = _accuracy(student, held_out)
    result = _train(student, EPOCHS, held_out)
    run = result.run_checkpoint
    assert run is not None and result.diagnostic is not None
    assert run.selected_on == "selection"
    assert len(run.epoch_selection_accuracies) == EPOCHS
    assert len(result.diagnostic.epoch_training_accuracies) == EPOCHS
    best, best_index = start, -1
    for index, accuracy in enumerate(run.epoch_selection_accuracies):
        if accuracy > best:
            best, best_index = accuracy, index
    assert result.diagnostic.best_epoch_index == best_index
    assert run.best_selection_accuracy == best
    assert _accuracy(student, held_out) == best  # the pocketed network is that epoch's
    if best_index >= 0:
        assert result.diagnostic.best_training_accuracy == result.diagnostic.epoch_training_accuracies[best_index]


def _accuracy(student: Any, rows: list[Any]) -> float:
    return sum(student.classify_state(state) == category for state, category in rows) / len(rows)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_selected_run_resumes_by_bits_through_its_file(implementation: str):
    _, held_out = _data()
    whole = _student(implementation)
    expected = _outcome(whole, _train(whole, EPOCHS, held_out))

    first = _student(implementation)
    rng = Random(3)
    stopped = _train(first, STOPPED_AFTER, held_out, rng=rng).run_checkpoint
    assert stopped is not None
    state = run_to_json(stopped, first)
    assert state["selected_on"] == "selection" and len(state["epoch_selection_accuracies"]) == STOPPED_AFTER
    second = _student(implementation)
    resumed = _train(second, EPOCHS, held_out, resume_from=run_from_json(state, second), rng=Random())
    assert _outcome(second, resumed) == expected


def test_a_run_without_a_selection_set_writes_no_selection_entries():
    student = _student("numpy")
    run = _train(student, 2, None).run_checkpoint
    assert run is not None and run.selected_on == "training" and run.epoch_selection_accuracies == ()
    state = run_to_json(run, student)
    assert not {"selected_on", "epoch_selection_accuracies", "best_selection_accuracy"} & state.keys()
    assert run_from_json(state, _student("numpy")).selected_on == "training"


@pytest.mark.parametrize("selected", [True, False])
def test_resuming_the_other_way_is_refused(selected: bool):
    _, held_out = _data()
    student = _student("numpy")
    run = _train(student, 2, held_out if selected else None).run_checkpoint
    with pytest.raises(ValueError, match="selected on its"):
        _train(_student("numpy"), 4, None if selected else held_out, resume_from=run)


def test_without_a_selection_set_the_trainer_is_unchanged_by_bits():
    # the default and an explicit None train alike; the pocket's key is the training accuracy
    a, b = _student("numpy"), _student("numpy")
    training, _ = _data()
    plain = train_backprop_network_mini_batch(
        a, training, BATCH_SIZE, learning_rate=SCHEDULE, epochs=EPOCHS, rng=Random(3)
    )
    none = _train(b, EPOCHS, None)
    assert _outcome(a, plain) == _outcome(b, none)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_selection_set_picks_its_own_epoch_not_the_training_sets(implementation: str):
    # on this problem the held-out set's best epoch is the second, the training set's the third
    _, held_out = _data()
    selected = _train(_student(implementation), EPOCHS, held_out).diagnostic
    trained = _train(_student(implementation), EPOCHS, None).diagnostic
    assert selected is not None and trained is not None
    assert (selected.best_epoch_index, trained.best_epoch_index) == (1, 2)
