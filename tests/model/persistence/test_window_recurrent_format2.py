"""
Window-recurrent networks' files and checkpoints (the window-recurrent adder workplan, stage 7), in
all three implementations: each fixture (tests/saved_model_fixtures.py, which
test_legacy_saved_models.py loads and test_format2.py resumes, numpy's and Rust's files into each
other) re-saves byte-identically and holds the recurrence entry; checkpoint()/restore_checkpoint()
resume training by bits under every rule, the dropout masks' generator included, also as nested
lists across a worker boundary; and a run file (training/run_checkpoint.py) round trips and refuses
a network of another recurrence.
"""

import json
import pickle
import random
from pathlib import Path
from typing import Any

import pytest

from indrajala_ml.model.networks.python.window_recurrent_backprop_network import WindowRecurrentBackpropNetwork
from indrajala_ml.model.networks.window_recurrent_array_network import (
    WindowRecurrentArrayNetwork,
    WindowRecurrentRustArrayNetwork,
)
from indrajala_ml.model.specs.update_rules import Adam, UpdateRule
from indrajala_ml.model.specs.window_recurrence import Codes, WindowRecurrence
from indrajala_ml.training.ensemble_train import _picklable_checkpoint  # pyright: ignore[reportPrivateUsage]
from indrajala_ml.training.run_checkpoint import RunCheckpoint, run_from_json, run_to_json
from tests.helpers import bits
from tests.model.persistence.test_checkpoint import IMPLEMENTATIONS, RULES, _seeded, _state_bits, _train
from tests.saved_model_fixtures import (
    FIXTURE_DIR,
    FIXTURES,
    WINDOW_COLUMNS,
    WINDOW_MODEL,
    WINDOW_RECURRENCE,
    fixture_class,
    random_label,
    random_state,
)

WINDOW_FIXTURES = [name for name in FIXTURES if name.startswith("WindowRecurrent")]
WINDOW_CLASSES: dict[str, Any] = {
    "python": WindowRecurrentBackpropNetwork,
    "numpy": WindowRecurrentArrayNetwork,
    "rust": WindowRecurrentRustArrayNetwork,
}


def _network(implementation: str, rule: UpdateRule, recurrence: WindowRecurrence = WINDOW_RECURRENCE) -> Any:
    return WINDOW_CLASSES[implementation]((WINDOW_COLUMNS,), WINDOW_MODEL, recurrence, rule)


def _rows(network: Any, count: int, seed: int) -> list[tuple[tuple[float, ...], Any]]:
    # two windows of token ids, and a class per column
    rng = random.Random(seed)
    return [(random_state(rng, network), random_label(rng, network, "predict_probabilities")) for _ in range(count)]


@pytest.mark.parametrize("name", WINDOW_FIXTURES)
def test_a_window_fixture_re_saves_byte_identically(name: str, tmp_path: Path):
    path = tmp_path / f"{name}.json"
    fixture_class(name).load(str(FIXTURE_DIR / f"{name}.json")).save(str(path))
    assert path.read_bytes() == (FIXTURE_DIR / f"{name}.json").read_bytes()
    saved = json.loads(path.read_text())
    assert saved["shape"] == "window_recurrent"
    assert saved["recurrence"] == {
        "state_size": 3,
        "codes": {"count": 4, "commitment": 0.5},
        "cell": {"kind": "gru", "update_bias": 0.5},
        "pooling": {"kind": "attention"},
    }
    assert [entry["kind"] for entry in saved["layers"]].count("state_input") == 2
    assert len(WINDOW_FIXTURES) == 3


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("across_workers", [False, True], ids=["in_memory", "as_lists"])
def test_a_restored_checkpoint_resumes_training_by_bits(implementation: str, rule: UpdateRule, across_workers: bool):
    # as test_checkpoint.py's networks: the window's dropout draws from the checkpoint's generator
    trained = _seeded(_network(implementation, rule), 2)
    trained.randomize()
    rows = _rows(trained, 8, seed=1)
    _train(trained, rows)
    checkpoint = trained.checkpoint()
    if across_workers:
        checkpoint = pickle.loads(pickle.dumps(_picklable_checkpoint(checkpoint)))
    _train(trained, rows)

    resumed = _seeded(_network(implementation, rule), 3)  # other weights and generator, overwritten
    resumed.randomize()
    resumed.restore_checkpoint(checkpoint)
    _train(resumed, rows)

    assert _state_bits(resumed) == _state_bits(trained)
    assert trained.optimizer.t == 2 * 5


def _run(network: Any) -> RunCheckpoint:
    # a run stopped after two epochs of the rows, its pocket at the first
    rows = _rows(network, 8, seed=1)
    network.learn_batch(0.1, rows)
    best = network.checkpoint()
    network.learn_batch(0.1, rows)
    return RunCheckpoint(
        network=network.checkpoint(),
        best=best,
        epochs=2,
        iterations=2,
        shuffle_state=random.Random(7).getstate(),
        best_epoch_index=0,
        best_training_accuracy=0.25,
        epoch_training_accuracies=(0.25, 0.125),
        convergence=((1, 0.5), (2, 0.25)),
    )


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_run_file_round_trips(implementation: str):
    student = _seeded(_network(implementation, Adam()), 2)
    student.randomize()
    run = _run(student)

    loaded = run_from_json(json.loads(json.dumps(run_to_json(run, student))), student)

    assert loaded.shuffle_state == run.shuffle_state
    for checkpoint, saved in ((loaded.network, run.network), (loaded.best, run.best)):
        assert bits(checkpoint.weights) == bits(saved.weights)
        assert checkpoint.optimizer.t == saved.optimizer.t
        assert bits(checkpoint.optimizer.layers) == bits(saved.optimizer.layers)
        assert checkpoint.rng == saved.rng
    resumed = _seeded(_network(implementation, Adam()), 3)
    resumed.restore_checkpoint(loaded.network)
    assert _state_bits(resumed) == _state_bits(student)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_run_file_refuses_a_network_of_another_recurrence(implementation: str):
    student = _seeded(_network(implementation, Adam()), 2)
    student.randomize()
    other = _network(implementation, Adam(), WindowRecurrence(3, Codes(5, commitment=0.5)))

    with pytest.raises(ValueError, match="recurrence"):
        run_from_json(run_to_json(_run(student), student), other)
