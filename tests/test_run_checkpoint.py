"""
Run checkpoints (run_checkpoint.py, the RNG generators workplan, D10): a run stopped after k
epochs, saved as a model file and a run file and finished in a new process, ends where the run
taken in one go ends, by bits. That covers the weights, the optimizer's state, the dropout masks'
generator, the shuffle and disagreement draws, the warmup schedule's batch count, the pocket, the
diagnostic and the convergence series, in all three implementations.
"""

import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from random import Random
from typing import Any

import pytest

from indrajala_ml.geometry import square_bounds
from indrajala_ml.lr_schedule import linear_warmup
from indrajala_ml.model.ensembles.ensemble_backprop_classifier_network import EnsembleBackpropClassifierNetwork
from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.networks.python.sequential_backprop_network import SequentialBackpropClassifierNetwork
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.layer_specs import Dense, LayerNorm, LayerSpec, Residual
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.pcg64 import default_rng, generator_state
from indrajala_ml.run_checkpoint import load_run, run_from_json, run_to_json, save_run
from indrajala_ml.train import (
    ConvergenceSeries,
    reachable_reference_and_training_data,
    train_backprop_network_mini_batch,
    train_linear_classifier_network,
)
from tests.helpers import bits

IMPLEMENTATIONS = ["python", "numpy", "rust"]
ROOT = Path(__file__).resolve().parent.parent

LAYERS: list[LayerSpec] = [Dense(5, dropout=0.25), Dense(1, output=True)]
# a residual block around a dropout layer (the residual-connections workplan, stage 5)
RESIDUAL_LAYERS: list[LayerSpec] = [
    Dense(5, activation="relu"),
    Residual((Dense(4, dropout=0.25), Dense(5, activation="linear", bias=True))),
    Dense(1, output=True),
]
# flat layer norms after a dropout layer and first in a residual body (the layer-norm and attention
# workplan, stage 5); a patch model's checkpoints are in tests/model/persistence/test_attention_format2.py, as its
# input is an image
LAYER_NORM_LAYERS: list[LayerSpec] = [
    Dense(5, dropout=0.25),
    LayerNorm(),
    Residual((LayerNorm(), Dense(4, activation="relu"), Dense(5, activation="linear", bias=True))),
    Dense(1, output=True),
]
EPOCHS, STOPPED_AFTER, BATCH_SIZE = 6, 3, 5  # 24 rows: four batches of 5 and a short one of 4


def _student(implementation: str, layers: list[LayerSpec] = LAYERS) -> Any:
    if implementation == "python":
        network: Any = SequentialBackpropClassifierNetwork((2,), layers, Adam(), square_bounds(10.0))
        network.rng = default_rng(2)
    else:
        backend = NUMPY if implementation == "numpy" else RUST
        network = SequentialArrayNetwork((2,), layers, Adam(), shape="single_output", backend=backend)
        network.rng = backend.default_rng(2)
    network.randomize()
    return network


def _problem() -> tuple[Any, list[tuple[tuple[float, ...], float]]]:
    return reachable_reference_and_training_data(1, 2, square_bounds(10.0), 24, rng=default_rng(1), data_rng=Random(1))


def _train(student: Any, epochs: int, rng: Random | None, resume_from: Any = None) -> ConvergenceSeries:
    reference, training_data = _problem()
    return train_backprop_network_mini_batch(
        student,
        training_data,
        BATCH_SIZE,
        learning_rate=linear_warmup(0.5, 8),
        epochs=epochs,
        reference_classifier=reference,
        rng=rng,
        resume_from=resume_from,
    )


def _fields_bits(value: Any) -> Any:
    # a dataclass's fields, nested ones too, as bits; dataclasses.asdict would deep-copy Rust
    # arrays, which don't pickle
    if hasattr(value, "__dataclass_fields__"):
        return {field.name: _fields_bits(getattr(value, field.name)) for field in fields(value)}
    return bits(value)


def _outcome(student: Any, result: ConvergenceSeries) -> Any:
    # the pocketed student and everything the run returns, as JSON-ready bits
    state = student.optimizer.state()
    diagnostic = result.diagnostic
    run = result.run_checkpoint
    assert run is not None
    return bits(
        {
            "weights": student.snapshot(),
            "t": state.t,
            "moments": state.layers,
            "generator": generator_state(student.rng),
            "accuracies": diagnostic.epoch_training_accuracies,
            "best": [diagnostic.best_epoch_index, diagnostic.best_training_accuracy],
            "convergence": list(result),
            "run": [run.epochs, run.iterations, run.shuffle_state, run.network.weights, run.network.rng],
        }
    )


def finish(model_path: str, run_path: str) -> None:
    """The new process's half: load both files, finish the run, print its outcome."""
    student = load_network(model_path)
    result = _train(student, EPOCHS, None, resume_from=load_run(run_path, student))
    print(json.dumps(_outcome(student, result)))


@pytest.mark.parametrize(
    "layers", [LAYERS, RESIDUAL_LAYERS, LAYER_NORM_LAYERS], ids=["dense", "residual", "layer norm"]
)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_run_resumed_in_a_new_process_matches_the_run_in_one_go(
    implementation: str, layers: list[LayerSpec], tmp_path: Path
):
    in_one_go = _student(implementation, layers)
    expected = _outcome(in_one_go, _train(in_one_go, EPOCHS, Random(7)))

    stopped = _student(implementation, layers)
    result = _train(stopped, STOPPED_AFTER, Random(7))
    assert result.run_checkpoint is not None
    model_path, run_path = str(tmp_path / "model.json"), str(tmp_path / "run.json")
    stopped.save(model_path)  # the pocketed model
    save_run(run_path, result.run_checkpoint, stopped)

    finished = subprocess.run(
        [sys.executable, "-c", f"from tests.test_run_checkpoint import finish; finish({model_path!r}, {run_path!r})"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(finished.stdout) == json.loads(json.dumps(expected))
    assert len(expected["accuracies"]) == EPOCHS and expected["run"][:2] == [EPOCHS, EPOCHS * 5]


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_run_file_round_trips(implementation: str):
    student = _student(implementation)
    run = _train(student, STOPPED_AFTER, Random(7)).run_checkpoint
    assert run is not None

    loaded = run_from_json(json.loads(json.dumps(run_to_json(run, student))), student)

    assert loaded.shuffle_state == run.shuffle_state
    assert _fields_bits(loaded) == _fields_bits(run)


def test_the_run_checkpoint_is_taken_before_the_pocket_restores():
    # the student ends at the pocket, the run checkpoint's best; resumed with no epochs left, the
    # run trains nothing and its checkpoint's network is again the last epoch's
    student = _student("numpy")
    run = _train(student, STOPPED_AFTER, Random(7)).run_checkpoint
    assert run is not None
    assert bits(student.snapshot()) == bits(run.best.weights)

    again = _train(student, STOPPED_AFTER, None, resume_from=run).run_checkpoint
    assert again is not None
    assert bits(again.network.weights) == bits(run.network.weights)
    assert again.shuffle_state == run.shuffle_state


def test_a_run_file_refuses_another_network():
    student = _student("numpy")
    run = _train(student, 1, Random(7)).run_checkpoint
    assert run is not None
    other = SequentialArrayNetwork((2,), [Dense(4), Dense(1, output=True)], Adam(), shape="single_output")

    with pytest.raises(ValueError, match="layers"):
        run_from_json(run_to_json(run, student), other)


def test_a_run_file_refuses_an_ensemble():
    ensemble = EnsembleBackpropClassifierNetwork([_student("python"), _student("python")])
    run = _train(_student("numpy"), 1, Random(7)).run_checkpoint
    assert run is not None

    with pytest.raises(ValueError, match="one network's run"):
        run_to_json(run, ensemble)


def test_the_linear_trainer_returns_no_run_checkpoint():
    reference, training_data = _problem()
    student = _student("numpy")
    result = train_linear_classifier_network(student, training_data, epochs=1, reference_classifier=reference)
    assert result.run_checkpoint is None
