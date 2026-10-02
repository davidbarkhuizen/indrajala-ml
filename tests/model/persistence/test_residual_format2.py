"""
Residual blocks in format 2 and checkpoints (the residual-connections workplan, stage 5): a block's
entry, {"kind": "residual", "body": [...]}, and the affine layer's "bias": true; a saved and loaded
residual network resumes training by bits under every rule, in all three implementations, through
its class's load and load_network; the weights and optimizer state per expanded layer (empty for a
fork and an add); numpy and Rust files load into each other; and checkpoint()/restore_checkpoint()
resume by bits, also as nested lists across a worker boundary.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import pytest

from indrajala_ml.model.persistence.format2_json import layer_from_json, layer_to_json
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.layer_specs import BatchNorm, Dense, LayerSpec, Residual
from indrajala_ml.model.specs.spec_shapes import InputShape
from indrajala_ml.model.specs.update_rules import Momentum, UpdateRule
from indrajala_ml.training.ensemble_train import _picklable_checkpoint  # pyright: ignore[reportPrivateUsage]
from tests.model.persistence.test_checkpoint import (
    IMPLEMENTATIONS,
    RULES,
    _network,
    _rows,
    _seeded,
    _state_bits,
    _train,
)
from tests.model.persistence.test_format2 import SEQUENTIAL, _file, _save_and_load, _train_batches

AFFINE = Dense(5, activation="linear", bias=True)
# two blocks in a row, a sigmoid body and a ReLU one, then the output layer: trains on single
# examples and batches
RESIDUAL: tuple[InputShape, list[LayerSpec]] = (
    (4,),
    [Dense(5), Residual((Dense(3), AFFINE)), Residual((Dense(4, activation="relu"), AFFINE)), Dense(3, output=True)],
)
# a batch-norm pair in a body, after a dropout layer: batches only
BATCH_NORM_RESIDUAL: tuple[InputShape, list[LayerSpec]] = (
    (4,),
    [
        Dense(5, dropout=0.25),
        Residual((Dense(3, activation="linear"), BatchNorm("relu"), AFFINE)),
        Dense(3, output=True, activation="softmax", loss="cross_entropy"),
    ],
)


def test_a_block_entry_holds_its_body_and_the_affine_layer_its_bias():
    block = Residual((Dense(3, activation="relu"), AFFINE))
    entry = {
        "kind": "residual",
        "body": [
            {"kind": "dense", "size": 3, "activation": "relu", "dropout": None, "output": False, "loss": "squared"},
            {
                "kind": "dense",
                "size": 5,
                "activation": "linear",
                "dropout": None,
                "output": False,
                "loss": "squared",
                "bias": True,
            },
        ],
    }
    assert layer_to_json(block) == entry
    assert layer_from_json(json.loads(json.dumps(entry))) == block
    # a body given as a list is the same block
    assert Residual([Dense(3, activation="relu"), AFFINE]) == block  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("architecture", [RESIDUAL, BATCH_NORM_RESIDUAL], ids=["residual", "batch norm"])
@pytest.mark.parametrize("loader", ["class", "load_network"])
def test_a_loaded_residual_network_resumes_training_by_bits(
    implementation: str,
    rule: UpdateRule,
    architecture: tuple[InputShape, list[LayerSpec]],
    loader: str,
    tmp_path: Path,
):
    # train N, save, load, train M: N + M steps without the save, by bits
    input_shape, layers = architecture
    train = _train if architecture is RESIDUAL else _train_batches
    rows = _rows(input_shape, 8, seed=1)
    trained = _seeded(_network(implementation, input_shape, layers, rule), 2)
    trained.randomize()
    train(trained, rows)

    loaded = _save_and_load(trained, tmp_path, load_network if loader == "load_network" else None)
    assert loaded.layer_specs == layers
    assert _state_bits(loaded) == _state_bits(trained)
    train(trained, rows)
    train(loaded, rows)

    assert _state_bits(loaded) == _state_bits(trained)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_residual_file_holds_an_entry_per_expanded_layer(implementation: str, tmp_path: Path):
    input_shape, layers = RESIDUAL
    network = _network(implementation, input_shape, layers, Momentum(0.9))
    network.randomize()
    _train(network, _rows(input_shape, 8, seed=1))

    saved = json.loads(Path(_file(network, tmp_path)).read_text())

    assert [entry["kind"] for entry in saved["layers"]] == ["dense", "residual", "residual", "dense"]
    # dense, fork, body, affine, add, fork, body, affine, add, output
    weights, state = saved["weights"], saved["optimizer_state"]["layers"]
    assert len(weights) == len(state) == 10
    assert [weights[i] for i in (1, 4, 5, 8)] == [[]] * 4
    assert all(not state[i] for i in (1, 4, 5, 8))  # null, or no weight sets in pure Python
    if implementation == "python":
        # per node [weights, bias]
        assert [len(node) for node in weights[3]] == [2] * 5
        assert all(set(node) == {"velocity_weights", "velocity_bias"} for node in state[3])
    else:
        assert len(weights[3]) == 2 and set(state[3]) == {"velocity_W", "velocity_b"}


@pytest.mark.parametrize("saved_by", ["numpy", "rust"])
def test_numpy_and_rust_residual_files_load_into_each_other(saved_by: str, tmp_path: Path):
    input_shape, layers = BATCH_NORM_RESIDUAL
    network = _seeded(_network(saved_by, input_shape, layers, Momentum(0.9)), 2)
    network.randomize()
    _train_batches(network, _rows(input_shape, 8, seed=1))
    loader = SEQUENTIAL["rust" if saved_by == "numpy" else "numpy", "multiclass"]

    loaded = _save_and_load(network, tmp_path, loader.load)

    assert type(loaded) is loader
    assert _state_bits(loaded) == _state_bits(network)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("across_workers", [False, True], ids=["in_memory", "as_lists"])
def test_a_restored_residual_checkpoint_resumes_training_by_bits(
    implementation: str, rule: UpdateRule, across_workers: bool
):
    input_shape, layers = RESIDUAL
    rows = _rows(input_shape, 8, seed=1)
    trained: Any = _seeded(_network(implementation, input_shape, layers, rule), 2)
    trained.randomize()
    _train(trained, rows)
    checkpoint = trained.checkpoint()
    if across_workers:
        checkpoint = pickle.loads(pickle.dumps(_picklable_checkpoint(checkpoint)))
    _train(trained, rows)

    resumed = _seeded(_network(implementation, input_shape, layers, rule), 3)
    resumed.randomize()
    resumed.restore_checkpoint(checkpoint)
    _train(resumed, rows)

    assert _state_bits(resumed) == _state_bits(trained)
