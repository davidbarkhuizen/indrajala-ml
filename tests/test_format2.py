"""
Format 2 (model/format2.py, the composable-layers workplan, stage 5): a saved and loaded network
resumes training by bits, for every rule in all three implementations and for every saveable class;
numpy files load into Rust and Rust files into numpy; a network refuses a file that isn't its own,
naming the difference; and load_network builds what a file describes.
"""

import json
import random
from pathlib import Path
from typing import Any

import pytest

from indrajala_ml.model.array_backend import NUMPY, RUST
from indrajala_ml.model.ensemble_array_backprop_classifier_network import EnsembleArrayBackpropClassifierNetwork
from indrajala_ml.model.ensemble_backprop_classifier_network import EnsembleBackpropClassifierNetwork
from indrajala_ml.model.ensemble_rust_array_backprop_classifier_network import (
    EnsembleRustArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.layer_specs import Dense, InputShape, LayerSpec
from indrajala_ml.model.load_network import load_network
from indrajala_ml.model.sequential_array_network import (
    SequentialArrayBackpropClassifierNetwork,
    SequentialArrayNetwork,
    SequentialRustArrayBackpropClassifierNetwork,
    SequentialRustArrayMultiClassBackpropClassifierNetwork,
    SequentialVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.sequential_backprop_network import (
    SequentialBackpropClassifierNetwork,
    SequentialMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.update_rules import SGD, UpdateRule
from indrajala_ml.seeding import seed_everything
from tests.saved_model_fixtures import CLASS_COUNT, FIXTURE_DIR, FIXTURES, MODEL_CLASSES, bits, outputs
from tests.test_checkpoint import CONV, DENSE, IMPLEMENTATIONS, RULES, _network, _rows, _state_bits, _train

ENSEMBLES = {name for name in FIXTURES if name.startswith("Ensemble")}
NETWORKS = sorted(set(FIXTURES) - ENSEMBLES)

# what load_network builds for a network file, by implementation and shape
SEQUENTIAL = {
    ("python", "multiclass"): SequentialMultiClassBackpropClassifierNetwork,
    ("python", "single_output"): SequentialBackpropClassifierNetwork,
    ("numpy", "multiclass"): SequentialVectorizedMultiClassBackpropClassifierNetwork,
    ("numpy", "single_output"): SequentialArrayBackpropClassifierNetwork,
    ("rust", "multiclass"): SequentialRustArrayMultiClassBackpropClassifierNetwork,
    ("rust", "single_output"): SequentialRustArrayBackpropClassifierNetwork,
}


def _save_and_load(network: Any, tmp_path: Path, load: Any = None) -> Any:
    path = str(tmp_path / "model.json")
    network.save(path)
    loader: Any = load or MODEL_CLASSES[type(network).__name__].load
    return loader(path)


def _file(network: Any, tmp_path: Path) -> str:
    path = str(tmp_path / "model.json")
    network.save(path)
    return path


def _examples(network: Any, predict: str, count: int, seed: int) -> list[tuple[tuple[float, ...], Any]]:
    rng = random.Random(seed)
    return [
        (
            tuple(rng.random() for _ in range(network.dimension)),
            rng.randrange(CLASS_COUNT) if predict == "predict_probabilities" else float(rng.randrange(2)),
        )
        for _ in range(count)
    ]


def _trained(name: str) -> Any:
    # a fixture's class, randomized and trained two batches, so its optimizer has state
    fixture = FIXTURES[name]
    seed_everything(1)
    network = fixture.build()
    network.randomize()
    examples = _examples(network, fixture.predict, 6, seed=2)
    network.learn_batch(0.1, examples[:3])
    network.learn_batch(0.1, examples[3:])
    return network


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("architecture", [DENSE, CONV], ids=["dense", "conv"])
@pytest.mark.parametrize("loader", ["class", "load_network"])
def test_a_loaded_network_resumes_training_by_bits(
    implementation: str,
    rule: UpdateRule,
    architecture: tuple[InputShape, list[LayerSpec]],
    loader: str,
    tmp_path: Path,
):
    # train N, save, load, train M: N + M steps without the save, by bits
    input_shape, layers = architecture
    rows = _rows(input_shape, 8, seed=1)
    seed_everything(2)
    trained = _network(implementation, input_shape, layers, rule)
    trained.randomize()
    _train(trained, rows)

    loaded = _save_and_load(trained, tmp_path, load_network if loader == "load_network" else None)
    assert _state_bits(loaded) == _state_bits(trained)
    _train(trained, rows)
    _train(loaded, rows)

    assert _state_bits(loaded) == _state_bits(trained)


@pytest.mark.parametrize("name", NETWORKS)
def test_every_saveable_network_resumes_training_by_bits(name: str, tmp_path: Path):
    trained = _trained(name)
    loaded = _save_and_load(trained, tmp_path)
    assert type(loaded) is type(trained)
    hyperparameters: tuple[str, ...] = MODEL_CLASSES[name].hyperparameters
    for hyperparameter in hyperparameters:
        assert getattr(loaded, hyperparameter) == getattr(trained, hyperparameter)
    assert _state_bits(loaded) == _state_bits(trained)

    examples = _examples(trained, FIXTURES[name].predict, 4, seed=3)
    seed_everything(4)  # the dropout masks
    trained.learn_batch(0.1, examples)
    seed_everything(4)
    loaded.learn_batch(0.1, examples)
    assert _state_bits(loaded) == _state_bits(trained)


@pytest.mark.parametrize("name", sorted(ENSEMBLES))
def test_an_ensemble_saves_its_sub_networks_optimizer_state(name: str, tmp_path: Path):
    ensemble = FIXTURES[name].build()
    for index, classifier in enumerate(ensemble.classifiers):
        classifier.randomize()
        classifier.learn_batch(0.1, _examples(classifier, "predict_probability", 4, seed=index))

    loaded = _save_and_load(ensemble, tmp_path)

    assert type(loaded) is type(ensemble)
    assert [_state_bits(classifier) for classifier in loaded.classifiers] == [
        _state_bits(classifier) for classifier in ensemble.classifiers
    ]
    assert all(classifier.optimizer.t == 1 for classifier in loaded.classifiers)


@pytest.mark.parametrize(
    "numpy_name, rust_name",
    [
        ("AdamVectorizedMultiClassBackpropClassifierNetwork", "AdamRustArrayMultiClassBackpropClassifierNetwork"),
        (
            "MomentumConvVectorizedMultiClassBackpropClassifierNetwork",
            "MomentumConvRustArrayMultiClassBackpropClassifierNetwork",
        ),
        ("ArrayBackpropClassifierNetwork", "RustArrayBackpropClassifierNetwork"),
        (
            "SequentialVectorizedMultiClassBackpropClassifierNetwork",
            "SequentialRustArrayMultiClassBackpropClassifierNetwork",
        ),
        ("EnsembleArrayBackpropClassifierNetwork", "EnsembleRustArrayBackpropClassifierNetwork"),
    ],
)
@pytest.mark.parametrize("saved_by", ["numpy", "rust"])
def test_numpy_and_rust_files_load_into_each_other(numpy_name: str, rust_name: str, saved_by: str, tmp_path: Path):
    saver, loader = (numpy_name, rust_name) if saved_by == "numpy" else (rust_name, numpy_name)
    if saver in ENSEMBLES:
        network = FIXTURES[saver].build()
        for index, classifier in enumerate(network.classifiers):
            classifier.randomize()
            classifier.learn_batch(0.1, _examples(classifier, "predict_probability", 4, seed=index))
    else:
        network = _trained(saver)

    loaded = _save_and_load(network, tmp_path, MODEL_CLASSES[loader].load)

    assert type(loaded) is MODEL_CLASSES[loader]
    pairs = zip(loaded.classifiers, network.classifiers) if saver in ENSEMBLES else [(loaded, network)]
    for loaded_network, saved_network in pairs:
        assert _state_bits(loaded_network) == _state_bits(saved_network)


@pytest.mark.parametrize(
    "saver, loader, difference",
    [
        # another rule, other layers, another shape
        (
            "MomentumVectorizedMultiClassBackpropClassifierNetwork",
            "VectorizedMultiClassBackpropClassifierNetwork",
            "update rule",
        ),
        (
            "ReLUVectorizedMultiClassBackpropClassifierNetwork",
            "VectorizedMultiClassBackpropClassifierNetwork",
            "layers",
        ),
        ("SoftmaxMultiClassBackpropClassifierNetwork", "MultiClassBackpropClassifierNetwork", "layers"),
        ("AdamBackpropClassifierNetwork", "BackpropClassifierNetwork", "update rule"),
        (
            "DropoutRustArrayMultiClassBackpropClassifierNetwork",
            "RustArrayMultiClassBackpropClassifierNetwork",
            "layers",
        ),
        (
            "SequentialArrayBackpropClassifierNetwork",
            "SequentialVectorizedMultiClassBackpropClassifierNetwork",
            "shape",
        ),
        # pure Python holds its weights per node, the array networks per layer
        ("MultiClassBackpropClassifierNetwork", "VectorizedMultiClassBackpropClassifierNetwork", "implementation"),
        ("VectorizedMultiClassBackpropClassifierNetwork", "MultiClassBackpropClassifierNetwork", "implementation"),
        (
            "SequentialMultiClassBackpropClassifierNetwork",
            "SequentialVectorizedMultiClassBackpropClassifierNetwork",
            "implementation",
        ),
        # a preset's hyperparameter the file doesn't have, and a file without a preset
        (
            "AdamVectorizedMultiClassBackpropClassifierNetwork",
            "MomentumVectorizedMultiClassBackpropClassifierNetwork",
            "has no momentum",
        ),
        (
            "SequentialVectorizedMultiClassBackpropClassifierNetwork",
            "VectorizedMultiClassBackpropClassifierNetwork",
            "without a preset",
        ),
    ],
)
def test_a_network_refuses_a_file_that_isnt_its_own(saver: str, loader: str, difference: str, tmp_path: Path):
    path = _file(_trained(saver), tmp_path)
    with pytest.raises(ValueError, match=difference):
        MODEL_CLASSES[loader].load(path)


def test_an_ensemble_refuses_a_sub_network_that_isnt_its_classifier_class(tmp_path: Path):
    # a pure-Python ensemble loads BackpropClassifierNetworks, whose rule is SGD
    adam = [MODEL_CLASSES["AdamBackpropClassifierNetwork"]([3], 4, [(0.0, 1.0)] * 4) for _ in range(2)]
    path = _file(EnsembleBackpropClassifierNetwork(adam), tmp_path)
    with pytest.raises(ValueError, match="update rule"):
        EnsembleBackpropClassifierNetwork.load(path)


def test_an_ensemble_of_fan_in_aware_sub_networks_loads(tmp_path: Path):
    # the MNIST ensemble's sub-networks (ensemble_train.py): only randomize differs from
    # BackpropClassifierNetwork's, so their specs and rule are its own, and the file loads
    fan_in_aware = MODEL_CLASSES["FanInAwareBackpropClassifierNetwork"]
    classifiers = [fan_in_aware.randomized([3], 4, [(0.0, 1.0)] * 4) for _ in range(2)]
    ensemble = EnsembleBackpropClassifierNetwork(classifiers)

    loaded = _save_and_load(ensemble, tmp_path)

    assert bits(loaded.snapshot()) == bits(ensemble.snapshot())


def test_a_legacy_file_loads_with_its_class_only(tmp_path: Path):
    with pytest.raises(ValueError, match="isn't a format-2 file"):
        load_network(str(FIXTURE_DIR / "VectorizedMultiClassBackpropClassifierNetwork.json"))
    with pytest.raises(ValueError, match="format 2 only"):
        SequentialVectorizedMultiClassBackpropClassifierNetwork.load(
            str(FIXTURE_DIR / "VectorizedMultiClassBackpropClassifierNetwork.json")
        )


@pytest.mark.parametrize("name", NETWORKS)
def test_load_network_builds_the_sequential_network_the_file_describes(name: str, tmp_path: Path):
    trained = _trained(name)
    path = _file(trained, tmp_path)
    with open(path) as f:
        saved = json.load(f)

    loaded = load_network(path)

    assert type(loaded) is SEQUENTIAL[saved["implementation"], saved["shape"]]
    assert loaded.layer_specs == trained.layer_specs and loaded.optimizer.rule == trained.optimizer.rule
    assert _state_bits(loaded) == _state_bits(trained)
    states = [state for state, _label in _examples(trained, FIXTURES[name].predict, 4, seed=5)]
    predict = FIXTURES[name].predict
    assert outputs(loaded, predict, states) == outputs(trained, predict, states)


@pytest.mark.parametrize(
    "name, ensemble_cls",
    [
        ("EnsembleBackpropClassifierNetwork", EnsembleBackpropClassifierNetwork),
        ("EnsembleArrayBackpropClassifierNetwork", EnsembleArrayBackpropClassifierNetwork),
        ("EnsembleRustArrayBackpropClassifierNetwork", EnsembleRustArrayBackpropClassifierNetwork),
    ],
)
def test_load_network_builds_an_ensemble_of_sequential_sub_networks(name: str, ensemble_cls: Any, tmp_path: Path):
    ensemble = FIXTURES[name].build()
    for classifier in ensemble.classifiers:
        classifier.randomize()

    loaded = load_network(_file(ensemble, tmp_path))

    assert type(loaded) is ensemble_cls
    implementation = FIXTURES[name].implementation
    assert all(type(classifier) is SEQUENTIAL[implementation, "single_output"] for classifier in loaded.classifiers)
    assert bits(loaded.snapshot()) == bits(ensemble.snapshot())


def test_a_sequential_network_saves_without_a_preset(tmp_path: Path):
    network = SequentialArrayNetwork((4,), [Dense(5), Dense(3, output=True)], SGD(), backend=NUMPY)
    saved = json.loads(Path(_file(network, tmp_path)).read_text())
    assert "preset" not in saved
    assert saved["layers"][0] == {
        "kind": "dense",
        "size": 5,
        "activation": "sigmoid",
        "dropout": None,
        "output": False,
        "loss": "squared",
    }
    assert saved["update_rule"] == {"rule": "sgd"}
    assert saved["optimizer_state"] == {"t": 0, "layers": [None, None]}
    assert RUST.name == "rust"  # the implementation names format 2 records
