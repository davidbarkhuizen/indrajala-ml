"""
Format 3 (model/persistence/format3.py, the export-and-hub workplan, stage 1): a network saved as a
directory, with its training state, loads and trains on by bits, through load_network and through
its class, for every numpy and Rust fixture and for the window architectures the fixtures lack
(linear codes, FiLM, branches); numpy directories load into Rust and Rust into numpy; a directory
without training state holds the weights alone; the tensors' names are format 2's parameter
names; and a directory refuses what isn't its own, a pure-Python network and a missing extra.
"""

import functools
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file  # pyright: ignore[reportUnknownVariableType]

from indrajala_ml.generation.tokenizer import CharacterTokenizer, tokenizer_path
from indrajala_ml.model.networks.window_recurrent_array_network import (
    WindowRecurrentArrayNetwork,
    WindowRecurrentRustArrayNetwork,
)
from indrajala_ml.model.persistence.format2 import network_to_json
from indrajala_ml.model.persistence.format3 import (
    CONFIG_FILE,
    OPTIMIZER_FILE,
    TENSORS_FILE,
    TOKENIZER_FILE,
    TRAINING_STATE_FILE,
    is_directory,
    read_directory,
    save_directory,
    tensor_names,
)
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.update_rules import Adam
from tests.helpers import bits
from tests.model.networks.test_window_recurrent_array_network import BRANCHED, BRANCHED_RECURRENCES, K
from tests.model.persistence.test_checkpoint import _state_bits
from tests.model.persistence.test_format2 import ENSEMBLES, _examples, _trained
from tests.model.test_codes import LINEAR, _rows_batch
from tests.model.test_codes import _network as codes_network
from tests.model.test_film_state_input import _network as film_network
from tests.saved_model_fixtures import FIXTURES, MODEL_CLASSES, fixture_class

ARRAY_NETWORKS = sorted(
    name for name in FIXTURES if name not in ENSEMBLES and FIXTURES[name].build().implementation != "python"
)
ARRAY_ENSEMBLES = sorted(name for name in ENSEMBLES if FIXTURES[name].build().classifiers[0].implementation != "python")


def _resume(network: Any, name: str, seed: int) -> None:
    # two more batches, so the optimizer's state and the generator are both read
    examples = _examples(network, FIXTURES[name].predict, 6, seed=seed)
    network.learn_batch(0.1, examples[:3])
    network.learn_batch(0.1, examples[3:])


def _trained_ensemble(name: str) -> Any:
    ensemble = FIXTURES[name].build()
    for index, classifier in enumerate(ensemble.classifiers):
        classifier.randomize()
        classifier.learn_batch(0.1, _examples(classifier, "predict_probability", 4, seed=index))
    return ensemble


@pytest.mark.parametrize("name", ARRAY_NETWORKS)
@pytest.mark.parametrize("loader", ["class", "load_network"])
def test_a_directory_with_training_state_resumes_by_bits(name: str, loader: str, tmp_path: Path):
    trained = _trained(name)
    save_directory(trained, tmp_path / "model", training_state=True)

    loaded = (load_network if loader == "load_network" else fixture_class(name).load)(tmp_path / "model")
    assert _state_bits(loaded) == _state_bits(trained)
    _resume(trained, name, seed=3)
    _resume(loaded, name, seed=3)

    assert _state_bits(loaded) == _state_bits(trained)


@pytest.mark.parametrize("name", ARRAY_NETWORKS)
def test_a_directory_holds_what_the_format_2_file_does(name: str, tmp_path: Path):
    # read back, the same document as format 2's, its arrays numpy's
    trained = _trained(name)
    save_directory(trained, tmp_path / "model", training_state=True)

    document = read_directory(tmp_path / "model")

    assert json.loads(json.dumps(document, default=lambda array: array.tolist())) == network_to_json(trained)


@pytest.mark.parametrize("name", ARRAY_ENSEMBLES)
def test_an_ensemble_directory_resumes_by_bits(name: str, tmp_path: Path):
    ensemble = _trained_ensemble(name)
    save_directory(ensemble, tmp_path / "model", training_state=True)

    for loaded in (load_network(tmp_path / "model"), MODEL_CLASSES[name].load(tmp_path / "model")):
        assert type(loaded) is type(ensemble)
        assert [_state_bits(c) for c in loaded.classifiers] == [_state_bits(c) for c in ensemble.classifiers]
        assert [c.checkpoint().rng for c in loaded.classifiers] == [c.checkpoint().rng for c in ensemble.classifiers]
    names = load_file(str(tmp_path / "model" / TENSORS_FILE))
    assert {name.split(".")[1] for name in names} == {str(c) for c in range(len(ensemble.classifiers))}


@pytest.mark.parametrize(
    "numpy_name, rust_name",
    [
        ("AdamVectorizedMultiClassBackpropClassifierNetwork", "AdamRustArrayMultiClassBackpropClassifierNetwork"),
        (
            "MomentumConvVectorizedMultiClassBackpropClassifierNetwork",
            "MomentumConvRustArrayMultiClassBackpropClassifierNetwork",
        ),
        ("SequentialSequenceArrayNetwork", "SequentialSequenceRustArrayNetwork"),
        ("WindowRecurrentArrayNetwork", "WindowRecurrentRustArrayNetwork"),
        ("EncoderDecoderArrayNetwork", "EncoderDecoderRustArrayNetwork"),
    ],
)
def test_numpy_into_rust_and_back_by_bits(numpy_name: str, rust_name: str, tmp_path: Path):
    # saved by numpy, loaded into Rust, trained there, saved, loaded into numpy: each load holds the
    # state the save had (the two backends train alike, not by bits)
    trained = _trained(numpy_name)
    save_directory(trained, tmp_path / "numpy", training_state=True)
    rust = fixture_class(rust_name).load(tmp_path / "numpy")
    assert type(rust) is fixture_class(rust_name)
    assert _state_bits(rust) == _state_bits(trained)
    _resume(rust, rust_name, seed=3)
    save_directory(rust, tmp_path / "rust", training_state=True)

    back = fixture_class(numpy_name).load(tmp_path / "rust")

    assert type(back) is fixture_class(numpy_name)
    assert _state_bits(back) == _state_bits(rust)
    assert back.checkpoint().rng == rust.checkpoint().rng


@pytest.mark.parametrize("name", ["WindowRecurrentRustArrayNetwork", "SequentialSequenceArrayNetwork"])
def test_a_directory_without_training_state_holds_the_weights_alone(name: str, tmp_path: Path):
    trained = _trained(name)
    save_directory(trained, tmp_path / "model")

    assert sorted(path.name for path in (tmp_path / "model").iterdir()) == [CONFIG_FILE, TENSORS_FILE]
    loaded = load_network(tmp_path / "model")
    assert bits(loaded.snapshot()) == bits(trained.snapshot())
    assert loaded.optimizer.state().t == 0
    assert loaded.optimizer.state().layers == {}


def test_a_rewritten_directory_drops_the_files_it_no_longer_holds(tmp_path: Path):
    trained = _trained("SequentialSequenceArrayNetwork")
    save_directory(trained, tmp_path / "model", training_state=True, tokenizer=CharacterTokenizer.of("t", "abcdefg"))
    save_directory(trained, tmp_path / "model")

    assert sorted(path.name for path in (tmp_path / "model").iterdir()) == [CONFIG_FILE, TENSORS_FILE]


def test_the_tokenizer_is_saved_inside_the_directory(tmp_path: Path):
    tokenizer = CharacterTokenizer.of("letters", "abcdefg")
    save_directory(_trained("SequentialSequenceArrayNetwork"), tmp_path / "model", tokenizer=tokenizer)

    assert tokenizer_path(tmp_path / "model") == tmp_path / "model" / TOKENIZER_FILE
    assert CharacterTokenizer.load(tokenizer_path(tmp_path / "model")) == tokenizer
    assert tokenizer_path(tmp_path / "model.json") == tmp_path / "model.tokenizer.json"


def test_the_config_is_format_2s_description(tmp_path: Path):
    trained = _trained("WindowRecurrentArrayNetwork")
    save_directory(trained, tmp_path / "model")

    config = json.loads((tmp_path / "model" / CONFIG_FILE).read_text())
    expected = network_to_json(trained)
    for key in ("weights", "optimizer_state", "rng"):
        del expected[key]
    assert config == {**expected, "format": 3, "tensors": TENSORS_FILE}
    assert is_directory(tmp_path / "model")


# the names (D2)


def test_a_window_networks_tensors_are_named_by_layer_and_part():
    network = FIXTURES["WindowRecurrentArrayNetwork"].build()

    names = tensor_names(network.layer_specs, network.recurrence)

    assert names[0] == ["layers.0.E"]
    assert names[-4:] == [
        ["recurrence.0.pooling.q"],
        ["recurrence.0.cell.W", "recurrence.0.cell.U", "recurrence.0.cell.b"],
        ["recurrence.0.codes.E"],
        ["recurrence.0.initial.c"],
    ]


@pytest.mark.parametrize("name", ARRAY_NETWORKS)
def test_the_optimizers_tensors_carry_format_2s_keys(name: str, tmp_path: Path):
    # a layer's or part's optimizer tensors are its weights' prefix and format 2's optimizer keys
    trained = _trained(name)
    save_directory(trained, tmp_path / "model", training_state=True)
    weights = tensor_names(trained.layer_specs, getattr(trained, "recurrence", None))
    layers = network_to_json(trained)["optimizer_state"]["layers"]

    expected: set[str] = {
        names[0].rsplit(".", 1)[0] + "." + key
        for names, entry in zip(weights, layers, strict=True)
        for key in cast("dict[str, Any]", entry or {})
    }

    assert set(load_file(str(tmp_path / "model" / OPTIMIZER_FILE))) == expected


# the window architectures the fixtures lack


def _branched(implementation: str, name: str) -> Any:
    cls = WindowRecurrentArrayNetwork if implementation == "numpy" else WindowRecurrentRustArrayNetwork
    return cls.randomized((K,), BRANCHED[name], BRANCHED_RECURRENCES[name], Adam(), seed=7)


def _linear(implementation: str) -> Any:
    return codes_network(implementation, LINEAR["two factors"])


def _film(implementation: str) -> Any:
    return film_network(implementation, True)


WINDOWS: dict[str, Callable[[str], Any]] = {
    "linear codes": _linear,
    "FiLM": _film,
    **{
        f"branches, {name}": functools.partial(_branched, name=name)
        for name in ("two factors", "two factors, recombined")
    },
}


@pytest.mark.parametrize("implementation", ["numpy", "rust"])
@pytest.mark.parametrize("window", WINDOWS)
def test_a_window_architecture_resumes_by_bits(implementation: str, window: str, tmp_path: Path):
    network = WINDOWS[window](implementation)
    network.learn_batch(0.01, _rows_batch(13))
    save_directory(network, tmp_path / "model", training_state=True)

    loaded = load_network(tmp_path / "model")
    assert type(loaded) is type(network)
    assert _state_bits(loaded) == _state_bits(network)
    for net in (network, loaded):
        net.learn_batch(0.01, _rows_batch(14))

    assert _state_bits(loaded) == _state_bits(network)


# refusals


def test_a_pure_python_network_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="pure-Python network"):
        save_directory(FIXTURES["SequentialSequenceBackpropNetwork"].build(), tmp_path / "model")


def test_a_missing_extra_is_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "safetensors.numpy", None)
    with pytest.raises(ImportError, match=r"indrajala-ml\[hub\]"):
        save_directory(_trained("SequentialSequenceArrayNetwork"), tmp_path / "model")


def test_a_directory_with_half_its_training_state_is_refused(tmp_path: Path):
    save_directory(_trained("SequentialSequenceArrayNetwork"), tmp_path / "model", training_state=True)
    (tmp_path / "model" / TRAINING_STATE_FILE).unlink()

    with pytest.raises(ValueError, match="without the other"):
        load_network(tmp_path / "model")


def test_a_tensor_the_config_doesnt_name_is_refused(tmp_path: Path):
    save_directory(_trained("SequentialSequenceArrayNetwork"), tmp_path / "model")
    path = str(tmp_path / "model" / TENSORS_FILE)
    tensors = load_file(path)
    save_file({**tensors, "layers.99.W": np.zeros(2)}, path)

    with pytest.raises(ValueError, match=r"layers\.99\.W"):
        load_network(tmp_path / "model")


def test_a_missing_tensor_is_named(tmp_path: Path):
    save_directory(_trained("SequentialSequenceArrayNetwork"), tmp_path / "model")
    path = str(tmp_path / "model" / TENSORS_FILE)
    tensors = load_file(path)
    del tensors["layers.0.E"]
    save_file(tensors, path)

    with pytest.raises(ValueError, match=r"no tensor 'layers\.0\.E'"):
        load_network(tmp_path / "model")
