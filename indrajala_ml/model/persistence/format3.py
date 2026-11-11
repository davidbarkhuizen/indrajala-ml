"""
Format 3, a network as a directory (the export-and-hub workplan, D1-D4, D6): format 2's document
(format2.py) as config.json, its arrays in safetensors, the same directory locally and on the
Hugging Face Hub.

    model/
      config.json               format 2's document, "format": 3, "tensors": "model.safetensors"
      model.safetensors         the weights, float64, named by layer and parameter (D2)
      optimizer.safetensors     with training_state=True: the optimizer's state per parameter
      training_state.json       with training_state=True: the step count, the layers with
                                optimizer state, and the generator's state
      indrajala_tokenizer.json  with a tokenizer: <model>.tokenizer.json's contents (D6)

- config.json is format2.description_to_json's document with "format": 3 and "tensors":
  "model.safetensors": no weights, optimizer state or generator state. An ensemble's is
  {"format": 3, "implementation": ..., "shape": "ensemble", "tensors": "model.safetensors",
  "classifiers": [...]}, each sub-network's description without "format" or "tensors".
- The tensors' names (D2) are format 2's own: an expanded layer's parameters, then its running
  state, as "layers.{i}.{name}" (layers.3.Wq, layers.1.running_mean), i over
  layer_specs.expand_specs (an encoder-decoder's encoder's then its decoder's); a window-recurrent
  network's parts after them, factor by factor, as "recurrence.{f}.pooling.q",
  "recurrence.{f}.cell.W" (U, b), "recurrence.{f}.codes.E" (a linear factor's W, then its Q),
  "recurrence.{f}.initial.c": every tensor "{part}.{parameter}". The optimizer's carry format 2's
  keys after the same part: "layers.3.m_Wq", "recurrence.0.cell.v_U". An ensemble's are prefixed "classifiers.{c}.".
- The dtype on disk is float64, always (D3): a directory resumes training by bits, as a format-2
  file does.
- Without training state (D4) a loaded network has a fresh optimizer and an OS-entropy generator,
  as a format-2 file without a generator does; the files present say which (the reader refuses one
  of the two without the other).

read_directory gives a format-2 document whose arrays are numpy arrays in place of nested lists:
format2.network_from_json reads it, and load_network builds from it, so loading is still built
from the specs, never from a class named in the file.

numpy and Rust networks only (D2): a pure-Python network holds its weights per node, and stays
format 2. safetensors is the [hub] extra's (D7), imported when a directory is read or written.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from indrajala_ml.model.persistence.format2 import (
    ENSEMBLE,
    FORMAT,
    PYTHON,
    Format2Network,
    description_to_json,
)
from indrajala_ml.model.persistence.format2_json import (
    layer_from_json,
    layer_parameter_names,
    recurrence_from_json,
    rng_to_json,
    rule_from_json,
    state_names,
)
from indrajala_ml.model.specs.layer_specs import BatchNorm, LayerSpec, expand_specs
from indrajala_ml.model.specs.window_recurrence import WindowRecurrence
from indrajala_ml.pcg64 import generator_state

DIRECTORY_FORMAT = 3
CONFIG_FILE = "config.json"
TENSORS_FILE = "model.safetensors"
OPTIMIZER_FILE = "optimizer.safetensors"
TRAINING_STATE_FILE = "training_state.json"
TOKENIZER_FILE = "indrajala_tokenizer.json"
_HUB_EXTRA = "pip install 'indrajala-ml[hub]'"


class Tokenizer(Protocol):
    """What a directory needs of a tokenizer: CharacterTokenizer's save."""

    def save(self, path: str | Path, /) -> None: ...


def _safetensors() -> Any:
    try:
        import safetensors.numpy
    except ImportError as error:
        raise ImportError(f"format 3 reads and writes safetensors, the [hub] extra's: {_HUB_EXTRA}") from error
    return safetensors.numpy


def is_directory(path: str | Path) -> bool:
    """Whether path is a format-3 directory: a directory with a config.json."""
    return (Path(path) / CONFIG_FILE).is_file()


# the names


def _layer_entries(
    specs: Sequence[LayerSpec], recurrence: WindowRecurrence | None
) -> list[tuple[str, tuple[str, ...]]]:
    # each snapshot entry's prefix and its arrays' names: its parameters, then its running state
    entries: list[tuple[str, tuple[str, ...]]] = []
    for index, (spec, parameters) in enumerate(zip(expand_specs(specs), layer_parameter_names(specs, None))):
        running = ("running_mean", "running_var") if isinstance(spec, BatchNorm) else ()
        entries.append((f"layers.{index}.", (*parameters, *running)))
    if recurrence is not None:
        for f, factor in enumerate(recurrence.factors):
            entries.append((f"recurrence.{f}.pooling.", ("q",)))
            entries.append((f"recurrence.{f}.cell.", ("W", "U", "b")))
            if factor.codes is not None:
                linear = factor.codes.parameterisation == "linear"
                entries.append((f"recurrence.{f}.codes.", ("W", "Q") if linear else ("E",)))
            entries.append((f"recurrence.{f}.initial.", ("c",)))
    return entries


def _parameter_entries(
    specs: Sequence[LayerSpec], recurrence: WindowRecurrence | None
) -> list[tuple[str, tuple[str, ...]]]:
    # each optimizer layer's prefix and its parameters' names (format2_json's, in the optimizers' order)
    prefixes = [prefix for prefix, _ in _layer_entries(specs, recurrence)]
    return list(zip(prefixes, layer_parameter_names(specs, recurrence), strict=True))


def tensor_names(specs: Sequence[LayerSpec], recurrence: WindowRecurrence | None) -> list[list[str]]:
    """The tensors' names (D2) of each entry of a network's snapshot()."""
    return [[prefix + name for name in names] for prefix, names in _layer_entries(specs, recurrence)]


# writing


def float64_array(value: Any) -> np.ndarray[Any, np.dtype[np.float64]]:
    """A backend's array (numpy's, or the crate's through tolist) as a contiguous float64 array."""
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(cast("np.ndarray[Any, Any]", value), dtype=np.float64)
    return np.array(value.tolist() if hasattr(value, "tolist") else value, dtype=np.float64)


def _description(network: Format2Network) -> dict[str, Any]:
    if network.implementation == PYTHON:
        raise ValueError(
            f"{type(network).__name__} is a pure-Python network, which holds its weights per node: "
            "it saves in format 2 only (network.save)"
        )
    return description_to_json(network)


def _network_tensors(
    network: Format2Network, prefix: str, training_state: bool
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    # (its weights, its optimizer's tensors, its training state) by format 3's names, under prefix
    recurrence: WindowRecurrence | None = getattr(network, "recurrence", None)
    snapshot = network.snapshot()
    names = tensor_names(network.layer_specs, recurrence)
    if len(snapshot) != len(names):
        raise AssertionError(
            f"{type(network).__name__}'s snapshot has {len(snapshot)} entries; its specs name {len(names)}"
        )
    weights: dict[str, Any] = {}
    for entry, entry_names in zip(snapshot, names):
        if len(entry) != len(entry_names):
            raise AssertionError(f"{entry_names}: the snapshot holds {len(entry)} arrays")
        weights.update({prefix + name: float64_array(array) for name, array in zip(entry_names, entry)})
    if not training_state:
        return weights, {}, None
    state = network.optimizer.state()
    rule_names = state_names(network.optimizer.rule)
    entries = _parameter_entries(network.layer_specs, recurrence)
    optimizer: dict[str, Any] = {}
    for index, layer_state in state.layers.items():
        entry_prefix, parameters = entries[index]
        for p, parameter in enumerate(parameters):
            for i, name in enumerate(rule_names):
                optimizer[f"{prefix}{entry_prefix}{name}_{parameter}"] = float64_array(
                    layer_state[p * len(rule_names) + i]
                )
    training = {"t": state.t, "layers": sorted(state.layers), "rng": rng_to_json(generator_state(network.rng))}
    return weights, optimizer, training


def save_directory(
    network: Any, directory: str | Path, *, training_state: bool = False, tokenizer: Tokenizer | None = None
) -> Path:
    """
    Writes network, a numpy or Rust network or ensemble, as a format-3 directory: its weights, with
    training_state its optimizer's and generator's state too (D4), and tokenizer's file (D6).
    Rewrites a directory already there, removing the optional files this save doesn't write.
    """
    st = _safetensors()
    directory = Path(directory)
    classifiers: Sequence[Format2Network] | None = getattr(network, "classifiers", None)
    weights: dict[str, Any] = {}
    optimizer: dict[str, Any] = {}
    if classifiers is None:
        config = _description(network)
        weights, optimizer, training = _network_tensors(network, "", training_state)
    else:
        if not classifiers:
            raise ValueError("an ensemble without sub-networks")
        descriptions: list[dict[str, Any]] = []
        trainings: list[dict[str, Any] | None] = []
        for c, classifier in enumerate(classifiers):
            description = _description(classifier)
            del description["format"]
            descriptions.append(description)
            classifier_weights, classifier_optimizer, classifier_training = _network_tensors(
                classifier, f"classifiers.{c}.", training_state
            )
            weights.update(classifier_weights)
            optimizer.update(classifier_optimizer)
            trainings.append(classifier_training)
        config = {
            "format": FORMAT,
            "implementation": classifiers[0].implementation,
            "shape": ENSEMBLE,
            "classifiers": descriptions,
        }
        training = {"classifiers": trainings} if training_state else None
    config["format"] = DIRECTORY_FORMAT
    config["tensors"] = TENSORS_FILE
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n")
    st.save_file(weights, str(directory / TENSORS_FILE))
    if training is None:
        for name in (OPTIMIZER_FILE, TRAINING_STATE_FILE):
            (directory / name).unlink(missing_ok=True)
    else:
        st.save_file(optimizer, str(directory / OPTIMIZER_FILE))
        (directory / TRAINING_STATE_FILE).write_text(json.dumps(training, indent=2) + "\n")
    if tokenizer is None:
        (directory / TOKENIZER_FILE).unlink(missing_ok=True)
    else:
        tokenizer.save(directory / TOKENIZER_FILE)
    return directory


# reading


class _Tensors:
    # a file's tensors, each taken once: read_directory refuses a name it never took
    def __init__(self, tensors: dict[str, Any], file: str) -> None:
        self._tensors = tensors
        self._file = file
        self._taken: set[str] = set()

    def take(self, name: str) -> Any:
        if name not in self._tensors:
            raise ValueError(f"{self._file} has no tensor {name!r}")
        self._taken.add(name)
        return self._tensors[name]

    def check_all_taken(self) -> None:
        left = sorted(set(self._tensors) - self._taken)
        if left:
            raise ValueError(f"{self._file} holds tensors its config doesn't name: {', '.join(left[:5])}")


def _network_state(
    description: dict[str, Any],
    prefix: str,
    weights: _Tensors,
    optimizer: _Tensors | None,
    training: dict[str, Any] | None,
) -> dict[str, Any]:
    # a format-2 document from a description and its tensors, its arrays numpy's
    state: dict[str, Any] = {key: value for key, value in description.items() if key != "tensors"}
    state["format"] = FORMAT
    # the names need the specs, which network_from_json reads again
    layers = [layer_from_json(spec) for spec in [*state.get("encoder", ()), *state["layers"]]]
    recurrence = recurrence_from_json(state["recurrence"]) if "recurrence" in state else None
    state["weights"] = [[weights.take(prefix + name) for name in names] for names in tensor_names(layers, recurrence)]
    entries = _parameter_entries(layers, recurrence)
    stateful: list[Any] = [None] * len(entries)
    if optimizer is not None and training is not None:
        rule_names = state_names(rule_from_json(state["update_rule"]))
        indices: list[int] = training["layers"]
        for index in indices:
            entry_prefix, parameters = entries[index]
            stateful[index] = {
                f"{name}_{parameter}": optimizer.take(f"{prefix}{entry_prefix}{name}_{parameter}")
                for parameter in parameters
                for name in rule_names
            }
        state["rng"] = training["rng"]
    state["optimizer_state"] = {"t": 0 if training is None else training["t"], "layers": stateful}
    return state


def read_directory(directory: str | Path) -> dict[str, Any]:
    """
    A format-3 directory as a format-2 document (format2.py), its arrays numpy arrays: what
    load_network and a network class's from_format2 read.
    """
    st = _safetensors()
    directory = Path(directory)
    config: dict[str, Any] = json.loads((directory / CONFIG_FILE).read_text())
    if config.get("format") != DIRECTORY_FORMAT:
        raise ValueError(f"{directory} isn't a format-3 directory: its config's format is {config.get('format')!r}")
    resumable = [(directory / name).is_file() for name in (OPTIMIZER_FILE, TRAINING_STATE_FILE)]
    if resumable[0] != resumable[1]:
        raise ValueError(f"{directory} has one of {OPTIMIZER_FILE} and {TRAINING_STATE_FILE} without the other")
    weights = _Tensors(st.load_file(str(directory / config["tensors"])), config["tensors"])
    optimizer = _Tensors(st.load_file(str(directory / OPTIMIZER_FILE)), OPTIMIZER_FILE) if resumable[0] else None
    training = json.loads((directory / TRAINING_STATE_FILE).read_text()) if resumable[1] else None
    if config["shape"] == ENSEMBLE:
        trainings: list[Any] = [None] * len(config["classifiers"]) if training is None else training["classifiers"]
        state: dict[str, Any] = {
            "format": FORMAT,
            "implementation": config["implementation"],
            "shape": ENSEMBLE,
            "classifiers": [
                _network_state(description, f"classifiers.{c}.", weights, optimizer, classifier_training)
                for c, (description, classifier_training) in enumerate(zip(config["classifiers"], trainings))
            ],
        }
    else:
        state = _network_state(config, "", weights, optimizer, training)
    weights.check_all_taken()
    if optimizer is not None:
        optimizer.check_all_taken()
    return state
