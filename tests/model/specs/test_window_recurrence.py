"""
A window-recurrent network's specs (the window-recurrent adder workplan, stage 2): a window, a
sequence model's specs holding StateInputs (W18), and its recurrence; their validation, the shape
walk, format 2's entries, and every builder's build of a window given its state size (numpy's since
stage 3, Rust's since stage 5, pure Python's since stage 6).
"""

import json
import math
from typing import Any

import pytest

from indrajala_ml.model.layers.array.array_layer_builder import build_array_layers
from indrajala_ml.model.layers.numpy.recurrent_array_layer import StateInputArrayLayer
from indrajala_ml.model.layers.python.python_layer_builder import build_python_layers
from indrajala_ml.model.layers.python.recurrent_layer import StateInputLayer
from indrajala_ml.model.layers.python.state_layer import StateLayer
from indrajala_ml.model.layers.rust.recurrent_rust_array_layer import StateInputRustArrayLayer
from indrajala_ml.model.persistence.checkpoint import OptimizerState
from indrajala_ml.model.persistence.format2 import network_from_json
from indrajala_ml.model.persistence.format2_json import (
    layer_from_json,
    layer_to_json,
    optimizer_state_from_json,
    optimizer_state_to_json,
    recurrence_from_json,
    recurrence_to_json,
    rule_to_json,
)
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    Dense,
    Dropout,
    Embedding,
    LayerNorm,
    LayerSpec,
    Patches,
    Position,
    Residual,
    StateInput,
    TokenMean,
)
from indrajala_ml.model.specs.spec_shapes import InputShape, SpecShape, spec_shapes
from indrajala_ml.model.specs.spec_validation import validate_layer_specs, validate_window_recurrent
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.model.specs.window_recurrence import GRU, AttentionPooling, Codes, WindowRecurrence

# a window of 3 columns, their 27 token ids as tokens of 16, a digit of 10 per column
IDS = Embedding(27, 16)
ATTENTION = Residual((LayerNorm(), Attention(heads=2, causal=True)))
FFN = Residual((LayerNorm(), Dense(32, activation="relu"), Dense(16, activation="linear", bias=True)))
DIGITS = Dense(10, output=True, activation="softmax", loss="cross_entropy")
STATE = StateInput()
WINDOW: InputShape = (3,)
RECURRENCE = WindowRecurrence(state_size=4, codes=Codes(8))

WINDOWS: dict[str, list[LayerSpec]] = {
    "W11's: the state before each attention block": [
        IDS,
        Position(),
        *[STATE, ATTENTION, FFN] * 3,
        LayerNorm(),
        DIGITS,
    ],
    "the state before every block": [IDS, Position(), STATE, ATTENTION, STATE, FFN, DIGITS],
    "the state once, before the first block": [IDS, Position(), STATE, ATTENTION, FFN, DIGITS],
    "after the embedding dropout": [IDS, Position(), Dropout(0.1), STATE, ATTENTION, DIGITS],
    "without a position": [IDS, STATE, ATTENTION, DIGITS],
    "before an FFN block only": [IDS, Position(), ATTENTION, STATE, FFN, DIGITS],
}

WINDOWS_INVALID: dict[str, list[LayerSpec]] = {
    "no state input": [IDS, Position(), ATTENTION, FFN, DIGITS],
    "a state input after the last block": [IDS, Position(), ATTENTION, STATE, DIGITS],
    "a state input before a layer norm": [IDS, Position(), ATTENTION, STATE, LayerNorm(), DIGITS],
    "two state inputs in a row": [IDS, Position(), STATE, STATE, ATTENTION, DIGITS],
    "a state input in a body": [IDS, Position(), Residual((STATE, LayerNorm(), Attention())), DIGITS],
    "a state input before the position": [IDS, STATE, Position(), ATTENTION, DIGITS],
    "a state input before the embedding dropout": [IDS, Position(), STATE, Dropout(0.1), ATTENTION, DIGITS],
    "a window of patches": [Patches(1), Dense(16, activation="linear", bias=True), STATE, ATTENTION, DIGITS],
    "a window ending in a mean": [IDS, STATE, ATTENTION, TokenMean(), Dense(10, output=True)],
    "a token-wise sigmoid output": [IDS, STATE, ATTENTION, Dense(10, output=True)],
    "a flat window": [Dense(16), DIGITS],
}


@pytest.mark.parametrize("specs", WINDOWS.values(), ids=WINDOWS.keys())
def test_windows_are_accepted(specs: list[LayerSpec]):
    validate_window_recurrent(specs, RECURRENCE, WINDOW)


@pytest.mark.parametrize("specs", WINDOWS_INVALID.values(), ids=WINDOWS_INVALID.keys())
def test_a_malformed_window_is_rejected(specs: list[LayerSpec]):
    with pytest.raises(AssertionError):
        validate_window_recurrent(specs, RECURRENCE, WINDOW)


@pytest.mark.parametrize("input_shape", [(0,), (3, 1, 1)], ids=["no columns", "an image"])
def test_a_window_reads_its_columns_token_ids(input_shape: Any):
    with pytest.raises(AssertionError, match="k columns' token ids"):
        validate_window_recurrent(WINDOWS["the state before every block"], RECURRENCE, input_shape)


def test_a_windows_sizes_are_checked_by_the_shape_walk():
    # 16 features don't split among 3 heads
    specs: list[LayerSpec] = [IDS, STATE, Residual((Attention(heads=3),)), DIGITS]
    with pytest.raises(AssertionError, match="heads divide it"):
        validate_window_recurrent(specs, RECURRENCE, WINDOW)


RECURRENCES_INVALID: dict[str, WindowRecurrence] = {
    "no state": WindowRecurrence(0, Codes(8)),
    "one code": WindowRecurrence(4, Codes(1)),
    "a negative commitment": WindowRecurrence(4, Codes(8, commitment=-0.25)),
    "a commitment of nan": WindowRecurrence(4, Codes(8, commitment=math.nan)),
    "an infinite update bias": WindowRecurrence(4, Codes(8), GRU(update_bias=math.inf)),
}


@pytest.mark.parametrize("recurrence", RECURRENCES_INVALID.values(), ids=RECURRENCES_INVALID.keys())
def test_a_malformed_recurrence_is_rejected(recurrence: WindowRecurrence):
    with pytest.raises(AssertionError):
        validate_window_recurrent(WINDOWS["the state before every block"], recurrence, WINDOW)


def test_a_commitment_of_zero_is_accepted():
    validate_window_recurrent(WINDOWS["the state before every block"], WindowRecurrence(4, Codes(2, 0.0)), WINDOW)


def test_the_recurrence_defaults_are_the_workplans():
    # W13's update bias toward keeping the state, W14's commitment of VQ-VAE's 0.25, W1b-i's pooling
    assert RECURRENCE == WindowRecurrence(4, Codes(8, commitment=0.25), GRU(update_bias=1.0), AttentionPooling())


def test_a_state_input_outside_a_window_is_rejected():
    with pytest.raises(AssertionError, match="validate_window_recurrent"):
        validate_layer_specs(WINDOWS["the state before every block"])
    with pytest.raises(AssertionError, match="acts on tokens"):
        validate_layer_specs([Dense(5), STATE, Dense(2, output=True)])


def test_a_state_input_keeps_its_tokens_shape():
    shapes = spec_shapes(WINDOWS["the state once, before the first block"], WINDOW)
    assert shapes[2] == SpecShape((3, 16), (3, 16))
    with pytest.raises(AssertionError, match="reads tokens"):
        spec_shapes([STATE, Dense(2, output=True)], (4,))


@pytest.mark.parametrize("specs", WINDOWS.values(), ids=WINDOWS.keys())
def test_format_2_round_trips_a_windows_specs(specs: list[LayerSpec]):
    assert [layer_from_json(json.loads(json.dumps(layer_to_json(spec)))) for spec in specs] == specs


def test_a_state_input_has_its_entry_and_its_projection_p():
    assert layer_to_json(STATE) == {"kind": "state_input"}
    specs = [IDS, STATE, ATTENTION, DIGITS]
    state = OptimizerState(2, {1: [[[0.5] * 4] * 16, [[0.25] * 4] * 16]})
    entry = optimizer_state_to_json(Adam(), False, state, specs)
    assert entry["layers"][1] == {"m_P": [[0.5] * 4] * 16, "v_P": [[0.25] * 4] * 16}
    assert optimizer_state_from_json(Adam(), False, entry, specs) == state


def test_a_recurrence_has_its_entry():
    entry = {
        "state_size": 4,
        "codes": {"count": 8, "commitment": 0.25},
        "cell": {"kind": "gru", "update_bias": 1.0},
        "pooling": {"kind": "attention"},
    }
    assert recurrence_to_json(RECURRENCE) == entry
    assert recurrence_from_json(json.loads(json.dumps(entry))) == RECURRENCE


@pytest.mark.parametrize("part", ["cell", "pooling"])
def test_an_unknown_cell_or_pooling_is_refused(part: str):
    entry = recurrence_to_json(RECURRENCE)
    entry[part]["kind"] = "lstm"
    with pytest.raises(ValueError, match=f"unknown {part} kind"):
        recurrence_from_json(entry)


def test_the_python_builder_builds_a_window_given_its_state_size():
    specs = WINDOWS["the state before every block"]
    layers = build_python_layers(specs, WINDOW, StateLayer(3, [(0.0, 26.0)] * 3), state_size=4)
    state_inputs = [layer for layer in layers if isinstance(layer, StateInputLayer)]
    assert [(len(layer.rows), len(layer.rows[0].weights)) for layer in state_inputs] == [(16, 4), (16, 4)]
    with pytest.raises(AssertionError, match="validate_window_recurrent"):
        build_python_layers(specs, WINDOW, StateLayer(3, [(0.0, 26.0)] * 3))
    with pytest.raises(AssertionError, match="at least one StateInput"):
        build_python_layers([IDS, ATTENTION, DIGITS], WINDOW, StateLayer(3, [(0.0, 26.0)] * 3), state_size=4)


@pytest.mark.parametrize(
    ("backend", "state_input_class"), [("numpy", StateInputArrayLayer), ("rust", StateInputRustArrayLayer)]
)
def test_an_array_builder_builds_a_window_given_its_state_size(backend: str, state_input_class: type[Any]):
    specs = WINDOWS["the state before every block"]
    layers = build_array_layers(specs, WINDOW, backend, state_size=4)
    state_inputs: list[Any] = [layer for layer in layers if isinstance(layer, state_input_class)]
    assert [tuple(layer.P.shape) for layer in state_inputs] == [(16, 4), (16, 4)]
    with pytest.raises(AssertionError, match="validate_window_recurrent"):
        build_array_layers(specs, WINDOW, backend)
    with pytest.raises(AssertionError, match="at least one StateInput"):
        build_array_layers([IDS, ATTENTION, DIGITS], WINDOW, backend, state_size=4)


def _window_file(**changes: Any) -> dict[str, Any]:
    specs = WINDOWS["the state before every block"]
    return {
        "format": 2,
        "implementation": "numpy",
        "shape": "window_recurrent",
        "input": {"dimension": 3},
        "layers": [layer_to_json(spec) for spec in specs],
        "recurrence": recurrence_to_json(RECURRENCE),
        "update_rule": rule_to_json(Adam()),
        "weights": [],
        "optimizer_state": {"t": 0, "layers": []},
        **changes,
    }


def test_a_window_recurrent_file_reads_back_its_window_and_recurrence():
    file = network_from_json(json.loads(json.dumps(_window_file())))
    assert file.shape == "window_recurrent"
    assert file.input_shape == WINDOW
    assert file.layers == WINDOWS["the state before every block"]
    assert file.recurrence == RECURRENCE


def test_only_a_window_recurrent_file_has_a_recurrence():
    without = _window_file()
    del without["recurrence"]
    with pytest.raises(ValueError, match="has a recurrence"):
        network_from_json(without)
    with pytest.raises(ValueError, match="has a recurrence"):
        network_from_json(_window_file(shape="sequence"))
