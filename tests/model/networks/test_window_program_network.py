# pyright: reportPrivateUsage=false
"""
The Rust window-recurrent network's window program (the window-program workplan, stage 3):
its inference passes, forward_windows, window_states, state_codes and window_step, run every
window in one crate call, by bits the per-layer path's (per_layer_windows, the
reference, which numpy's network runs): for every layer kind a window holds, one, two and three
factors (continuous and crossed among them, the poolings reading the output or the tokens before
the first StateInput), the studies' windows (E2b''s linear window, the division family's) and
their certificates. Every layer class a window can hold has an op.
"""

import random
from typing import Any

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import RUST
from indrajala_ml.model.layers.array.array_layer_builder import LAYER_CLASSES, TOKEN_LAYER_CLASSES
from indrajala_ml.model.networks.window_recurrent_array_network import (
    WindowRecurrentRustArrayNetwork,
    _program_op,
    per_layer_path,
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
from indrajala_ml.model.specs.spec_validation import validate_window_specs
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.model.specs.window_recurrence import OUTPUT
from indrajala_ml.pcg64 import generator_state
from indrajala_ml.studies import modular_family as mf
from indrajala_ml.studies import window_recurrent_certificate as cert
from indrajala_ml.studies import window_recurrent_study as wrs
from indrajala_ml.studies import window_tasks as wt
from tests.helpers import bits
from tests.model.networks.test_window_recurrent_array_network import (
    DIGITS,
    RECURRENCES,
    SNAPPED,
    WINDOW,
    K,
    _batch,
    covering_consolidation,
)

# every layer kind a window holds: dropout outside and inside attention, a token-wise ReLU and
# affine pair in a residual block; and rotary attention, which stands in for a Position
EVERY_KIND: list[LayerSpec] = [
    Embedding(27, 8),
    Position(),
    Dropout(0.2),
    StateInput(),
    Residual((LayerNorm(), Attention(heads=2, causal=True, dropout=0.1))),
    StateInput(),
    Residual((LayerNorm(), Dense(16, activation="relu"), Dense(8, activation="linear", bias=True))),
    LayerNorm(),
    DIGITS,
]
ROTARY: list[LayerSpec] = [
    Embedding(27, 8),
    StateInput(),
    Residual((LayerNorm(), Attention(heads=2, causal=False, positions="rotary"))),
    LayerNorm(),
    DIGITS,
]
WINDOWS = {"the tests' window": WINDOW, "every kind": EVERY_KIND, "rotary": ROTARY}


def _passes(network: Any, X: Any) -> dict[str, Any]:
    # every inference pass's results, as bits and codes
    outputs, states, cells = network.window_states(X)
    result: dict[str, Any] = {
        "forward_windows": bits(network.forward_windows(X)),
        "window_states": [bits(outputs), [bits(s) for s in states], [bits(c) for c in cells]],
    }
    if network.recurrence.snapped:
        # a machine's: every factor snapped
        result["state_codes"] = network.state_codes(X)
        output, cell, codes = network.window_step(RUST.columns(X, 0, network.columns), network._first_state(X.shape[0]))
        result["window_step"] = [bits(output), bits(cell), codes]
    return result


def _assert_program_is_per_layer(network: Any, X: Any) -> None:
    program = _passes(network, X)
    with per_layer_path(network):
        per_layer = _passes(network, X)
    assert program == per_layer


@pytest.mark.parametrize("recurrence", RECURRENCES.values(), ids=list(RECURRENCES))
@pytest.mark.parametrize("window", WINDOWS.values(), ids=list(WINDOWS))
def test_the_inference_passes_are_the_per_layer_paths(window: list[LayerSpec], recurrence: Any):
    network = WindowRecurrentRustArrayNetwork.randomized((K,), window, recurrence, Adam(), seed=7)
    X, labels = _batch(4, examples=5)
    # a step trained first, so the poolings' queries and the initial state aren't zero
    network.learn_batch(0.01, [(tuple(x), label) for x, label in zip(X, labels, strict=True)])
    _assert_program_is_per_layer(network, RUST.matrix(X.tolist()))


@pytest.mark.parametrize("window", list(wrs.WINDOWS))
def test_the_studys_windows_are_the_per_layer_path(window: str):
    # E2b''s cell (the linear window), k 1, a state of 16 features and 16 codes, and the others
    network = wrs.build(wt.ADDITION, 1, 16, 16, seed=3, window=window)
    rng = np.random.default_rng(5)
    _assert_program_is_per_layer(network, RUST.matrix(rng.integers(0, 27, (6, 40)).astype(float).tolist()))


def test_the_division_family_is_the_per_layer_path():
    # the fork probe's cell: p 7, two factors of 32 codes, the window's output read
    encoding = mf.Encoding(7, ("sum", "difference"))
    network = mf.network(encoding, [(16, 32), (16, 32)], seed=2, reads=OUTPUT)
    rng = random.Random(6)
    rows = [encoding.example(mf.columns(20, rng))[0] for _ in range(4)]
    _assert_program_is_per_layer(network, RUST.matrix([list(row) for row in rows]))


def test_a_certificate_is_the_per_layer_paths():
    network = wrs.build(wt.ADDITION, 1, 4, 8, seed=4, window="linear")
    program = cert.certify(cert.NetworkMachine(network)).to_json()
    with per_layer_path(network):
        per_layer = cert.certify(cert.NetworkMachine(network)).to_json()
    program.pop("seconds")
    per_layer.pop("seconds")
    assert program == per_layer


def test_training_runs_the_per_layer_path_and_the_program_refuses_it():
    network = WindowRecurrentRustArrayNetwork.randomized((K,), EVERY_KIND, RECURRENCES["one factor"], Adam(), seed=7)
    X = RUST.matrix(_batch(2)[0].tolist())
    network._set_training_mode(True)
    with pytest.raises(AssertionError, match="inference only"):
        network.forward_windows(X)


def test_every_layer_class_a_window_holds_has_an_op():
    # every Rust token layer class but those a window can't hold, and the residual block's fork and add
    window_kinds = {
        name: getattr(TOKEN_LAYER_CLASSES["rust"], name)
        for name in TOKEN_LAYER_CLASSES["rust"].__dataclass_fields__
        if name not in ("patches", "token_mean")
    }
    held = {
        type(layer)
        for window in (EVERY_KIND, ROTARY)
        for layer in WindowRecurrentRustArrayNetwork.randomized(
            (K,), window, RECURRENCES["one factor"], Adam(), seed=7
        ).layers
    }
    assert set(window_kinds.values()) | {LAYER_CLASSES["rust"].fork, LAYER_CLASSES["rust"].add} == held
    networks = [
        WindowRecurrentRustArrayNetwork.randomized((K,), window, RECURRENCES["one factor"], Adam(), seed=7)
        for window in (EVERY_KIND, ROTARY)
    ]
    for layer in (layer for network in networks for layer in network.layers):
        assert _program_op(layer)[0]
    # what a window can't hold: patches (an image's) and a token mean (the output is token-wise)
    with pytest.raises(AssertionError):
        validate_window_specs([Embedding(27, 8), StateInput(), TokenMean(), Dense(3, output=True)], (K,))
    with pytest.raises(AssertionError):
        validate_window_specs([Patches(2), StateInput(), DIGITS], (K,))


def _twins(window: list[LayerSpec], recurrence: Any) -> tuple[Any, Any]:
    return tuple(  # type: ignore[return-value]
        WindowRecurrentRustArrayNetwork.randomized((K,), window, recurrence, Adam(), seed=7) for _ in range(2)
    )


def _gradients(network: Any) -> list[list[bytes]]:
    return [bits(array) for array in network._program_gradients()]


def _reset(network: Any) -> None:
    for layer in [*network.layers, *network.parts]:
        if hasattr(layer, "reset_gradient_accum"):
            layer.reset_gradient_accum()


@pytest.mark.parametrize("recurrence", RECURRENCES.values(), ids=list(RECURRENCES))
@pytest.mark.parametrize("window", WINDOWS.values(), ids=list(WINDOWS))
def test_a_training_steps_gradients_are_the_per_layer_paths(window: list[LayerSpec], recurrence: Any):
    program, per_layer = _twins(window, recurrence)
    for windows in (1, 3):
        X, labels = _batch(windows, examples=4, seed=windows)
        for network in (program, per_layer):
            _reset(network)
        X_rust = RUST.matrix(X.tolist())
        program._accumulate_gradients(X_rust, program._target_batch_array(labels))
        with per_layer_path(per_layer):
            per_layer._accumulate_gradients(X_rust, per_layer._target_batch_array(labels))
        assert _gradients(program) == _gradients(per_layer)
        # the dropout masks drawn once, in the per-layer path's order (D6)
        assert generator_state(program.rng) == generator_state(per_layer.rng)


@pytest.mark.parametrize("recurrence", SNAPPED.values(), ids=list(SNAPPED))
@pytest.mark.parametrize("window", WINDOWS.values(), ids=list(WINDOWS))
def test_a_consolidated_training_step_is_the_per_layer_paths(window: list[LayerSpec], recurrence: Any):
    # the program's lookup of each window's commitment targets against the per-layer path's (D1 of
    # the code-consolidation workplan), the initial state's through the same codes
    program, per_layer = _twins(window, recurrence)
    for network in (program, per_layer):
        network.consolidate(covering_consolidation(recurrence))
    for windows in (1, 3):
        X, labels = _batch(windows, examples=6, seed=windows + 30)
        for network in (program, per_layer):
            _reset(network)
        X_rust = RUST.matrix(X.tolist())
        program._accumulate_gradients(X_rust, program._target_batch_array(labels))
        with per_layer_path(per_layer):
            per_layer._accumulate_gradients(X_rust, per_layer._target_batch_array(labels))
        assert _gradients(program) == _gradients(per_layer)
        assert generator_state(program.rng) == generator_state(per_layer.rng)


@pytest.mark.parametrize("window", WINDOWS.values(), ids=list(WINDOWS))
def test_training_through_the_program_keeps_the_per_layer_paths_weights(window: list[LayerSpec]):
    program, per_layer = _twins(window, RECURRENCES["two factors"])
    for step in range(4):
        X, labels = _batch(2 + step % 3, examples=5, seed=20 + step)
        rows = [(tuple(x), label) for x, label in zip(X, labels, strict=True)]
        program.learn_batch(0.01, rows)
        with per_layer_path(per_layer):
            per_layer.learn_batch(0.01, rows)
    assert [[bits(a) for a in entry] for entry in program.snapshot()] == [
        [bits(a) for a in entry] for entry in per_layer.snapshot()
    ]
    assert generator_state(program.rng) == generator_state(per_layer.rng)


def test_the_studys_and_the_familys_training_steps_are_the_per_layer_paths():
    encoding = mf.Encoding(7, ("sum", "difference"))
    rng = random.Random(9)
    family = [mf.network(encoding, [(16, 32), (16, 32)], seed=2, reads=OUTPUT) for _ in range(2)]
    examples = [encoding.example(mf.columns(15, rng)) for _ in range(6)]
    linear = [wrs.build(wt.ADDITION, 1, 16, 16, seed=3, window="linear") for _ in range(2)]
    digits = np.random.default_rng(10)
    sums = [
        (tuple(float(v) for v in digits.integers(0, 27, 19)), tuple(int(v) for v in digits.integers(0, 3, 19)))
        for _ in range(8)
    ]
    for (program, per_layer), rows in ((family, examples), (linear, sums)):
        program.learn_batch(0.003, rows)
        with per_layer_path(per_layer):
            per_layer.learn_batch(0.003, rows)
        assert [[bits(a) for a in entry] for entry in program.snapshot()] == [
            [bits(a) for a in entry] for entry in per_layer.snapshot()
        ]


def test_the_program_refuses_a_step_past_its_memory_budget():
    network = WindowRecurrentRustArrayNetwork.randomized((K,), WINDOW, RECURRENCES["one factor"], Adam(), seed=7)
    X, labels = _batch(3)
    X_rust = RUST.matrix(X.tolist())
    with pytest.raises(ValueError, match="memory_budget"):
        pa.window_program_learn(
            network._program,
            X_rust,
            network._target_batch_array(labels),
            network._program_parameters(),
            network._program_gradients(),
            network._first_state(X.shape[0]),
            network.rng,
            1 / 3,
            network.recurrence_layer.commitments,
            memory_budget=1000,
        )
