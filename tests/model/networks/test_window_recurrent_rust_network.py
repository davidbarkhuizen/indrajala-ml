# pyright: reportPrivateUsage=false
"""
The window-recurrent network on Rust (the window-recurrent adder workplan, stage 5), against the
numpy network (tests/model/networks/test_window_recurrent_array_network.py, which checks the
gradient against central differences). The two draw the same weights by bits
(tests/model/networks/test_seeded_init_parity.py); training through one, two and three windows,
with dropout, keeps numpy's weights to within the products' rounding (the crate's against BLAS),
and each step under Adam has numpy's gradients to within it. Exactly: a recomputed window draws its
masks again, one window with the state's projections at zero is the Rust sequence network's digits
by bits, and a saved network loads and resumes by bits.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.layers.numpy import attention_array_layer, recurrent_array_layer
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.networks.window_recurrent_array_network import (
    WindowRecurrentArrayNetwork,
    WindowRecurrentRustArrayNetwork,
)
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.layer_specs import Dropout, Embedding, LayerSpec, Position, StateInput
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule
from indrajala_ml.model.specs.window_recurrence import Factor, WindowRecurrence
from indrajala_ml.pcg64 import generator_state
from tests.helpers import bits, exp_by_crate, patching, to_numpy
from tests.model.networks.test_window_recurrent_array_network import (
    ATTENTION,
    DIGITS,
    RECURRENCE,
    RECURRENCES,
    SNAPPED,
    WINDOW,
    K,
    _batch,
    covering_consolidation,
)

# the window with dropout before the state's first input and in its attention
DROPOUT_WINDOW: list[LayerSpec] = [Embedding(27, 8), Position(), Dropout(0.2), *WINDOW[2:]]
WINDOWS = {"plain": WINDOW, "with dropout": DROPOUT_WINDOW}

# numpy's exp as the crate's in both the attention and the window's parts
crate_exp_attention = patching(attention_array_layer, "exp", exp_by_crate)
crate_exp_parts = patching(recurrent_array_layer, "exp", exp_by_crate)


def _pair(
    window: list[LayerSpec], rule: UpdateRule, seed: int = 7, recurrence: WindowRecurrence = RECURRENCE
) -> tuple[Any, Any]:
    return (
        WindowRecurrentRustArrayNetwork.randomized((K,), window, recurrence, rule, seed=seed),
        WindowRecurrentArrayNetwork.randomized((K,), window, recurrence, rule, seed=seed),
    )


def _rows(windows: int, examples: int, seed: int) -> list[tuple[tuple[float, ...], tuple[int, ...]]]:
    X, labels = _batch(windows, examples, seed)
    return [(tuple(x), label) for x, label in zip(X.tolist(), labels, strict=True)]


def _scales(snapshot: list[tuple[Any, ...]]) -> list[float]:
    # each entry's largest magnitude, the yardstick of its rounding
    return [max((float(np.abs(to_numpy(a)).max()) for a in entry), default=0.0) for entry in snapshot]


def _assert_close(expected: list[tuple[Any, ...]], actual: list[tuple[Any, ...]], tolerance: float) -> None:
    assert len(expected) == len(actual)
    for scale, expected_entry, actual_entry in zip(_scales(expected), expected, actual, strict=True):
        for values, rust_values in zip(expected_entry, actual_entry, strict=True):
            np.testing.assert_allclose(to_numpy(rust_values), to_numpy(values), rtol=0.0, atol=tolerance * scale)


@pytest.mark.usefixtures("crate_exp_attention", "crate_exp_parts")
@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9)], ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("windows", [1, 2, 3])
@pytest.mark.parametrize("name", WINDOWS)
@pytest.mark.parametrize("recurrence", RECURRENCES)
def test_training_keeps_numpys_weights_within_the_products_rounding(
    recurrence: str, name: str, windows: int, rule: UpdateRule
):
    # measured: within 5.8e-13 of each entry's scale after 20 steps
    rust, numpy = _pair(WINDOWS[name], rule, recurrence=RECURRENCES[recurrence])
    rows = _rows(windows, 4, windows)
    for _ in range(20):
        rust.learn_batch(0.1, rows)
        numpy.learn_batch(0.1, rows)
    _assert_close(numpy.snapshot(), rust.snapshot(), 1e-10)
    # the dropout masks drew alike, so the generators are where each other's are
    assert rust.rng.state == numpy.rng.bit_generator.state


@pytest.mark.usefixtures("crate_exp_attention", "crate_exp_parts")
@pytest.mark.parametrize("name", WINDOWS)
@pytest.mark.parametrize("recurrence", RECURRENCES)
def test_every_step_under_adam_has_numpys_gradients(recurrence: str, name: str):
    # Adam's trajectory is chaotic near its epsilon, so each step's gradients are compared from
    # numpy's weights, on networks fresh from them, from generators in one state
    trainer = WindowRecurrentArrayNetwork.randomized((K,), WINDOWS[name], RECURRENCES[recurrence], Adam(), seed=3)
    for step in range(10):
        rows = _rows(1 + step % 3, 4, step)
        X, labels = np.array([x for x, _ in rows]), [label for _, label in rows]
        rust, numpy = _pair(WINDOWS[name], SGD(), recurrence=RECURRENCES[recurrence])
        for network in (rust, numpy):
            network.restore(trainer.snapshot())
        rust.rng, numpy.rng = RUST.default_rng(step), NUMPY.default_rng(step)
        numpy._accumulate_gradients(X, numpy._target_batch_array(labels))
        rust._accumulate_gradients(RUST.matrix(X.tolist()), rust._target_batch_array(labels))
        expected = [tuple(layer.gradients()) for layer in [*numpy.layers, *numpy.parts] if hasattr(layer, "gradients")]
        actual = [tuple(layer.gradients()) for layer in [*rust.layers, *rust.parts] if hasattr(layer, "gradients")]
        _assert_close(expected, actual, 1e-10)
        trainer.learn_batch(0.01, rows)


@pytest.mark.usefixtures("crate_exp_attention", "crate_exp_parts")
@pytest.mark.parametrize("name", WINDOWS)
@pytest.mark.parametrize("recurrence", SNAPPED)
def test_consolidated_training_keeps_numpys_weights_within_the_products_rounding(recurrence: str, name: str):
    # the code-consolidation workplan's D1 through the program, against numpy's per-layer path
    rust, numpy = _pair(WINDOWS[name], Adam(), recurrence=RECURRENCES[recurrence])
    for network in (rust, numpy):
        network.consolidate(covering_consolidation(RECURRENCES[recurrence]))
    for step in range(10):
        rows = _rows(1 + step % 3, 4, step)
        X, labels = np.array([x for x, _ in rows]), [label for _, label in rows]
        fresh_rust, fresh_numpy = _pair(WINDOWS[name], SGD(), recurrence=RECURRENCES[recurrence])
        for network in (fresh_rust, fresh_numpy):
            network.restore(numpy.snapshot())
            network.consolidate(covering_consolidation(RECURRENCES[recurrence]))
        fresh_rust.rng, fresh_numpy.rng = RUST.default_rng(step), NUMPY.default_rng(step)
        fresh_numpy._accumulate_gradients(X, fresh_numpy._target_batch_array(labels))
        fresh_rust._accumulate_gradients(RUST.matrix(X.tolist()), fresh_rust._target_batch_array(labels))
        expected = [
            tuple(layer.gradients())
            for layer in [*fresh_numpy.layers, *fresh_numpy.parts]
            if hasattr(layer, "gradients")
        ]
        actual = [
            tuple(layer.gradients()) for layer in [*fresh_rust.layers, *fresh_rust.parts] if hasattr(layer, "gradients")
        ]
        _assert_close(expected, actual, 1e-10)
        numpy.learn_batch(0.01, rows)


def test_a_recomputed_window_draws_its_dropout_masks_again():
    window: list[LayerSpec] = [Embedding(27, 8), Position(), Dropout(0.3), StateInput(), ATTENTION, DIGITS]
    trained = WindowRecurrentRustArrayNetwork.randomized((K,), window, RECURRENCE, SGD(), seed=7)
    forward = WindowRecurrentRustArrayNetwork.randomized((K,), window, RECURRENCE, SGD(), seed=7)
    X, labels = _batch(3)
    trained._accumulate_gradients(RUST.matrix(X.tolist()), trained._target_batch_array(labels))
    # the per-layer path in training (the window program runs inference only)
    forward._set_training_mode(True)
    X_rust = RUST.matrix(X.tolist())
    forward.per_layer_windows(X_rust, forward._first_state(3), False, False)
    assert generator_state(trained.rng) == generator_state(forward.rng)


def test_one_window_with_the_projections_at_zero_is_the_rust_sequence_networks_digits_by_bits():
    network = WindowRecurrentRustArrayNetwork.randomized((K,), WINDOW, RECURRENCE, SGD(), seed=7)
    for state_input in network._state_inputs:
        state_input.P = RUST.zeros(tuple(state_input.P.shape))
    plain_specs = [spec for spec in WINDOW if not isinstance(spec, StateInput)]
    plain = SequentialArrayNetwork((K,), plain_specs, SGD(), shape="sequence", backend=RUST)
    weights = [entry for layer, entry in zip(network.layers, network.snapshot()) if layer not in network._state_inputs]
    plain.restore(weights)
    X, _ = _batch(1, examples=5)
    expected = [
        row
        for chunk in plain.forward_rows(plain.prepare_dataset([(tuple(x), (0,) * K) for x in X]))
        for row in chunk.tolist()
    ]
    assert bits(network.forward_windows(RUST.matrix(X.tolist()))) == bits(expected)


def test_the_network_reads_any_window_count_and_classifies_as_numpy_does():
    rust, numpy = _pair(WINDOW, SGD())
    for windows in (1, 2, 5):
        X, _ = _batch(windows)
        assert rust.forward_windows(RUST.matrix(X.tolist())).shape == (3, windows * K * 3)
        assert rust.classify_state(tuple(X[0])) == numpy.classify_state(tuple(X[0]))
    with pytest.raises(AssertionError, match="whole windows"):
        rust.forward_windows(RUST.zeros((1, 3)))


def test_a_saved_rust_network_loads_and_resumes_training_by_bits(tmp_path: Path):
    network = WindowRecurrentRustArrayNetwork.randomized((K,), WINDOW, RECURRENCE, Adam(), seed=7)
    rows = _rows(2, 3, 11)
    network.learn_batch(0.01, rows)
    path = str(tmp_path / "window.json")
    network.save(path)
    loaded = load_network(path)
    assert isinstance(loaded, WindowRecurrentRustArrayNetwork)
    assert loaded.recurrence == RECURRENCE
    assert bits(loaded.snapshot()) == bits(network.snapshot())
    network.learn_batch(0.01, rows)
    loaded.learn_batch(0.01, rows)
    assert bits(loaded.snapshot()) == bits(network.snapshot())


@pytest.mark.parametrize("recurrence", ["one factor", "two factors"])
def test_state_codes_are_numpys(recurrence: str):
    rust, numpy = _pair(WINDOW, SGD(), recurrence=RECURRENCES[recurrence])
    X, _ = _batch(3, examples=5)
    assert rust.state_codes(RUST.matrix(X.tolist())) == numpy.state_codes(X)


@pytest.mark.usefixtures("crate_exp_attention", "crate_exp_parts")
@pytest.mark.parametrize("windows", [1, 3])
def test_a_continuous_state_trains_as_numpys(windows: int):
    # the window-recurrent claims workplan, stage 4: no codes, the cell's output the next state
    recurrence = WindowRecurrence((Factor(3, None),))
    rust = WindowRecurrentRustArrayNetwork.randomized((K,), DROPOUT_WINDOW, recurrence, Momentum(0.9), seed=7)
    numpy = WindowRecurrentArrayNetwork.randomized((K,), DROPOUT_WINDOW, recurrence, Momentum(0.9), seed=7)
    rows = _rows(windows, 4, windows)
    for _ in range(20):
        rust.learn_batch(0.1, rows)
        numpy.learn_batch(0.1, rows)
    _assert_close(numpy.snapshot(), rust.snapshot(), 1e-10)
    assert rust.rng.state == numpy.rng.bit_generator.state
