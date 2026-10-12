# pyright: reportPrivateUsage=false
"""
The window-recurrent network in pure Python (the window-recurrent adder workplan, stage 6), against
the numpy network (tests/model/networks/test_window_recurrent_array_network.py, which checks the
gradient against central differences), with numpy's exps as math.exp. The two draw the same weights
by bits (tests/model/networks/test_seeded_init_parity.py); training through one, two and three
windows, with dropout, keeps numpy's weights to within the products' rounding (pure Python's folds
against BLAS), and each step under Adam has numpy's gradients to within it. The codes are numpy's
by bits, and the other parts within the products' rounding. Exactly: a recomputed window draws its
masks again, one window with the state's projections at zero is the pure-Python sequence network's
digits by bits, and a saved network loads and resumes by bits.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.numpy import attention_array_layer, recurrent_array_layer, token_array_layer
from indrajala_ml.model.layers.python import recurrent_layer
from indrajala_ml.model.networks.python.sequential_backprop_network import SequentialSequenceBackpropNetwork
from indrajala_ml.model.networks.python.window_recurrent_backprop_network import WindowRecurrentBackpropNetwork
from indrajala_ml.model.networks.window_recurrent_array_network import WindowRecurrentArrayNetwork
from indrajala_ml.model.persistence.format2 import network_to_json
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.layer_specs import Dropout, Embedding, LayerSpec, Position, StateInput
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from indrajala_ml.pcg64 import default_rng, generator_state
from tests.helpers import bits, exp_by_math, patching
from tests.model.networks.test_window_recurrent_array_network import (
    ATTENTION,
    DIGITS,
    RECURRENCE,
    WINDOW,
    K,
    _batch,
)
from tests.python_array_snapshot import as_array_snapshot, assert_seeded_alike

# the window with dropout before the state's first input and in its attention
DROPOUT_WINDOW: list[LayerSpec] = [Embedding(27, 8), Position(), Dropout(0.2), *WINDOW[2:]]
WINDOWS = {"plain": WINDOW, "with dropout": DROPOUT_WINDOW}

# numpy's exps, the attention's, the token-wise output's and the window's parts', as math.exp
attention_exp = patching(attention_array_layer, "exp", exp_by_math)
token_exp = patching(token_array_layer, "exp", exp_by_math)
parts_exp = patching(recurrent_array_layer, "exp", exp_by_math)


def _pair(window: list[LayerSpec], rule: UpdateRule, seed: int = 7) -> tuple[Any, Any]:
    return (
        WindowRecurrentBackpropNetwork.randomized((K,), window, RECURRENCE, rule, seed=seed),
        WindowRecurrentArrayNetwork.randomized((K,), window, RECURRENCE, rule, seed=seed),
    )


def _rows(windows: int, examples: int, seed: int) -> list[tuple[tuple[float, ...], tuple[int, ...]]]:
    X, labels = _batch(windows, examples, seed)
    return [(tuple(x), label) for x, label in zip(X.tolist(), labels, strict=True)]


def _assert_close(expected: list[tuple[Any, ...]], actual: list[tuple[Any, ...]], tolerance: float) -> None:
    # within tolerance of each entry's largest magnitude, the yardstick of its rounding
    assert len(expected) == len(actual)
    for expected_entry, actual_entry in zip(expected, actual, strict=True):
        scale = max((float(np.abs(np.asarray(a)).max()) for a in expected_entry), default=0.0)
        for values, python_values in zip(expected_entry, actual_entry, strict=True):
            np.testing.assert_allclose(
                np.asarray(python_values, dtype=np.float64),
                np.asarray(values, dtype=np.float64),
                rtol=0.0,
                atol=tolerance * scale,
            )


def _gradients(python: WindowRecurrentBackpropNetwork) -> list[tuple[Any, ...]]:
    # the accumulated gradients in snapshot()'s shape, then as numpy's
    entries = [
        [
            (list(row.weight_gradient_accum), row.bias_gradient_accum)
            if row.has_bias
            else (list(row.weight_gradient_accum),)
            for row in layer.weight_sets()
        ]
        for layer in [*python.trainable_layers, *python.parts]
    ]
    return as_array_snapshot(python, entries)


@pytest.mark.usefixtures("attention_exp", "token_exp", "parts_exp")
@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), WeightDecay(0.01)], ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("windows", [1, 2, 3])
@pytest.mark.parametrize("name", WINDOWS)
def test_training_keeps_numpys_weights_within_the_products_rounding(name: str, windows: int, rule: UpdateRule):
    # measured: within 4.4e-13 of each entry's scale after 20 steps
    python, numpy = _pair(WINDOWS[name], rule)
    assert_seeded_alike(python, numpy)
    rows = _rows(windows, 4, windows)
    for _ in range(20):
        python.learn_batch(0.1, rows)
        numpy.learn_batch(0.1, rows)
    _assert_close(numpy.snapshot(), as_array_snapshot(python), 1e-10)
    # the dropout masks drew alike, so the generators are where each other's are
    assert generator_state(python.rng) == generator_state(numpy.rng)


@pytest.mark.usefixtures("attention_exp", "token_exp", "parts_exp")
@pytest.mark.parametrize("name", WINDOWS)
def test_every_step_under_adam_has_numpys_gradients(name: str):
    # Adam's trajectory is chaotic near its epsilon, so each step's gradients are compared from
    # the trainer's weights, on networks fresh from them, from generators in one state. Measured:
    # within 3.9e-15 of each entry's scale
    trainer = WindowRecurrentBackpropNetwork.randomized((K,), WINDOWS[name], RECURRENCE, Adam(), seed=3)
    for step in range(10):
        rows = _rows(1 + step % 3, 4, step)
        X, labels = np.array([x for x, _ in rows]), [label for _, label in rows]
        python, numpy = _pair(WINDOWS[name], SGD())
        python.restore(trainer.snapshot())
        numpy.restore(as_array_snapshot(trainer))
        python.rng, numpy.rng = default_rng(step), np.random.default_rng(step)
        python._accumulate_batch_gradients([x for x, _ in rows], labels)
        numpy._accumulate_gradients(X, numpy._target_batch_array(labels))
        expected = [
            tuple(layer.gradients()) if hasattr(layer, "gradients") else () for layer in [*numpy.layers, *numpy.parts]
        ]
        _assert_close(expected, _gradients(python), 1e-10)
        trainer.learn_batch(0.01, rows)


def test_a_recomputed_window_draws_its_dropout_masks_again():
    window: list[LayerSpec] = [Embedding(27, 8), Position(), Dropout(0.3), StateInput(), ATTENTION, DIGITS]
    trained = WindowRecurrentBackpropNetwork.randomized((K,), window, RECURRENCE, SGD(), seed=7)
    forward = WindowRecurrentBackpropNetwork.randomized((K,), window, RECURRENCE, SGD(), seed=7)
    X, labels = _batch(3)
    trained._accumulate_batch_gradients(X.tolist(), labels)
    forward._set_training_mode(True)
    forward.forward_windows(X.tolist())
    assert generator_state(trained.rng) == generator_state(forward.rng)


def test_one_window_with_the_projections_at_zero_is_the_sequence_networks_digits_by_bits():
    network = WindowRecurrentBackpropNetwork.randomized((K,), WINDOW, RECURRENCE, SGD(), seed=7)
    for state_input in network._state_inputs:
        for row in state_input.rows:
            row.set_weights([0.0] * len(row.weights))
    plain_specs = [spec for spec in WINDOW if not isinstance(spec, StateInput)]
    plain = SequentialSequenceBackpropNetwork((K,), plain_specs, SGD())
    weights = [
        entry
        for layer, entry in zip(network.trainable_layers, network.snapshot())
        if layer not in network._state_inputs
    ]
    plain.restore(weights)
    X, _ = _batch(1, examples=5)
    assert bits(network.forward_windows(X.tolist())) == bits([plain._forward(tuple(x)) for x in X.tolist()])


@pytest.mark.usefixtures("attention_exp", "token_exp", "parts_exp")
def test_the_network_reads_any_window_count_and_classifies_as_numpy_does():
    python, numpy = _pair(WINDOW, SGD())
    for windows in (1, 2, 5):
        X, _ = _batch(windows)
        outputs = python.forward_windows(X.tolist())
        assert [len(row) for row in outputs] == [windows * K * 3] * 3
        np.testing.assert_allclose(outputs, numpy.forward_windows(X), rtol=0.0, atol=1e-14)
        assert python.classify_state(tuple(X[0])) == numpy.classify_state(tuple(X[0]))
    with pytest.raises(AssertionError, match="whole windows"):
        python.forward_windows([(0.0,) * 3])
    with pytest.raises(AssertionError, match="the same window count"):
        python.forward_windows([(0.0,) * 2, (0.0,) * 4])


@pytest.mark.usefixtures("attention_exp", "token_exp", "parts_exp")
def test_the_loss_is_numpys_within_the_products_rounding():
    python, numpy = _pair(WINDOW, SGD())
    X, labels = _batch(2, examples=4)
    assert python.loss_batch(X.tolist(), labels) == pytest.approx(numpy.loss_batch(X, labels), rel=1e-13)


def test_a_saved_python_network_loads_and_resumes_training_by_bits(tmp_path: Path):
    network = WindowRecurrentBackpropNetwork.randomized((K,), DROPOUT_WINDOW, RECURRENCE, Adam(), seed=7)
    rows = _rows(2, 3, 11)
    network.learn_batch(0.01, rows)
    path = str(tmp_path / "window.json")
    network.save(path)
    state = network_to_json(network)
    assert state["implementation"] == "python" and state["shape"] == "window_recurrent"

    loaded = load_network(path)
    assert isinstance(loaded, WindowRecurrentBackpropNetwork)
    assert isinstance(WindowRecurrentBackpropNetwork.load(path), WindowRecurrentBackpropNetwork)
    with pytest.raises(ValueError, match="implementation python"):
        WindowRecurrentArrayNetwork.load(path)
    assert loaded.recurrence == RECURRENCE
    assert bits(loaded.snapshot()) == bits(network.snapshot())
    network.learn_batch(0.01, rows)
    loaded.learn_batch(0.01, rows)
    assert bits(loaded.snapshot()) == bits(network.snapshot())
    assert generator_state(loaded.rng) == generator_state(network.rng)


# the parts


def _uniform(shape: tuple[int, ...], seed: int) -> Any:
    return np.random.default_rng(seed).uniform(-1.0, 1.0, shape)


def test_the_codes_are_numpys_by_bits():
    # the distances and the code losses are folds in both, and the gradients elementwise
    python, array = recurrent_layer.CodesLayer(5, 3, 0.3), recurrent_array_layer.CodesArrayLayer(5, 3, 0.3)
    E, H, dQ = _uniform((5, 3), 0), _uniform((6, 3), 1), _uniform((6, 3), 2)
    python.restore_state([(row,) for row in E.tolist()])
    array.E = E
    for _ in range(2):
        assert bits(python.forward_batch(H.tolist())) == bits(array.forward_batch(H).tolist())
        assert python.indices == array.indices.tolist()
        assert bits(python.loss_batch(0.5)) == bits(array.loss_batch(0.5))
        assert bits(python.backward_batch(dQ.tolist(), 0.5)) == bits(array.backward_batch(dQ, 0.5).tolist())
    assert bits([row.weight_gradient_accum for row in python.codes]) == bits(array.grad_E.tolist())


def test_a_tie_snaps_to_the_lowest_index_code():
    codes = recurrent_layer.CodesLayer(3, 2)
    codes.restore_state([([1.0, 0.0],), ([0.0, 1.0],), ([1.0, 0.0],)])
    assert codes.nearest([0.5, 0.5]) == 0 and codes.nearest([2.0, 0.0]) == 0


def test_the_initial_state_is_numpys_by_bits():
    python, array = recurrent_layer.InitialStateLayer(3), recurrent_array_layer.InitialStateArrayLayer(3)
    c, dC = _uniform((3,), 0), _uniform((5, 3), 1) * 1e8
    python.restore_state([(c.tolist(),)])
    array.c = c
    assert bits(python.forward_batch(5)) == bits(array.forward_batch(5).tolist())
    for _ in range(2):
        python.backward_batch(dC.tolist())
        array.backward_batch(dC)
    assert bits(python.state.weight_gradient_accum) == bits(array.grad_c.tolist())


@pytest.mark.usefixtures("parts_exp")
def test_the_pooling_and_the_cell_are_numpys_within_the_products_rounding():
    pooling, pooling_array = (
        recurrent_layer.AttentionPoolingLayer(4, 3),
        recurrent_array_layer.AttentionPoolingArrayLayer(4, 3),
    )
    q = _uniform((3,), 0)
    pooling.restore_state([(q.tolist(),)])
    pooling_array.q = q
    X, dU = _uniform((5, 12), 1), _uniform((5, 3), 2)
    np.testing.assert_allclose(pooling.forward_batch(X.tolist()), pooling_array.forward_batch(X), rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        pooling.backward_batch(dU.tolist()), pooling_array.backward_batch(dU), rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(pooling.query.weight_gradient_accum, pooling_array.grad_q, rtol=0, atol=1e-15)

    cell, cell_array = recurrent_layer.GRULayer(3, 2, 1.0), recurrent_array_layer.GRUArrayLayer(3, 2, 1.0)
    cell.draw(default_rng(4))
    cell_array.draw(np.random.default_rng(4))
    assert bits([row.bias for row in cell.rows]) == bits(cell_array.b.tolist())
    U_in, C, dH = _uniform((5, 3), 5), _uniform((5, 2), 6), _uniform((5, 2), 7)
    np.testing.assert_allclose(
        cell.forward_batch(U_in.tolist(), C.tolist()), cell_array.forward_batch(U_in, C), rtol=0, atol=1e-15
    )
    for python_values, array_values in zip(
        cell.backward_batch(dH.tolist()), cell_array.backward_batch(dH), strict=True
    ):
        np.testing.assert_allclose(python_values, array_values, rtol=0, atol=1e-15)
    W, U, b = (np.asarray(values) for values in _gradients_of_cell(cell))
    for values, array_values in zip((W, U, b), cell_array.gradients(), strict=True):
        np.testing.assert_allclose(values, array_values, rtol=0, atol=1e-15)


def _gradients_of_cell(cell: recurrent_layer.GRULayer) -> tuple[Any, Any, Any]:
    d = cell.input_size
    return (
        [row.weight_gradient_accum[:d] for row in cell.rows],
        [row.weight_gradient_accum[d:] for row in cell.rows],
        [row.bias_gradient_accum for row in cell.rows],
    )


def test_the_cells_tanh_and_sigmoid_take_numpys_values_where_exp_overflows():
    # numpy's exp gives inf there, and its forms exactly -1, 1 and 0
    z = np.array([400.0, -400.0, 800.0, -800.0])
    assert bits([recurrent_layer.tanh(v) for v in z.tolist()]) == bits(recurrent_array_layer.tanh(z).tolist())
    assert bits([recurrent_layer.sigmoid(v) for v in z.tolist()]) == bits(recurrent_array_layer.sigmoid(z).tolist())
    assert [recurrent_layer.tanh(v) for v in z.tolist()] == [1.0, -1.0, 1.0, -1.0]
