# pyright: reportConstantRedefinition=false, reportPrivateUsage=false
"""
The numpy window-recurrent network (the window-recurrent adder workplan, stage 4): the gradient
through one, two and three windows against central differences, the codes made the identity so the
loss is differentiable; the codes' losses' terms through the network; the dropout masks a
recomputed window draws again; the plain sequence network's digits by bits with the state's
projections at zero and one window; and saving, loading and resuming by bits.
"""

import itertools
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pytest

from indrajala_ml.model.layers.array.branch_layers import BranchArrayLayer
from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.networks.window_recurrent_array_network import WindowRecurrentArrayNetwork
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    Branches,
    Dense,
    Dropout,
    Embedding,
    LayerNorm,
    LayerSpec,
    Position,
    Residual,
    StateInput,
)
from indrajala_ml.model.specs.update_rules import SGD, Adam
from indrajala_ml.model.specs.window_recurrence import (
    BEFORE_STATE,
    OUTPUT,
    AttentionPooling,
    Codes,
    Factor,
    WindowRecurrence,
)
from indrajala_ml.pcg64 import generator_state

K = 2
ATTENTION = Residual((LayerNorm(), Attention(heads=2, causal=True)))
FFN = Residual((LayerNorm(), Dense(16, activation="relu"), Dense(8, activation="linear", bias=True)))
DIGITS = Dense(3, output=True, activation="softmax", loss="cross_entropy")
WINDOW: list[LayerSpec] = [
    Embedding(27, 8),
    Position(),
    StateInput(),
    ATTENTION,
    StateInput(),
    FFN,
    LayerNorm(),
    DIGITS,
]
RECURRENCE = WindowRecurrence((Factor(3, Codes(4)),))
# factored states (the factored-state workplan, stage 4): two factors of their own sizes, pooling
# the tokens before the first StateInput (D10); and three, one continuous, each cell reading the
# others' states
FACTORED = WindowRecurrence((Factor(2, Codes(4)), Factor(3, Codes(3))), pooling=AttentionPooling(BEFORE_STATE))
FACTORED_CROSS = WindowRecurrence((Factor(2, Codes(4)), Factor(1, None), Factor(3, Codes(3))), cross=True)
RECURRENCES = {"one factor": RECURRENCE, "two factors": FACTORED, "three factors, crossed": FACTORED_CROSS}
SNAPPED = {"one factor": RECURRENCE, "two factors": FACTORED}

# windows with branches (the window branches workplan): a shared trunk, then a branch a factor and
# one for the answers, each ending in a LayerNorm; with two factors reading their branches' ends,
# and three crossed, one continuous


def _branched(factors: int) -> list[LayerSpec]:
    branch = (StateInput(), FFN, LayerNorm())
    first = (StateInput(), ATTENTION, LayerNorm())
    return [
        Embedding(27, 8),
        Position(),
        StateInput(),
        ATTENTION,
        Branches((first, *[branch] * factors)),
        DIGITS,
    ]


BRANCHED_RECURRENCES = {
    "two factors": WindowRecurrence((Factor(2, Codes(4)), Factor(3, Codes(3)))),
    "three factors, crossed": WindowRecurrence((Factor(2, Codes(4)), Factor(1, None), Factor(3, Codes(3))), cross=True),
}
BRANCHED = {name: _branched(len(r.factors)) for name, r in BRANCHED_RECURRENCES.items()}


def covering_consolidation(recurrence: WindowRecurrence) -> dict[tuple[int, ...], tuple[int, ...]]:
    # a consolidation for the tests (the code-consolidation workplan, D1): every joint code onto the
    # next in its first factor, but those at the first factor's last code, which keep their own, so
    # a fresh network's states, on one code or a few, are held elsewhere
    counts = [f.codes.count for f in recurrence.factors if f.codes is not None]
    last = counts[0] - 1
    return {code: (code[0] + 1, *code[1:]) for code in itertools.product(*map(range, counts)) if code[0] != last}


STEP = 1e-6


def _network(
    window: list[LayerSpec] = WINDOW, seed: int = 7, rule: Any = None, recurrence: WindowRecurrence = RECURRENCE
) -> WindowRecurrentArrayNetwork:
    return WindowRecurrentArrayNetwork.randomized((K,), window, recurrence, rule or SGD(), seed=seed)


def _batch(windows: int, examples: int = 3, seed: int = 11) -> tuple[FloatArray, list[tuple[int, ...]]]:
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 27, size=(examples, windows * K)).astype(np.float64)
    labels = [tuple(int(d) for d in row) for row in rng.integers(0, 3, size=(examples, windows * K))]
    return X, labels


class _Trained(Protocol):
    def parameters(self) -> tuple[FloatArray, ...]: ...

    def gradients(self) -> tuple[FloatArray, ...]: ...


def _trained(network: WindowRecurrentArrayNetwork) -> list[_Trained]:
    # every layer and part with parameters, in the optimizer's order
    return [cast("_Trained", layer) for layer in [*network.layers, *network.parts] if hasattr(layer, "parameters")]


def _factor(network: WindowRecurrentArrayNetwork) -> Any:
    # the state's one factor's parts: its pooling, cell, codes and initial state
    return network.recurrence_layer.factors[0]


def _identity_codes(network: WindowRecurrentArrayNetwork) -> None:
    # the codes as the identity, so every state is the cell's output and the loss differentiable
    def forward(H: FloatArray) -> FloatArray:
        return H.copy()

    def backward(dQ: FloatArray, weight: float) -> FloatArray:
        return dQ

    for factor in network.recurrence_layer.factors:
        if factor.codes is not None:
            factor.codes.forward_batch = forward
            factor.codes.backward_batch = backward


def _gradients(network: WindowRecurrentArrayNetwork, X: FloatArray, labels: list[tuple[int, ...]]) -> list[FloatArray]:
    network._accumulate_gradients(X, network._target_batch_array(labels))
    return [gradient.copy() for layer in _trained(network) for gradient in layer.gradients()]


def _numeric(network: WindowRecurrentArrayNetwork, X: FloatArray, labels: list[tuple[int, ...]]) -> list[FloatArray]:
    gradients: list[FloatArray] = []
    for value in (value for layer in _trained(network) for value in layer.parameters()):
        gradient = np.zeros(value.shape)
        for index in np.ndindex(value.shape):
            kept = float(value[index])
            value[index] = kept + STEP
            up = network.loss_batch(X, labels)
            value[index] = kept - STEP
            down = network.loss_batch(X, labels)
            value[index] = kept
            gradient[index] = (up - down) / (2 * STEP)
        gradients.append(gradient)
    return gradients


@pytest.mark.parametrize("windows", [1, 2, 3])
@pytest.mark.parametrize("recurrence", RECURRENCES)
def test_the_gradient_through_the_windows_matches_central_differences(recurrence: str, windows: int):
    network = _network(recurrence=RECURRENCES[recurrence])
    rng = np.random.default_rng(3)
    # every part away from its starting point, so no gradient is zero by symmetry
    for factor in network.recurrence_layer.factors:
        factor.pooling.q = rng.normal(size=factor.pooling.q.shape)
        factor.initial_state.c = rng.normal(size=factor.initial_state.c.shape)
    _identity_codes(network)
    X, labels = _batch(windows)
    analytic = _gradients(network, X, labels)
    numeric = _numeric(network, X, labels)
    assert len(analytic) == len(numeric)
    for a, n in zip(analytic, numeric, strict=True):
        np.testing.assert_allclose(a, n, rtol=1e-5, atol=1e-7)
    # the state carries gradient back: the initial state's and the first StateInput's are not zero
    assert np.any(_factor(network).initial_state.c != 0.0)


def _fresh_gradients(
    network: WindowRecurrentArrayNetwork, X: FloatArray, labels: list[tuple[int, ...]]
) -> list[FloatArray]:
    for layer in _trained(network):
        for gradient in layer.gradients():
            gradient.fill(0.0)
    return _gradients(network, X, labels)


@pytest.mark.parametrize("recurrence", SNAPPED)
def test_a_consolidation_holds_each_duplicates_state_to_its_representatives_codes(recurrence: str):
    # after each snap, every factor's codes hold their commitment targets: a duplicate joint code's
    # representative's index, every other code's own (the code-consolidation workplan, D1)
    network = _network(recurrence=RECURRENCES[recurrence])
    table = covering_consolidation(network.recurrence)
    X, labels = _batch(3, examples=8)
    plain = _fresh_gradients(network, X, labels)
    network.consolidate(table)
    assert network.consolidation == table
    layer = network.recurrence_layer
    network.state_codes(X)  # the last window's snap
    joint = list(zip(*layer.indices, strict=True))
    for g, f in enumerate(layer.factors):
        assert f.codes.targets.tolist() == [table.get(tuple(map(int, c)), c)[g] for c in joint]
    assert any(tuple(map(int, c)) in table for c in joint)
    consolidated = _fresh_gradients(network, X, labels)
    assert any(not np.array_equal(a, b) for a, b in zip(plain, consolidated, strict=True))
    # an empty consolidation clears the targets: training as without one, by bits
    network.consolidate({})
    network.state_codes(X)
    assert all(f.codes.targets is None for f in layer.factors)
    assert all(np.array_equal(a, b) for a, b in zip(plain, _fresh_gradients(network, X, labels), strict=True))


@pytest.mark.parametrize("recurrence", SNAPPED)
def test_set_commitments_changes_training_and_the_specs_values_restore_it_by_bits(recurrence: str):
    # the commitment-control workplan's diagnostic: commitment 0 drops the commitment's gradient
    network = _network(recurrence=RECURRENCES[recurrence])
    X, labels = _batch(3, examples=8)
    plain = _fresh_gradients(network, X, labels)
    specs = [0.0 if f.codes is None else f.codes.commitment for f in network.recurrence.factors]
    network.set_commitments([0.0] * len(specs))
    assert all(f.codes.commitment == 0.0 for f in network.recurrence_layer.factors)
    off = _fresh_gradients(network, X, labels)
    assert any(not np.array_equal(a, b) for a, b in zip(plain, off, strict=True))
    network.set_commitments(specs)
    assert all(np.array_equal(a, b) for a, b in zip(plain, _fresh_gradients(network, X, labels), strict=True))


def test_a_consolidation_is_refused_unless_it_fits_the_recurrence():
    with pytest.raises(AssertionError, match="every factor snapped"):
        _network(recurrence=FACTORED_CROSS).consolidate({(1, 0, 0): (0, 0, 0)})
    network = _network(recurrence=FACTORED)
    for table in ({(1,): (0,)}, {(1, 3): (0, 0)}, {(4, 0): (0, 0)}, {(1, 0): (0, -1)}):
        with pytest.raises(AssertionError, match="a code index per factor"):
            network.consolidate(table)
    assert network.consolidation == {}


@pytest.mark.parametrize("windows", [1, 3])
def test_the_codes_losses_count_every_snapped_state_once_at_one_over_the_windows(windows: int):
    # the codes' gradient is the codebook loss's, -2 (h - e) / W at each state's code, summed over
    # the initial state's snap and each window's; and each snap passes its gradient straight
    # through plus the commitment's, 2 * commitment * (h - e) / W
    network = _network()
    codes = _factor(network).codes
    backward = codes.backward_batch
    snaps: list[tuple[FloatArray, FloatArray, Any, FloatArray, FloatArray]] = []

    def recorded(dQ: FloatArray, weight: float) -> FloatArray:
        H, Q, indices = codes._H.copy(), codes._Q.copy(), codes.indices.copy()
        dH = backward(dQ, weight)
        assert weight == 1.0 / windows
        snaps.append((H, Q, indices, dQ.copy(), dH))
        return dH

    codes.backward_batch = recorded  # type: ignore[method-assign]
    X, labels = _batch(windows)
    network._accumulate_gradients(X, network._target_batch_array(labels))
    assert len(snaps) == windows + 1
    expected = np.zeros(codes.E.shape)
    for H, Q, indices, dQ, dH in snaps:
        np.add.at(expected, indices, -2.0 * (H - Q) / windows)
        np.testing.assert_allclose(dH, dQ + 2.0 * codes.commitment * (H - Q) / windows, rtol=1e-15)
    np.testing.assert_allclose(codes.grad_E, expected, rtol=1e-12, atol=1e-15)


def test_a_training_step_lowers_the_loss_on_its_batch():
    network = _network(rule=Adam())
    X, labels = _batch(3, examples=8)
    before = network.loss_batch(X, labels)
    for _ in range(30):
        network._learn_batch_input(0.01, X, labels)
    assert network.loss_batch(X, labels) < before


def test_a_recomputed_window_draws_its_dropout_masks_again():
    # the generator ends a step where one forward pass over the windows leaves it, and a window's
    # recompute draws the masks its forward pass drew: the gradient is the masks' network's
    window: list[LayerSpec] = [Embedding(27, 8), Position(), Dropout(0.3), StateInput(), ATTENTION, DIGITS]
    trained, forward = _network(window), _network(window)
    X, labels = _batch(3)
    trained._accumulate_gradients(X, trained._target_batch_array(labels))
    forward._set_training_mode(True)
    forward.forward_windows(X)
    assert generator_state(trained.rng) == generator_state(forward.rng)


def test_one_window_with_the_projections_at_zero_is_the_sequence_networks_digits_by_bits():
    network = _network()
    for state_input in network._state_inputs:
        state_input.P = np.zeros(state_input.P.shape)
    plain_specs = [spec for spec in WINDOW if not isinstance(spec, StateInput)]
    plain = SequentialArrayNetwork((K,), plain_specs, SGD(), shape="sequence")
    weights = [entry for layer, entry in zip(network.layers, network.snapshot()) if layer not in network._state_inputs]
    plain.restore(weights)
    X, _ = _batch(1, examples=5)
    expected = np.concatenate(list(plain.forward_rows(plain.prepare_dataset([(tuple(x), (0,) * K) for x in X]))))
    assert np.array_equal(network.forward_windows(X), expected)


def test_the_network_reads_any_window_count_and_classifies_each_column():
    network = _network()
    for windows in (1, 2, 5):
        X, _ = _batch(windows)
        assert network.forward_windows(X).shape == (3, windows * K * 3)
        digits = network.classify_state(tuple(X[0]))
        assert len(digits) == windows * K and all(d in (0, 1, 2) for d in digits)
    with pytest.raises(AssertionError, match="whole windows"):
        network.forward_windows(np.zeros((1, 3)))


def test_a_saved_network_loads_and_resumes_training_by_bits(tmp_path: Path):
    network = _network(rule=Adam())
    X, labels = _batch(2)
    network._learn_batch_input(0.01, X, labels)
    path = str(tmp_path / "window.json")
    network.save(path)
    loaded = load_network(path)
    assert isinstance(loaded, WindowRecurrentArrayNetwork)
    assert loaded.recurrence == RECURRENCE
    for a, b in zip(network.snapshot(), loaded.snapshot(), strict=True):
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    network._learn_batch_input(0.01, X, labels)
    loaded._learn_batch_input(0.01, X, labels)
    for a, b in zip(network.snapshot(), loaded.snapshot(), strict=True):
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert WindowRecurrentArrayNetwork.load(path).snapshot()[-1][0].shape == (RECURRENCE.state_size,)


def test_state_codes_are_each_states_nearest_code_through_the_windows():
    network = _network()
    X, _ = _batch(3, examples=4)
    codes = network.state_codes(X)
    assert len(codes) == 4 and all(len(example) == 3 + 1 for example in codes)
    # the same states, window by window, through the parts
    C = _factor(network).initial_state.forward_batch(4)
    expected = [_factor(network).codes.nearest(C).tolist()]
    C = _factor(network).codes.forward_batch(C)
    for w in range(3):
        _, C = network._window_forward(network._window(X, w), C)
        expected.append(_factor(network).codes.indices.tolist())
    assert codes == [[(code,) for code in example] for example in zip(*expected, strict=True)]


@pytest.mark.parametrize("recurrence", [RECURRENCE, WindowRecurrence((Factor(3, None),))])
def test_window_states_are_the_forward_pass_and_each_state_it_passes_on(recurrence: WindowRecurrence):
    network = _network(recurrence=recurrence)
    X, _ = _batch(3, examples=4)
    outputs, states, cells = network.window_states(X)
    assert np.array_equal(outputs, network.forward_windows(X))
    C = network._first_state(4)
    expected, expected_cells = [C], [network.recurrence_layer.H]
    for w in range(3):
        _, C = network._window_forward(network._window(X, w), C)
        expected.append(C)
        expected_cells.append(network.recurrence_layer.H)
    assert len(states) == len(cells) == 3 + 1
    assert all(np.array_equal(s, e) for s, e in zip(states, expected, strict=True))
    assert all(np.array_equal(c, e) for c, e in zip(cells, expected_cells, strict=True))


def test_the_snap_walls_gate_is_each_examples_every_digit_right_for_one_batch():
    network = _network()
    codes = _factor(network).codes
    codes.wall = 1.0
    X, _ = _batch(3, examples=6)
    predicted = network.backend.argmax_token_rows(network.forward_windows(X), 3)
    # half the examples labelled as the network answers, half with a digit changed
    labels = [p if i % 2 == 0 else ((p[0] + 1) % 3, *p[1:]) for i, p in enumerate(predicted)]
    gates: list[Any] = []
    backward = codes.backward_batch

    def recorded(dQ: FloatArray, weight: float) -> FloatArray:
        gates.append(None if codes.gate is None else codes.gate.copy())
        return backward(dQ, weight)

    codes.backward_batch = recorded  # type: ignore[method-assign]
    network._accumulate_gradients(X, network._target_batch_array(labels))
    assert len(gates) == 3 + 1 and all(np.array_equal(g, [1.0, 0.0] * 3) for g in gates)
    assert codes.gate is None


# a continuous state (the window-recurrent claims workplan, stage 4)

CONTINUOUS = WindowRecurrence((Factor(3, None),))


@pytest.mark.parametrize("windows", [1, 3])
def test_a_continuous_states_gradient_matches_central_differences(windows: int):
    network = WindowRecurrentArrayNetwork.randomized((K,), WINDOW, CONTINUOUS, SGD(), seed=7)
    assert _factor(network).codes is None and len(network.parts) == 3
    rng = np.random.default_rng(3)
    _factor(network).pooling.q = rng.normal(size=_factor(network).pooling.q.shape)
    _factor(network).initial_state.c = rng.normal(size=_factor(network).initial_state.c.shape)
    X, labels = _batch(windows)
    for a, n in zip(_gradients(network, X, labels), _numeric(network, X, labels), strict=True):
        np.testing.assert_allclose(a, n, rtol=1e-5, atol=1e-7)


def test_a_continuous_state_is_the_snapped_networks_with_identity_codes_by_bits():
    # the same weights but the codes': a training step and the digits after it agree by bits
    snapped = _network(rule=Adam())
    _identity_codes(snapped)
    continuous = WindowRecurrentArrayNetwork((K,), WINDOW, CONTINUOUS, Adam())
    codes_entry = len(snapped.layers) + snapped.parts.index(_factor(snapped).codes)
    continuous.restore([entry for i, entry in enumerate(snapped.snapshot()) if i != codes_entry])
    X, labels = _batch(3)
    for network in (snapped, continuous):
        network._learn_batch_input(0.01, X, labels)
    kept = [entry for i, entry in enumerate(snapped.snapshot()) if i != codes_entry]
    for a, b in zip(kept, continuous.snapshot(), strict=True):
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert np.array_equal(snapped.forward_windows(X), continuous.forward_windows(X))


def test_a_continuous_state_has_no_codes_to_report():
    network = WindowRecurrentArrayNetwork.randomized((K,), WINDOW, CONTINUOUS, SGD(), seed=7)
    X, _ = _batch(2)
    for call in (lambda: network.state_codes(X), network.first_step):
        with pytest.raises(AssertionError, match="no codes"):
            call()


# the factored state (the factored-state workplan, stage 4)


def test_a_factored_states_codes_are_a_tuple_of_each_factors():
    network = _network(recurrence=FACTORED)
    X, _ = _batch(3, examples=4)
    states = network.state_codes(X)
    assert len(states) == 4 and all(len(example) == 3 + 1 for example in states)
    first, second = network.recurrence_layer.factors
    C = network.recurrence_layer.first_state(4)
    expected = [list(zip(first.codes.indices.tolist(), second.codes.indices.tolist(), strict=True))]
    for w in range(3):
        _, C = network._window_forward(network._window(X, w), C)
        expected.append(list(zip(first.codes.indices.tolist(), second.codes.indices.tolist(), strict=True)))
    assert states == [list(example) for example in zip(*expected, strict=True)]
    # each factor's codes are its own: its state's columns snapped to them
    assert network.recurrence_layer.H.shape == (4, 5)
    assert np.array_equal(C[:, :2], first.codes.E[first.codes.indices])
    assert np.array_equal(C[:, 2:], second.codes.E[second.codes.indices])


@pytest.mark.parametrize("cross", [False, True])
@pytest.mark.parametrize("reads", [BEFORE_STATE, OUTPUT])
def test_a_factors_next_state_reads_the_others_only_through_the_cross_term_or_the_output_layers_tokens(
    reads: str, cross: bool
):
    # the cross-term off (D1's default) and the poolings before the first StateInput (D10), factor
    # 1's next state is a function of its own state and the window's columns: the second factor's
    # state changed leaves it by bits; with the cross-term, or pooling the tokens the StateInputs
    # have added the whole state to, it moves
    recurrence = WindowRecurrence((Factor(2, None), Factor(3, None)), cross=cross, pooling=AttentionPooling(reads))
    network = _network(recurrence=recurrence)
    rng = np.random.default_rng(5)
    X, _ = _batch(1, examples=4)
    C = rng.normal(size=(4, 5))
    moved = C.copy()
    moved[:, 2:] = rng.normal(size=(4, 3))
    _, before = network._window_forward(X, C)
    _, after = network._window_forward(X, moved)
    assert np.array_equal(before[:, :2], after[:, :2]) == (reads == BEFORE_STATE and not cross)
    assert not np.array_equal(before[:, 2:], after[:, 2:])


def test_a_factored_network_saves_and_loads_by_bits(tmp_path: Path):
    network = _network(rule=Adam(), recurrence=FACTORED_CROSS)
    X, labels = _batch(2)
    network._learn_batch_input(0.01, X, labels)
    path = str(tmp_path / "factored.json")
    network.save(path)
    loaded = load_network(path)
    assert loaded.recurrence == FACTORED_CROSS
    for a, b in zip(network.snapshot(), loaded.snapshot(), strict=True):
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert np.array_equal(loaded.forward_windows(X), network.forward_windows(X))


@pytest.mark.parametrize("windows", [1, 2])
@pytest.mark.parametrize("recurrence", BRANCHED_RECURRENCES)
def test_a_branched_windows_gradient_matches_central_differences(recurrence: str, windows: int):
    # the window branches workplan: each factor's pooling reads its branch's end, the answers' branch
    # runs on into the output layer, and the split's downstream sums its branches'
    network = _network(BRANCHED[recurrence], recurrence=BRANCHED_RECURRENCES[recurrence])
    rng = np.random.default_rng(4)
    for factor in network.recurrence_layer.factors:
        factor.pooling.q = rng.normal(size=factor.pooling.q.shape)
        factor.initial_state.c = rng.normal(size=factor.initial_state.c.shape)
    _identity_codes(network)
    X, labels = _batch(windows)
    analytic = _gradients(network, X, labels)
    numeric = _numeric(network, X, labels)
    assert len(analytic) == len(numeric)
    for a, n in zip(analytic, numeric, strict=True):
        np.testing.assert_allclose(a, n, rtol=1e-5, atol=1e-7)


def test_each_factor_reads_its_own_branch_and_the_answers_theirs():
    # with the cross-term off, a factor's next state is a function of its own state and its branch's
    # end: another factor's branch, or the answers', doesn't move it
    recurrence = BRANCHED_RECURRENCES["two factors"]
    network = _network(BRANCHED["two factors"], recurrence=recurrence)
    reads = network._pooling_reads
    assert len(set(reads)) == 2 and all(isinstance(network.layers[r], BranchArrayLayer) for r in reads)
    X = _batch(2)[0]
    _, _, cells = network.window_states(X)
    # the second factor's branch's first FFN layer, and the answers' branch's, perturbed
    second = next(i for i in range(reads[0] + 1, reads[1]) if hasattr(network.layers[i], "W"))
    answers = next(i for i in range(reads[1] + 1, len(network.layers) - 1) if hasattr(network.layers[i], "W"))
    for index in (second, answers):
        layer: Any = network.layers[index]
        layer.W = layer.W + 1.0
    # the first window's cells read the initial state alone; each later one's, the window before's
    # snapped state, which the first factor's own branch gives: its cells' outputs are unchanged
    _, _, moved = network.window_states(X)
    first_size = recurrence.factors[0].state_size
    np.testing.assert_array_equal(cells[1][:, :first_size], moved[1][:, :first_size])
    assert not np.array_equal(cells[1][:, first_size:], moved[1][:, first_size:])


def test_a_branched_network_saves_and_loads_by_bits(tmp_path: Path):
    network = _network(BRANCHED["two factors"], rule=Adam(), recurrence=BRANCHED_RECURRENCES["two factors"])
    X, labels = _batch(2)
    rows = [(tuple(x), label) for x, label in zip(X.tolist(), labels, strict=True)]
    network.learn_batch(0.01, rows)
    path = tmp_path / "branched.json"
    network.save(str(path))
    loaded = load_network(str(path))
    assert isinstance(loaded, WindowRecurrentArrayNetwork) and loaded._pooling_reads == network._pooling_reads
    np.testing.assert_array_equal(loaded.forward_windows(X), network.forward_windows(X))
    network.learn_batch(0.01, rows)
    loaded.learn_batch(0.01, rows)
    for a, b in zip(network.snapshot(), loaded.snapshot(), strict=True):
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize(
    "branches, recurrence, message",
    [
        (((StateInput(), FFN, LayerNorm()),) * 2, "two factors", "one a factor and one for the answers"),
        (((StateInput(), FFN, LayerNorm()),) * 3, "before state", "reads its branch's end"),
        (((StateInput(), FFN),) * 3, "two factors", "ends in a LayerNorm"),
        (((Position(), StateInput(), FFN, LayerNorm()),) * 3, "two factors", "not Position"),
    ],
)
def test_a_window_refuses_malformed_branches(branches: Any, recurrence: str, message: str):
    recurrences = {**BRANCHED_RECURRENCES, "before state": FACTORED}
    window: list[LayerSpec] = [Embedding(27, 8), Position(), StateInput(), ATTENTION, Branches(branches), DIGITS]
    with pytest.raises(AssertionError, match=message):
        _network(window, recurrence=recurrences[recurrence])
    # branches stand last, right after a residual block
    with pytest.raises(AssertionError, match="right before the output"):
        _network(
            [
                Embedding(27, 8),
                StateInput(),
                ATTENTION,
                Branches(((StateInput(), FFN, LayerNorm()),) * 3),
                LayerNorm(),
                DIGITS,
            ],
            recurrence=BRANCHED_RECURRENCES["two factors"],
        )
