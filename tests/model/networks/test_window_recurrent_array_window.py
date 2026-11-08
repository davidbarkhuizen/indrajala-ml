# pyright: reportConstantRedefinition=false, reportPrivateUsage=false
"""
The window-recurrent network with an array window (the array window workplan, S2, S3): a conv
body over each step's observation, the state entering after it as one token, learning from a given
gradient at every step, from stored states with resets. The gradient through 1-3 steps against
central differences (the codes as the identity; soft codes, coupled; a continuous factor), through
resets and from a stored state; the rollout's reading against the training step's; a 2-step memory
learned with every factor snapped; the refusals; and saving, loading and resuming by bits. The
central differences run on numpy; every other check runs on both backends, and the Rust per-layer
path's steps and gradients are numpy's to within the products' rounding (S3).

A snap's analytic gradients are the exact gradients of a surrogate (CodesArrayLayer): a hard snap
passes its gradient straight through, q = h + sg(e - h), and the codes' losses hold their
stop-gradients (sg[h], sg[q], the soft weights). The central differences run that surrogate, each
snap's stop-gradient values recorded at the point the gradient is taken (_Snaps).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.layers.numpy import activations, recurrent_array_layer
from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.layers.numpy.recurrent_array_layer import CodesArrayLayer
from indrajala_ml.model.networks.python.window_recurrent_backprop_network import WindowRecurrentBackpropNetwork
from indrajala_ml.model.networks.window_recurrent_array_network import (
    WindowRecurrentArrayNetwork,
    WindowRecurrentRustArrayNetwork,
)
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    BatchNorm,
    Conv,
    Dense,
    LayerSpec,
    Pool,
    StateInput,
)
from indrajala_ml.model.specs.update_rules import SGD, Adam
from indrajala_ml.model.specs.window_recurrence import (
    BEFORE_STATE,
    AttentionPooling,
    Codes,
    Factor,
    WindowRecurrence,
)
from tests.helpers import exp_by_crate, to_numpy

SHAPE = (6, 7, 2)  # a step's observation: 6 x 7 cells of 2 planes
STEP_WIDTH = 6 * 7 * 2
OUTPUTS = 3
GIVEN = Dense(OUTPUTS, "linear", output=True, loss="given")
# the policy's body in small (the machine-policy plan's): conv, conv, pool, then the state as one
# token of the conv stack's output (2 x 3 x 3), a relu mixing state and screen, the given output
WINDOW: list[LayerSpec] = [Conv(1, 3), Conv(2, 3), Pool(2), StateInput(), Dense(6, "relu"), GIVEN]
RECURRENCE = WindowRecurrence((Factor(3, Codes(4)),))
FACTORED = WindowRecurrence((Factor(2, Codes(4)), Factor(2, Codes(3))), pooling=AttentionPooling(BEFORE_STATE))
CONTINUOUS = WindowRecurrence((Factor(2, Codes(3)), Factor(2, None)), cross=True)
STEP = 1e-6


BACKENDS: dict[str, Any] = {"numpy": WindowRecurrentArrayNetwork, "rust": WindowRecurrentRustArrayNetwork}


def _network(
    recurrence: WindowRecurrence = RECURRENCE,
    window: list[LayerSpec] = WINDOW,
    seed: int = 7,
    rule: Any = None,
    backend: str = "numpy",
) -> Any:
    return BACKENDS[backend].randomized(SHAPE, window, recurrence, rule or SGD(), seed=seed)


def _steps(steps: int, rows: int = 3, seed: int = 11) -> FloatArray:
    return np.random.default_rng(seed).normal(size=(rows, steps * STEP_WIDTH))


def _moved(network: WindowRecurrentArrayNetwork) -> None:
    # every part away from its starting point, so no gradient is zero by symmetry (one token's
    # pooling has none: its query's gradient is exactly 0)
    rng = np.random.default_rng(3)
    for factor in network.recurrence_layer.factors:
        factor.pooling.q = network.backend.vector(rng.normal(size=factor.pooling.q.shape).tolist())
        factor.initial_state.c = network.backend.vector(rng.normal(size=factor.initial_state.c.shape).tolist())


def _identity_codes(network: WindowRecurrentArrayNetwork) -> None:
    # the codes as the identity, so every state is the cell's output
    def forward(H: FloatArray, temperature: float | None = None) -> FloatArray:
        return H.copy()

    def backward(dQ: FloatArray, weight: float) -> FloatArray:
        return dQ

    for factor in network.recurrence_layer.factors:
        if factor.codes is not None:
            factor.codes.forward_batch = forward
            factor.codes.backward_batch = backward


class _Snaps:
    """
    Each codes layer's snaps' stop-gradient values, recorded in a forward pass at the gradient's
    point, then replayed (replay) as the surrogate whose exact gradients the analytic ones are.
    """

    def __init__(self, network: WindowRecurrentArrayNetwork, weight: float) -> None:
        self.weight = weight
        self.layers: list[CodesArrayLayer] = [f.codes for f in network.recurrence_layer.factors if f.codes is not None]
        self.records: list[list[tuple[FloatArray, FloatArray, Any, FloatArray | None, float | None]]] = [
            [] for _ in self.layers
        ]
        self.loss = 0.0
        self._originals = [layer.forward_batch for layer in self.layers]
        self._calls = [0] * len(self.layers)

    def record(self) -> None:
        for g, layer in enumerate(self.layers):
            original = self._originals[g]

            def recorded(H: FloatArray, temperature: float | None = None, *, _g: int = g, _f: Any = original) -> Any:
                Q = _f(H, temperature)
                codes = self.layers[_g]
                W = None if codes._W is None else codes._W.copy()
                self.records[_g].append((H.copy(), Q.copy(), codes.indices.copy(), W, temperature))
                return Q

            layer.forward_batch = recorded  # type: ignore[method-assign]

    def start(self) -> None:
        """A forward pass's replay from its first snap."""
        self._calls = [0] * len(self.layers)
        self.loss = 0.0

    def replay(self) -> None:
        for g, layer in enumerate(self.layers):

            def surrogate(H: FloatArray, temperature: float | None = None, *, _g: int = g) -> Any:
                H0, Q0, indices, W0, recorded = self.records[_g][self._calls[_g]]
                self._calls[_g] += 1
                assert temperature == recorded
                codes = self.layers[_g]
                codes.indices = indices
                c, w = codes.commitment, self.weight
                if temperature is None:
                    # straight through, its losses w (||sg[h] - e||^2 + c ||h - sg[e]||^2)
                    self.loss += w * (np.sum((H0 - codes.E[indices]) ** 2) + c * np.sum((H - Q0) ** 2))
                    return H + (Q0 - H0)
                assert W0 is not None
                differences = H[:, np.newaxis, :] - codes.E[np.newaxis, :, :]
                D = np.sum(differences * differences, axis=2)
                A = np.exp(-(D - D.min(axis=1)[:, np.newaxis]) / temperature)
                Q = (A / A.sum(axis=1)[:, np.newaxis]) @ codes.E
                pulled = H0[:, np.newaxis, :] - codes.E[np.newaxis, :, :]
                self.loss += w * (c * np.sum((H - Q0) ** 2) + np.sum(W0[:, :, np.newaxis] * pulled**2))
                return Q

            layer.forward_batch = surrogate  # type: ignore[method-assign]


def _trained(network: WindowRecurrentArrayNetwork) -> list[Any]:
    return [layer for layer in [*network.layers, *network.parts] if hasattr(layer, "parameters")]


def _given(G: FloatArray) -> Callable[[FloatArray], FloatArray]:
    def gradient(Y: FloatArray) -> FloatArray:
        assert Y.shape == G.shape
        return G

    return gradient


def _check_central_differences(
    network: WindowRecurrentArrayNetwork,
    X: FloatArray,
    state: FloatArray | None = None,
    starts: Any = None,
    snapped: bool = True,
) -> list[FloatArray]:
    # the analytic gradient of sum(G * Y) (each row's loss G_i . Y_i) plus the codes' losses, against
    # central differences of its surrogate; returns the analytic gradients
    steps = X.shape[1] // STEP_WIDTH
    G = np.random.default_rng(5).normal(size=(X.shape[0], steps * OUTPUTS))
    resumed = network._resumed(X, state, starts)
    snaps = _Snaps(network, 1.0 / steps)
    if snapped:
        snaps.record()
    network.per_layer_gradients(X, gradient=_given(G), state=state, starts=resumed)
    analytic = [gradient.copy() for layer in _trained(network) for gradient in layer.gradients()]
    if snapped:
        # the records of the first forward pass (the initial state's snap, then each step's) are
        # the surrogate's: the backward pass's recomputes are dropped
        snaps.records = [r[: steps + 1] for r in snaps.records]
        snaps.replay()

    def objective() -> float:
        snaps.start()
        Y = network.run_steps(X, state=state, starts=starts).outputs
        return float(np.sum(G * Y)) + snaps.loss

    for value, a in zip((v for layer in _trained(network) for v in layer.parameters()), analytic, strict=True):
        numeric = np.zeros(value.shape)
        for index in np.ndindex(value.shape):
            kept = float(value[index])
            value[index] = kept + STEP
            up = objective()
            value[index] = kept - STEP
            down = objective()
            value[index] = kept
            numeric[index] = (up - down) / (2 * STEP)
        np.testing.assert_allclose(a, numeric, rtol=1e-5, atol=1e-7)
    return analytic


# central differences through the steps (the workplan's test 2)


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_the_gradient_through_the_steps_matches_central_differences_with_identity_codes(steps: int):
    network = _network(FACTORED)
    _moved(network)
    _identity_codes(network)
    analytic = _check_central_differences(network, _steps(steps), snapped=False)
    # the state carries gradient back: the initial states' are not zero
    assert all(np.any(f.initial_state.grad_c != 0.0) for f in network.recurrence_layer.factors)
    assert any(np.any(a != 0.0) for a in analytic)


@pytest.mark.parametrize("steps", [1, 3])
def test_the_gradient_through_soft_coupled_codes_matches_central_differences(steps: int):
    network = _network(FACTORED)
    _moved(network)
    network.set_temperatures([0.7, 0.4])
    network.set_couplings([0.5, 0.5])
    _check_central_differences(network, _steps(steps))


@pytest.mark.parametrize("steps", [1, 3])
def test_the_gradient_through_hard_codes_matches_their_straight_through_surrogate(steps: int):
    network = _network()
    _moved(network)
    _check_central_differences(network, _steps(steps))


@pytest.mark.parametrize("steps", [1, 2])
def test_the_gradient_with_a_continuous_factor_matches_central_differences(steps: int):
    network = _network(CONTINUOUS)
    _moved(network)
    network.set_temperatures([0.6, None])
    _check_central_differences(network, _steps(steps))


# stored states and resets (test 3)


def test_the_gradient_through_resets_and_from_a_stored_state_matches_central_differences():
    network = _network(FACTORED)
    _moved(network)
    network.set_temperatures([0.7, None])
    X = _steps(3, rows=4)
    state = network.run_steps(_steps(2, rows=4, seed=2)).states[-1]
    starts = [[False, True, False], [False, False, True], [True, False, True], [False, False, False]]
    _check_central_differences(network, X, state=state, starts=starts)


def test_resets_gradient_matches_central_differences_from_the_initial_state():
    network = _network()
    _moved(network)
    starts = [[False, True, False], [True, False, False], [False, False, False]]
    _check_central_differences(network, _steps(3), starts=starts)


def _in(network: Any, A: FloatArray) -> Any:
    # a numpy array as the network's backend's
    return network.backend.matrix(A.tolist())


def _equal(a: Any, b: Any) -> bool:
    return np.array_equal(to_numpy(a), to_numpy(b))


def _columns(network: Any, X: FloatArray, start: int, stop: int) -> Any:
    # steps start to stop of X, as the network's
    return _in(network, X[:, start * STEP_WIDTH : stop * STEP_WIDTH])


def _zeros(network: Any) -> Callable[[Any], Any]:
    return lambda Y: network.backend.zeros(tuple(Y.shape))


def _backend_given(network: Any, G: FloatArray) -> Callable[[Any], Any]:
    given = _in(network, G)

    def gradient(Y: Any) -> Any:
        assert tuple(Y.shape) == G.shape
        return given

    return gradient


@pytest.mark.parametrize("machine", [False, True])
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_sequence_from_a_stored_state_is_the_steps_run_after_those_before_it_by_bits(backend: str, machine: bool):
    network = _network(FACTORED, backend=backend)
    network.set_temperatures([0.7, None])
    X = _steps(5)
    whole = network.run_steps(_in(network, X), machine=machine)
    before = network.run_steps(_columns(network, X, 0, 2), machine=machine)
    after = network.run_steps(_columns(network, X, 2, 5), state=before.states[-1], machine=machine)
    assert _equal(whole.outputs, np.concatenate([to_numpy(before.outputs), to_numpy(after.outputs)], axis=1))
    for a, b in zip(whole.states, [*before.states, *after.states], strict=True):
        assert _equal(a, b)
    assert whole.codes == [*before.codes, *after.codes]


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_reset_at_a_step_is_a_fresh_sequence_from_it_by_bits(backend: str):
    network = _network(FACTORED, backend=backend)
    network.set_couplings([0.5, 1.0])
    X = _steps(4)
    t = 2
    starts = np.zeros((3, 4), dtype=bool)
    starts[[0, 2], t] = True
    reset = to_numpy(network.run_steps(_in(network, X), starts=starts.tolist()).outputs)
    fresh = to_numpy(network.run_steps(_columns(network, X, t, 4)).outputs)
    plain = network.run_steps(_in(network, X))
    for row in (0, 2):
        assert np.array_equal(reset[row, t * OUTPUTS :], fresh[row])
    # the row that doesn't start again runs on, as without resets
    assert np.array_equal(reset[1], to_numpy(plain.outputs)[1])
    # the stored state is replaced where a row starts at the first step
    first = np.zeros((3, 4), dtype=bool)
    first[:, 0] = True
    replaced = network.run_steps(_in(network, X), state=plain.states[-1], starts=first.tolist())
    assert _equal(replaced.outputs, plain.outputs)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_stored_state_takes_no_gradient_and_a_reset_at_the_first_step_is_the_initial_states_by_bits(
    backend: str,
):
    X = _steps(3)
    stored, alone = _network(backend=backend), _network(backend=backend)
    state = stored.run_steps(_in(stored, _steps(1, seed=4))).states[-1]
    G = np.random.default_rng(5).normal(size=(3, 3 * OUTPUTS))
    stored.per_layer_gradients(_in(stored, X), gradient=_backend_given(stored, G), state=state)
    # the initial state's gradient is its snap's codes' losses' alone: none comes from the steps
    alone._first_state(3)
    alone.recurrence_layer.first_state_backward(alone.backend.zeros((3, 3)), 1.0 / 3)
    assert _equal(
        stored.recurrence_layer.factors[0].initial_state.grad_c, alone.recurrence_layer.factors[0].initial_state.grad_c
    )

    initial, restarted = _network(backend=backend), _network(backend=backend)
    initial.per_layer_gradients(_in(initial, X), gradient=_backend_given(initial, G))
    starts = np.zeros((3, 3), dtype=bool)
    starts[:, 0] = True
    resumed = restarted._resumed(_in(restarted, X), state, starts.tolist())
    restarted.per_layer_gradients(_in(restarted, X), gradient=_backend_given(restarted, G), state=state, starts=resumed)
    for a, b in zip(_trained(initial), _trained(restarted), strict=True):
        assert all(_equal(x, y) for x, y in zip(a.gradients(), b.gradients(), strict=True))


# the rollout's reading (test 4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_run_steps_one_step_at_a_time_is_one_call_by_bits(backend: str):
    network = _network(FACTORED, backend=backend)
    network.set_temperatures([0.5, None])
    network.set_couplings([0.7, 1.0])
    X = _steps(4)
    whole = network.run_steps(_in(network, X))
    state = None
    for t in range(4):
        step = network.run_steps(_columns(network, X, t, t + 1), state=state)
        assert np.array_equal(to_numpy(step.outputs), to_numpy(whole.outputs)[:, t * OUTPUTS : (t + 1) * OUTPUTS])
        assert _equal(step.states[0], whole.states[t])
        assert _equal(step.cells[0], whole.cells[t])
        state = step.states[0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_training_steps_outputs_are_run_steps_by_bits(backend: str):
    network = _network(FACTORED, rule=Adam(), backend=backend)
    network.set_temperatures([0.5, None])
    network.set_couplings([0.7, 1.0])
    X = _in(network, _steps(3))
    starts = [[False, True, False], [False, False, False], [False, False, True]]
    state = network.run_steps(_in(network, _steps(1, seed=9))).states[-1]
    expected = network.run_steps(X, state=state, starts=starts).outputs
    seen = network.learn_batch_output_gradient(0.01, X, _zeros(network), state=state, starts=starts)
    assert _equal(seen, expected)
    # the machine's reading differs while the codes are soft
    assert not _equal(network.run_steps(X, machine=True).outputs, network.run_steps(X).outputs)


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_machines_codes_are_state_codes(backend: str):
    network = _network(FACTORED, backend=backend)
    network.set_temperatures([0.5, 0.5])
    X = _in(network, _steps(3))
    machine = network.run_steps(X, machine=True)
    codes = network.state_codes(X)
    assert [list(step) for step in zip(*(example[1:] for example in codes), strict=True)] == machine.codes
    assert network.run_steps(X).codes == []


# learning (test 5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_given_loss_falls_over_a_few_steps(backend: str):
    network = _network(rule=Adam(), backend=backend)
    X = _in(network, _steps(3, rows=8))
    T = np.random.default_rng(6).normal(size=(8, 3 * OUTPUTS))

    def loss(Y: Any) -> float:
        return float(np.mean(np.sum((to_numpy(Y) - T) ** 2, axis=1)))

    def gradient(Y: Any) -> Any:
        return _in(network, 2.0 * (to_numpy(Y) - T))

    before = loss(network.run_steps(X).outputs)
    for _ in range(30):
        network.learn_batch_output_gradient(0.01, X, gradient)
    assert loss(network.run_steps(X).outputs) < before


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_two_step_memory_is_learned_with_every_factor_snapped(backend: str):
    # a cue at step 1 (one of two planes lit), a blank screen at step 2, whose answer is the cue:
    # only the snapped state carries it, through a given cross-entropy at step 2. Trained through the
    # coupling's handover from the cell's output to the code (the state formation workplan's D25;
    # hard codes from the start collapse onto one code on some seeds: 8 of 8 seeds learn it so)
    network = BACKENDS[backend].randomized(
        (3, 3, 2),
        [Conv(1, 4), StateInput(), Dense(8, "relu"), Dense(2, "linear", output=True, loss="given")],
        WindowRecurrence((Factor(4, Codes(4)),)),
        Adam(),
        seed=1,
    )
    rows = 16
    cues = np.arange(rows) % 2
    screens = np.zeros((rows, 2 * 18))
    for i, cue in enumerate(cues):
        screens[i, cue * 9 : (cue + 1) * 9] = 1.0  # channel-major: plane cue's 9 cells
    X = _in(network, screens)

    def gradient(Y: Any) -> Any:
        # the second step's softmax cross-entropy, each row's mean over its 2 steps
        logits = to_numpy(Y)[:, 2:]
        P = np.exp(logits - logits.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        G = np.zeros((rows, 4))
        G[:, 2:] = (P - np.eye(2)[cues]) / 2.0
        return _in(network, G)

    for epoch in range(400):
        network.set_couplings([min(epoch / 300, 1.0)])
        network.learn_batch_output_gradient(0.01, X, gradient)
    answers = to_numpy(network.run_steps(X, machine=True).outputs)[:, 2:].argmax(axis=1)
    assert np.array_equal(answers, cues)


# refusals (test 6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_given_loss_window_refuses_labels_and_loss_batch(backend: str):
    network = _network(backend=backend)
    X = _steps(2)
    with pytest.raises(ValueError, match="given loss"):
        network.learn_batch(0.01, [(tuple(X[0]), (0, 1))])
    with pytest.raises(ValueError, match="given loss"):
        network.loss_batch(_in(network, X), [(0, 1)] * 3)


@pytest.mark.parametrize(
    ("window", "message"),
    [
        ([Conv(1, 3), Attention(heads=1), StateInput(), Dense(6, "relu"), GIVEN], "acts on tokens"),
        (
            [Conv(1, 3), StateInput(), Dense(6, "relu"), Dense(3, "softmax", output=True, loss="cross_entropy")],
            "given loss",
        ),
        ([Conv(1, 3), Dense(6, "relu"), GIVEN], "at least one StateInput"),
        ([Conv(1, 3), Dense(6, "relu"), StateInput(), GIVEN, StateInput()], "before its output layer"),
        ([StateInput(), Conv(1, 3), Dense(6, "relu"), GIVEN], "Embedding"),
        (
            [Conv(1, 3, activation="linear"), BatchNorm(activation="relu"), StateInput(), Dense(6, "relu"), GIVEN],
            "BatchNorm",
        ),
    ],
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_an_array_window_refuses_what_it_cant_run(backend: str, window: list[LayerSpec], message: str):
    with pytest.raises(AssertionError, match=message):
        _network(window=window, backend=backend)


def test_the_python_implementation_refuses_an_array_window():
    with pytest.raises(AssertionError, match="no conv front end"):
        WindowRecurrentBackpropNetwork(SHAPE, WINDOW, RECURRENCE, SGD())


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_stored_state_or_starts_of_the_wrong_shape_is_refused(backend: str):
    network = _network(backend=backend)
    X = _in(network, _steps(2))
    with pytest.raises(AssertionError, match="a stored state"):
        network.run_steps(X, state=network.backend.zeros((3, 4)))
    with pytest.raises(AssertionError, match="a flag per row and step"):
        network.run_steps(X, starts=[[False, True, False]] * 3)
    with pytest.raises(AssertionError, match="a flag per row and step"):
        network.learn_batch_output_gradient(0.01, X, _zeros(network), starts=[[False, True]] * 2)


# saving (test 7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_array_window_saves_loads_and_resumes_training_by_bits(backend: str, tmp_path: Path):
    network = _network(FACTORED, rule=Adam(), backend=backend)
    X = _in(network, _steps(2))
    G = np.random.default_rng(8).normal(size=(3, 2 * OUTPUTS))
    network.learn_batch_output_gradient(0.01, X, _backend_given(network, G))
    path = str(tmp_path / "array-window.json")
    network.save(path)
    loaded = load_network(path)
    assert isinstance(loaded, BACKENDS[backend]) and loaded.array_window
    for a, b in zip(network.snapshot(), loaded.snapshot(), strict=True):
        assert all(_equal(x, y) for x, y in zip(a, b, strict=True))
    starts = [[False, True]] * 3
    network.learn_batch_output_gradient(0.01, X, _backend_given(network, G), starts=starts)
    loaded.learn_batch_output_gradient(0.01, X, _backend_given(loaded, G), starts=starts)
    for a, b in zip(network.snapshot(), loaded.snapshot(), strict=True):
        assert all(_equal(x, y) for x, y in zip(a, b, strict=True))
    assert _equal(network.run_steps(X).outputs, loaded.run_steps(X).outputs)


# numpy against Rust (test 8): the Rust per-layer path's every value within the products' rounding
# of numpy's, which central differences check above


@pytest.fixture
def crate_exp_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    # the window's parts' exp, and the cell's tanh's, the crate's in numpy too
    monkeypatch.setattr(recurrent_array_layer, "exp", exp_by_crate)
    monkeypatch.setattr(activations, "exp", exp_by_crate)


def _close(expected: Any, actual: Any, scale: float = 1.0) -> None:
    np.testing.assert_allclose(to_numpy(actual), to_numpy(expected), rtol=0.0, atol=1e-12 * max(scale, 1.0))


@pytest.mark.usefixtures("crate_exp_parts")
@pytest.mark.parametrize("recurrence", ["one factor", "two factors, soft", "continuous, crossed"])
def test_rusts_steps_and_gradients_are_numpys_through_resets_and_a_stored_state(recurrence: str):
    chosen = {"one factor": RECURRENCE, "two factors, soft": FACTORED, "continuous, crossed": CONTINUOUS}[recurrence]
    numpy, rust = _network(chosen), _network(chosen, backend="rust")
    for network in (numpy, rust):
        _moved(network)
        if recurrence != "one factor":
            network.set_temperatures([0.6, None])
            network.set_couplings([0.5, 1.0])
    X = _steps(3, rows=4)
    first = _steps(2, rows=4, seed=2)
    starts = [[False, True, False], [False, False, True], [True, False, True], [False, False, False]]
    G = np.random.default_rng(5).normal(size=(4, 3 * OUTPUTS))
    for machine in (False, True) if chosen.snapped else (False,):
        expected = numpy.run_steps(X, machine=machine)
        actual = rust.run_steps(_in(rust, X), machine=machine)
        _close(expected.outputs, actual.outputs)
        for a, b in zip([*expected.states, *expected.cells], [*actual.states, *actual.cells], strict=True):
            _close(a, b)
        assert expected.codes == actual.codes
    numpy_state = numpy.run_steps(first).states[-1]
    rust_state = rust.run_steps(_in(rust, first)).states[-1]
    _close(numpy_state, rust_state)
    numpy.per_layer_gradients(
        X, gradient=_backend_given(numpy, G), state=numpy_state, starts=numpy._resumed(X, numpy_state, starts)
    )
    X_rust = _in(rust, X)
    rust.per_layer_gradients(
        X_rust,
        gradient=_backend_given(rust, G),
        state=rust_state,
        starts=rust._resumed(X_rust, rust_state, starts),
    )
    for expected_layer, actual_layer in zip(_trained(numpy), _trained(rust), strict=True):
        for a, b in zip(expected_layer.gradients(), actual_layer.gradients(), strict=True):
            _close(a, b, float(np.abs(to_numpy(a)).max()))


def test_rusts_select_rows_is_numpys_by_bits():
    A: FloatArray = np.random.default_rng(1).normal(size=(5, 3))
    B: FloatArray = np.random.default_rng(2).normal(size=(5, 3))
    mask = [True, False, False, True, True]
    expected = NUMPY.select_rows(mask, A, B)
    assert np.array_equal(to_numpy(RUST.select_rows(mask, RUST.matrix(A.tolist()), RUST.matrix(B.tolist()))), expected)
    assert np.array_equal(expected, np.where(np.array(mask)[:, np.newaxis], A, B))
