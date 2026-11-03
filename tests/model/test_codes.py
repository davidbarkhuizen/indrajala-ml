"""
The k-means codes (model/codes.py; the audit-remedies workplan, stage 3, D3).

The library's kmeans is the harness's by bits: tests/fixtures/codes/harness_kmeans.json holds the
harness's own centres and assignments (indrajala-studies' modular_family.kmeans, run by this
module's main from a studies checkout) on five cases: separated clusters; more codes than
clusters; fewer distinct points than codes and all points one (the empty clusters' perturbation,
and k-means++'s zero-distance draw); and a window network's own cell outputs. initialise_codes
sets each snapped factor's codes on every implementation from its own states, reading them in the
harness's order, and leaves a continuous factor's untouched, drawing nothing for it.

Re-seeding (stage 4, D4): a code no sample state snaps to moves onto a sample state, drawn by the
D^2 rule against the live codes, and its optimizer rows are zeroed, on every implementation and
under momentum and Adam; a live code, its optimizer rows and every other parameter keep their bits.

The codebook's learning-rate scale (stage 5, D5): a factor's codes at scale k step as at the rate
k * lr, by bits, under SGD, momentum and Adam, and every other parameter as at lr; a scale of 1.0
keeps every bit.

    .venv/bin/python -m tests.model.test_codes ../indrajala-studies   # re-records the fixture
"""

import ast
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from indrajala_ml.model.codes import (
    d2_draw,
    factor_states,
    fit_codes,
    initialise_codes,
    kmeans,
    reseed_dead_codes,
    snap_indices,
)
from indrajala_ml.model.networks.python.window_recurrent_backprop_network import WindowRecurrentBackpropNetwork
from indrajala_ml.model.networks.window_recurrent_array_network import (
    WindowRecurrentArrayNetwork,
    WindowRecurrentRustArrayNetwork,
)
from indrajala_ml.model.persistence.format2 import network_to_json
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule
from indrajala_ml.model.specs.window_recurrence import WindowRecurrence
from tests.helpers import bits
from tests.model.networks.test_window_recurrent_array_network import RECURRENCES, WINDOW, K, _batch

STEP = 1e-6
FIXTURE = Path(__file__).parents[1] / "fixtures" / "codes" / "harness_kmeans.json"
NETWORKS: dict[str, Any] = {
    "numpy": WindowRecurrentArrayNetwork,
    "rust": WindowRecurrentRustArrayNetwork,
    "python": WindowRecurrentBackpropNetwork,
}
ITERATIONS = 10


def _hex(a: Any) -> Any:
    return [_hex(x) for x in cast("list[Any]", a)] if isinstance(a, list) else float(a).hex()


def _floats(a: Any) -> Any:
    return [_floats(x) for x in cast("list[Any]", a)] if isinstance(a, list) else float.fromhex(a)


def _network(implementation: str, recurrence: WindowRecurrence, seed: int = 7, rule: UpdateRule | None = None) -> Any:
    return NETWORKS[implementation].randomized((K,), WINDOW, recurrence, rule or Adam(), seed=seed)


def _batches() -> list[list[list[float]]]:
    # two batches of three windows' rows, as the harness reads them
    return [_batch(3, 4, seed)[0].tolist() for seed in (11, 12)]


def _cases() -> dict[str, tuple[np.ndarray, int, int]]:
    """Each case's points, codes and seed (the kmeans generator's)."""
    rng = np.random.default_rng(5)
    centres = rng.normal(0.0, 3.0, size=(4, 3))
    clusters = np.concatenate([c + rng.normal(0.0, 0.3, size=(15, 3)) for c in centres])
    spread = rng.normal(0.0, 1.0, size=(40, 3))
    few = np.repeat(rng.normal(0.0, 1.0, size=(3, 2)), 4, axis=0)
    one = np.repeat(rng.normal(0.0, 1.0, size=(1, 3)), 6, axis=0)
    network = _network("numpy", RECURRENCES["one factor"])
    (states,) = factor_states(network, _batches())
    return {
        "separated clusters": (clusters, 4, 1),
        "more codes than clusters": (spread, 8, 2),
        "fewer distinct points than codes": (few, 5, 3),
        "one point": (one, 3, 4),
        "a window network's states": (states, 4, 5),
    }


def _record(studies: Path) -> None:
    # the harness's kmeans, its source alone (the harness's module imports the studies' whole stack)
    source = (studies / "indrajala_studies" / "modular_family.py").read_text()
    function = next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "kmeans"
    )
    namespace: dict[str, Any] = {"np": np}
    # the harness's own source, from the studies checkout the recorder names: the fixture is its bits
    exec(compile(ast.Module([function], []), "modular_family.kmeans", "exec"), namespace)  # noqa: S102
    cases: dict[str, Any] = {}
    for name, (points, k, seed) in _cases().items():
        C, nearest = namespace["kmeans"](points, k, np.random.default_rng(seed), ITERATIONS)
        cases[name] = {
            "points": _hex(points.tolist()),
            "k": k,
            "seed": seed,
            "centres": _hex(C.tolist()),
            "nearest": nearest.tolist(),
        }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps({"iterations": ITERATIONS, "cases": cases}, indent=1) + "\n")


FIXTURE_CASES: dict[str, Any] = json.loads(FIXTURE.read_text())["cases"] if FIXTURE.exists() else {}


@pytest.mark.parametrize("name", FIXTURE_CASES)
def test_kmeans_is_the_harnesss_by_bits(name: str):
    case = FIXTURE_CASES[name]
    points = np.array(_floats(case["points"]))
    C, nearest = kmeans(points, case["k"], np.random.default_rng(case["seed"]), ITERATIONS)
    assert _hex(C.tolist()) == case["centres"]
    assert nearest.tolist() == case["nearest"]


def test_the_fixtures_inputs_are_this_modules_cases():
    # the drawn points by bits; the window network's states, which the library reads as the fixture
    # was recorded, within the forward pass's rounding, which a machine's BLAS moves
    for name, (points, k, seed) in _cases().items():
        recorded = np.array(_floats(FIXTURE_CASES[name]["points"]))
        if name == "a window network's states":
            np.testing.assert_allclose(points, recorded, rtol=0, atol=1e-12)
        else:
            assert FIXTURE_CASES[name]["points"] == _hex(points.tolist()), name
        assert (FIXTURE_CASES[name]["k"], FIXTURE_CASES[name]["seed"]) == (k, seed)


@pytest.mark.parametrize("name", ["fewer distinct points than codes", "one point"])
def test_an_empty_cluster_is_a_drawn_point_moved_so_no_two_codes_are_equal(name: str):
    points, k, seed = _cases()[name]
    C, _ = kmeans(points, k, np.random.default_rng(seed), ITERATIONS)
    assert len({tuple(c) for c in C.tolist()}) == k
    nearest_point = np.min(np.sum((C[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=2), axis=1)
    assert np.all(nearest_point < 1e-4)


@pytest.mark.parametrize("implementation", NETWORKS)
@pytest.mark.parametrize("recurrence", RECURRENCES.values(), ids=RECURRENCES.keys())
def test_each_snapped_factors_codes_are_its_states_kmeans_centres(implementation: str, recurrence: WindowRecurrence):
    network = _network(implementation, recurrence)
    batches = _batches()
    states = factor_states(network, batches)
    before = network.snapshot()
    fits = initialise_codes(network, batches, np.random.default_rng(3), ITERATIONS)
    expected = np.random.default_rng(3)
    codes = _codes(network)
    for g, factor in enumerate(recurrence.factors):
        if factor.codes is None:
            assert fits[g] is None and codes[g] is None
            continue
        C, nearest = kmeans(states[g], factor.codes.count, expected, ITERATIONS)
        assert codes[g] == C.tolist()
        fit = fits[g]
        assert fit is not None and fit.in_use == len(np.unique(nearest)) and fit.within_lambda_max >= 0.0
    # nothing but the codes moved
    after = network.snapshot()
    unchanged = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if bits(a) == bits(b)]
    assert len(unchanged) == len(before) - sum(f.codes is not None for f in recurrence.factors)


def test_a_continuous_factor_draws_nothing():
    # the crossed recurrence's continuous factor, between two snapped ones: the generator's state
    # after initialise_codes is fit_codes' over the snapped factors' states alone
    recurrence = RECURRENCES["three factors, crossed"]
    network = _network("numpy", recurrence)
    states = factor_states(network, _batches())
    rng = np.random.default_rng(3)
    fit_codes(network, states, rng, ITERATIONS)
    expected = np.random.default_rng(3)
    for g, factor in enumerate(recurrence.factors):
        if factor.codes is not None:
            kmeans(states[g], factor.codes.count, expected, ITERATIONS)
    assert rng.bit_generator.state == expected.bit_generator.state


def test_the_states_are_read_batch_by_batch_then_window_by_window():
    network = _network("numpy", RECURRENCES["one factor"])
    batches = _batches()
    (states,) = factor_states(network, batches)
    expected = [row for batch in batches for cell in network.cell_outputs(batch) for row in cell]
    assert states.tolist() == expected
    assert len(states) == 2 * 3 * 4  # batches, windows, examples


def test_numpy_and_rust_set_the_same_codes_within_the_states_rounding():
    recurrence = RECURRENCES["two factors"]
    numpy_network, rust_network = (_network(i, recurrence) for i in ("numpy", "rust"))
    for network in (numpy_network, rust_network):
        initialise_codes(network, _batches(), np.random.default_rng(3), ITERATIONS)
    for a, b in zip(_codes(numpy_network), _codes(rust_network), strict=True):
        assert a is not None and b is not None
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)


def test_set_codes_refuses_a_continuous_factor_and_a_wrong_shape():
    network = _network("numpy", RECURRENCES["three factors, crossed"])
    with pytest.raises(AssertionError, match="continuous"):
        network.set_codes(1, [[0.0]])
    with pytest.raises(AssertionError, match="codes are"):
        network.set_codes(0, [[0.0, 0.0]])


def _codes(network: Any) -> list[list[list[float]] | None]:
    # each factor's codes as lists, on every implementation
    codes: list[list[list[float]] | None] = []
    for factor in network.recurrence_layer.factors:
        if factor.codes is None:
            codes.append(None)
        elif isinstance(network, WindowRecurrentBackpropNetwork):
            codes.append([list(code.weights) for code in factor.codes.codes])
        else:
            codes.append(factor.codes.E.tolist())
    return codes


FAR = 100.0  # a code this far from every state is dead


def _stepped(implementation: str, rule: UpdateRule) -> Any:
    # the two-factor network, one batch stepped so its optimizer has state, then factor 0's code 1
    # moved far from every state, so it is dead
    network = _network(implementation, RECURRENCES["two factors"], rule=rule)
    X, labels = _batch(3, 4, 13)
    network.learn_batch(0.01, [(tuple(x), label) for x, label in zip(X.tolist(), labels, strict=True)])
    codes = network.factor_codes(0)
    codes[1] = [FAR] * len(codes[1])
    network.set_codes(0, codes)
    return network


def _codes_state(network: Any, factor: int) -> list[Any]:
    # the optimizer's state for factor's codes part, as lists: per array, or per weight set
    if isinstance(network, WindowRecurrentBackpropNetwork):
        index = len(network.trainable_layers) + network.parts.index(network.recurrence_layer.factors[factor].codes)
        return [list(weights) for weights, _bias in network.optimizer.state().layers[index]]
    index = len(network.layers) + network.parts.index(network.recurrence_layer.factors[factor].codes)
    return [array.tolist() for array in network.optimizer.state().layers[index]]


@pytest.mark.parametrize("rule", [Adam(), Momentum(0.9)], ids=["adam", "momentum"])
@pytest.mark.parametrize("implementation", NETWORKS)
def test_a_dead_code_moves_onto_a_sample_state_with_its_optimizer_rows_zeroed(implementation: str, rule: UpdateRule):
    network = _stepped(implementation, rule)
    batches = _batches()
    before_codes = [network.factor_codes(g) for g in range(2)]
    before_state = [_codes_state(network, g) for g in range(2)]
    before = network.snapshot()
    states = factor_states(network, batches, first=True)
    uses = [np.bincount(snap_indices(h, np.array(c)), minlength=len(c)) for h, c in zip(states, before_codes)]
    expected_dead = [[i for i, u in enumerate(use) if u == 0] for use in uses]
    assert 1 in expected_dead[0]
    # the dead code's optimizer rows aren't zero before: the step moved it (so the reset is seen)
    assert np.abs(np.array(before_state[0][1] if implementation == "python" else before_state[0][0][1])).max() > 0

    moved = reseed_dead_codes(network, batches, np.random.default_rng(3))

    assert moved == expected_dead
    after_codes = [network.factor_codes(g) for g in range(2)]
    for g in range(2):
        for i, code in enumerate(after_codes[g]):
            if i in moved[g]:
                assert code in states[g].tolist()  # onto a sample state
            else:
                assert code == before_codes[g][i]  # a live code by bits
        _assert_rows(network, g, moved[g], before_state[g])
    # every other parameter by bits
    after = network.snapshot()
    codes_entries = {i for i, (a, b) in enumerate(zip(bits(before), bits(after), strict=True)) if a != b}
    assert len(codes_entries) == sum(bool(m) for m in moved)


def _assert_rows(network: Any, factor: int, moved: list[int], before: list[Any]) -> None:
    after = _codes_state(network, factor)
    if isinstance(network, WindowRecurrentBackpropNetwork):
        # per weight set (a code): its state lists
        for row, (old, new) in enumerate(zip(before, after, strict=True)):
            assert (all(v == 0.0 for values in new for v in values)) if row in moved else bits(new) == bits(old)
        return
    for old, new in zip(before, after, strict=True):  # each state array, (count, size)
        for row, (old_row, new_row) in enumerate(zip(old, new, strict=True)):
            assert (all(v == 0.0 for v in new_row)) if row in moved else bits(new_row) == bits(old_row)


def test_dead_codes_are_drawn_in_order_by_the_d2_rule_against_the_live_ones():
    network = _stepped("numpy", Adam())
    batches = _batches()
    states = factor_states(network, batches, first=True)
    codes = [np.array(network.factor_codes(g)) for g in range(2)]
    expected = np.random.default_rng(3)
    moved = reseed_dead_codes(network, batches, np.random.default_rng(3))
    for g in range(2):
        anchors = [codes[g][i] for i in range(len(codes[g])) if i not in moved[g]]
        for i in moved[g]:
            point = states[g][d2_draw(states[g], anchors, expected)]
            assert network.factor_codes(g)[i] == point.tolist()
            anchors.append(point)


def test_min_use_counts_a_rarely_used_code_dead():
    network = _network("numpy", RECURRENCES["one factor"])
    batches = _batches()
    (h,) = factor_states(network, batches, first=True)
    uses = np.bincount(snap_indices(h, np.array(network.factor_codes(0))), minlength=4)
    threshold = int(uses[uses > 0].min())
    moved = reseed_dead_codes(network, batches, np.random.default_rng(3), min_use=threshold)
    assert moved == [[i for i in range(4) if uses[i] <= threshold]]


def test_a_network_with_every_code_live_is_untouched():
    network = _network("numpy", RECURRENCES["one factor"])
    batches = _batches()
    initialise_codes(network, batches, np.random.default_rng(3), ITERATIONS)
    (h,) = factor_states(network, batches, first=True)
    uses = np.bincount(snap_indices(h, np.array(network.factor_codes(0))), minlength=4)
    before = bits(network.snapshot())
    rng = np.random.default_rng(3)
    moved = reseed_dead_codes(network, batches, rng)
    assert moved == [[i for i in range(4) if uses[i] == 0]]
    if not moved[0]:
        assert bits(network.snapshot()) == before
        assert rng.bit_generator.state == np.random.default_rng(3).bit_generator.state


def test_numpy_and_rust_move_the_same_codes():
    numpy_network, rust_network = (_stepped(i, Adam()) for i in ("numpy", "rust"))
    moved = [reseed_dead_codes(n, _batches(), np.random.default_rng(3)) for n in (numpy_network, rust_network)]
    assert moved[0] == moved[1]
    for g in range(2):
        np.testing.assert_allclose(numpy_network.factor_codes(g), rust_network.factor_codes(g), rtol=0, atol=1e-9)


def test_the_snap_indices_are_the_networks_own_snaps():
    network = _network("numpy", RECURRENCES["one factor"])
    X = _batches()[0]
    (h,) = factor_states(network, [X])
    indices = [code[0] for example in network.state_codes(network.backend.matrix(X)) for code in example[1:]]
    # state_codes is example-major, factor_states window-major
    windows = len(X[0]) // K
    by_window = [indices[e * windows + w] for w in range(windows) for e in range(len(X))]
    assert snap_indices(h, np.array(network.factor_codes(0))).tolist() == by_window


# SimVQ's codes, E = Q W (stage 6, D6)

LINEAR = {
    name: WindowRecurrence(
        tuple(
            dataclasses.replace(
                f, codes=None if f.codes is None else dataclasses.replace(f.codes, parameterisation="linear")
            )
            for f in recurrence.factors
        ),
        cross=recurrence.cross,
        cell=recurrence.cell,
        pooling=recurrence.pooling,
    )
    for name, recurrence in RECURRENCES.items()
}


def _rows_batch(seed: int = 13) -> list[tuple[tuple[float, ...], tuple[int, ...]]]:
    X, labels = _batch(3, 4, seed)
    return [(tuple(x), label) for x, label in zip(X.tolist(), labels, strict=True)]


@pytest.mark.parametrize("implementation", NETWORKS)
@pytest.mark.parametrize("recurrence", RECURRENCES)
def test_at_the_identity_a_linear_network_is_the_direct_ones_by_bits(implementation: str, recurrence: str):
    direct = _network(implementation, RECURRENCES[recurrence])
    linear = _network(implementation, LINEAR[recurrence])
    assert _codes(linear) == _codes(direct)
    rows = [state for state, _ in _rows_batch()]
    assert bits(linear.forward_windows(_inputs(linear, rows))) == bits(direct.forward_windows(_inputs(direct, rows)))


def _inputs(network: Any, rows: list[tuple[float, ...]]) -> Any:
    return rows if isinstance(network, WindowRecurrentBackpropNetwork) else network.backend.matrix(rows)


@pytest.mark.parametrize("implementation", NETWORKS)
def test_one_step_moves_every_linear_code_and_only_the_selected_direct_ones(implementation: str):
    moved: dict[str, int] = {}
    for kind, recurrence in (("direct", RECURRENCES["one factor"]), ("linear", LINEAR["one factor"])):
        network = _network(implementation, recurrence)
        before = network.factor_codes(0)
        network.learn_batch(0.01, _rows_batch()[:1])  # one example: few codes selected
        after = network.factor_codes(0)
        moved[kind] = sum(a != b for a, b in zip(before, after, strict=True))
    assert moved["linear"] == 4
    assert moved["direct"] < 4


def test_ws_gradient_is_qt_de_and_matches_central_differences():
    from indrajala_ml.model.layers.numpy.recurrent_array_layer import LinearCodesArrayLayer

    rng = np.random.default_rng(4)
    # no commitment: its term, ||h - sg[e]||^2, gives E no gradient, so the loss is the codebook's alone
    part = LinearCodesArrayLayer(5, 3, commitment=0.0)
    part.Q = rng.normal(size=(5, 3))
    part.W = np.eye(3) + 0.1 * rng.normal(size=(3, 3))
    part.after_step()
    H = rng.normal(size=(7, 3))
    weight = 0.5

    def loss() -> float:
        part.forward_batch(H)
        return part.loss_batch(weight)

    loss()
    part.reset_gradient_accum()
    part.backward_batch(np.zeros((7, 3)), weight)
    part.before_step()
    np.testing.assert_array_equal(part.grad_W, part.Q.T @ part.grad_E)
    numeric = np.zeros((3, 3))
    for index in np.ndindex(3, 3):
        kept = part.W[index]
        part.W[index] = kept + STEP
        part.after_step()
        up = loss()
        part.W[index] = kept - STEP
        part.after_step()
        down = loss()
        part.W[index] = kept
        part.after_step()
        numeric[index] = (up - down) / (2 * STEP)
    np.testing.assert_allclose(part.grad_W, numeric, rtol=1e-6, atol=1e-9)


def test_numpy_and_rust_train_linear_codes_alike():
    networks = [_network(i, LINEAR["two factors"]) for i in ("numpy", "rust")]
    for seed in (13, 14, 15):
        for network in networks:
            network.learn_batch(0.01, _rows_batch(seed))
    for g in range(2):
        np.testing.assert_allclose(networks[0].factor_codes(g), networks[1].factor_codes(g), rtol=0, atol=1e-9)


@pytest.mark.parametrize("implementation", NETWORKS)
def test_a_linear_network_saves_and_loads_and_trains_on_by_bits(implementation: str, tmp_path: Path):
    network = _network(implementation, LINEAR["two factors"])
    network.learn_batch(0.01, _rows_batch(13))
    path = tmp_path / "linear.json"
    network.save(str(path))
    saved = json.loads(path.read_text())
    assert [f["codes"]["parameterisation"] for f in saved["recurrence"]["factors"]] == ["linear", "linear"]
    loaded: Any = NETWORKS[implementation].load(str(path))
    assert bits(loaded.snapshot()) == bits(network.snapshot())
    for net in (network, loaded):
        net.learn_batch(0.01, _rows_batch(14))
    assert bits(loaded.snapshot()) == bits(network.snapshot())


def test_a_direct_codes_entry_is_as_before_it_had_a_parameterisation():
    network = _network("numpy", RECURRENCES["one factor"])
    assert network_to_json(network)["recurrence"]["factors"][0]["codes"] == {"count": 4, "commitment": 0.25}


@pytest.mark.parametrize("implementation", NETWORKS)
def test_k_means_codes_set_a_linear_factors_q_and_w_to_the_identity(implementation: str):
    network = _network(implementation, LINEAR["one factor"])
    network.learn_batch(0.01, _rows_batch(13))  # W away from the identity
    batches = _batches()
    (states,) = factor_states(network, batches)
    initialise_codes(network, batches, np.random.default_rng(3), ITERATIONS)
    C, _ = kmeans(states, 4, np.random.default_rng(3), ITERATIONS)
    np.testing.assert_allclose(network.factor_codes(0), C, rtol=0, atol=1e-12)
    part = network.recurrence_layer.factors[0].codes
    W = [list(row.weights) for row in part.basis] if implementation == "python" else part.W.tolist()
    assert W == np.eye(3).tolist()


@pytest.mark.parametrize("implementation", NETWORKS)
def test_a_reseeded_linear_code_lands_on_its_state_and_w_is_kept(implementation: str):
    network = _network(implementation, LINEAR["two factors"])
    network.learn_batch(0.01, _rows_batch(13))
    codes = network.factor_codes(0)
    codes[1] = [FAR] * len(codes[1])
    part = network.recurrence_layer.factors[0].codes
    # a far code through Q's row, W kept
    network.reseed_codes(0, {1: codes[1]})
    W_before = [list(row.weights) for row in part.basis] if implementation == "python" else part.W.tolist()
    batches = _batches()
    states = factor_states(network, batches, first=True)
    moved = reseed_dead_codes(network, batches, np.random.default_rng(3))
    assert 1 in moved[0]
    # each moved code on a sample state, within the solve's rounding
    for i in moved[0]:
        gaps = np.sum((states[0] - np.array(network.factor_codes(0)[i])) ** 2, axis=1)
        assert gaps.min() < 1e-20
    W_after = [list(row.weights) for row in part.basis] if implementation == "python" else part.W.tolist()
    assert W_after == W_before


# the codebook's learning-rate scale (stage 5, D5)


def _one_step(implementation: str, rule: UpdateRule, rate: float, scales: list[float] | None) -> Any:
    network = _network(implementation, RECURRENCES["three factors, crossed"], rule=rule)
    if scales is not None:
        network.set_code_learning_rate_scale(scales)
    X, labels = _batch(3, 4, 13)
    network.learn_batch(rate, [(tuple(x), label) for x, label in zip(X.tolist(), labels, strict=True)])
    return network


def _code_entries(network: Any) -> set[int]:
    # the snapshot's entries that are codes parts
    offset = len(network.trainable_layers if isinstance(network, WindowRecurrentBackpropNetwork) else network.layers)
    return {offset + network.parts.index(f.codes) for f in network.recurrence_layer.factors if f.codes is not None}


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), Adam()], ids=["sgd", "momentum", "adam"])
@pytest.mark.parametrize("implementation", NETWORKS)
def test_a_scaled_codebook_steps_as_at_the_scaled_rate_and_the_rest_at_the_rate(implementation: str, rule: UpdateRule):
    scaled = bits(_one_step(implementation, rule, 0.01, [4.0, 1.0, 4.0]).snapshot())
    plain = bits(_one_step(implementation, rule, 0.01, None).snapshot())
    fast = bits(_one_step(implementation, rule, 0.01 * 4.0, None).snapshot())
    codes = _code_entries(_network(implementation, RECURRENCES["three factors, crossed"]))
    for i, entry in enumerate(scaled):
        assert entry == (fast[i] if i in codes else plain[i]), i
    assert any(scaled[i] != plain[i] for i in codes)


@pytest.mark.parametrize("implementation", NETWORKS)
def test_a_scale_of_one_keeps_every_bit(implementation: str):
    ones = _one_step(implementation, Adam(), 0.01, [1.0, 1.0, 1.0])
    assert ones.optimizer.learning_rate_scales == {}
    assert bits(ones.snapshot()) == bits(_one_step(implementation, Adam(), 0.01, None).snapshot())


def test_a_scale_is_refused_on_a_continuous_factor_or_unless_positive_one_per_factor():
    network = _network("numpy", RECURRENCES["three factors, crossed"])
    with pytest.raises(AssertionError, match="continuous"):
        network.set_code_learning_rate_scale([1.0, 2.0, 1.0])
    with pytest.raises(AssertionError, match="positive"):
        network.set_code_learning_rate_scale([0.0, 1.0, 1.0])
    with pytest.raises(AssertionError, match="a scale per factor"):
        network.set_code_learning_rate_scale([2.0])
    network.set_code_learning_rate_scale([2.0, 1.0, 1.0])
    network.set_code_learning_rate_scale([1.0, 1.0, 1.0])  # back to the default
    assert network.optimizer.learning_rate_scales == {}


if __name__ == "__main__":
    _record(Path(sys.argv[1]))
