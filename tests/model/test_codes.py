"""
The k-means codes (model/codes.py; the audit-remedies workplan, stage 3, D3).

The library's kmeans is the harness's by bits: tests/fixtures/codes/harness_kmeans.json holds the
harness's own centres and assignments (indrajala-studies' modular_family.kmeans, run by this
module's main from a studies checkout) on five cases: separated clusters; more codes than
clusters; fewer distinct points than codes and all points one (the empty clusters' perturbation,
and k-means++'s zero-distance draw); and a window network's own cell outputs. initialise_codes
sets each snapped factor's codes on every implementation from its own states, reading them in the
harness's order, and leaves a continuous factor's untouched, drawing nothing for it.

    .venv/bin/python -m tests.model.test_codes ../indrajala-studies   # re-records the fixture
"""

import ast
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from indrajala_ml.model.codes import factor_states, fit_codes, initialise_codes, kmeans
from indrajala_ml.model.networks.python.window_recurrent_backprop_network import WindowRecurrentBackpropNetwork
from indrajala_ml.model.networks.window_recurrent_array_network import (
    WindowRecurrentArrayNetwork,
    WindowRecurrentRustArrayNetwork,
)
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.model.specs.window_recurrence import WindowRecurrence
from tests.helpers import bits
from tests.model.networks.test_window_recurrent_array_network import RECURRENCES, WINDOW, K, _batch

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


def _network(implementation: str, recurrence: WindowRecurrence, seed: int = 7) -> Any:
    return NETWORKS[implementation].randomized((K,), WINDOW, recurrence, Adam(), seed=seed)


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


if __name__ == "__main__":
    _record(Path(sys.argv[1]))
