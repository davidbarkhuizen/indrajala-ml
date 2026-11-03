# pyright: reportPrivateUsage=false
"""
The FiLM state input (the audit-remedies workplan, stage 9, D9): StateInput(scale=True), x_t * (1 +
G c) + P c. G starts at zero and draws nothing, so a FiLM network from a seed has the additive
one's weights and, at G = 0, its outputs by bits, in all three implementations; with G away from
zero the gradient matches central differences (numpy); numpy, Rust (the window program) and pure
Python train alike; format 2 writes scale only when set, and a FiLM network saves, loads and trains
on by bits.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.networks.python.window_recurrent_backprop_network import WindowRecurrentBackpropNetwork
from indrajala_ml.model.networks.window_recurrent_array_network import (
    WindowRecurrentArrayNetwork,
    WindowRecurrentRustArrayNetwork,
)
from indrajala_ml.model.persistence.format2_json import layer_from_json, layer_to_json
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    Dense,
    Embedding,
    LayerNorm,
    LayerSpec,
    Position,
    Residual,
    StateInput,
)
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.model.specs.window_recurrence import Codes, Factor, WindowRecurrence
from tests.helpers import bits
from tests.model.networks.test_window_recurrent_array_network import (
    _batch,
    _gradients,
    _identity_codes,
    _numeric,
)

K = 2
NETWORKS: dict[str, Any] = {
    "numpy": WindowRecurrentArrayNetwork,
    "rust": WindowRecurrentRustArrayNetwork,
    "python": WindowRecurrentBackpropNetwork,
}
RECURRENCE = WindowRecurrence((Factor(3, Codes(4)),))


def _window(scale: bool) -> list[LayerSpec]:
    return [
        Embedding(27, 8),
        Position(),
        StateInput(scale=scale),
        Residual((LayerNorm(), Attention(heads=2, causal=True))),
        StateInput(scale=scale),
        Residual((LayerNorm(), Dense(16, activation="relu"), Dense(8, activation="linear", bias=True))),
        LayerNorm(),
        Dense(3, output=True, activation="softmax", loss="cross_entropy"),
    ]


def _network(implementation: str, scale: bool, seed: int = 7) -> Any:
    return NETWORKS[implementation].randomized((K,), _window(scale), RECURRENCE, Adam(), seed=seed)


def _rows(seed: int) -> list[tuple[tuple[float, ...], tuple[int, ...]]]:
    X, labels = _batch(3, 4, seed)
    return [(tuple(x), label) for x, label in zip(X.tolist(), labels, strict=True)]


def _film_layers(network: Any) -> list[Any]:
    layers = network.trainable_layers if isinstance(network, WindowRecurrentBackpropNetwork) else network.layers
    return [layer for layer in layers if hasattr(layer, "G") or hasattr(layer, "scales")]


def _G(layer: Any) -> list[list[float]]:
    return [list(row.weights) for row in layer.scales] if hasattr(layer, "scales") else layer.G.tolist()


def _inputs(network: Any, rows: list[tuple[float, ...]]) -> Any:
    return rows if isinstance(network, WindowRecurrentBackpropNetwork) else network.backend.matrix(rows)


@pytest.mark.parametrize("implementation", NETWORKS)
def test_at_g_zero_a_film_network_is_the_additive_one_by_bits(implementation: str):
    film, additive = _network(implementation, True), _network(implementation, False)
    layers = _film_layers(film)
    assert len(layers) == 2 and all(v == 0.0 for layer in layers for row in _G(layer) for v in row)
    # the same draws: every additive weight is the FiLM network's, G aside
    rows = [state for state, _ in _rows(13)]
    assert bits(film.forward_windows(_inputs(film, rows))) == bits(additive.forward_windows(_inputs(additive, rows)))


def test_the_gradient_with_g_away_from_zero_matches_central_differences():
    network = _network("numpy", True)
    rng = np.random.default_rng(3)
    for layer in _film_layers(network):
        layer.G = rng.normal(0.0, 0.5, size=layer.G.shape)
    for factor in network.recurrence_layer.factors:
        factor.pooling.q = rng.normal(size=factor.pooling.q.shape)
        factor.initial_state.c = rng.normal(size=factor.initial_state.c.shape)
    _identity_codes(network)
    X, labels = _batch(3)
    analytic = _gradients(network, X, labels)
    numeric = _numeric(network, X, labels)
    for a, n in zip(analytic, numeric, strict=True):
        np.testing.assert_allclose(a, n, rtol=1e-5, atol=1e-7)
    assert any(np.abs(layer.grad_G).max() > 0 for layer in _film_layers(network))


def test_numpy_rust_and_pure_python_train_alike():
    networks = [_network(i, True) for i in NETWORKS]
    for seed in (13, 14, 15):
        for network in networks:
            network.learn_batch(0.05, _rows(seed))
    G = [np.array([_G(layer) for layer in _film_layers(n)]) for n in networks]
    assert np.abs(G[0]).max() > 0.0
    np.testing.assert_allclose(G[1], G[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(G[2], G[0], rtol=0, atol=1e-12)


def test_format_2_writes_scale_only_when_set():
    assert layer_to_json(StateInput()) == {"kind": "state_input"}
    assert layer_to_json(StateInput(scale=True)) == {"kind": "state_input", "scale": True}
    assert layer_from_json({"kind": "state_input", "scale": True}) == StateInput(scale=True)
    assert layer_from_json({"kind": "state_input"}) == StateInput()


@pytest.mark.parametrize("implementation", NETWORKS)
def test_a_film_network_saves_loads_and_trains_on_by_bits(implementation: str, tmp_path: Path):
    network = _network(implementation, True)
    network.learn_batch(0.05, _rows(13))
    path = tmp_path / "film.json"
    network.save(str(path))
    loaded: Any = NETWORKS[implementation].load(str(path))
    assert bits(loaded.snapshot()) == bits(network.snapshot())
    for net in (network, loaded):
        net.learn_batch(0.05, _rows(14))
    assert bits(loaded.snapshot()) == bits(network.snapshot())
