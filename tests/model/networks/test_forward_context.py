"""
The forward context (the segments workplan, stage 2, D3): every network builds one per batch and
passes it to each of its layers' forward, backward and gradient passes, in all three
implementations; nothing reads it yet.
"""

from collections.abc import Callable
from typing import Any

import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.layers.forward_context import NO_CONTEXT, ForwardContext
from indrajala_ml.model.networks.python.sequential_backprop_network import SequentialSequenceBackpropNetwork
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.layer_specs import Attention, Dense, Embedding, LayerNorm, LayerSpec, Residual
from indrajala_ml.model.specs.update_rules import Adam

TOKENS, VOCABULARY = 4, 5
SPECS: list[LayerSpec] = [
    Embedding(VOCABULARY, 4),
    Residual((LayerNorm(), Attention(heads=2, key_size=2, causal=True, positions="rotary"))),
    Residual((LayerNorm(), Dense(6, activation="relu"), Dense(4, activation="linear", bias=True))),
    LayerNorm(),
    Dense(VOCABULARY, output=True, activation="softmax", loss="cross_entropy"),
]
ARRAY_PASSES = [
    "forward",
    "forward_batch",
    "compute_output_delta",
    "compute_output_delta_batch",
    "compute_hidden_delta",
    "compute_hidden_delta_batch",
    "accumulate_gradient",
    "accumulate_gradient_batch",
]
PYTHON_PASSES = ["forward", "compute_hidden_deltas", "accumulate_gradients"]
BATCH = [((0.0, 1.0, 2.0, 3.0), (1, 2, 3, 4)), ((4.0, 3.0, 2.0, 1.0), (3, 2, 1, 0))]


def _network(implementation: str) -> Any:
    if implementation == "python":
        network: Any = SequentialSequenceBackpropNetwork((TOKENS,), SPECS, Adam())
    else:
        backend = NUMPY if implementation == "numpy" else RUST
        network = SequentialArrayNetwork((TOKENS,), SPECS, Adam(), "sequence", backend)
        network.rng = backend.default_rng(0)
    network.randomize()
    return network


def _spied(network: Any, implementation: str, sentinel: ForwardContext) -> list[tuple[str, str, object]]:
    # every pass of every layer records the context it was given, and the network builds sentinel
    calls: list[tuple[str, str, object]] = []
    layers = network.trainable_layers if implementation == "python" else network.layers
    passes = PYTHON_PASSES if implementation == "python" else ARRAY_PASSES

    def spy(layer: Any, name: str, inner: Callable[..., Any]) -> Callable[..., Any]:
        def recorded(*args: Any, context: ForwardContext = NO_CONTEXT, **kwargs: Any) -> Any:
            calls.append((type(layer).__name__, name, context))
            return inner(*args, context=context, **kwargs)

        return recorded

    for layer in layers:
        for name in passes:
            if hasattr(layer, name):
                setattr(layer, name, spy(layer, name, getattr(layer, name)))

    def built(*_args: object) -> ForwardContext:
        return sentinel

    network._forward_context = built
    return calls


@pytest.mark.parametrize("implementation", ["python", "numpy", "rust"])
def test_every_layer_pass_gets_the_networks_context(implementation: str):
    network = _network(implementation)
    sentinel = ForwardContext()
    calls = _spied(network, implementation, sentinel)

    network.learn_batch(0.01, BATCH)
    network.learn(0.01, *BATCH[0])
    network.predict_probabilities(BATCH[0][0])

    assert calls
    assert {name for _layer, name, _context in calls} >= (
        {"forward", "compute_hidden_deltas", "accumulate_gradients"}
        if implementation == "python"
        else {"forward", "forward_batch", "compute_hidden_delta_batch", "accumulate_gradient_batch"}
    )
    strays = [(layer, name) for layer, name, context in calls if context is not sentinel]
    assert not strays


def test_the_empty_context_is_one_frozen_value():
    assert (NO_CONTEXT.segments, NO_CONTEXT.positions, NO_CONTEXT.counted, NO_CONTEXT.count) == (None, None, None, 0)
    with pytest.raises(AttributeError):
        NO_CONTEXT.segments = None  # pyright: ignore[reportAttributeAccessIssue]
