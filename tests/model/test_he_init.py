"""
He initialisation (the audit-remedies workplan, stage 8, D8): a hidden Dense or Conv with
init="he" draws its W uniform within sqrt(6 / fan_in), variance 2 / fan_in, and a zero bias, by the
uniform draw all three generators share, so numpy, Rust and pure Python draw the same W from one
seed by bits; "fan_in", the default, keeps every bit; format 2 writes init only when "he"; the
output layer's is refused.
"""

import math
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.networks.python.sequential_backprop_network import (
    SequentialMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.persistence.format2_json import layer_from_json, layer_to_json
from indrajala_ml.model.specs.layer_specs import BatchNorm, Conv, Dense, LayerSpec, Pool, Residual
from indrajala_ml.model.specs.spec_validation import validate_layer_specs
from indrajala_ml.model.specs.update_rules import Adam
from indrajala_ml.pcg64 import default_rng
from tests.helpers import bits

INPUT = (8, 8, 1)
OUTPUT = Dense(3, output=True, activation="softmax", loss="cross_entropy")


def _specs(init: Any) -> list[LayerSpec]:
    # every kind a dense or conv spec builds: a ReLU conv, a linear conv before its batch norm, a
    # ReLU dense layer, a residual body's dense and affine layers, a linear dense layer
    return [
        Conv(3, 4, init=init),
        Conv(2, 3, activation="linear", init=init),
        BatchNorm("relu"),
        Pool(2),
        Dense(6, activation="relu", init=init),
        Residual((Dense(5, activation="relu", init=init), Dense(6, activation="linear", bias=True, init=init))),
        Dense(4, activation="linear", init=init),
        BatchNorm(),
        OUTPUT,
    ]


def _network(implementation: str, specs: list[LayerSpec], seed: int = 3) -> Any:
    if implementation == "python":
        network: Any = SequentialMultiClassBackpropClassifierNetwork(
            INPUT, specs, Adam(), [(0.0, 1.0)] * math.prod(INPUT)
        )
        network.rng = default_rng(seed)
    else:
        backend = NUMPY if implementation == "numpy" else RUST
        network = SequentialArrayNetwork(INPUT, specs, Adam(), "multiclass", backend)
        network.rng = backend.default_rng(seed)
    network.randomize()
    return network


def test_the_draw_is_within_sqrt_6_over_fan_in_with_variance_2_over_fan_in_and_a_zero_bias():
    network = SequentialArrayNetwork((400,), [Dense(300, activation="relu", init="he"), OUTPUT], Adam(), "multiclass")
    network.rng = NUMPY.default_rng(5)
    network.randomize()
    W, b = (np.array(a.tolist()) for a in network.snapshot()[0])
    limit = math.sqrt(6.0 / 400)
    assert np.abs(W).max() <= limit
    assert np.abs(W).max() > 0.99 * limit
    assert abs(W.var() - 2.0 / 400) < 0.02 * (2.0 / 400)
    assert np.all(b == 0.0)


def test_numpy_rust_and_pure_python_draw_the_same_he_weights_by_bits():
    numpy, rust = (_network(i, _specs("he")) for i in ("numpy", "rust"))
    assert bits(rust.snapshot()) == bits(numpy.snapshot())
    # pure Python's per weight set: each dense or conv layer's W rows are numpy's W rows
    python = _network("python", _specs("he"))
    compared = 0
    for layer, entry in zip(python.trainable_layers, numpy.snapshot(), strict=True):
        if not entry or not hasattr(layer, "weight_sets") or len(entry) not in (1, 2):
            continue
        sets = layer.weight_sets()
        if not sets or len(sets) != len(entry[0].tolist()):
            continue
        assert bits([list(s.weights) for s in sets]) == bits(entry[0].tolist())
        compared += 1
    assert compared == 7  # the six dense and conv layers drawn by He, and the output layer


@pytest.mark.parametrize("implementation", ["numpy", "rust", "python"])
def test_fan_in_is_the_default_and_keeps_every_bit(implementation: str):
    explicit = _network(implementation, _specs("fan_in"))
    default = _network(
        implementation,
        [
            Conv(3, 4),
            Conv(2, 3, activation="linear"),
            BatchNorm("relu"),
            Pool(2),
            Dense(6, activation="relu"),
            Residual((Dense(5, activation="relu"), Dense(6, activation="linear", bias=True))),
            Dense(4, activation="linear"),
            BatchNorm(),
            OUTPUT,
        ],
    )
    assert bits(explicit.snapshot()) == bits(default.snapshot())


@pytest.mark.parametrize("implementation", ["numpy", "rust", "python"])
def test_he_draws_differ_from_fan_in_and_the_biases_are_zero(implementation: str):
    he, fan_in = (_network(implementation, _specs(init)) for init in ("he", "fan_in"))
    assert bits(he.snapshot()) != bits(fan_in.snapshot())
    if implementation != "python":
        first = he.snapshot()[0]  # the ReLU conv: (W, b)
        assert all(v == 0.0 for v in first[1].tolist())


def test_format_2_writes_init_only_when_he():
    assert "init" not in layer_to_json(Dense(5, activation="relu"))
    assert "init" not in layer_to_json(Conv(3, 4))
    for spec in (Dense(5, activation="relu", init="he"), Conv(3, 4, init="he")):
        entry = layer_to_json(spec)
        assert entry["init"] == "he"
        assert layer_from_json(entry) == spec


def test_an_output_layer_with_he_is_refused():
    with pytest.raises(AssertionError, match="hidden layer's"):
        validate_layer_specs([Dense(5, activation="relu"), Dense(3, output=True, loss="squared", init="he")])
