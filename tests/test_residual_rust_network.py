"""
Residual blocks on Rust (the residual-connections workplan, stage 4): the layer before a fork takes
its hidden delta from the fused skip op (D8), whose bits are the unfused downstream, add and
derivative; the fork sums lazily; and training matches numpy within the dense layers' rounding. The
gradient check, identity blocks and learn against a batch of one run on both backends in
tests/test_residual_array_network.py.
"""

from typing import Any

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model.array_backend import NUMPY, RUST
from indrajala_ml.model.layer_specs import BatchNorm, Dense, LayerSpec
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from tests.test_residual_array_network import INPUT, NETWORKS, Shape, network, output, rows

RULES = [SGD(), Momentum(0.9), Adam(), WeightDecay(0.01)]


def _numpy(values: Any) -> Any:
    return np.array(values.tolist())


@pytest.mark.parametrize("name", ["sigmoid body", "two blocks in a row", "relu body"])
def test_the_layer_before_a_fork_takes_the_skip_ops_bits_and_the_fork_sums_lazily(name: str):
    built = network(name, rule=Momentum(0.9), backend=RUST)
    before, fork = built.layers[0], built.layers[1]
    body, add = fork.body_first, fork.add
    # the W the backward pass read: the optimizer's step rebinds body.W, never writes it
    body_w = body.W
    built.learn_batch(0.5, rows(5))

    downstream = _numpy(pa.layer_downstream_batch(body_w, body.delta_batch)) + _numpy(add.delta_batch)
    A = _numpy(before.A)
    expected = downstream * A * (1.0 - A) if name != "relu body" else np.where(A > 0.0, downstream, 0.0)
    assert _numpy(before.delta_batch).tobytes() == expected.tobytes()
    # the fused predecessor read body_first and add itself, so the fork never summed
    assert not hasattr(fork, "delta_batch") and not hasattr(fork, "delta")


def _max_relative_gap(layers: list[LayerSpec], shape: Shape, rule: UpdateRule) -> float:
    networks: list[Any] = []
    for backend in (NUMPY, RUST):
        built = SequentialArrayNetwork((INPUT,), [*layers, output(shape)], rule, shape=shape, backend=backend)
        built.rng = backend.default_rng(3)
        built.randomize()
        networks.append(built)
    data = rows(40, shape)
    for step in range(50):
        batch = data[(step * 5) % 40 :][:5]
        for built in networks:
            built.learn_batch(0.3, batch)

    gap = 0.0
    for expected_entry, actual_entry in zip(networks[0].snapshot(), networks[1].snapshot(), strict=True):
        for expected, actual in zip(expected_entry, actual_entry, strict=True):
            gap = max(gap, float((np.abs(_numpy(actual) - expected) / np.abs(expected)).max()))
    return gap


# no residual block, the same widths: the dense layers' own gap
CONTROLS: dict[str, list[LayerSpec]] = {
    "sigmoid": [Dense(INPUT), Dense(5)],
    "relu": [Dense(INPUT, activation="relu"), Dense(5, activation="relu")],
    "deep": [Dense(INPUT), Dense(5), Dense(INPUT), Dense(3, activation="relu"), Dense(INPUT)],
    "batch norm": [Dense(INPUT, activation="relu"), Dense(5, activation="linear"), BatchNorm("relu")],
}


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("shape", ["multiclass", "single_output"])
@pytest.mark.parametrize("name", NETWORKS)
def test_training_matches_numpy_within_the_dense_layers_rounding(name: str, shape: Shape, rule: UpdateRule):
    # the fork's and add's sums are one IEEE addition each, the same bits on both; the dense and
    # affine products aren't (numpy's BLAS against the crate's FMA chains), as without blocks. 50
    # steps in, the residual networks were at most 2.9e-11 apart relative when measured (Adam,
    # batch norm in the body), and the controls 9.5e-12 (momentum, batch norm) and 1.0e-12 (Adam,
    # ReLU): the same kind of gap, from the same products
    assert _max_relative_gap(NETWORKS[name], shape, rule) < 1e-10


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("shape", ["multiclass", "single_output"])
@pytest.mark.parametrize("name", CONTROLS)
def test_the_dense_layers_alone_have_the_same_kind_of_gap(name: str, shape: Shape, rule: UpdateRule):
    assert _max_relative_gap(CONTROLS[name], shape, rule) < 1e-10
