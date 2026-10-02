"""
Residual blocks on Rust (the residual-connections workplan, stage 4): the layer before a fork takes
its hidden delta from the fused skip op (D8), whose bits are the unfused downstream, add and
derivative; the fork sums lazily; and training matches numpy within the dense layers' rounding.
Identity blocks run on both array backends in tests/test_residual_array_network.py; the gradient
check, the block's layers and learn against a batch of one are tests/test_residual_network.py's.
"""

from typing import Any

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.layer_specs import BatchNorm, Dense, LayerSpec
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from tests.helpers import max_relative_gap, to_numpy
from tests.test_residual_array_network import INPUT, NETWORKS, Shape, network, output, rows

RULES = [SGD(), Momentum(0.9), Adam(), WeightDecay(0.01)]


@pytest.mark.parametrize("name", ["sigmoid body", "two blocks in a row", "relu body"])
def test_the_layer_before_a_fork_takes_the_skip_ops_bits_and_the_fork_sums_lazily(name: str):
    built = network(name, rule=Momentum(0.9), backend=RUST)
    before, fork = built.layers[0], built.layers[1]
    body, add = fork.body_first, fork.add
    # the W the backward pass read: the optimizer's step rebinds body.W, never writes it
    body_w = body.W
    built.learn_batch(0.5, rows(5))

    downstream = to_numpy(pa.layer_downstream_batch(body_w, body.delta_batch)) + to_numpy(add.delta_batch)
    A = to_numpy(before.A)
    expected = downstream * A * (1.0 - A) if name != "relu body" else np.where(A > 0.0, downstream, 0.0)
    assert to_numpy(before.delta_batch).tobytes() == expected.tobytes()
    # the fused predecessor read body_first and add itself, so the fork never summed
    assert not hasattr(fork, "delta_batch") and not hasattr(fork, "delta")


def _trained(layers: list[LayerSpec], shape: Shape, rule: UpdateRule, steps: int) -> tuple[Any, Any, list[Any]]:
    networks: list[Any] = []
    for backend in (NUMPY, RUST):
        built = SequentialArrayNetwork((INPUT,), [*layers, output(shape)], rule, shape=shape, backend=backend)
        built.rng = backend.default_rng(3)
        built.randomize()
        networks.append(built)
    before = networks[0].snapshot()
    data = rows(40, shape)
    for step in range(steps):
        batch = data[(step * 5) % 40 :][:5]
        for built in networks:
            built.learn_batch(0.3, batch)
    return networks[0], networks[1], before


def _gap_after_training(layers: list[LayerSpec], shape: Shape, rule: UpdateRule) -> float:
    expected_network, actual_network, _ = _trained(layers, shape, rule, steps=50)
    return max_relative_gap(expected_network.snapshot(), actual_network.snapshot())


def _max_ulps_after_one_step(layers: list[LayerSpec], shape: Shape, rule: UpdateRule) -> float:
    # in ulps of the step's operands, the weight before and the step: a weight the step takes
    # near zero has tiny ulps of its own, which would say nothing about the step
    expected_network, actual_network, before = _trained(layers, shape, rule, steps=1)
    ulps = 0.0
    entries = zip(expected_network.snapshot(), actual_network.snapshot(), before, strict=True)
    for expected_entry, actual_entry, before_entry in entries:
        for expected, actual, start in zip(expected_entry, actual_entry, before_entry, strict=True):
            operands = np.maximum(np.abs(start), np.abs(expected - start))
            ulps = max(ulps, float((np.abs(to_numpy(actual) - expected) / np.spacing(operands)).max()))
    return ulps


# no residual block, the same widths: the dense layers' own gap
CONTROLS: dict[str, list[LayerSpec]] = {
    "sigmoid": [Dense(INPUT), Dense(5)],
    "relu": [Dense(INPUT, activation="relu"), Dense(5, activation="relu")],
    "deep": [Dense(INPUT), Dense(5), Dense(INPUT), Dense(3, activation="relu"), Dense(INPUT)],
    "batch norm": [Dense(INPUT, activation="relu"), Dense(5, activation="linear"), BatchNorm("relu")],
}


# the rules whose step is linear in the gradient, so a step's ulps are its gradient's. Adam's,
# lr * g / (|g| + epsilon), has slope lr * epsilon / (|g| + epsilon)**2, which turns a gradient
# near epsilon's size into a steep step: one ulp of such a gradient was ~1000 of its step (the
# deep control, g ~ 8e-9, OpenBLAS's Sandybridge kernel). Adam's step from the same gradient is
# pinned against numpy in tests/model/layers/test_adam_fused_layer_ops.py
LINEAR_RULES = [rule for rule in RULES if not isinstance(rule, Adam)]


@pytest.mark.parametrize("rule", LINEAR_RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("shape", ["multiclass", "single_output"])
@pytest.mark.parametrize("name", [*NETWORKS, *CONTROLS])
def test_one_step_matches_numpy_to_a_few_ulps(name: str, shape: Shape, rule: UpdateRule):
    # the fork's and add's sums are one IEEE addition each, the same bits on both; the dense and
    # affine products aren't (numpy's BLAS against the crate's FMA chains), as without blocks. The
    # layer before a fork sums the skip's and the body's gradients, which can cancel (parts up to
    # ~40x their sum, measured), so its products' rounding counts in ulps of the parts. Measured
    # after one step: residual networks 5 ulps (batch norm before the block), the controls 1;
    # with OpenBLAS's Sandybridge kernel 6 and 2
    layers = {**NETWORKS, **CONTROLS}[name]
    assert _max_ulps_after_one_step(layers, shape, rule) <= 32


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("shape", ["multiclass", "single_output"])
@pytest.mark.parametrize("name", NETWORKS)
def test_training_matches_numpy_within_the_dense_layers_rounding(name: str, shape: Shape, rule: UpdateRule):
    # 50 steps in, the per-step rounding has compounded: under Adam, which turns a gradient's
    # rounding into its step's, a residual block makes the trajectory itself sensitive. numpy
    # against numpy with the first layer's W nudged by one ulp drifts 7.9e-12 apart (batch norm in
    # the body), and 2.2e-14 with the same layers and weights and the skip path cut; numpy against
    # Rust measured 2.9e-11 relative here (Haswell's kernel; Sandybridge's 2.6e-11) and 1.6e-10 on
    # CI's. The other rules stay under 1.3e-12, the controls' kind of gap
    bound = 1e-9 if isinstance(rule, Adam) else 1e-10
    assert _gap_after_training(NETWORKS[name], shape, rule) < bound


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("shape", ["multiclass", "single_output"])
@pytest.mark.parametrize("name", CONTROLS)
def test_the_dense_layers_alone_have_the_same_kind_of_gap(name: str, shape: Shape, rule: UpdateRule):
    assert _gap_after_training(CONTROLS[name], shape, rule) < 1e-10
