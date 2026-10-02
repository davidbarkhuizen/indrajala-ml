"""
Residual blocks in pure Python (the residual-connections workplan, stage 3; README, Residual
connections): the gradient check, identity blocks by bits, learn against a batch of one, the
layer-major batch path (for a block holding batch norm) against the example-major loop, the block's
layers and their wiring, and parity with numpy after 50 steps. The cases are
tests/test_residual_array_network.py's.
"""

import math
import random
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layer_specs import Dense, LayerSpec
from indrajala_ml.model.residual_layer import AddLayer, AffineLayer, ForkLayer
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from indrajala_ml.pcg64 import default_rng
from tests.gradient_check import analytic_gradients, check_gradients
from tests.helpers import bits
from tests.test_batch_norm_python_network import CLASSES, _as_array_snapshot
from tests.test_residual_array_network import INPUT, NETWORKS, Shape, block, output, rows

# the cases without batch norm, which train example by example; with it, layer-major
EXAMPLE_MAJOR = [name for name in NETWORKS if "batch norm" not in name]


def network(name: str, shape: Shape = "multiclass", rule: UpdateRule | None = None, seed: int = 3) -> Any:
    built = CLASSES[shape]((INPUT,), [*NETWORKS[name], output(shape)], SGD() if rule is None else rule)
    built.rng = default_rng(seed)
    built.randomize()
    return built


@pytest.mark.parametrize("shape", ["multiclass", "single_output"])
@pytest.mark.parametrize("name", NETWORKS)
def test_every_gradient_matches_its_finite_difference(name: str, shape: Shape):
    built = network(name, shape)
    built.learn_batch(0.5, rows(6, shape, seed=2))
    batch = rows(5, shape)

    check_gradients(built, [state for state, _ in batch], [label for _, label in batch])


def test_a_block_builds_a_fork_its_body_and_an_add_wired_together():
    built = network("sigmoid body")
    dense, fork, body, end, add, out = built.trainable_layers

    assert isinstance(fork, ForkLayer) and isinstance(add, AddLayer) and isinstance(end, AffineLayer)
    assert fork.body_first is body and fork.add is add and add.fork is fork
    assert fork.input_layer is dense and body.input_layer is fork and add.input_layer is end and out.input_layer is add
    assert [node.input_node for node in fork.nodes] == list(dense.nodes)
    assert [(node.body_node, node.fork_node) for node in add.nodes] == list(zip(end.nodes, fork.nodes, strict=True))
    assert fork.snapshot_state() == [] and add.snapshot_state() == [] and fork.weight_sets() == []


def test_randomize_draws_weights_then_bias_per_node_and_nothing_for_the_fork_or_add():
    built = network("sigmoid body")
    rng = default_rng(3)
    expected: list[Any] = []
    for size, fan_in in [(INPUT, INPUT), (5, INPUT), (INPUT, 5), (3, INPUT)]:
        limit = 1 / math.sqrt(fan_in)
        expected.append(
            [([rng.uniform(-limit, limit) for _ in range(fan_in)], rng.uniform(-limit, limit)) for _ in range(size)]
        )

    assert bits([entry for entry in built.snapshot() if entry]) == bits(expected)


def _identity_pair(body: Dense, shape: Shape) -> tuple[Any, Any]:
    plain = CLASSES[shape]((INPUT,), [Dense(INPUT), Dense(6), output(shape)], SGD())
    plain.rng = default_rng(5)
    plain.randomize()
    residual = CLASSES[shape]((INPUT,), [Dense(INPUT), block(body), Dense(6), output(shape)], SGD())
    residual.rng = default_rng(7)
    residual.randomize()

    first, second, out = plain.snapshot()
    snapshot = residual.snapshot()
    zero = [([0.0] * len(weights), 0.0) for weights, _bias in snapshot[3]]
    residual.restore([first, [], snapshot[2], zero, [], second, out])
    return plain, residual


@pytest.mark.parametrize("shape", ["multiclass", "single_output"])
@pytest.mark.parametrize("body", [Dense(5), Dense(5, activation="relu")], ids=["sigmoid", "relu"])
def test_an_identity_block_changes_no_output_and_no_other_layers_gradient_by_bits(body: Dense, shape: Shape):
    plain, residual = _identity_pair(body, shape)
    batch = rows(5, shape)
    states, labels = [state for state, _ in batch], [label for _, label in batch]

    for state in states:
        assert bits(residual._forward(state)) == bits(plain._forward(state))

    residual_gradients = analytic_gradients(residual, states, labels)
    assert bits([residual_gradients[i] for i in (0, 5, 6)]) == bits(analytic_gradients(plain, states, labels))

    plain.learn(0.5, *batch[0])
    residual.learn(0.5, *batch[0])
    trained = residual.snapshot()
    assert bits([trained[i] for i in (0, 5, 6)]) == bits(plain.snapshot())


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), Adam()], ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", ["sigmoid body", "relu body", "two blocks in a row", "block first"])
def test_learn_and_a_learn_batch_of_one_example_agree_by_bits(name: str, rule: UpdateRule):
    single, batched = network(name, rule=rule), network(name, rule=rule)

    for state, label in rows(6, seed=4):
        single.learn(0.5, state, label)
        batched.learn_batch(0.5, [(state, label)])

    assert bits(single.snapshot()) == bits(batched.snapshot())


@pytest.mark.parametrize("rule", [Momentum(0.9), Adam()], ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", [name for name in EXAMPLE_MAJOR if "dropout" not in name])
def test_the_layer_major_path_is_the_example_major_loop_by_bits(name: str, rule: UpdateRule):
    # the lanes of the fork and add, which read each other's (layer_major.py)
    example_major, layer_major = network(name, rule=rule), network(name, rule=rule)
    rng = random.Random(2)

    for size in (3, 5, 4):
        batch = [(tuple(rng.uniform(-1.0, 1.0) for _ in range(INPUT)), i % 3) for i in range(size)]
        example_major.learn_batch(0.3, batch)
        layer_major._learn_batch_layer_major(0.3, batch)

    assert bits(layer_major.snapshot()) == bits(example_major.snapshot())
    assert bits(list(layer_major.optimizer.state().layers.values())) == bits(
        list(example_major.optimizer.state().layers.values())
    )


RULES = [SGD(), Momentum(0.9), Adam(), WeightDecay(0.01)]


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("shape", ["multiclass", "single_output"])
@pytest.mark.parametrize("name", NETWORKS)
def test_training_matches_numpy_within_the_dense_layers_rounding(name: str, shape: Shape, rule: UpdateRule):
    # the fork's and add's sums are one IEEE addition each, the same bits in both; the dense and
    # affine layers' aren't (pure Python's weighted sums are the builtin sum, compensated since
    # Python 3.12, against numpy's BLAS products), as without blocks, so the networks agree within
    # the tolerance every pure-Python parity test allows (assert_array_network_weights_match)
    python = network(name, shape, rule)
    specs: list[LayerSpec] = [*NETWORKS[name], output(shape)]
    array = SequentialArrayNetwork((INPUT,), specs, rule, shape=shape)
    array.restore(_as_array_snapshot(python))
    data = rows(40, shape)

    for step in range(50):
        batch = data[(step * 5) % 40 :][:5]
        python.learn_batch(0.3, batch)
        array.learn_batch(0.3, batch)

    for expected, actual in zip(_as_array_snapshot(python), array.snapshot(), strict=True):
        for values, array_values in zip(expected, actual, strict=True):
            np.testing.assert_allclose(array_values, values, rtol=1e-9, atol=1e-9)
