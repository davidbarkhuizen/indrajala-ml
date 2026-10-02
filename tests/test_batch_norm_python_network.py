"""
Dense batch norm in pure Python (the batch-norm workplan, stage 2): BatchNormLayer and LinearLayer
against the README's expressions and numpy's layer by bits, the layer-major batch path (D3) against
the example-major loop by bits, gradient checks under every rule, the running averages in training
and inference, snapshot and checkpoint, the one-example refusal (D4) with train.py's final batch of
one, weight decay (D7), and parity with numpy's networks.
"""

import math
import random
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model import batch_norm_array_layer
from indrajala_ml.model.backprop_node import sigmoid
from indrajala_ml.model.base_node import AbstractNode
from indrajala_ml.model.batch_norm_array_layer import BatchNormArrayLayer
from indrajala_ml.model.batch_norm_layer import BatchNormLayer
from indrajala_ml.model.layer_major import LayerMajorBatch
from indrajala_ml.model.layer_specs import BatchNorm, Conv, Dense, LayerSpec, Pool
from indrajala_ml.model.linear_conv_layer import LinearConvLayer
from indrajala_ml.model.linear_layer import LinearLayer
from indrajala_ml.model.relu_layer import relu_activation
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.sequential_backprop_network import (
    SequentialBackpropClassifierNetwork,
    SequentialMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.state_layer import StateLayer
from indrajala_ml.model.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from indrajala_ml.pcg64 import default_rng
from indrajala_ml.train import train_backprop_network_mini_batch
from tests.gradient_check import check_gradients
from tests.helpers import bits, exp_by_math, patching, sigmoid_by
from tests.test_batch_norm_array_network import (
    BETA,
    EPSILON,
    GAMMA,
    INPUT,
    NETWORKS,
    RATE,
    RULES,
    SOFTMAX,
    X,
    _reference,
    _rows,
)

DOWNSTREAM = [[0.3, -0.7], [-0.1, 0.2], [0.5, 0.4]]


class _Input(AbstractNode):
    def value(self) -> float:
        return 0.0


class _InputLayer:
    def __init__(self, size: int) -> None:
        self.nodes = [_Input() for _ in range(size)]


def _layer(activation: Any = "sigmoid") -> BatchNormLayer:
    layer = BatchNormLayer(_InputLayer(2), activation, EPSILON, RATE)
    for node, gamma, beta in zip(layer.channels, GAMMA.tolist(), BETA.tolist()):
        node.gamma, node.beta = gamma, beta
    layer.set_training_mode(True)
    return layer


@pytest.mark.parametrize("activation", ["sigmoid", "relu"])
def test_every_expression_is_the_readmes_by_bits(activation: str):
    layer = _layer(activation)
    layer.forward_batch(X.tolist())
    layer.backward_batch(DOWNSTREAM)
    layer.accumulate_gradients()

    for feature, node in enumerate(layer.channels):
        downstream = [row[feature] for row in DOWNSTREAM]
        if activation == "sigmoid":
            delta = [ds * a * (1.0 - a) for ds, a in zip(downstream, node.activations)]
        else:
            delta = [ds * (1.0 if a > 0.0 else 0.0) for ds, a in zip(downstream, node.activations)]
        reference = _reference(X[:, feature].tolist(), GAMMA[feature], BETA[feature], delta)
        activate = sigmoid if activation == "sigmoid" else relu_activation
        assert bits(node.xhat) == bits(reference["xhat"])
        assert bits(node.activations) == bits([activate(y) for y in reference["y"]])
        assert bits(node.deltas) == bits(delta)
        assert bits(node.dxs) == bits(reference["dx"])
        assert bits(node.weight_gradient_accum) == bits([reference["grad_gamma"]])
        assert bits(node.bias_gradient_accum) == bits(reference["grad_beta"])
        assert bits([node.running_mean, node.running_var]) == bits(
            [reference["running_mean"], reference["running_var"]]
        )


def test_the_forward_pass_normalizes_each_feature_over_the_batch_by_hand():
    layer = _layer()
    layer.forward_batch(X.tolist())

    # as the numpy test's: feature 0 has mean 7/3 and biased variance 14/9, feature 1 mean 1/2
    # and variance 25/6
    for node, (mean, variance), gamma, beta, column in zip(
        layer.channels, [(7 / 3, 14 / 9), (1 / 2, 25 / 6)], GAMMA, BETA, X.T.tolist()
    ):
        xhat = [(x - mean) / math.sqrt(variance + EPSILON) for x in column]
        assert node.xhat == pytest.approx(xhat, rel=1e-14)
        assert node.activations == pytest.approx([sigmoid(gamma * v + beta) for v in xhat], rel=1e-14)
    assert [node.running_mean for node in layer.channels] == pytest.approx([0.1 * 7 / 3, 0.1 * 1 / 2], rel=1e-14)
    assert [node.running_var for node in layer.channels] == pytest.approx(
        [0.9 + 0.1 * 1.5 * 14 / 9, 0.9 + 0.1 * 1.5 * 25 / 6], rel=1e-14
    )


class _Next:
    def __init__(self, downstream: Any) -> None:
        self._downstream = downstream

    def downstream_batch(self) -> Any:
        return self._downstream


# the numpy layer's sigmoid with math.exp: backprop_node's
math_exp = patching(batch_norm_array_layer, "sigmoid", sigmoid_by(exp_by_math))


@pytest.mark.usefixtures("math_exp")
@pytest.mark.parametrize("activation", ["sigmoid", "relu"])
@pytest.mark.parametrize("batch_size", [2, 3, 8, 33])
def test_the_layer_is_numpys_by_bits(activation: Any, batch_size: int):
    # the same inputs, parameters and downstream: every value batch norm computes is the same, in
    # both implementations, since both follow the README's expressions and fold order
    rng = random.Random(batch_size)
    inputs = [[rng.uniform(-3.0, 3.0) for _ in range(4)] for _ in range(batch_size)]
    downstream = [[rng.uniform(-1.0, 1.0) for _ in range(4)] for _ in range(batch_size)]
    python = BatchNormLayer(_InputLayer(4), activation, EPSILON, RATE)
    array = BatchNormArrayLayer(4, activation, EPSILON, RATE)
    for node in python.channels:
        node.gamma, node.beta = rng.uniform(0.5, 2.0), rng.uniform(-1.0, 1.0)
    array.gamma = np.array([node.gamma for node in python.channels])
    array.beta = np.array([node.beta for node in python.channels])
    for layer in (python, array):
        layer.set_training_mode(True)

    python.forward_batch(inputs)
    python.backward_batch(downstream)
    python.accumulate_gradients()
    activations = array.forward_batch(np.array(inputs))
    array.compute_hidden_delta_batch(_Next(np.array(downstream)))
    array.accumulate_gradient_batch(np.array(inputs))
    dx = array.downstream_batch()

    def columns(values: Any) -> list[list[float]]:
        return np.asarray(values).T.tolist()

    assert bits([node.activations for node in python.channels]) == bits(columns(activations))
    assert bits([node.dxs for node in python.channels]) == bits(columns(dx))
    assert bits([node.weight_gradient_accum[0] for node in python.channels]) == bits(array.grad_gamma.tolist())
    assert bits([node.bias_gradient_accum for node in python.channels]) == bits(array.grad_beta.tolist())
    assert bits([node.running_mean for node in python.channels]) == bits(array.running_mean.tolist())
    assert bits([node.running_var for node in python.channels]) == bits(array.running_var.tolist())


def test_inference_normalizes_with_the_running_averages():
    inputs = StateLayer(2, [(0.0, 1.0)] * 2)
    layer = BatchNormLayer(inputs, "sigmoid", EPSILON, RATE)
    for node, gamma, beta, mean, var in zip(layer.channels, GAMMA, BETA, [0.5, -1.0], [2.0, 0.25]):
        node.gamma, node.beta, node.running_mean, node.running_var = gamma, beta, mean, var

    inputs.update_state((2.0, 0.5))
    layer.forward()

    expected = [
        sigmoid(gamma * ((x - mean) / math.sqrt(var + EPSILON)) + beta)
        for x, gamma, beta, mean, var in zip((2.0, 0.5), GAMMA, BETA, [0.5, -1.0], [2.0, 0.25])
    ]
    assert bits([node.value() for node in layer.nodes]) == bits(expected)
    assert [(node.running_mean, node.running_var) for node in layer.channels] == [(0.5, 2.0), (-1.0, 0.25)]


def test_a_layer_refuses_one_example_in_training():
    layer = _layer()
    with pytest.raises(ValueError, match="D4"):
        layer.forward_batch(X[:1].tolist())
    with pytest.raises(ValueError, match="D4"):
        layer.forward()


def test_the_linear_layer_has_no_bias_and_its_delta_is_the_downstream():
    inputs = StateLayer(3, [(0.0, 1.0)] * 3)
    layer = LinearLayer(2, inputs)
    for node, weights in zip(layer.nodes, [[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]]):
        node.update_input_weights(weights)
    inputs.update_state((1.0, 0.0, 2.0))

    layer.forward()
    assert [node.value() for node in layer.nodes] == [7.0, -1.0]

    class Norm:
        def downstream_sum(self, own_index: int) -> float:
            return [0.1, -0.4][own_index]

    layer.compute_hidden_deltas(Norm())
    layer.accumulate_gradients()
    assert [node.delta for node in layer.nodes] == [0.1, -0.4]
    assert [node.weight_gradient_accum for node in layer.nodes] == [[0.1, 0.0, 0.2], [-0.4, -0.0, -0.8]]
    assert [node.bias_gradient_accum for node in layer.nodes] == [0.0, 0.0]
    assert layer.snapshot_state() == [([1.0, 2.0, 3.0],), ([-1.0, 0.5, 0.0],)]
    assert not layer.nodes[0].has_bias and layer.nodes[0].weights_decayed


def test_randomize_draws_the_linear_layers_weights_only_and_nothing_for_batch_norm():
    network = _network()
    linear, norm, output = network.trainable_layers
    rng = default_rng(3)
    limit = 1 / math.sqrt(4)
    linear_weights = [[rng.uniform(-limit, limit) for _ in range(4)] for _ in range(5)]
    limit = 1 / math.sqrt(5)
    output_weights_and_biases = [
        ([rng.uniform(-limit, limit) for _ in range(5)], rng.uniform(-limit, limit)) for _ in range(3)
    ]

    assert bits(linear.snapshot_state()) == bits([(weights,) for weights in linear_weights])
    assert bits(output.snapshot_state()) == bits(output_weights_and_biases)
    assert norm.snapshot_state() == [([1.0], 0.0, 0.0, 1.0)] * 5


# networks

CLASSES = {
    "multiclass": SequentialMultiClassBackpropClassifierNetwork,
    "single_output": SequentialBackpropClassifierNetwork,
}
FRONT_END_INPUT = (6, 6, 1)
FRONT_END: list[LayerSpec] = [Conv(3, 2), Pool(2), Dense(5, activation="linear"), BatchNorm("relu"), SOFTMAX]
AFTER_BATCH_NORM_DROPOUT: list[LayerSpec] = [
    Dense(5, activation="linear"),
    BatchNorm(),
    Dense(4, dropout=0.3),
    Dense(3, output=True),
]


def _network(name: str = "sigmoid", rule: UpdateRule | None = None, seed: int = 3) -> Any:
    layers, shape = NETWORKS[name]
    network = CLASSES[shape](INPUT, layers, SGD() if rule is None else rule)
    network.rng = default_rng(seed)
    network.randomize()
    return network


def _image_rows(count: int, seed: int = 1) -> list[tuple[tuple[float, ...], int]]:
    rng = random.Random(seed)
    return [(tuple(rng.random() for _ in range(36)), i % 3) for i in range(count)]


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", NETWORKS)
@pytest.mark.parametrize("batch_size", [2, 5])
def test_every_gradient_matches_its_finite_difference(name: str, rule: UpdateRule, batch_size: int):
    network = _network(name, rule)
    rows = _rows(batch_size, NETWORKS[name][1])
    # moved running averages and a trained step, so gamma and beta aren't at their initial values
    network.learn_batch(0.5, _rows(6, NETWORKS[name][1], seed=2))

    check_gradients(network, [state for state, _ in rows], [label for _, label in rows])


def test_every_gradient_through_a_conv_front_end_matches_its_finite_difference():
    # the conv and pool layers' lanes (layer_major.py) hold each example's activations, deltas and
    # winning slots
    network = SequentialMultiClassBackpropClassifierNetwork(FRONT_END_INPUT, FRONT_END, Momentum(0.9))
    network.rng = default_rng(4)
    network.randomize()
    network.learn_batch(0.5, _image_rows(6, seed=2))
    rows = _image_rows(4)

    check_gradients(network, [state for state, _ in rows], [label for _, label in rows])


def test_every_gradient_through_dropout_after_batch_norm_matches_its_finite_difference(
    monkeypatch: pytest.MonkeyPatch,
):
    # the dropout layer's lanes hold each example's mask: with the same masks in every forward
    # pass (the network's generator reseeded before each), the loss is a function of the weights
    # again
    forward = LayerMajorBatch.forward

    def seeded(self: LayerMajorBatch) -> None:
        network.rng = default_rng(11)
        forward(self)

    monkeypatch.setattr(LayerMajorBatch, "forward", seeded)
    network = SequentialMultiClassBackpropClassifierNetwork(INPUT, AFTER_BATCH_NORM_DROPOUT, SGD())
    network.rng = default_rng(5)
    network.randomize()
    rows = _rows(5)

    check_gradients(network, [state for state, _ in rows], [label for _, label in rows])

    # a unit dropped for some examples and kept for others: the masks are per example
    batch = LayerMajorBatch(network.input_layer, network.trainable_layers, [state for state, _ in rows])
    network._set_training_mode(True)
    batch.forward()
    network._set_training_mode(False)
    kept = [[fields["_kept"] for fields in nodes] for _own, nodes in batch.lanes[2]]
    assert any(any(unit) and not all(unit) for unit in zip(*kept))


# the layer-major path's lanes against the example-major loop, over every layer kind but dropout
# (whose draw order the two paths don't share)
LAYER_MAJOR_CASES: dict[str, tuple[Any, list[LayerSpec], str]] = {
    "sigmoid": ((4,), [Dense(5), Dense(4), Dense(3, output=True)], "multiclass"),
    "relu softmax": ((4,), [Dense(5, activation="relu"), SOFTMAX], "multiclass"),
    "cross-entropy single output": ((4,), [Dense(5), Dense(1, output=True, loss="cross_entropy")], "single_output"),
    "conv pool conv": (FRONT_END_INPUT, [Conv(3, 2), Pool(2), Conv(2, 3), Dense(4), SOFTMAX], "multiclass"),
}


@pytest.mark.parametrize("rule", [Momentum(0.9), Adam()], ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", LAYER_MAJOR_CASES)
def test_the_layer_major_path_is_the_example_major_loop_by_bits(name: str, rule: UpdateRule):
    input_shape, specs, shape = LAYER_MAJOR_CASES[name]
    example_major, layer_major = (CLASSES[shape](input_shape, specs, rule) for _ in range(2))
    example_major.rng = default_rng(6)
    example_major.randomize()
    layer_major.restore(example_major.snapshot())
    rng = random.Random(2)
    dimension = math.prod(input_shape)

    for size in (3, 5, 4):
        batch: list[tuple[tuple[float, ...], Any]] = [
            (tuple(rng.random() for _ in range(dimension)), float(i % 2) if shape == "single_output" else i % 3)
            for i in range(size)
        ]
        example_major.learn_batch(0.3, batch)
        layer_major._learn_batch_layer_major(0.3, batch)

    assert bits(layer_major.snapshot()) == bits(example_major.snapshot())
    assert bits(list(layer_major.optimizer.state().layers.values())) == bits(
        list(example_major.optimizer.state().layers.values())
    )


def test_the_running_averages_move_in_training_forward_passes_only():
    network = _network()
    rows = _rows(6)
    norm = network.trainable_layers[1]

    network.learn_batch(0.5, rows)
    trained = bits(norm.snapshot_state())
    assert trained != bits([([1.0], 0.0, 0.0, 1.0)] * 5)

    for state, _ in rows:
        network.classify_state(state)
        network.predict_probabilities(state)
    assert bits(norm.snapshot_state()) == trained


def test_classifying_normalizes_with_the_running_averages():
    network = _network()
    rows = _rows(6)
    network.learn_batch(0.5, rows)
    linear, norm, output = network.trainable_layers

    for state, _ in rows:
        z = [sum([x * w for x, w in zip(state, node.input_node_weights)]) for node in linear.nodes]
        y = [
            sigmoid(node.gamma * ((z_j - node.running_mean) / math.sqrt(node.running_var + EPSILON)) + node.beta)
            for z_j, node in zip(z, norm.channels)
        ]
        expected = [
            sigmoid(sum([a * w for a, w in zip(y, node.input_node_weights)]) + node.bias) for node in output.nodes
        ]
        assert bits(network.predict_probabilities(state)) == bits(expected)


def test_a_training_step_is_the_rules_step_on_gamma_and_beta():
    network = _network(rule=Momentum(0.9))
    norm = network.trainable_layers[1]
    rows = _rows(6)
    network.learn_batch(0.5, rows)
    before = [(node.gamma, node.beta) for node in norm.channels]

    network.learn_batch(0.5, rows)

    # per feature: [gamma]'s velocity, then beta's
    velocities = network.optimizer.state().layers[1]
    for node, (gamma, beta), (weight_state, bias_state) in zip(norm.channels, before, velocities):
        assert node.gamma == gamma - 0.5 * weight_state[0][0]
        assert node.beta == beta - 0.5 * bias_state[0]


def test_the_optimizers_state_has_no_bias_for_a_linear_node():
    network = _network(rule=Adam())
    network.learn_batch(0.1, _rows(6))
    linear, norm, output = (network.optimizer.state().layers[i] for i in range(3))

    assert [([len(values) for values in weights], len(bias)) for weights, bias in linear] == [([4, 4], 0)] * 5
    assert [([len(values) for values in weights], len(bias)) for weights, bias in norm] == [([1, 1], 2)] * 5
    assert [([len(values) for values in weights], len(bias)) for weights, bias in output] == [([5, 5], 2)] * 3


def test_weight_decay_decays_the_linear_layers_weights_and_neither_gamma_nor_beta():
    rows = _rows(6)
    sgd, decayed = _network(rule=SGD()), _network(rule=WeightDecay(0.1))

    for network in (sgd, decayed):
        network.learn_batch(0.5, rows)

    # gamma and beta step with plain SGD (D7), and nothing before them differs
    gamma_and_beta = [[entry[:2] for entry in network.snapshot()[1]] for network in (sgd, decayed)]
    assert bits(gamma_and_beta[1]) == bits(gamma_and_beta[0])
    assert bits(decayed.snapshot()[0]) != bits(sgd.snapshot()[0])


@pytest.mark.parametrize("name", ["after a sigmoid layer", "single output"])
@pytest.mark.parametrize("method", ["learn", "learn_batch"])
def test_a_one_example_training_step_is_refused_naming_the_layer(name: str, method: str):
    network = _network(name)
    rows = _rows(3, NETWORKS[name][1])
    before = bits(network.snapshot())
    index = 2 if name == "after a sigmoid layer" else 1

    with pytest.raises(ValueError, match=rf"layer {index}, BatchNorm\(.*D4"):
        if method == "learn":
            network.learn(0.5, *rows[0])
        else:
            network.learn_batch(0.5, rows[:1])
    assert bits(network.snapshot()) == before


def test_train_drops_a_final_batch_of_one_for_batch_norm():
    network = _network()
    sizes: list[int] = []
    learn_batch = network.learn_batch

    def record(learning_rate: float, batch: Any) -> None:
        sizes.append(len(batch))
        learn_batch(learning_rate, batch)

    network.learn_batch = record
    train_backprop_network_mini_batch(network, _rows(7), 3, epochs=2, rng=random.Random(0))

    assert sizes == [3, 3, 3, 3]


def test_snapshot_carries_the_running_averages_and_restore_returns_them():
    network = _network()
    network.learn_batch(0.5, _rows(6))
    snapshot = network.snapshot()
    norm = network.trainable_layers[1]

    assert [len(entry) for entry in snapshot] == [5, 5, 3]
    assert [len(node) for node in snapshot[0]] == [1] * 5
    assert snapshot[1] == [([node.gamma], node.beta, node.running_mean, node.running_var) for node in norm.channels]

    network.learn_batch(0.5, _rows(6, seed=5))
    network.restore([[list(node) for node in entry] for entry in snapshot])  # as nested lists
    assert bits(network.snapshot()) == bits(snapshot)


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
def test_a_checkpoint_resumes_training_by_bits(rule: UpdateRule):
    network = _network("two pairs", rule)
    network.learn_batch(0.1, _rows(6))
    checkpoint = network.checkpoint()

    network.learn_batch(0.1, _rows(5, seed=7))
    network.learn_batch(0.1, _rows(4, seed=8))
    trained = bits(network.snapshot())

    network.restore_checkpoint(checkpoint)
    network.learn_batch(0.1, _rows(5, seed=7))
    network.learn_batch(0.1, _rows(4, seed=8))
    assert bits(network.snapshot()) == trained


# parity with numpy


def _matching_numpy_network(python: Any, name: str, rule: UpdateRule) -> Any:
    layers, shape = NETWORKS[name]
    array = SequentialArrayNetwork(INPUT, layers, rule, shape=shape)
    array.restore(_as_array_snapshot(python))
    return array


def _as_array_snapshot(python: Any) -> list[tuple[list[Any], ...]]:
    # the pure-Python snapshot, per node or feature, as numpy's, per parameter
    snapshot: list[tuple[list[Any], ...]] = []
    for layer, entry in zip(python.trainable_layers, python.snapshot()):
        if isinstance(layer, BatchNormLayer):
            gamma, beta, mean, var = zip(*entry)
            snapshot.append(([g for (g,) in gamma], list(beta), list(mean), list(var)))
        elif isinstance(layer, LinearLayer | LinearConvLayer):
            snapshot.append(([weights for (weights,) in entry],))
        else:
            snapshot.append(tuple(list(values) for values in zip(*entry)))
    return snapshot


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", NETWORKS)
def test_training_matches_numpy_within_the_dense_layers_rounding(name: str, rule: UpdateRule):
    # batch norm computes the same bits in both (test_the_layer_is_numpys_by_bits); the dense
    # layers don't: pure Python's weighted sums are the builtin sum, compensated since Python 3.12,
    # and its gradients accumulate example by example, where numpy's are BLAS products. So the
    # networks agree within the tolerance every pure-Python parity test allows
    # (assert_array_network_weights_match), 50 steps in: at most 6e-10 relative when measured,
    # Adam's steps amplifying it most, as they do for networks without batch norm
    python = _network(name, rule)
    array = _matching_numpy_network(python, name, rule)
    rows = _rows(40, NETWORKS[name][1])

    for step in range(50):
        batch = rows[(step * 5) % 40 :][:5]
        python.learn_batch(0.3, batch)
        array.learn_batch(0.3, batch)

    for expected, actual in zip(_as_array_snapshot(python), array.snapshot()):
        for values, array_values in zip(expected, actual):
            np.testing.assert_allclose(array_values, values, rtol=1e-9, atol=1e-9)
