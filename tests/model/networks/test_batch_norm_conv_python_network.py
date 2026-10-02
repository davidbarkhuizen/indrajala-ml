"""
Conv batch norm in pure Python (the batch-norm workplan, stage 4b): LinearConvLayer and
BatchNormLayer over a conv layer's channels, against tests/model/networks/test_batch_norm_array_network.py's
scalar transcription of the README's expressions and against numpy's layers (stage 4a) by bits,
gradient checks under every rule through the layer-major path, the running averages, snapshot and
checkpoint, and whole-network parity with numpy. Reuses tests/model/networks/test_batch_norm_conv_array_network.py's
cases.
"""

import random
from typing import Any, cast

import numpy as np
import pytest

from indrajala_ml.model.layers.numpy.batch_norm_array_layer import BatchNormArrayLayer
from indrajala_ml.model.layers.python.backprop_node import sigmoid
from indrajala_ml.model.layers.python.base_node import AbstractNode
from indrajala_ml.model.layers.python.batch_norm_layer import BatchNormLayer, BatchNormPosition
from indrajala_ml.model.layers.python.conv_layer import ConvLayer
from indrajala_ml.model.layers.python.fan_in_aware_init import fan_in_aware_weights
from indrajala_ml.model.layers.python.linear_conv_layer import LinearConvKernel, LinearConvLayer
from indrajala_ml.model.layers.python.relu_layer import relu_activation
from indrajala_ml.model.layers.python.state_layer import StateLayer
from indrajala_ml.model.networks.python.sequential_backprop_network import SequentialMultiClassBackpropClassifierNetwork
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.update_rules import SGD, Adam, UpdateRule, WeightDecay
from indrajala_ml.pcg64 import default_rng
from tests.gradient_check import check_gradients
from tests.helpers import bits
from tests.model.networks.test_batch_norm_array_network import EPSILON, RATE, RULES, _reference
from tests.model.networks.test_batch_norm_conv_array_network import (
    BETA,
    DOWNSTREAM,
    GAMMA,
    INPUT,
    NETWORKS,
    POSITIONS,
    X,
    _rows,
)
from tests.model.networks.test_batch_norm_python_network import (  # math_exp: a fixture
    _as_array_snapshot,
    math_exp,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)


class _Input(AbstractNode):
    def value(self) -> float:
        return 0.0


class _InputLayer:
    def __init__(self, size: int) -> None:
        self.nodes = [_Input() for _ in range(size)]


def _layer(activation: Any = "relu") -> BatchNormLayer:
    layer = BatchNormLayer(_InputLayer(8), activation, EPSILON, RATE, POSITIONS)
    for channel, gamma, beta in zip(layer.channels, GAMMA.tolist(), BETA.tolist()):
        channel.gamma, channel.beta = gamma, beta
    layer.set_training_mode(True)
    return layer


def _channel(rows: list[list[float]], channel: int) -> list[float]:
    """A channel's values in the README's order: example by example, then position by position."""
    return [row[channel * POSITIONS + p] for row in rows for p in range(POSITIONS)]


def test_a_conv_layer_has_a_node_per_position_and_a_weight_set_per_channel():
    layer = _layer()

    assert len(layer.nodes) == 8 and all(isinstance(node, BatchNormPosition) for node in layer.nodes)
    assert layer.weight_sets() == layer.channels and len(layer.channels) == 2
    assert [node.channel for node in layer.nodes] == [layer.channels[0]] * 4 + [layer.channels[1]] * 4


@pytest.mark.parametrize("activation", ["sigmoid", "relu"])
def test_every_expression_is_the_readmes_over_each_channel_by_bits(activation: str):
    layer = _layer(activation)
    layer.forward_batch(X.tolist())
    layer.backward_batch(DOWNSTREAM.tolist())
    layer.accumulate_gradients()

    activate = sigmoid if activation == "sigmoid" else relu_activation
    for c, channel in enumerate(layer.channels):
        downstream = _channel(DOWNSTREAM.tolist(), c)
        if activation == "sigmoid":
            delta = [ds * a * (1.0 - a) for ds, a in zip(downstream, channel.activations)]
        else:
            delta = [ds * (1.0 if a > 0.0 else 0.0) for ds, a in zip(downstream, channel.activations)]
        reference = _reference(_channel(X.tolist(), c), GAMMA[c], BETA[c], delta)
        assert bits(channel.activations) == bits([activate(y) for y in reference["y"]])
        assert bits(channel.deltas) == bits(delta)
        assert bits(channel.dxs) == bits(reference["dx"])
        assert bits(channel.weight_gradient_accum) == bits([reference["grad_gamma"]])
        assert bits(channel.bias_gradient_accum) == bits(reference["grad_beta"])
        assert bits([channel.running_mean, channel.running_var]) == bits(
            [reference["running_mean"], reference["running_var"]]
        )


def test_selecting_an_example_gives_each_position_its_values():
    layer = _layer()
    layer.forward_batch(X.tolist())
    layer.backward_batch(DOWNSTREAM.tolist())

    for example in range(3):
        layer.select_example(example)
        for i, node in enumerate(layer.nodes):
            channel, k = layer.channels[i // POSITIONS], example * POSITIONS + i % POSITIONS
            assert (node.value(), node.delta, node.dx) == (channel.activations[k], channel.deltas[k], channel.dxs[k])
            assert layer.downstream_sum(i) == channel.dxs[k]


class _Next:
    def __init__(self, downstream: Any) -> None:
        self._downstream = downstream

    def downstream_batch(self) -> Any:
        return self._downstream


@pytest.mark.usefixtures("math_exp")
@pytest.mark.parametrize("activation", ["sigmoid", "relu"])
@pytest.mark.parametrize("batch_size", [2, 3, 9])
def test_the_layer_is_numpys_by_bits(activation: Any, batch_size: int):
    # the same inputs, parameters and downstream: both follow the README's expressions in its order,
    # the numpy layer on its (N * P, C) rows. math_exp: the numpy layer's sigmoid takes math.exp
    rng = random.Random(batch_size)
    inputs = [[rng.uniform(-3.0, 3.0) for _ in range(12)] for _ in range(batch_size)]
    downstream = [[rng.uniform(-1.0, 1.0) for _ in range(12)] for _ in range(batch_size)]
    python = BatchNormLayer(_InputLayer(12), activation, EPSILON, RATE, 4)
    array = BatchNormArrayLayer(12, activation, EPSILON, RATE, 4)
    for channel in python.channels:
        channel.gamma, channel.beta = rng.uniform(0.5, 2.0), rng.uniform(-1.0, 1.0)
    array.gamma = np.array([channel.gamma for channel in python.channels])
    array.beta = np.array([channel.beta for channel in python.channels])
    for layer in (python, array):
        layer.set_training_mode(True)

    python.forward_batch(inputs)
    python.backward_batch(downstream)
    python.accumulate_gradients()
    activations = array.forward_batch(np.array(inputs))
    array.compute_hidden_delta_batch(_Next(np.array(downstream)))
    array.accumulate_gradient_batch(np.array(inputs))
    dx = array.downstream_batch()

    for example in range(batch_size):
        python.select_example(example)
        assert bits([node.value() for node in python.nodes]) == bits(activations[example].tolist())
        assert bits([node.dx for node in python.nodes]) == bits(dx[example].tolist())
    for name, values in (("weight_gradient_accum", array.grad_gamma), ("bias_gradient_accum", array.grad_beta)):
        actual = [getattr(channel, name) for channel in python.channels]
        assert bits([value[0] if isinstance(value, list) else value for value in actual]) == bits(values.tolist())
    assert bits([channel.running_mean for channel in python.channels]) == bits(array.running_mean.tolist())
    assert bits([channel.running_var for channel in python.channels]) == bits(array.running_var.tolist())


def test_inference_normalizes_each_position_with_its_channels_running_averages():
    inputs = StateLayer(8, [(0.0, 1.0)] * 8)
    layer = BatchNormLayer(inputs, "relu", EPSILON, RATE, POSITIONS)
    for channel, gamma, beta, mean, var in zip(layer.channels, GAMMA, BETA, [0.5, -1.0], [2.0, 0.25]):
        channel.gamma, channel.beta, channel.running_mean, channel.running_var = gamma, beta, mean, var

    inputs.update_state(tuple(X[1].tolist()))
    layer.forward()

    expected = [
        relu_activation(
            GAMMA[i // POSITIONS] * ((x - [0.5, -1.0][i // POSITIONS]) / np.sqrt([2.0, 0.25][i // POSITIONS] + EPSILON))
            + BETA[i // POSITIONS]
        )
        for i, x in enumerate(X[1].tolist())
    ]
    assert [node.value() for node in layer.nodes] == pytest.approx(expected, rel=1e-15)


def test_a_layer_refuses_one_example_in_training_although_it_has_several_positions():
    with pytest.raises(ValueError, match="D4"):
        _layer().forward_batch(X[:1].tolist())


def _conv_pair() -> tuple[ConvLayer, LinearConvLayer, StateLayer]:
    inputs = StateLayer(50, [(0.0, 1.0)] * 50)
    conv = ConvLayer(inputs, 5, 5, kernel_size=3, channel_count=3, stride=2, input_channels=2)
    linear = LinearConvLayer(inputs, 5, 5, kernel_size=3, channel_count=3, stride=2, input_channels=2)
    rng = random.Random(0)
    for kernel, linear_kernel in zip(conv.kernels, linear.kernels):
        kernel.weights = [rng.uniform(-1.0, 1.0) for _ in kernel.weights]
        linear_kernel.weights = list(kernel.weights)
    inputs.update_state(tuple(rng.uniform(-1.0, 1.0) for _ in range(50)))
    return conv, linear, inputs


def test_the_linear_conv_layer_is_the_conv_layers_weighted_sums_without_bias_or_relu():
    conv, linear, _inputs = _conv_pair()
    conv.forward()
    linear.forward()

    # conv's bias is 0.0, and z + 0.0 is z but for -0.0
    assert [unit.value() for unit in linear.nodes] == [unit.z() for unit in conv.nodes]
    assert any(unit.value() < 0.0 for unit in linear.nodes)
    assert all(type(kernel) is LinearConvKernel and not kernel.has_bias for kernel in linear.kernels)
    assert linear.snapshot_state() == [(list(kernel.weights),) for kernel in linear.kernels]


def test_the_linear_conv_layers_delta_is_its_downstream_and_its_gradient_has_no_bias():
    conv, linear, _inputs = _conv_pair()
    for layer in (conv, linear):
        layer.forward()
    downstream = [0.1 * (i % 7) - 0.3 for i in range(len(linear.nodes))]

    class Next:
        def downstream_sum(self, own_index: int) -> float:
            return downstream[own_index]

    linear.compute_hidden_deltas(cast(Any, Next()))
    linear.accumulate_gradients()
    for unit, value in zip(conv.nodes, downstream):
        unit.delta = value
    conv.accumulate_gradients()

    assert [unit.delta for unit in linear.nodes] == downstream
    assert [k.weight_gradient_accum for k in linear.kernels] == [k.weight_gradient_accum for k in conv.kernels]
    assert [k.bias_gradient_accum for k in linear.kernels] == [0.0] * 3
    assert [linear.downstream_sum(i) for i in range(50)] == [conv.downstream_sum(i) for i in range(50)]


def test_randomize_draws_the_linear_conv_kernels_weights_only_and_nothing_for_batch_norm():
    network = _network()
    linear, norm, *_rest = network.trainable_layers
    rng = default_rng(4)

    # each kernel draws 9 weights, in kernel order, and no bias
    assert [kernel.weights for kernel in linear.kernels] == [fan_in_aware_weights(rng, 9) for _ in range(2)]
    assert norm.snapshot_state() == [([1.0], 0.0, 0.0, 1.0)] * 2


# networks


def _network(name: str = "conv pool", rule: UpdateRule | None = None, seed: int = 4) -> Any:
    # not seed 3: it draws "after a relu conv"'s first conv dead on every input (both biases
    # about -0.4 on inputs in [0, 1)), so batch norm normalizes a constant zero and its ReLU sits
    # exactly on the kink, where the finite difference sees half the slope
    network = SequentialMultiClassBackpropClassifierNetwork(INPUT, NETWORKS[name], SGD() if rule is None else rule)
    network.rng = default_rng(seed)
    network.randomize()
    return network


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", NETWORKS)
def test_every_gradient_matches_its_finite_difference(name: str, rule: UpdateRule):
    network = _network(name, rule)
    rows = _rows(3)
    # moved running averages and a trained step, so gamma and beta aren't at their initial values
    network.learn_batch(0.5, _rows(4, seed=2))

    check_gradients(network, [state for state, _ in rows], [label for _, label in rows])


def test_the_running_averages_move_in_training_forward_passes_only():
    network = _network()
    rows = _rows(6)
    norm = network.trainable_layers[1]

    network.learn_batch(0.5, rows)
    trained = norm.snapshot_state()
    assert trained != [([1.0], 0.0, 0.0, 1.0)] * 2

    for state, _ in rows:
        network.classify_state(state)
    assert bits(norm.snapshot_state()) == bits(trained)


def test_the_optimizers_state_is_per_channel_with_no_bias_for_a_linear_kernel():
    network = _network(rule=Adam())
    network.learn_batch(0.1, _rows(4))
    state = network.optimizer.state().layers

    # per weight set, (the weights' moments, the bias's): a linear kernel has none for its bias
    assert [len(biases) for _weights, biases in state[0]] == [0, 0]
    assert [(len(weights[0]), len(biases)) for weights, biases in state[1]] == [(1, 2), (1, 2)]


def test_weight_decay_decays_the_linear_conv_kernels_and_neither_gamma_nor_beta():
    rows = _rows(6)
    sgd, decayed = _network(rule=SGD()), _network(rule=WeightDecay(0.1))

    for network in (sgd, decayed):
        network.learn_batch(0.5, rows)

    assert bits(decayed.snapshot()[1]) == bits(sgd.snapshot()[1])
    assert decayed.snapshot()[0] != sgd.snapshot()[0]


def test_a_one_example_training_step_is_refused_naming_the_layer():
    network = _network()
    with pytest.raises(ValueError, match=r"layer 1, BatchNorm\(activation='relu'.*D4"):
        network.learn_batch(0.5, _rows(1))


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
def test_a_checkpoint_resumes_training_by_bits(rule: UpdateRule):
    network = _network("two conv pairs", rule)
    network.learn_batch(0.1, _rows(4))
    checkpoint = network.checkpoint()

    network.learn_batch(0.1, _rows(3, seed=7))
    network.learn_batch(0.1, _rows(2, seed=8))
    trained = bits(network.snapshot())

    network.restore_checkpoint(checkpoint)
    network.learn_batch(0.1, _rows(3, seed=7))
    network.learn_batch(0.1, _rows(2, seed=8))
    assert bits(network.snapshot()) == trained


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", NETWORKS)
def test_training_matches_numpy_within_the_parity_tolerance(name: str, rule: UpdateRule):
    # batch norm computes the same bits in both (test_the_layer_is_numpys_by_bits); the conv and
    # dense layers' sums don't: the builtin sum and example-order accumulation against BLAS. 20
    # steps in: at most 1.1e-12 relative when measured, within every pure-Python parity test's 1e-9
    python = _network(name, rule)
    array = SequentialArrayNetwork(INPUT, NETWORKS[name], rule)
    array.restore(_as_array_snapshot(python))
    rows = _rows(20)

    for step in range(20):
        batch = rows[(step * 4) % 20 :][:4]
        python.learn_batch(0.3, batch)
        array.learn_batch(0.3, batch)

    for expected, actual in zip(_as_array_snapshot(python), array.snapshot()):
        for values, array_values in zip(expected, actual):
            np.testing.assert_allclose(array_values, values, rtol=1e-9, atol=1e-9)
