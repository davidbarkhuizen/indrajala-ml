"""
Layer norm in numpy (the layer-norm and attention workplan, stage 2; README, Layer norm and
attention): LayerNormArrayLayer against hand-computed values and a scalar transcription of the
README's expressions, by bits, over tokens and over a flat layer; the gradient check on dense
networks with a flat LayerNorm (after and before sigmoid, ReLU and dropout layers, in and around
residual blocks, after a conv front end); and learn against a batch of one, which layer norm,
unlike batch norm, trains on.
"""

import math
import random
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.array_backend import NUMPY
from indrajala_ml.model.array_layer import FloatArray
from indrajala_ml.model.dropout_array_layer import DropoutArrayLayer
from indrajala_ml.model.layer_norm_array_layer import LayerNormArrayLayer
from indrajala_ml.model.layer_specs import LayerSpec, batch_norm_index
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.update_rules import SGD, Adam, Momentum, UpdateRule
from tests.gradient_check import check_gradients
from tests.helpers import bits
from tests.test_layer_specs import FLAT_LAYER_NORM, _input_shape  # pyright: ignore[reportPrivateUsage]


def network(specs: list[LayerSpec], rule: UpdateRule | None = None) -> Any:
    built = SequentialArrayNetwork(_input_shape(specs), specs, SGD() if rule is None else rule, backend=NUMPY)
    built.rng = NUMPY.default_rng(3)
    built.randomize()
    return built


def rows(specs: list[LayerSpec], count: int, seed: int = 1) -> list[tuple[tuple[float, ...], int]]:
    rng = random.Random(seed)
    size = math.prod(_input_shape(specs))
    return [(tuple(rng.uniform(-1.0, 1.0) for _ in range(size)), i % 3) for i in range(count)]


def test_a_layer_norm_matches_hand_computed_values():
    # x = [1, 3]: mu 2, c [-1, 1], var 1, and epsilon 3 makes std exactly 2
    layer = LayerNormArrayLayer(1, 2, 3.0)
    layer.gamma, layer.beta = np.array([2.0, 4.0]), np.array([1.0, -1.0])
    assert layer.forward_batch(np.array([[1.0, 3.0]])).tolist() == [[0.0, 1.0]]

    # dxhat [2, 4], m1 3, m2 0.5: dx ((2 - 3) + 0.25) / 2 and ((4 - 3) - 0.25) / 2
    layer.delta_batch = np.array([[1.0, 1.0]])
    assert layer.downstream_batch().tolist() == [[-0.375, 0.375]]
    layer.accumulate_gradient_batch(np.array([[1.0, 3.0]]))
    assert (layer.grad_gamma.tolist(), layer.grad_beta.tolist()) == ([-0.5, 0.5], [1.0, 1.0])


def _sum(values: list[float]) -> float:
    total = 0.0
    for value in values:
        total += value
    return total


def _reference(x: list[float], gamma: list[float], beta: list[float], delta: list[float], epsilon: float) -> Any:
    """The README's expressions for one token, in Python floats, in their grouping."""
    d = len(x)
    mu = _sum(x) / d
    c = [x_j - mu for x_j in x]
    var = _sum([c_j * c_j for c_j in c]) / d
    std = math.sqrt(var + epsilon)
    xhat = [c_j / std for c_j in c]
    y = [g * xhat_j + b for g, xhat_j, b in zip(gamma, xhat, beta)]
    dxhat = [delta_j * g for delta_j, g in zip(delta, gamma)]
    m1 = _sum(dxhat) / d
    m2 = _sum([dxhat_j * xhat_j for dxhat_j, xhat_j in zip(dxhat, xhat)]) / d
    dx = [((dxhat_j - m1) - xhat_j * m2) / std for dxhat_j, xhat_j in zip(dxhat, xhat)]
    return y, dx, xhat


@pytest.mark.parametrize("tokens", [1, 3], ids=["flat", "tokens"])
def test_a_layer_norm_computes_the_readmes_expressions_by_bits(tokens: int):
    features, n, epsilon = 7, 4, 1e-5
    rng = np.random.default_rng(5)
    layer = LayerNormArrayLayer(tokens, features, epsilon)
    layer.gamma, layer.beta = rng.uniform(0.5, 1.5, features), rng.uniform(-0.5, 0.5, features)
    # values of mixed scales, so a fold order other than the README's would show
    X = rng.uniform(-1.0, 1.0, (n, tokens * features)) * 10.0 ** rng.integers(-3, 4, (n, tokens * features))
    delta = rng.uniform(-1.0, 1.0, (n, tokens * features))

    Y = layer.forward_batch(X)
    layer.delta_batch = delta
    dX = layer.downstream_batch()
    layer.accumulate_gradient_batch(X)

    gamma, beta = layer.gamma.tolist(), layer.beta.tolist()
    expected_y: list[float] = []
    expected_dx: list[float] = []
    grad_gamma, grad_beta = [0.0] * features, [0.0] * features
    for i in range(n):
        for t in range(tokens):
            span = slice(t * features, (t + 1) * features)
            y, dx, xhat = _reference(X[i, span].tolist(), gamma, beta, delta[i, span].tolist(), epsilon)
            expected_y += y
            expected_dx += dx
            # summed over the (N * T, d) rows, examples then tokens
            for j in range(features):
                grad_gamma[j] += delta[i, t * features + j] * xhat[j]
                grad_beta[j] += delta[i, t * features + j]

    assert bits(Y) == bits(np.array(expected_y).reshape(Y.shape))
    assert bits(dX) == bits(np.array(expected_dx).reshape(dX.shape))
    # the layer's gradients start at zero, so its += adds a fold onto 0.0, as the reference does
    assert bits([layer.grad_gamma, layer.grad_beta]) == bits([np.array(grad_gamma), np.array(grad_beta)])


class _FrozenGenerator:
    """A dropout layer's generator drawing the same values on every pass, so the gradient check's
    perturbed passes drop the same nodes as its training step."""

    def random(self, shape: tuple[int, int]) -> FloatArray:
        return np.random.default_rng(11).random(shape)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("name", FLAT_LAYER_NORM)
def test_every_gradient_matches_its_finite_difference(name: str, batch_size: int):
    specs = FLAT_LAYER_NORM[name]
    if batch_size == 1 and batch_norm_index(specs) is not None:
        pytest.skip("batch norm trains on batches only (the batch-norm workplan, D4)")
    built = network(specs)
    for layer in built.layers:
        if isinstance(layer, DropoutArrayLayer):
            layer.set_rng(_FrozenGenerator())  # pyright: ignore[reportArgumentType]
    # a trained step first, so gamma and beta aren't at their initial values
    built.learn_batch(0.5, rows(specs, 4, seed=2))

    batch = rows(specs, batch_size)
    check_gradients(built, [state for state, _ in batch], [label for _, label in batch])


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), Adam()], ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", ["alone", "after sigmoid", "after ReLU", "a body's first layer"])
def test_learn_and_a_learn_batch_of_one_example_agree(name: str, rule: UpdateRule):
    specs = FLAT_LAYER_NORM[name]
    single, batched = network(specs, rule), network(specs, rule)
    for state, label in rows(specs, 6, seed=4):
        single.learn(0.5, state, label)
        batched.learn_batch(0.5, [(state, label)])

    # layer norm's single-example pass is its batch pass on a batch of one, but a dense layer's
    # W @ x and X @ W.T are different BLAS calls, which can differ in the last bit
    # (test_residual_array_network)
    for one, other in zip(single.snapshot(), batched.snapshot(), strict=True):
        for a, b in zip(one, other, strict=True):
            assert a == pytest.approx(b, rel=1e-12, abs=1e-15)
