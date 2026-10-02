"""
Layer norm in pure Python (the layer-norm and attention workplan, stage 3; README, Layer norm and
attention): LayerNormLayer against hand-computed values and against numpy's layer by bits, over
tokens and over a flat layer; the gradient check on dense networks with a flat LayerNorm; learn
against a batch of one; and parity with numpy after 50 steps. The cases are
tests/test_layer_norm_array_network.py's.
"""

import math
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.array_backend import NUMPY
from indrajala_ml.model.layer_norm_array_layer import LayerNormArrayLayer
from indrajala_ml.model.layer_norm_layer import LayerNormLayer
from indrajala_ml.model.layer_specs import LayerSpec, batch_norm_index
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.sequential_backprop_network import SequentialMultiClassBackpropClassifierNetwork
from indrajala_ml.model.state_layer import StateLayer
from indrajala_ml.model.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from indrajala_ml.pcg64 import default_rng
from tests.gradient_check import check_gradients
from tests.test_attention_python_network import (
    as_array_snapshot,
    assert_every_step_has_numpys_gradients,
    downstream,
)
from tests.test_batch_norm_python_network import _bits
from tests.test_layer_norm_array_network import rows
from tests.test_layer_specs import FLAT_LAYER_NORM, _input_shape


def network(specs: list[LayerSpec], rule: UpdateRule | None = None) -> Any:
    built = SequentialMultiClassBackpropClassifierNetwork(_input_shape(specs), specs, SGD() if rule is None else rule)
    built.rng = default_rng(3)
    built.randomize()
    return built


def _layer(tokens: int, features: int, epsilon: float) -> tuple[LayerNormLayer, StateLayer]:
    size = tokens * features
    inputs = StateLayer(size, [(-1e20, 1e20)] * size)
    return LayerNormLayer(inputs, tokens, features, epsilon), inputs


def test_a_layer_norm_matches_hand_computed_values():
    # x = [1, 3]: mu 2, c [-1, 1], var 1, and epsilon 3 makes std exactly 2
    layer, inputs = _layer(1, 2, 3.0)
    layer.restore_state([([2.0], 1.0), ([4.0], -1.0)])
    inputs.update_state((1.0, 3.0))
    layer.forward()
    assert [node.value() for node in layer.nodes] == [0.0, 1.0]

    # dxhat [2, 4], m1 3, m2 0.5: dx ((2 - 3) + 0.25) / 2 and ((4 - 3) - 0.25) / 2
    layer.compute_hidden_deltas(downstream([1.0, 1.0]))
    assert [layer.downstream_sum(i) for i in range(2)] == [-0.375, 0.375]
    layer.accumulate_gradients()
    assert [(channel.weight_gradient_accum, channel.bias_gradient_accum) for channel in layer.channels] == [
        ([-0.5], 1.0),
        ([0.5], 1.0),
    ]


@pytest.mark.parametrize("tokens", [1, 3], ids=["flat", "tokens"])
def test_a_layer_norm_is_numpys_by_bits(tokens: int):
    # every sum a fold in both, with no product: the same bits, example by example against the batch
    features, n, epsilon = 7, 4, 1e-5
    rng = np.random.default_rng(5)
    gamma, beta = rng.uniform(0.5, 1.5, features), rng.uniform(-0.5, 0.5, features)
    # values of mixed scales, so a fold order other than the README's would show
    X = rng.uniform(-1.0, 1.0, (n, tokens * features)) * 10.0 ** rng.integers(-3, 4, (n, tokens * features))
    delta = rng.uniform(-1.0, 1.0, (n, tokens * features))

    array = LayerNormArrayLayer(tokens, features, epsilon)
    array.gamma, array.beta = gamma, beta
    Y = array.forward_batch(X)
    array.delta_batch = delta
    dX = array.downstream_batch()
    array.accumulate_gradient_batch(X)

    layer, inputs = _layer(tokens, features, epsilon)
    layer.restore_state([([g], b) for g, b in zip(gamma.tolist(), beta.tolist(), strict=True)])
    for i in range(n):
        inputs.update_state(tuple(X[i].tolist()))
        layer.forward()
        assert _bits([node.value() for node in layer.nodes]) == _bits(Y[i].tolist())
        layer.compute_hidden_deltas(downstream(delta[i].tolist()))
        assert _bits([layer.downstream_sum(j) for j in range(tokens * features)]) == _bits(dX[i].tolist())
        layer.accumulate_gradients()

    gradients = [channel.weight_gradient_accum[0] for channel in layer.channels]
    assert _bits(gradients) == _bits(array.grad_gamma.tolist())
    assert _bits([channel.bias_gradient_accum for channel in layer.channels]) == _bits(array.grad_beta.tolist())


def _freeze_dropout(built: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # the generator reseeded before each batch pass, so the gradient check's perturbed passes drop
    # the same nodes as its training step: the loss is a function of the weights again
    for name in ("_learn_batch", "_forward_batch_outputs"):
        method = getattr(built, name)

        def seeded(*args: Any, method: Any = method) -> Any:
            built.rng = default_rng(11)
            return method(*args)

        monkeypatch.setattr(built, name, seeded)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("name", FLAT_LAYER_NORM)
def test_every_gradient_matches_its_finite_difference(name: str, batch_size: int, monkeypatch: pytest.MonkeyPatch):
    specs = FLAT_LAYER_NORM[name]
    if batch_size == 1 and batch_norm_index(specs) is not None:
        pytest.skip("batch norm trains on batches only (the batch-norm workplan, D4)")
    built = network(specs)
    # a trained step first, so gamma and beta aren't at their initial values
    built.learn_batch(0.5, rows(specs, 4, seed=2))
    _freeze_dropout(built, monkeypatch)

    batch = rows(specs, batch_size)
    check_gradients(built, [state for state, _ in batch], [label for _, label in batch])


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), Adam()], ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", ["alone", "after sigmoid", "after ReLU", "a body's first layer"])
def test_learn_and_a_learn_batch_of_one_example_agree_by_bits(name: str, rule: UpdateRule):
    specs = FLAT_LAYER_NORM[name]
    single, batched = network(specs, rule), network(specs, rule)
    for state, label in rows(specs, 6, seed=4):
        single.learn(0.5, state, label)
        batched.learn_batch(0.5, [(state, label)])

    assert _bits(single.snapshot()) == _bits(batched.snapshot())


# the rules whose step is linear in the gradient, as for attention
# (tests/test_attention_python_network.py): under Adam a gradient at or under epsilon's size turns
# its rounding into its step's. Measured on "before a block": every step's gradients within 1e-10
# of numpy's, but by step 29 the first layer had one at 3e-8 (g / B), and 50 steps in the two had
# drifted 4.3e-5 apart; every other case stayed within the tolerance below
LINEAR_RULES = [SGD(), Momentum(0.9), WeightDecay(0.01)]


@pytest.mark.parametrize("rule", LINEAR_RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", FLAT_LAYER_NORM)
def test_training_matches_numpy_within_the_dense_layers_rounding(name: str, rule: UpdateRule):
    # layer norm computes the same bits in both (test_a_layer_norm_is_numpys_by_bits); the dense
    # layers don't, as without it, so the networks agree within the tolerance every pure-Python
    # parity test allows. Dropout's masks are the same draws in both
    specs = FLAT_LAYER_NORM[name]
    python = network(specs, rule)
    array = SequentialArrayNetwork(_input_shape(specs), specs, rule, backend=NUMPY)
    array.restore(as_array_snapshot(python))
    array.rng = NUMPY.default_rng(7)
    python.rng = default_rng(7)
    data = rows(specs, 40)

    for step in range(50):
        batch = data[(step * 5) % 40 :][:5]
        python.learn_batch(0.3, batch)
        array.learn_batch(0.3, batch)

    for expected, actual in zip(as_array_snapshot(python), array.snapshot(), strict=True):
        for values, array_values in zip(expected, actual, strict=True):
            np.testing.assert_allclose(array_values, values, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("name", FLAT_LAYER_NORM)
def test_every_step_under_adam_has_numpys_gradients(name: str):
    specs = FLAT_LAYER_NORM[name]
    python = network(specs, Adam())
    array = SequentialArrayNetwork(_input_shape(specs), specs, SGD(), backend=NUMPY)
    assert_every_step_has_numpys_gradients(python, array, rows(specs, 40))


def test_a_layer_norm_after_a_conv_front_end_normalizes_the_flat_image_as_one_token():
    built = network(FLAT_LAYER_NORM["after a conv front end"])
    norm = built.trainable_layers[2]
    assert isinstance(norm, LayerNormLayer)
    assert (norm.tokens, norm.features) == (1, math.prod((3, 3, 2)))
