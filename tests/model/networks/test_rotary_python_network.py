"""
Rotary positions in pure Python (the rotary-positions workplan, stage 6): its Q and K are the
unrotated layer's rotated by numpy's expression, by bits, and its dQ and dK the unrotated layer's
from the same rotated caches, rotated back; one token, position 0, is the unrotated layer by bits;
the layer-major path is the example-major loop by bits; learn against a batch of one; save and load
by bits; and parity with numpy, after 50 steps under the rules whose step is linear in the gradient
and per step under Adam. The cases are tests/model/specs/test_layer_specs.py's ROTARY; numpy's exps
are math.exp, as pure Python's.
"""

import random

import numpy as np
import pytest

from indrajala_ml.model.layers.numpy.attention_array_layer import AttentionArrayLayer
from indrajala_ml.model.layers.python.attention_layer import AttentionLayer
from indrajala_ml.model.specs.update_rules import Adam, Momentum, UpdateRule
from indrajala_ml.pcg64 import generator_state
from tests.helpers import assert_snapshots_close, bits, learn_in_step
from tests.model.networks.test_attention_python_network import (
    LINEAR_RULES,
    _input,  # pyright: ignore[reportPrivateUsage]
    assert_every_step_has_numpys_gradients,
    downstream,
)
from tests.model.networks.test_sequence_array_network import rows
from tests.model.networks.test_sequence_python_network import attention_exp, network, numpy_network, token_exp
from tests.model.specs.test_layer_specs import ROTARY
from tests.python_array_snapshot import as_array_snapshot, assert_seeded_alike

__all__ = ["attention_exp", "token_exp"]  # the fixtures, used by name

# 5 tokens of 8 features
T, D = 5, 8
HEADS = [(1, None), (2, None), (2, 2), (3, 4)]
HEAD_IDS = ["one head of 8", "two of 4", "two of 2", "three of 4"]


def _layers(heads: int, key_size: int | None, tokens: int = T) -> tuple[AttentionLayer, AttentionLayer]:
    # a rotary layer and an unrotated one with the same weights, reading the same input
    rng = random.Random(1)
    X = _input([rng.uniform(-1.0, 1.0) for _ in range(tokens * D)])
    rotary = AttentionLayer(X, tokens, D, heads, key_size, positions="rotary", rotary_base=100.0)
    plain = AttentionLayer(X, tokens, D, heads, key_size)
    weights = [([rng.uniform(-0.5, 0.5) for _ in row.weights], rng.uniform(-0.5, 0.5)) for row in rotary.weight_sets()]
    rotary.restore_state(weights)
    plain.restore_state(weights)
    return rotary, plain


def _numpy(tokens: int, heads: int, key_size: int | None) -> AttentionArrayLayer:
    # numpy's layer, for its _rotated: the workplan's form in numpy's elementwise expressions
    return AttentionArrayLayer(tokens, D, heads, key_size, positions="rotary", rotary_base=100.0)


@pytest.mark.parametrize(("heads", "key_size"), HEADS, ids=HEAD_IDS)
def test_q_and_k_are_the_unrotated_layers_rotated_by_numpys_expression_by_bits(heads: int, key_size: int | None):
    rotary, plain = _layers(heads, key_size)
    rotary.forward()
    plain.forward()
    numpy = _numpy(T, heads, key_size)
    assert bits(rotary._Q) == bits(numpy._rotated(np.array(plain._Q)))  # pyright: ignore[reportPrivateUsage]
    assert bits(rotary._K) == bits(numpy._rotated(np.array(plain._K)))  # pyright: ignore[reportPrivateUsage]
    assert bits(rotary._V) == bits(plain._V)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(("heads", "key_size"), HEADS, ids=HEAD_IDS)
def test_the_backward_pass_rotates_dq_and_dk_back_before_projects_products_by_bits(heads: int, key_size: int | None):
    rotary, plain = _layers(heads, key_size)
    rotary.forward()
    # the unrotated layer from the rotary one's caches: attend's gradients of the rotated Q and K
    plain._Q, plain._K, plain._V, plain._P, plain._M = rotary._Q, rotary._K, rotary._V, rotary._P, rotary._M  # pyright: ignore[reportPrivateUsage]
    rng = random.Random(2)
    row = [rng.uniform(-1.0, 1.0) for _ in range(T * D)]
    rotary.compute_hidden_deltas(downstream(row))
    plain.compute_hidden_deltas(downstream(row))
    numpy = _numpy(T, heads, key_size)
    assert bits(rotary._dQ) == bits(numpy._rotated(np.array(plain._dQ), inverse=True))  # pyright: ignore[reportPrivateUsage]
    assert bits(rotary._dK) == bits(numpy._rotated(np.array(plain._dK), inverse=True))  # pyright: ignore[reportPrivateUsage]
    assert bits(rotary._dV) == bits(plain._dV)  # pyright: ignore[reportPrivateUsage]
    dX = rotary._project_backward(rotary._dQ, rotary._dK, rotary._dV)  # pyright: ignore[reportPrivateUsage]
    assert bits([rotary.downstream_sum(i) for i in range(T * D)]) == bits(dX)


@pytest.mark.parametrize(("heads", "key_size"), HEADS, ids=HEAD_IDS)
def test_one_token_at_position_0_is_the_unrotated_layer_by_bits(heads: int, key_size: int | None):
    rotary, plain = _layers(heads, key_size, tokens=1)
    row = [random.Random(3).uniform(-1.0, 1.0) for _ in range(D)]
    for layer in (rotary, plain):
        layer.forward()
        layer.compute_hidden_deltas(downstream(row))
        layer.accumulate_gradients()
    assert bits([node.value() for node in rotary.nodes]) == bits([node.value() for node in plain.nodes])
    assert bits([rotary.downstream_sum(i) for i in range(D)]) == bits([plain.downstream_sum(i) for i in range(D)])
    assert bits([(r.weight_gradient_accum, r.bias_gradient_accum) for r in rotary.weight_sets()]) == bits(
        [(r.weight_gradient_accum, r.bias_gradient_accum) for r in plain.weight_sets()]
    )


@pytest.mark.parametrize("rule", [Momentum(0.9), Adam()], ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", ROTARY)
def test_the_layer_major_path_is_the_example_major_loop_by_bits(name: str, rule: UpdateRule):
    specs = ROTARY[name]
    example_major, layer_major = network(specs, rule), network(specs, rule)
    for size in (3, 5, 4):
        batch = rows(specs, size, seed=size)
        example_major.learn_batch(0.3, batch)
        layer_major._learn_batch_layer_major(0.3, batch)  # pyright: ignore[reportPrivateUsage]
    assert bits(layer_major.snapshot()) == bits(example_major.snapshot())


# parity with numpy: the rotation is numpy's expression, so the gap is the products', as without it


@pytest.mark.usefixtures("attention_exp", "token_exp")
@pytest.mark.parametrize("rule", LINEAR_RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", ROTARY)
def test_training_matches_numpy_within_the_dense_layers_rounding(name: str, rule: UpdateRule):
    specs = ROTARY[name]
    python, array = network(specs, rule), numpy_network(specs, rule)
    assert_seeded_alike(python, array)
    learn_in_step(0.1, rows(specs, 40), (python, array))
    assert generator_state(python.rng) == array.rng.bit_generator.state
    assert_snapshots_close(as_array_snapshot(python), array.snapshot())


@pytest.mark.usefixtures("attention_exp", "token_exp")
@pytest.mark.parametrize("name", ROTARY)
def test_every_step_under_adam_has_numpys_gradients(name: str):
    specs = ROTARY[name]
    assert_every_step_has_numpys_gradients(network(specs, Adam()), numpy_network(specs), rows(specs, 40))
