"""
Rotary positions on Rust (the rotary-positions workplan, stage 5): the layer hands the crate
rotary_table's bits (D4); its Q and K are the unrotated layer's rotated by numpy's expression, by
bits, and its dQ and dK the downstream op's rotated back; one token, position 0, is the unrotated
layer by bits; learn against a batch of one; save and load by bits; and parity with numpy, after 50
steps under the rules whose step is linear in the gradient and per step under Adam. The cases are
tests/model/specs/test_layer_specs.py's ROTARY; the gradient check is
tests/model/networks/test_rotary_array_network.py's, on numpy.
"""

from pathlib import Path

import numpy as np
import pytest

from indrajala_ml.model.layers.numpy.attention_array_layer import AttentionArrayLayer
from indrajala_ml.model.layers.rust.attention_rust_array_layer import AttentionRustArrayLayer
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.model.specs.rotary_table import rotary_table
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule
from tests.helpers import assert_learn_and_a_batch_of_one_agree, bits, to_numpy
from tests.model.networks.test_attention_rust_network import (
    LINEAR_RULES,
    _assert_every_adam_step_has_numpys_gradients,  # pyright: ignore[reportPrivateUsage]
    _assert_training_matches_numpy,  # pyright: ignore[reportPrivateUsage]
)
from tests.model.networks.test_sequence_array_network import rows
from tests.model.networks.test_sequence_rust_network import (  # pyright: ignore[reportPrivateUsage]
    _array,
    attention_exp,
    network,
    token_exp,
)
from tests.model.specs.test_layer_specs import ROTARY

__all__ = ["attention_exp", "token_exp"]  # the fixtures, used by name

# a batch of 3 examples of 5 tokens of 8 features
N, T, D = 3, 5, 8
HEADS = [(1, None), (2, None), (2, 2), (3, 4)]
HEAD_IDS = ["one head of 8", "two of 4", "two of 2", "three of 4"]


def _layers(
    heads: int, key_size: int | None, tokens: int = T
) -> tuple[AttentionRustArrayLayer, AttentionRustArrayLayer]:
    # a rotary layer and an unrotated one with the same parameters
    rotary = AttentionRustArrayLayer(tokens, D, heads, key_size, positions="rotary", rotary_base=100.0)
    plain = AttentionRustArrayLayer(tokens, D, heads, key_size)
    rng = np.random.default_rng(0)
    parameters = [_array(rng.uniform(-1.0, 1.0, to_numpy(p).shape)) for p in rotary.parameters()]
    rotary.set_parameters(parameters)
    plain.set_parameters(parameters)
    return rotary, plain


def _numpy_rotation(tokens: int, heads: int, key_size: int | None) -> AttentionArrayLayer:
    # numpy's layer, for its _rotated: the workplan's form in numpy's elementwise expressions
    return AttentionArrayLayer(tokens, D, heads, key_size, positions="rotary", rotary_base=100.0)


def _X(tokens: int = T, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1.0, 1.0, (N, tokens * D))


def test_the_layer_hands_the_crate_rotary_tables_bits():
    rotary, plain = _layers(2, None)
    cos, sin = rotary_table(T, 4, 100.0)
    assert bits(rotary._table["cos"]) == bits(cos)  # pyright: ignore[reportPrivateUsage]
    assert bits(rotary._table["sin"]) == bits(sin)  # pyright: ignore[reportPrivateUsage]
    assert plain._table == {}  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(("heads", "key_size"), HEADS, ids=HEAD_IDS)
def test_q_and_k_are_the_unrotated_layers_rotated_by_numpys_expression_by_bits(heads: int, key_size: int | None):
    rotary, plain = _layers(heads, key_size)
    X = _array(_X())
    rotary.forward_batch(X)
    plain.forward_batch(X)
    numpy = _numpy_rotation(T, heads, key_size)
    Q, K = to_numpy(plain._Q), to_numpy(plain._K)  # pyright: ignore[reportPrivateUsage]
    assert bits(rotary._Q) == bits(numpy._rotated(Q))  # pyright: ignore[reportPrivateUsage]
    assert bits(rotary._K) == bits(numpy._rotated(K))  # pyright: ignore[reportPrivateUsage]
    assert bits(rotary._V) == bits(plain._V)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(("heads", "key_size"), HEADS, ids=HEAD_IDS)
def test_one_examples_pass_is_its_batch_of_ones(heads: int, key_size: int | None):
    rotary, _ = _layers(heads, key_size)
    X = _X()
    one = to_numpy(rotary.forward(_array(X[0])))
    batch = to_numpy(rotary.forward_batch(_array(X)))
    assert bits(one) == bits(batch[0])


@pytest.mark.parametrize(("heads", "key_size"), HEADS, ids=HEAD_IDS)
def test_one_token_at_position_0_is_the_unrotated_layer_by_bits(heads: int, key_size: int | None):
    rotary, plain = _layers(heads, key_size, tokens=1)
    X, delta = _array(_X(tokens=1)), _array(_X(tokens=1, seed=2))
    for layer in (rotary, plain):
        layer.forward_batch(X)
        layer._backward(delta)  # pyright: ignore[reportPrivateUsage]
        layer.delta_batch = delta
        layer.accumulate_gradient_batch(X)
    assert bits(rotary.A) == bits(plain.A)
    assert bits(rotary.downstream_batch()) == bits(plain.downstream_batch())
    assert bits(list(rotary.gradients())) == bits(list(plain.gradients()))


def test_the_downstream_rotates_dq_and_dk_back():
    # against the unrotated op's gradients from the same rotated caches, rotated back by numpy
    rotary, plain = _layers(2, None)
    X, delta = _array(_X()), _array(_X(seed=2))
    rotary.forward_batch(X)
    rotary._backward(delta)  # pyright: ignore[reportPrivateUsage]
    for name in ("_Q", "_K", "_V", "_P", "_M"):
        setattr(plain, name, getattr(rotary, name))
    plain._backward(delta)  # pyright: ignore[reportPrivateUsage]
    numpy = _numpy_rotation(T, 2, None)
    assert bits(rotary._dQ) == bits(numpy._rotated(to_numpy(plain._dQ), inverse=True))  # pyright: ignore[reportPrivateUsage]
    assert bits(rotary._dK) == bits(numpy._rotated(to_numpy(plain._dK), inverse=True))  # pyright: ignore[reportPrivateUsage]
    assert bits(rotary._dV) == bits(plain._dV)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), Adam()], ids=["sgd", "momentum", "adam"])
@pytest.mark.parametrize("name", ROTARY)
def test_learn_and_a_batch_of_one_agree(name: str, rule: UpdateRule):
    specs = ROTARY[name]
    assert_learn_and_a_batch_of_one_agree("rust", network(specs, rule), network(specs, rule), rows(specs, 4))


@pytest.mark.parametrize("name", ROTARY)
def test_a_saved_rotary_network_loads_and_trains_on_by_bits(name: str, tmp_path: Path):
    specs = ROTARY[name]
    built = network(specs, Adam())
    data = rows(specs, 6)
    built.learn_batch(0.1, data[:3])
    path = str(tmp_path / "rotary.json")
    built.save(path)
    loaded = load_network(path)
    assert loaded.layer_specs == specs
    built.learn_batch(0.1, data[3:])
    loaded.learn_batch(0.1, data[3:])
    assert bits([to_numpy(a) for entry in loaded.snapshot() for a in entry]) == bits(
        [to_numpy(a) for entry in built.snapshot() for a in entry]
    )


# parity with numpy: the rotation is elementwise, so the gap is the products', as without it


@pytest.mark.usefixtures("attention_exp", "token_exp")
@pytest.mark.parametrize("rule", LINEAR_RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", ROTARY)
def test_training_matches_numpy_within_the_dense_layers_rounding(name: str, rule: UpdateRule):
    _assert_training_matches_numpy(ROTARY[name], rule, 0.1, network, rows)


@pytest.mark.usefixtures("attention_exp", "token_exp")
@pytest.mark.parametrize("name", ROTARY)
def test_every_step_under_adam_has_numpys_gradients(name: str):
    _assert_every_adam_step_has_numpys_gradients(ROTARY[name], 0.1, network, rows)
