"""
Patch models and flat layer norms on Rust (the layer-norm and attention workplan, stage 4; README,
Layer norm and attention): randomize's draws against numpy's by bits, the exact tests (one token,
uniform attention, an identity attention block; with heads, identical and silent heads too), the
layer before a LayerNorm taking its downstream through a mask op (D5), and parity with numpy: after
50 steps under the rules whose step is linear in the gradient, and per step under Adam, one head
and many (the multi-head attention workplan, stage 5). The cases are
tests/model/specs/test_layer_specs.py's and tests/model/networks/test_attention_network.py's
MULTI_HEAD; the gradient check, wiring and learn against a batch of one are
tests/model/networks/test_attention_network.py's (patch models) and
tests/model/networks/test_layer_norm_network.py's (flat layer norms).
"""

import math
import random
from typing import Any

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.layers.numpy import attention_array_layer
from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.layers.rust.attention_rust_array_layer import AttentionRustArrayLayer
from indrajala_ml.model.layers.rust.batch_norm_rust_array_layer import BatchNormRustArrayLayer
from indrajala_ml.model.layers.rust.dropout_rust_array_layer import DropoutRustArrayLayer
from indrajala_ml.model.layers.rust.relu_rust_array_layer import ReLURustArrayLayer
from indrajala_ml.model.layers.rust.rust_array_layer import RustArrayLayer
from indrajala_ml.model.layers.rust.token_rust_array_layer import (
    TokenMeanRustArrayLayer,
)
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.layer_specs import BatchNorm, Dense, LayerNorm, LayerSpec, Residual, TokenMean
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from tests.gradient_check import analytic_gradients
from tests.helpers import bits, exp_by_crate, patching, split, to_numpy
from tests.model.networks.test_attention_array_network import HEAD_IDS, HEADS
from tests.model.networks.test_attention_network import MULTI_HEAD
from tests.model.specs.test_layer_specs import (
    AFFINE_5,
    ATTENTION_BLOCK,
    EMBED,
    FFN_BLOCK,
    FLAT_LAYER_NORM,
    LINEAR,
    OUTPUT,
    PATCHES,
    SOFTMAX,
    TOKENS,
    _input_shape,  # pyright: ignore[reportPrivateUsage]
)

# the flat layer norms; the gradient check, wiring and learn against a batch of one are
# tests/model/networks/test_attention_network.py's (patch models) and tests/model/networks/test_layer_norm_network.py's (flat)
FLAT = {f"flat, {name}": specs for name, specs in FLAT_LAYER_NORM.items()}
CASES = TOKENS | FLAT


def network(specs: list[LayerSpec], rule: UpdateRule | None = None, backend: Any = RUST, seed: int = 3) -> Any:
    built = SequentialArrayNetwork(_input_shape(specs), specs, SGD() if rule is None else rule, backend=backend)
    built.rng = backend.default_rng(seed)
    built.randomize()
    return built


def rows(specs: list[LayerSpec], count: int, seed: int = 1) -> list[tuple[tuple[float, ...], int]]:
    rng = random.Random(seed)
    size = math.prod(_input_shape(specs))
    return [(tuple(rng.uniform(-1.0, 1.0) for _ in range(size)), i % 3) for i in range(count)]


def _as_numpy(snapshot: list[tuple[Any, ...]]) -> list[tuple[FloatArray, ...]]:
    return [tuple(to_numpy(array) for array in entry) for entry in snapshot]


# numpy's softmax exp as the crate's (Rust's f64::exp)
crate_exp = patching(attention_array_layer, "exp", exp_by_crate)


@pytest.mark.parametrize("name", CASES)
def test_randomize_draws_numpys_parameters_by_bits(name: str):
    rust, numpy = network(CASES[name]), network(CASES[name], backend=NUMPY)
    assert bits([a for entry in _as_numpy(rust.snapshot()) for a in entry]) == bits(
        [a for entry in numpy.snapshot() for a in entry]
    )


def _attention(
    tokens: int, features: int, seed: int, heads: int = 1, key_size: int | None = None
) -> AttentionRustArrayLayer:
    layer = AttentionRustArrayLayer(tokens, features, heads, key_size)
    rng = np.random.default_rng(seed)
    layer.set_parameters([pa.Array(rng.uniform(-0.5, 0.5, to_numpy(p).shape).tolist()) for p in layer.parameters()])
    return layer


def _head(rows: FloatArray, i: int, width: int) -> FloatArray:
    # head i's columns of packed (N * T, h * width) rows
    return rows[:, i * width : (i + 1) * width]


@pytest.mark.parametrize(("heads", "key_size"), HEADS, ids=HEAD_IDS)
def test_one_token_attends_only_to_itself_so_attention_is_two_affine_maps_by_bits(heads: int, key_size: int | None):
    layer = _attention(1, 6, 1, heads, key_size)
    *_, Wv, bv, Wo, bo = layer.parameters()
    X = pa.Array(np.random.default_rng(2).uniform(-1.0, 1.0, (4, 6)).tolist())

    out = layer.forward_batch(X)
    assert bits(to_numpy(layer._P)) == bits(np.ones((4, heads)))  # pyright: ignore[reportPrivateUsage]
    assert bits(to_numpy(out)) == bits(to_numpy(pa.affine_forward_batch(Wo, pa.affine_forward_batch(Wv, X, bv), bo)))


@pytest.mark.parametrize(
    ("heads", "key_size"), [(1, None), (2, None), (4, None), (4, 3)], ids=["1", "2", "4", "4 of 3"]
)
def test_zero_queries_and_keys_weigh_every_token_exactly_one_sixteenth(heads: int, key_size: int | None):
    # T = 16: every score is 0, every weight 1/16, and each head's H the token mean of its V, the
    # same fold
    layer = _attention(16, 8, 3, heads, key_size)
    parameters = list(layer.parameters())
    for i in range(4):  # Wq, bq, Wk, bk
        parameters[i] = pa.Array(np.zeros(to_numpy(parameters[i]).shape).tolist())
    layer.set_parameters(parameters)
    X = pa.Array(np.random.default_rng(4).uniform(-1.0, 1.0, (3, 16 * 8)).tolist())

    layer.forward_batch(X)
    assert bits(to_numpy(layer._P)) == bits(np.full((48, heads * 16), 1 / 16))  # pyright: ignore[reportPrivateUsage]
    V, H = to_numpy(layer._V), to_numpy(layer._H)  # pyright: ignore[reportPrivateUsage]
    d_k = layer.key_size
    for i in range(heads):
        V_i = pa.Array(_head(V, i, d_k).reshape(3, -1).tolist())
        mean = to_numpy(TokenMeanRustArrayLayer(16, d_k).forward_batch(V_i))
        assert bits(_head(H, i, d_k)) == bits(np.repeat(mean, 16, axis=0))


@pytest.mark.parametrize(("heads", "key_size"), [(2, None), (3, None), (2, 4)], ids=["2", "3", "2 of 4"])
def test_identical_heads_weigh_and_mix_alike_by_bits(heads: int, key_size: int | None):
    layer = _attention(4, 6, 5, heads, key_size)
    d_k = layer.key_size
    parameters = [to_numpy(p) for p in layer.parameters()]
    for i in range(6):  # each head's blocks of Wq, bq, Wk, bk, Wv, bv: head 0's
        parameters[i] = np.concatenate([parameters[i][:d_k]] * heads)
    layer.set_parameters([pa.Array(p.tolist()) for p in parameters])
    X = pa.Array(np.random.default_rng(6).uniform(-1.0, 1.0, (3, 4 * 6)).tolist())

    layer.forward_batch(X)
    P, H = to_numpy(layer._P), to_numpy(layer._H)  # pyright: ignore[reportPrivateUsage]
    for i in range(1, heads):
        assert bits(_head(P, i, 4)) == bits(_head(P, 0, 4))
        assert bits(_head(H, i, d_k)) == bits(_head(H, 0, d_k))


@pytest.mark.parametrize(("heads", "key_size"), [(2, None), (3, None), (2, 4)], ids=["2", "3", "2 of 4"])
def test_a_silent_heads_projection_gradients_are_exactly_zero(heads: int, key_size: int | None):
    # with head i's columns of Wo zero, dH[i] is exactly zero, and so is all that flows from it
    layer = _attention(4, 6, 7, heads, key_size)
    d_k, silent = layer.key_size, heads - 1
    parameters = [to_numpy(p) for p in layer.parameters()]
    parameters[6][:, silent * d_k : (silent + 1) * d_k] = 0.0
    layer.set_parameters([pa.Array(p.tolist()) for p in parameters])
    rng = np.random.default_rng(8)
    X = pa.Array(rng.uniform(-1.0, 1.0, (3, 4 * 6)).tolist())
    delta = pa.Array(rng.uniform(-0.5, 0.5, (3, 4 * 6)).tolist())
    layer.forward_batch(X)
    layer._backward(delta)  # pyright: ignore[reportPrivateUsage]
    layer._accumulate(delta, X)  # pyright: ignore[reportPrivateUsage]

    gradients = [to_numpy(g) for g in layer.gradients()]
    for weight, bias in ((0, 1), (2, 3), (4, 5)):  # Wq, bq; Wk, bk; Wv, bv
        assert not np.any(gradients[weight][silent * d_k : (silent + 1) * d_k])
        assert not np.any(gradients[bias][silent * d_k : (silent + 1) * d_k])
        assert np.any(gradients[weight][: silent * d_k])  # the others' blocks are not


def test_an_identity_attention_block_changes_no_output_and_no_other_layers_gradient_by_bits():
    # with Wo and bo zero the block adds 0 forward, and backward its body's downstream is 0
    plain_specs: list[LayerSpec] = [PATCHES, EMBED, FFN_BLOCK, TokenMean(), LayerNorm(), SOFTMAX]
    plain = network(plain_specs, seed=5)
    plain.learn_batch(0.5, rows(plain_specs, 4, seed=2))  # gamma and beta off their initial values
    block_specs: list[LayerSpec] = [*plain_specs[:2], ATTENTION_BLOCK, *plain_specs[2:]]
    blocked = network(block_specs, seed=7)

    snapshot = plain.snapshot()
    attention = list(blocked.snapshot()[4])
    attention[6], attention[7] = pa.Array.zeros((6, 6)), pa.Array.zeros(6)
    blocked.restore([*snapshot[:2], (), blocked.snapshot()[3], tuple(attention), (), *snapshot[2:]])

    states, labels = split(rows(block_specs, 5))
    for state in states:
        assert bits(to_numpy(blocked._forward(state))) == bits(to_numpy(plain._forward(state)))  # pyright: ignore[reportPrivateUsage]

    outside = [0, 1, *range(6, 14)]
    plain_gradients = analytic_gradients(plain, states, labels)
    blocked_gradients = analytic_gradients(blocked, states, labels)
    assert [np.asarray(g).tobytes() for i in outside for g in blocked_gradients[i]] == [
        np.asarray(g).tobytes() for layer in plain_gradients for g in layer
    ]


# the layer right before a LayerNorm, or a fork whose body starts with one (D5), and its mask
BEFORE_LAYER_NORM: dict[str, tuple[list[LayerSpec], type]] = {
    "sigmoid": ([Dense(5), LayerNorm(), OUTPUT], RustArrayLayer),
    "ReLU": ([Dense(5, activation="relu"), LayerNorm(), OUTPUT], ReLURustArrayLayer),
    "dropout": ([Dense(5, dropout=0.3), LayerNorm(), OUTPUT], DropoutRustArrayLayer),
    "batch norm, sigmoid": ([LINEAR, BatchNorm(), LayerNorm(), OUTPUT], BatchNormRustArrayLayer),
    "sigmoid, a fork": (
        [Dense(5), Residual((LayerNorm(), Dense(8, activation="relu"), AFFINE_5)), OUTPUT],
        RustArrayLayer,
    ),
    "ReLU, a fork": (
        [Dense(5, activation="relu"), Residual((LayerNorm(), Dense(8, activation="relu"), AFFINE_5)), OUTPUT],
        ReLURustArrayLayer,
    ),
    "dropout, a fork": (
        [Dense(5, dropout=0.3), Residual((LayerNorm(), Dense(8, activation="relu"), AFFINE_5)), OUTPUT],
        DropoutRustArrayLayer,
    ),
}


@pytest.mark.parametrize("single", [False, True], ids=["batch", "single"])
@pytest.mark.parametrize("name", BEFORE_LAYER_NORM)
def test_the_layer_before_a_layer_norm_masks_its_downstream(name: str, single: bool):
    specs, kind = BEFORE_LAYER_NORM[name]
    if single and kind is BatchNormRustArrayLayer:
        pytest.skip("batch norm trains on batches only (the batch-norm workplan, D4)")
    built = network(specs)
    index = next(i for i, layer in enumerate(built.layers) if isinstance(layer, kind))
    before, after = built.layers[index], built.layers[index + 1]
    data = rows(specs, 1 if single else 4)
    # a learning rate of 0, so the step leaves gamma, beta and every W as the backward pass read them
    if single:
        built.learn(0.0, *data[0])
    else:
        built.learn_batch(0.0, data)

    downstream = to_numpy(after.downstream() if single else after.downstream_batch())
    delta = to_numpy(before.delta if single else before.delta_batch)
    if isinstance(before, DropoutRustArrayLayer):
        base = to_numpy(before._base_activation if single else before._base_activation_batch)  # pyright: ignore[reportPrivateUsage]
        mask = to_numpy(before._mask if single else before._mask_batch)  # pyright: ignore[reportPrivateUsage]
        expected = downstream * (base * (1.0 - base)) * (mask / 0.7)
    else:
        A = to_numpy(before.a if single else before.A)
        expected = (
            np.where(A > 0.0, downstream, 0.0) if isinstance(before, ReLURustArrayLayer) else downstream * A * (1.0 - A)
        )
    assert bits(delta) == bits(expected)


def _layer_scales(snapshot: list[tuple[FloatArray, ...]]) -> list[float]:
    # each layer's largest parameter: the scale its rounding is measured in. A parameter whose
    # gradient is rounding noise (attention's bk, D6; a layer norm's beta before a batch-norm pair,
    # which the batch norm's mean cancels) stays near 0, so its own scale would say nothing
    return [max((float(np.abs(a).max()) for a in entry if a.size), default=0.0) for entry in snapshot]


LINEAR_RULES = [SGD(), Momentum(0.9), WeightDecay(0.01)]


def _assert_training_matches_numpy(specs: list[LayerSpec], rule: UpdateRule, rate: float) -> None:
    # outside the products both compute the README's expressions by bits (the crate's ops tests),
    # and the products are numpy's BLAS against the crate's, as the dense layers'. Measured after 50
    # steps: within 1.7e-12 of each layer's scale (a flat layer norm after ReLU, Momentum)
    rust, numpy = network(specs, rule), network(specs, rule, backend=NUMPY)
    data = rows(specs, 40)
    for step in range(50):
        batch = data[(step * 5) % 40 :][:5]
        rust.learn_batch(rate, batch)
        numpy.learn_batch(rate, batch)

    expected, actual = numpy.snapshot(), _as_numpy(rust.snapshot())
    for scale, expected_entry, actual_entry in zip(_layer_scales(expected), expected, actual, strict=True):
        for values, rust_values in zip(expected_entry, actual_entry, strict=True):
            np.testing.assert_allclose(rust_values, values, rtol=0.0, atol=1e-10 * scale)


@pytest.mark.usefixtures("crate_exp")
@pytest.mark.parametrize("rule", LINEAR_RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", CASES)
def test_training_matches_numpy_within_the_dense_layers_rounding(name: str, rule: UpdateRule):
    _assert_training_matches_numpy(CASES[name], rule, 0.3)


@pytest.mark.usefixtures("crate_exp")
@pytest.mark.parametrize("rule", LINEAR_RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", MULTI_HEAD)
def test_multi_head_training_matches_numpy_within_the_dense_layers_rounding(name: str, rule: UpdateRule):
    # at 0.1, as pure Python's: two multi-head layers under Momentum diverge at 0.3
    _assert_training_matches_numpy(MULTI_HEAD[name], rule, 0.1)


def _assert_every_adam_step_has_numpys_gradients(specs: list[LayerSpec], rate: float) -> None:
    # Adam's steep step at |g| near epsilon makes the trajectory chaotic (the stage 3 parity tests:
    # numpy against numpy with a one-ulp nudge drifts as far), so each of 50 steps is checked from
    # numpy's weights: Rust's gradients within 1e-10 of each layer's largest gradient (measured
    # 7e-13, the dense part after the mean). Attention's bk, rounding noise (D6), is compared apart:
    # on both sides within 1e-12 of attention's largest gradient (measured 6.9e-14; pure Python's
    # folds against numpy's BLAS gave 4e-16, the crate's products against BLAS more)
    rust, numpy = network(specs, SGD()), network(specs, Adam(), backend=NUMPY)
    data = rows(specs, 40)
    for step in range(50):
        batch = data[(step * 5) % 40 :][:5]
        states, labels = split(batch)
        rust.restore(numpy.snapshot())
        numpy.rng, rust.rng = NUMPY.default_rng(step), RUST.default_rng(step)
        expected = analytic_gradients(numpy, states, labels)
        actual = analytic_gradients(rust, states, labels)
        for layer, numpy_layer, rust_layer in zip(numpy.layers, expected, actual, strict=True):
            scale = max((float(np.abs(np.asarray(g)).max()) for g in numpy_layer), default=0.0)
            for i, (values, rust_values) in enumerate(zip(numpy_layer, rust_layer, strict=True)):
                values, rust_values = np.asarray(values), np.asarray(rust_values)
                if type(layer).__name__ == "AttentionArrayLayer" and i == 3:
                    assert max(np.abs(values).max(), np.abs(rust_values).max()) <= 1e-12 * scale
                    continue
                np.testing.assert_allclose(rust_values, values, rtol=0.0, atol=1e-10 * scale)
        numpy.rng = NUMPY.default_rng(step)
        numpy.learn_batch(rate, batch)


@pytest.mark.usefixtures("crate_exp")
@pytest.mark.parametrize("name", CASES)
def test_every_step_under_adam_has_numpys_gradients(name: str):
    _assert_every_adam_step_has_numpys_gradients(CASES[name], 0.3)


@pytest.mark.usefixtures("crate_exp")
@pytest.mark.parametrize("name", MULTI_HEAD)
def test_every_multi_head_step_under_adam_has_numpys_gradients(name: str):
    _assert_every_adam_step_has_numpys_gradients(MULTI_HEAD[name], 0.1)
