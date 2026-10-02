"""
Patch models and flat layer norms on Rust (the layer-norm and attention workplan, stage 4; README,
Layer norm and attention): the layers the builder wires, randomize's draws against numpy's by bits,
the gradient check on every accepted patch model and flat layer norm, the exact tests (one token,
uniform attention, an identity attention block), learn against a batch of one, the layer before a
LayerNorm taking its downstream through a mask op (D5), and parity with numpy: after 50 steps under
the rules whose step is linear in the gradient, and per step under Adam. The cases are
tests/test_layer_specs.py's.
"""

import math
import random
from typing import Any

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model import attention_array_layer
from indrajala_ml.model.array_backend import NUMPY, RUST
from indrajala_ml.model.array_layer import FloatArray
from indrajala_ml.model.attention_rust_array_layer import AttentionRustArrayLayer
from indrajala_ml.model.batch_norm_rust_array_layer import BatchNormRustArrayLayer
from indrajala_ml.model.dropout_rust_array_layer import DropoutRustArrayLayer
from indrajala_ml.model.layer_norm_rust_array_layer import LayerNormRustArrayLayer
from indrajala_ml.model.layer_specs import BatchNorm, Dense, LayerNorm, LayerSpec, Residual, TokenMean, batch_norm_index
from indrajala_ml.model.relu_rust_array_layer import ReLURustArrayLayer
from indrajala_ml.model.residual_rust_array_layer import AddRustArrayLayer, ForkRustArrayLayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.token_rust_array_layer import (
    PatchesRustArrayLayer,
    PositionRustArrayLayer,
    TokenDenseRustArrayLayer,
    TokenMeanRustArrayLayer,
)
from indrajala_ml.model.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from tests.gradient_check import analytic_gradients, check_gradients
from tests.test_layer_specs import (
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
from tests.test_residual_array_network import bits

CASES = TOKENS | {f"flat, {name}": specs for name, specs in FLAT_LAYER_NORM.items()}


def network(specs: list[LayerSpec], rule: UpdateRule | None = None, backend: Any = RUST, seed: int = 3) -> Any:
    built = SequentialArrayNetwork(_input_shape(specs), specs, SGD() if rule is None else rule, backend=backend)
    built.rng = backend.default_rng(seed)
    built.randomize()
    return built


def rows(specs: list[LayerSpec], count: int, seed: int = 1) -> list[tuple[tuple[float, ...], int]]:
    rng = random.Random(seed)
    size = math.prod(_input_shape(specs))
    return [(tuple(rng.uniform(-1.0, 1.0) for _ in range(size)), i % 3) for i in range(count)]


def _split(batch: list[tuple[tuple[float, ...], int]]) -> tuple[list[tuple[float, ...]], list[int]]:
    return [state for state, _ in batch], [label for _, label in batch]


def _numpy(values: Any) -> FloatArray:
    return np.array(values.tolist())


def _as_numpy(snapshot: list[tuple[Any, ...]]) -> list[tuple[FloatArray, ...]]:
    return [tuple(_numpy(array) for array in entry) for entry in snapshot]


@pytest.fixture
def crate_exp(monkeypatch: pytest.MonkeyPatch) -> None:
    # numpy's softmax exp as the crate's (Rust's f64::exp): np.exp picks its implementation by CPU
    # and can differ in the last bit
    def exp_with_crate_exp(values: FloatArray) -> FloatArray:
        return _numpy(pa.exp(pa.Array(values.reshape(-1).tolist()))).reshape(values.shape)

    monkeypatch.setattr(attention_array_layer, "exp", exp_with_crate_exp)


def _freeze(layer: DropoutRustArrayLayer) -> None:
    # a fresh generator before every forward pass, so the gradient check's perturbed passes drop the
    # same nodes as its training step (the crate draws the mask from a crate Generator itself)
    forward, forward_batch = layer.forward, layer.forward_batch

    def frozen(x: Any) -> Any:
        layer.set_rng(pa.default_rng(11))
        return forward(x)

    def frozen_batch(X: Any) -> Any:
        layer.set_rng(pa.default_rng(11))
        return forward_batch(X)

    layer.forward, layer.forward_batch = frozen, frozen_batch  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("name", CASES)
def test_every_gradient_matches_its_finite_difference(name: str, batch_size: int):
    if batch_size == 1 and batch_norm_index(CASES[name]) is not None:
        pytest.skip("batch norm trains on batches only (the batch-norm workplan, D4)")
    built = network(CASES[name])
    for layer in built.layers:
        if isinstance(layer, DropoutRustArrayLayer):
            _freeze(layer)
    # a trained step first, so gamma, beta and the positions aren't at their initial values
    built.learn_batch(0.5, rows(CASES[name], 4, seed=2))

    check_gradients(built, *_split(rows(CASES[name], batch_size)))


def test_the_readme_model_builds_its_layers_wired_together():
    built = network(TOKENS["the README's model"])
    kinds: list[type] = [type(layer) for layer in built.layers]
    assert kinds == [
        PatchesRustArrayLayer,
        TokenDenseRustArrayLayer,
        PositionRustArrayLayer,
        ForkRustArrayLayer,
        LayerNormRustArrayLayer,
        AttentionRustArrayLayer,
        AddRustArrayLayer,
        ForkRustArrayLayer,
        LayerNormRustArrayLayer,
        TokenDenseRustArrayLayer,
        TokenDenseRustArrayLayer,
        AddRustArrayLayer,
        TokenMeanRustArrayLayer,
        LayerNormRustArrayLayer,
        type(built.layers[-1]),
    ]
    fork, norm, add = built.layers[3], built.layers[4], built.layers[6]
    assert fork.body_first is norm and fork.add is add
    assert [len(entry) for entry in built.snapshot()] == [0, 2, 1, 0, 2, 8, 0, 0, 2, 2, 2, 0, 0, 2, 2]


@pytest.mark.parametrize("name", CASES)
def test_randomize_draws_numpys_parameters_by_bits(name: str):
    rust, numpy = network(CASES[name]), network(CASES[name], backend=NUMPY)
    assert bits([a for entry in _as_numpy(rust.snapshot()) for a in entry]) == bits(
        [a for entry in numpy.snapshot() for a in entry]
    )


def _attention(tokens: int, features: int, seed: int) -> AttentionRustArrayLayer:
    layer = AttentionRustArrayLayer(tokens, features)
    rng = np.random.default_rng(seed)
    layer.set_parameters([pa.Array(rng.uniform(-0.5, 0.5, _numpy(p).shape).tolist()) for p in layer.parameters()])
    return layer


def test_one_token_attends_only_to_itself_so_attention_is_two_affine_maps_by_bits():
    layer = _attention(1, 5, 1)
    *_, Wv, bv, Wo, bo = layer.parameters()
    X = pa.Array(np.random.default_rng(2).uniform(-1.0, 1.0, (4, 5)).tolist())

    out = layer.forward_batch(X)
    assert bits(_numpy(layer._P)) == bits(np.ones((4, 1)))  # pyright: ignore[reportPrivateUsage]
    assert bits(_numpy(out)) == bits(_numpy(pa.affine_forward_batch(Wo, pa.affine_forward_batch(Wv, X, bv), bo)))


def test_zero_queries_and_keys_weigh_every_token_exactly_one_sixteenth():
    # T = 16: every score is 0, every weight 1/16, and H is the token mean of V, the same fold
    layer = _attention(16, 8, 3)
    parameters = list(layer.parameters())
    for i in range(4):  # Wq, bq, Wk, bk
        parameters[i] = pa.Array(np.zeros(_numpy(parameters[i]).shape).tolist())
    layer.set_parameters(parameters)
    X = pa.Array(np.random.default_rng(4).uniform(-1.0, 1.0, (3, 16 * 8)).tolist())

    layer.forward_batch(X)
    assert bits(_numpy(layer._P)) == bits(np.full((48, 16), 1 / 16))  # pyright: ignore[reportPrivateUsage]
    V, H = _numpy(layer._V), _numpy(layer._H)  # pyright: ignore[reportPrivateUsage]
    mean = _numpy(TokenMeanRustArrayLayer(16, 8).forward_batch(pa.Array(V.reshape(3, -1).tolist())))
    assert bits(H) == bits(np.repeat(mean, 16, axis=0))


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

    states, labels = _split(rows(block_specs, 5))
    for state in states:
        assert bits(_numpy(blocked._forward(state))) == bits(_numpy(plain._forward(state)))  # pyright: ignore[reportPrivateUsage]

    outside = [0, 1, *range(6, 14)]
    plain_gradients = analytic_gradients(plain, states, labels)
    blocked_gradients = analytic_gradients(blocked, states, labels)
    assert [np.asarray(g).tobytes() for i in outside for g in blocked_gradients[i]] == [
        np.asarray(g).tobytes() for layer in plain_gradients for g in layer
    ]


SINGLE = ["the README's model", "token-wise layers after a block", "flat, after sigmoid", "flat, before a block"]


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), Adam()], ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", SINGLE)
def test_learn_and_a_learn_batch_of_one_example_agree_by_bits(name: str, rule: UpdateRule):
    # each token layer's and layer norm's single-example ops give a batch of one's bits, and so do
    # the crate's dense ops (W @ x against X @ W.T, outer against delta^T X at one row)
    single, batched = network(CASES[name], rule), network(CASES[name], rule)
    for state, label in rows(CASES[name], 6, seed=4):
        single.learn(0.5, state, label)
        batched.learn_batch(0.5, [(state, label)])

    assert bits([a for entry in _as_numpy(single.snapshot()) for a in entry]) == bits(
        [a for entry in _as_numpy(batched.snapshot()) for a in entry]
    )


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

    downstream = _numpy(after.downstream() if single else after.downstream_batch())
    delta = _numpy(before.delta if single else before.delta_batch)
    if isinstance(before, DropoutRustArrayLayer):
        base = _numpy(before._base_activation if single else before._base_activation_batch)  # pyright: ignore[reportPrivateUsage]
        mask = _numpy(before._mask if single else before._mask_batch)  # pyright: ignore[reportPrivateUsage]
        expected = downstream * (base * (1.0 - base)) * (mask / 0.7)
    else:
        A = _numpy(before.a if single else before.A)
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


@pytest.mark.usefixtures("crate_exp")
@pytest.mark.parametrize("rule", LINEAR_RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", CASES)
def test_training_matches_numpy_within_the_dense_layers_rounding(name: str, rule: UpdateRule):
    # outside the products both compute the README's expressions by bits (the crate's ops tests),
    # and the products are numpy's BLAS against the crate's, as the dense layers'. Measured after 50
    # steps: within 1.7e-12 of each layer's scale (a flat layer norm after ReLU, Momentum)
    rust, numpy = network(CASES[name], rule), network(CASES[name], rule, backend=NUMPY)
    data = rows(CASES[name], 40)
    for step in range(50):
        batch = data[(step * 5) % 40 :][:5]
        rust.learn_batch(0.3, batch)
        numpy.learn_batch(0.3, batch)

    expected, actual = numpy.snapshot(), _as_numpy(rust.snapshot())
    for scale, expected_entry, actual_entry in zip(_layer_scales(expected), expected, actual, strict=True):
        for values, rust_values in zip(expected_entry, actual_entry, strict=True):
            np.testing.assert_allclose(rust_values, values, rtol=0.0, atol=1e-10 * scale)


@pytest.mark.usefixtures("crate_exp")
@pytest.mark.parametrize("name", CASES)
def test_every_step_under_adam_has_numpys_gradients(name: str):
    # Adam's steep step at |g| near epsilon makes the trajectory chaotic (the stage 3 parity tests:
    # numpy against numpy with a one-ulp nudge drifts as far), so each of 50 steps is checked from
    # numpy's weights: Rust's gradients within 1e-10 of each layer's largest gradient (measured
    # 7e-13, the dense part after the mean). Attention's bk, rounding noise (D6), is compared apart:
    # on both sides within 1e-12 of attention's largest gradient (measured 6.9e-14; pure Python's
    # folds against numpy's BLAS gave 4e-16, the crate's products against BLAS more)
    rust, numpy = network(CASES[name], SGD()), network(CASES[name], Adam(), backend=NUMPY)
    data = rows(CASES[name], 40)
    for step in range(50):
        batch = data[(step * 5) % 40 :][:5]
        states, labels = _split(batch)
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
        numpy.learn_batch(0.3, batch)
