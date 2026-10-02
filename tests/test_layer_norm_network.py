"""
Flat layer norms on each implementation, numpy, Rust and pure Python (README, Layer norm and
attention): the gradient check on dense networks with a flat LayerNorm (after and before sigmoid,
ReLU and dropout layers, in and around residual blocks, after a conv front end), and learn against a
batch of one, which layer norm, unlike batch norm, trains on. The layer itself, and what one
implementation alone has (the parity tests, Rust's masked downstream), stay in
tests/test_layer_norm_{array,python}_network.py and tests/test_attention_rust_network.py.
"""

from typing import Any

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model.array_layer import FloatArray
from indrajala_ml.model.dropout_array_layer import DropoutArrayLayer
from indrajala_ml.model.dropout_rust_array_layer import DropoutRustArrayLayer
from indrajala_ml.model.specs.layer_specs import LayerSpec
from indrajala_ml.model.specs.single_example import batch_norm_index
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule
from indrajala_ml.pcg64 import default_rng
from tests.gradient_check import check_gradients
from tests.helpers import Implementation, bits, randomized, split
from tests.model.specs.test_layer_specs import FLAT_LAYER_NORM, _input_shape  # pyright: ignore[reportPrivateUsage]
from tests.test_layer_norm_array_network import rows


def network(implementation: Implementation, specs: list[LayerSpec], rule: UpdateRule | None = None) -> Any:
    return randomized(implementation, _input_shape(specs), specs, rule)


class _FrozenGenerator:
    """A numpy dropout layer's generator drawing the same values on every pass."""

    def random(self, shape: tuple[int, int]) -> FloatArray:
        return np.random.default_rng(11).random(shape)


def _freeze(layer: DropoutRustArrayLayer) -> None:
    # a fresh generator before every forward pass (the crate draws the mask from a crate Generator
    # itself)
    forward, forward_batch = layer.forward, layer.forward_batch

    def frozen(x: Any) -> Any:
        layer.set_rng(pa.default_rng(11))
        return forward(x)

    def frozen_batch(X: Any) -> Any:
        layer.set_rng(pa.default_rng(11))
        return forward_batch(X)

    layer.forward, layer.forward_batch = frozen, frozen_batch  # pyright: ignore[reportAttributeAccessIssue]


def _freeze_dropout(implementation: Implementation, built: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dropout draws the same mask on every pass, so the gradient check's perturbed passes drop the
    same nodes as its training step: the loss is a function of the weights again."""
    if implementation == "python":
        # the network's generator reseeded before each batch pass
        for name in ("_learn_batch", "_forward_batch_outputs"):
            method = getattr(built, name)

            def seeded(*args: Any, method: Any = method) -> Any:
                built.rng = default_rng(11)
                return method(*args)

            monkeypatch.setattr(built, name, seeded)
        return
    for layer in built.layers:
        if isinstance(layer, DropoutArrayLayer):
            layer.set_rng(_FrozenGenerator())  # pyright: ignore[reportArgumentType]
        elif isinstance(layer, DropoutRustArrayLayer):
            _freeze(layer)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("name", FLAT_LAYER_NORM)
def test_every_gradient_matches_its_finite_difference(
    name: str, batch_size: int, implementation: Implementation, monkeypatch: pytest.MonkeyPatch
):
    specs = FLAT_LAYER_NORM[name]
    if batch_size == 1 and batch_norm_index(specs) is not None:
        pytest.skip("batch norm trains on batches only (the batch-norm workplan, D4)")
    built = network(implementation, specs)
    _freeze_dropout(implementation, built, monkeypatch)
    # a trained step first, so gamma and beta aren't at their initial values
    built.learn_batch(0.5, rows(specs, 4, seed=2))

    check_gradients(built, *split(rows(specs, batch_size)))


@pytest.mark.parametrize("rule", [SGD(), Momentum(0.9), Adam()], ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("name", ["alone", "after sigmoid", "after ReLU", "a body's first layer", "before a block"])
def test_learn_and_a_learn_batch_of_one_example_agree(name: str, rule: UpdateRule, implementation: Implementation):
    specs = FLAT_LAYER_NORM[name]
    single, batched = network(implementation, specs, rule), network(implementation, specs, rule)
    for state, label in rows(specs, 6, seed=4):
        single.learn(0.5, state, label)
        batched.learn_batch(0.5, [(state, label)])

    if implementation == "numpy":
        # layer norm's single-example pass is its batch pass on a batch of one, but a dense layer's
        # W @ x and X @ W.T are different BLAS calls, which can differ in the last bit
        # (test_residual_network)
        for one, other in zip(single.snapshot(), batched.snapshot(), strict=True):
            for a, b in zip(one, other, strict=True):
                assert a == pytest.approx(b, rel=1e-12, abs=1e-15)
    else:
        # by bits: pure Python's single-example step is its batch of one's, and so are Rust's layer
        # norm's single-example ops and the crate's dense ops (W @ x against X @ W.T, outer against
        # delta^T X at one row)
        assert bits(single.snapshot()) == bits(batched.snapshot())
