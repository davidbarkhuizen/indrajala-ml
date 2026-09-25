"""
RustArrayLayer.forward_batch computes X @ W.T with the crate's matmul_nt: one dot product per
output, the same calls W @ x makes in forward. So each row of forward_batch is bit-identical to
forward on that row, for every dense Rust layer. Checked exactly (tolist() ==, not approx),
including the one-row batch.
"""

import importlib
import pkgutil
from collections.abc import Callable

import indrajala_math_rust as pa
import numpy as np
import pytest

import indrajala_ml.model
from indrajala_ml.model.adam_rust_array_layer import AdamRustArrayLayer
from indrajala_ml.model.cross_entropy_rust_array_layer import CrossEntropyRustArrayLayer
from indrajala_ml.model.dropout_rust_array_layer import DropoutRustArrayLayer
from indrajala_ml.model.l2_rust_array_layer import L2RustArrayLayer
from indrajala_ml.model.momentum_rust_array_layer import MomentumRustArrayLayer
from indrajala_ml.model.relu_rust_array_layer import ReLURustArrayLayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from indrajala_ml.model.softmax_rust_array_layer import SoftmaxRustArrayLayer
from tests.helpers import all_subclasses

LAYER_CLASSES: dict[str, Callable[[int, int], RustArrayLayer]] = {
    "plain": lambda size, input_size: RustArrayLayer(size, input_size),
    "relu": lambda size, input_size: ReLURustArrayLayer(size, input_size),
    "softmax": lambda size, input_size: SoftmaxRustArrayLayer(size, input_size),
    "cross-entropy": lambda size, input_size: CrossEntropyRustArrayLayer(size, input_size),
    "dropout": lambda size, input_size: DropoutRustArrayLayer(size, input_size, 0.5),
    "momentum": lambda size, input_size: MomentumRustArrayLayer(size, input_size, 0.9),
    "adam": lambda size, input_size: AdamRustArrayLayer(size, input_size, 0.9, 0.999, 1e-8),
    "l2": lambda size, input_size: L2RustArrayLayer(size, input_size, 0.01),
}

# (size, input_size): a small layer, the dense production layers (784 -> 30 -> 10), and the conv
# tail (the dense layer after a ConvSpec(3, 8) layer on 28x28 input)
SHAPES: list[tuple[int, int]] = [(7, 11), (30, 784), (10, 30), (32, 5408)]


def _layer(name: str, size: int, input_size: int, rng: np.random.Generator) -> RustArrayLayer:
    layer = LAYER_CLASSES[name](size, input_size)
    layer.W = pa.Array(rng.uniform(-0.3, 0.3, (size, input_size)).tolist())
    layer.b = pa.Array(rng.uniform(-0.3, 0.3, size).tolist())
    return layer


@pytest.mark.parametrize("name", LAYER_CLASSES)
@pytest.mark.parametrize("size, input_size", SHAPES)
@pytest.mark.parametrize("batch", [1, 5])
def test_forward_batch_rows_are_bit_identical_to_forward(name: str, size: int, input_size: int, batch: int):
    rng = np.random.default_rng(size * 10_000 + input_size + batch)
    layer = _layer(name, size, input_size, rng)
    X = rng.uniform(0.0, 1.0, (batch, input_size))

    rows = layer.forward_batch(pa.Array(X.tolist())).tolist()
    assert len(rows) == batch
    for i in range(batch):
        assert rows[i] == layer.forward(pa.Array(X[i].tolist())).tolist()


@pytest.mark.parametrize("size, input_size", SHAPES)
def test_dropout_training_base_activation_rows_are_bit_identical_to_forward(size: int, input_size: int):
    # at training time the mask is random, but the pre-mask sigmoid each call keeps is not
    rng = np.random.default_rng(size + input_size)
    layer = _layer("dropout", size, input_size, rng)
    assert isinstance(layer, DropoutRustArrayLayer)
    layer.set_training_mode(True)
    X = rng.uniform(0.0, 1.0, (3, input_size))

    layer.forward_batch(pa.Array(X.tolist()))
    rows = layer._base_activation_batch.tolist()
    for i in range(3):
        layer.forward(pa.Array(X[i].tolist()))
        assert rows[i] == layer._base_activation.tolist()


def test_every_dense_rust_layer_is_covered():
    for module in pkgutil.iter_modules(indrajala_ml.model.__path__):
        importlib.import_module(f"indrajala_ml.model.{module.name}")
    covered = {type(factory(2, 3)) for factory in LAYER_CLASSES.values()}
    assert {RustArrayLayer, *all_subclasses(RustArrayLayer)} == covered
