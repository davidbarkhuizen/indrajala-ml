"""
The gated FFN, GatedDense (the FFN activations workplan, D3, D4), on each implementation: its
parameters are one (2 * size, d) weight, the gate's rows then the value's, and one (2 * size,)
bias, one pair to the optimizers with the weight decayed and the bias not, and at d 64 a SwiGLU FFN
of size 170 has 33,044 parameters to a ReLU FFN's 33,088 at size 256 (D4). Its gradient check,
parity, persistence and decoding run with the shared spec tables (tests/model/specs/test_layer_specs.py:
TOKENS, SEQUENCE, DROPOUT) and tests/generation.
"""

from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.numpy.token_array_layer import GatedDenseArrayLayer
from indrajala_ml.model.layers.python.token_layer import GatedDenseLayer
from indrajala_ml.model.layers.rust.token_rust_array_layer import GatedDenseRustArrayLayer
from indrajala_ml.model.specs.layer_specs import Dense, Embedding, GatedDense, LayerNorm, LayerSpec, Residual
from tests.helpers import Implementation, randomized

D = 64
TOKENS = 4
OUTPUT = Dense(7, output=True, activation="softmax", loss="cross_entropy")


def _model(hidden: LayerSpec) -> list[LayerSpec]:
    return [Embedding(7, D), Residual((LayerNorm(), hidden, Dense(D, activation="linear", bias=True))), OUTPUT]


def _ffn_parameters(implementation: Implementation, hidden: LayerSpec) -> int:
    # the FFN's hidden layer's and its affine layer's parameters: expanded layers 3 and 4 (Embedding,
    # Fork, LayerNorm, hidden, affine, Add, output)
    network = randomized(implementation, (TOKENS,), _model(hidden), shape="sequence")
    snapshot = network.snapshot()
    return sum(
        int(np.size(np.array(value.tolist() if hasattr(value, "tolist") else value)))
        for i in (3, 4)
        for value in _flat(snapshot[i])
    )


def _flat(entry: Any) -> list[Any]:
    # a layer's snapshot entry, its arrays or (pure Python) its (weights, bias) rows, as arrays
    if entry and isinstance(entry[0], tuple):
        return [np.array(weights + [bias]) for weights, bias in entry]
    return list(entry)


@pytest.mark.parametrize(("hidden", "expected"), [(Dense(256, activation="relu"), 33_088), (GatedDense(170), 33_044)])
def test_d4s_parameter_parity_at_d_64(hidden: LayerSpec, expected: int, implementation: Implementation):
    assert _ffn_parameters(implementation, hidden) == expected


@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_tanh"])
def test_the_array_layers_are_one_decayed_weight_and_an_undecayed_bias(activation: Any):
    for layer in (GatedDenseArrayLayer(5, 6, TOKENS, activation), GatedDenseRustArrayLayer(5, 6, TOKENS, activation)):
        weight, bias = layer.parameters()
        assert tuple(weight.shape) == (10, 6) and tuple(bias.shape) == (10,)
        assert layer.decayed == (True, False)
        assert tuple(layer.gradients()[0].shape) == (10, 6)


def test_the_pure_python_layer_has_twice_its_size_in_decayed_units():
    class _Input:
        nodes = [object()] * (TOKENS * 6)

    layer = GatedDenseLayer(_Input(), 5, TOKENS, "silu")  # pyright: ignore[reportArgumentType]
    assert len(layer.units) == 10 and len(layer.nodes) == TOKENS * 5
    assert all(unit.weights_decayed and unit.has_bias for unit in layer.units)
