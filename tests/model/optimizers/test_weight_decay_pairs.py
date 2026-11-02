"""
Weight decay reaches every decayed parameter on both array backends, a layer of three parameters
included: the GRU cell's (W, U, b), whose W and U are decayed and b isn't. The Rust optimizer steps
parameters in fused-op pairs whose second member is stepped as a bias, so a decayed parameter
never takes that place (rust_optimizer._pairs).

With zero gradients, a step leaves only the decay: AdamW's w * (1 - lr * λ), and WeightDecay's
w - lr * λ * w; an undecayed parameter doesn't move.
"""

from collections.abc import Callable
from typing import Any

import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.layers.numpy.recurrent_array_layer import GRUArrayLayer
from indrajala_ml.model.layers.rust.recurrent_rust_array_layer import GRURustArrayLayer
from indrajala_ml.model.specs.update_rules import Adam, UpdateRule, WeightDecay

RATE = 0.1
DECAY = 0.5


def _cell(backend: Any) -> Any:
    cell: Any = GRUArrayLayer(3, 2) if backend is NUMPY else GRURustArrayLayer(3, 2)
    cell.draw(backend.default_rng(7))
    return cell


def _adamw(w: float) -> float:
    return w * (1.0 - RATE * DECAY)


def _weight_decay(w: float) -> float:
    return w - RATE * (DECAY * w)


@pytest.mark.parametrize("backend", [NUMPY, RUST], ids=["numpy", "rust"])
@pytest.mark.parametrize(
    "rule, expected",
    [(Adam(weight_decay=DECAY), _adamw), (WeightDecay(DECAY), _weight_decay)],
    ids=["adamw", "weight_decay"],
)
def test_every_decayed_parameter_of_a_gru_cell_is_decayed_and_its_bias_is_not(
    backend: Any, rule: UpdateRule, expected: Callable[[float], float]
):
    cell = _cell(backend)
    before = [parameter.tolist() for parameter in cell.parameters()]
    backend.optimizer(rule).step([cell], RATE, 1)

    for after, initial, decayed in zip(cell.parameters(), before, cell.decayed, strict=True):
        if not decayed:
            assert after.tolist() == initial
            continue
        assert after.tolist() == [[expected(w) for w in row] for row in initial]


@pytest.mark.parametrize("rule", [Adam(weight_decay=DECAY), WeightDecay(DECAY)], ids=["adamw", "weight_decay"])
def test_the_two_backends_decay_a_gru_cell_alike(rule: UpdateRule):
    cells = [_cell(backend) for backend in (NUMPY, RUST)]
    for cell, backend in zip(cells, (NUMPY, RUST), strict=True):
        backend.optimizer(rule).step([cell], RATE, 1)
    numpy_cell, rust_cell = cells
    for a, b in zip(numpy_cell.parameters(), rust_cell.parameters(), strict=True):
        assert a.tolist() == b.tolist()
