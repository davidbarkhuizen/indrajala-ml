"""
Residual blocks on the array backends (the residual-connections workplan, stages 2 and 4; README,
Residual connections): randomize's draws, identity blocks (an affine layer of zeros makes a block
the identity, by bits, in the forward and the backward pass), and the fork's input left as it was.
The cases are shared: the gradient check, the block's layers and learn against a batch of one are
tests/test_residual_network.py's, on every implementation.
"""

import random
from typing import Any, Literal

import numpy as np
import pytest

from indrajala_ml.model.array_backend import NUMPY, RUST
from indrajala_ml.model.layer_specs import BatchNorm, Dense, LayerSpec, Residual
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.update_rules import SGD, UpdateRule
from tests.gradient_check import analytic_gradients
from tests.helpers import bits

INPUT = 4

Shape = Literal["multiclass", "single_output"]
BACKENDS = [NUMPY, RUST]


def affine(size: int = INPUT) -> Dense:
    return Dense(size, activation="linear", bias=True)


def output(shape: Shape) -> Dense:
    return Dense(3, output=True) if shape == "multiclass" else Dense(1, output=True)


def block(*body: LayerSpec) -> Residual:
    return Residual((*body, affine()))


# hidden layers, then the shape's output layer
NETWORKS: dict[str, list[LayerSpec]] = {
    "sigmoid body": [Dense(INPUT), block(Dense(5))],
    "relu body": [Dense(INPUT, activation="relu"), block(Dense(5, activation="relu"))],
    "affine body": [Dense(INPUT), Residual((affine(),))],
    "two blocks in a row": [Dense(INPUT), block(Dense(5)), block(Dense(3, activation="relu"))],
    "block first": [block(Dense(5)), Dense(6)],
    "batch norm in the body": [
        Dense(INPUT, activation="relu"),
        block(Dense(5, activation="linear"), BatchNorm("relu")),
    ],
    "batch norm before the block": [Dense(INPUT, activation="linear"), BatchNorm(), block(Dense(5))],
    # batch norm's ReLU branch reads the fork's downstream (on Rust, its lazy sum)
    "relu batch norm before the block": [Dense(INPUT, activation="linear"), BatchNorm("relu"), block(Dense(5))],
    "dropout before the block": [Dense(INPUT, dropout=0.0), block(Dense(5, dropout=0.0))],
}


def network(name: str, shape: Shape = "multiclass", rule: UpdateRule | None = None, backend: Any = NUMPY) -> Any:
    specs = [*NETWORKS[name], output(shape)]
    built = SequentialArrayNetwork((INPUT,), specs, SGD() if rule is None else rule, shape=shape, backend=backend)
    built.rng = backend.default_rng(3)
    built.randomize()
    return built


def rows(count: int, shape: str = "multiclass", seed: int = 1) -> list[tuple[tuple[float, ...], Any]]:
    rng = random.Random(seed)
    return [
        (tuple(rng.uniform(-1.0, 1.0) for _ in range(INPUT)), float(i % 2) if shape == "single_output" else i % 3)
        for i in range(count)
    ]


def test_randomize_draws_w_then_b_per_weighted_layer_and_nothing_for_the_fork_or_add():
    built = network("sigmoid body")
    rng = np.random.default_rng(3)
    expected: list[tuple[Any, Any]] = []
    for rows_, fan_in in [(INPUT, INPUT), (5, INPUT), (INPUT, 5), (3, INPUT)]:
        limit = 1.0 / np.sqrt(fan_in)
        expected.append((rng.uniform(-limit, limit, (rows_, fan_in)), rng.uniform(-limit, limit, rows_)))

    weighted = [entry for entry in built.snapshot() if entry]
    assert bits(weighted) == bits(expected)


def _identity_pair(name: str, shape: Shape, backend: Any) -> tuple[Any, Any]:
    # a network with a block whose affine layer is zero, and the same network without the block,
    # sharing every other layer's weights
    plain_specs = [Dense(INPUT), Dense(6), output(shape)]
    plain = SequentialArrayNetwork((INPUT,), plain_specs, SGD(), shape=shape, backend=backend)
    plain.rng = backend.default_rng(5)
    plain.randomize()
    body = {"sigmoid": Dense(5), "relu": Dense(5, activation="relu")}[name]
    specs = [Dense(INPUT), block(body), Dense(6), output(shape)]
    residual = SequentialArrayNetwork((INPUT,), specs, SGD(), shape=shape, backend=backend)
    residual.rng = backend.default_rng(7)
    residual.randomize()

    first, second, out = plain.snapshot()
    snapshot = residual.snapshot()
    end = snapshot[3]
    residual.restore(
        [first, (), snapshot[2], tuple(np.zeros_like(np.asarray(a.tolist())) for a in end), (), second, out]
    )
    return plain, residual


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend.name)
@pytest.mark.parametrize("shape", ["multiclass", "single_output"])
@pytest.mark.parametrize("body", ["sigmoid", "relu"])
def test_an_identity_block_changes_no_output_and_no_other_layers_gradient_by_bits(
    body: str, shape: Shape, backend: Any
):
    # x + 0 = x forward; backward the body's downstream is 0, so the fork's delta is 0 + skip = skip
    plain, residual = _identity_pair(body, shape, backend)
    batch = rows(5, shape)
    states, labels = [state for state, _ in batch], [label for _, label in batch]

    for state in states:
        assert bits(residual._forward(state)) == bits(plain._forward(state))
    prepared = plain.prepare_dataset(batch)
    assert residual.classify_rows(residual.prepare_dataset(batch)) == plain.classify_rows(prepared)

    plain_gradients = analytic_gradients(plain, states, labels)
    residual_gradients = analytic_gradients(residual, states, labels)
    outside = [residual_gradients[i] for i in (0, 5, 6)]
    assert [np.asarray(g).tobytes() for layer in outside for g in layer] == [
        np.asarray(g).tobytes() for layer in plain_gradients for g in layer
    ]

    # and one example's step (after it the affine layer isn't zero, so the block isn't the identity)
    plain.learn(0.5, *batch[0])
    residual.learn(0.5, *batch[0])
    trained = residual.snapshot()
    assert bits([trained[i] for i in (0, 5, 6)]) == bits(plain.snapshot())


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda backend: backend.name)
def test_the_fork_passes_its_input_on_and_nothing_writes_it_in_place(backend: Any):
    built = network("two blocks in a row", backend=backend)
    batch = rows(5)
    X = backend.matrix([state for state, _ in batch])
    before = bits(X)

    built._set_training_mode(True)
    try:
        hidden = built.layers[0].forward_batch(X)
        kept = bits(hidden)
        assert built.layers[1].forward_batch(hidden) is hidden
    finally:
        built._set_training_mode(False)

    built.learn_batch(0.5, batch)
    built.learn(0.5, *batch[0])
    assert bits(X) == before and bits(hidden) == kept
