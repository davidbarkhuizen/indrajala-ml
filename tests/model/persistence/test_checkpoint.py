"""
checkpoint()/restore_checkpoint() (model/persistence/checkpoint.py) and the optimizers' state()/load_state():
a restored checkpoint resumes training by bits, for every update rule in all three
implementations, dense, conv and a causal sequence model, also after crossing a worker boundary as nested lists
(ensemble_train._picklable_checkpoint); a checkpoint is a copy; and train.py's pocket restores the
best epoch's optimizer state along with its weights.
"""

import pickle
import random
from typing import Any

import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.networks.python.backprop_network_base import BackpropNetworkBase
from indrajala_ml.model.networks.python.sequential_backprop_network import (
    SequentialMultiClassBackpropClassifierNetwork,
    SequentialSequenceBackpropNetwork,
)
from indrajala_ml.model.networks.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.specs.layer_specs import (
    Attention,
    Conv,
    Dense,
    Dropout,
    Embedding,
    LayerNorm,
    LayerSpec,
    Pool,
    Position,
    Residual,
    token_wise_output,
)
from indrajala_ml.model.specs.spec_shapes import InputShape
from indrajala_ml.model.specs.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from indrajala_ml.pcg64 import default_rng
from indrajala_ml.training.ensemble_train import _picklable_checkpoint  # pyright: ignore[reportPrivateUsage]
from indrajala_ml.training.train import train_backprop_network_mini_batch
from tests.helpers import bits

RULES: list[UpdateRule] = [SGD(), Momentum(0.9), Adam(), WeightDecay(0.01)]
IMPLEMENTATIONS = ["python", "numpy", "rust"]

DENSE: tuple[InputShape, list[LayerSpec]] = ((4,), [Dense(5), Dense(3, output=True)])
# a pool layer between weighted ones: the optimizers key state by layer index, and skip the pool
CONV: tuple[InputShape, list[LayerSpec]] = ((6, 6, 1), [Conv(2, 3), Pool(2), Dense(3, output=True)])
# a dropout layer: its masks draw from the network's generator, which a checkpoint holds
DROPOUT: tuple[InputShape, list[LayerSpec]] = ((4,), [Dense(5, dropout=0.25), Dense(3, output=True)])
# a causal sequence model over 5 token ids of a vocabulary of 7: its label is one class per token
SEQUENCE: tuple[InputShape, list[LayerSpec]] = (
    (5,),
    [
        Embedding(7, 4),
        Position(),
        Residual((LayerNorm(), Attention(heads=2, causal=True))),
        Dense(7, output=True, activation="softmax", loss="cross_entropy"),
    ],
)
# the sequence model with GPT's three dropouts (the attention-dropout workplan, D7): the masks of the
# attention weights and the token dropouts draw from the network's generator too
ATTENTION_DROPOUT: tuple[InputShape, list[LayerSpec]] = (
    (5,),
    [
        Embedding(7, 4),
        Position(),
        Dropout(0.2),
        Residual((LayerNorm(), Attention(heads=2, causal=True, dropout=0.2), Dropout(0.2))),
        Dense(7, output=True, activation="softmax", loss="cross_entropy"),
    ],
)


def _network(implementation: str, input_shape: InputShape, layers: list[LayerSpec], rule: UpdateRule) -> Any:
    sequence = token_wise_output(layers)
    if implementation == "python":
        cls = SequentialSequenceBackpropNetwork if sequence else SequentialMultiClassBackpropClassifierNetwork
        return cls(input_shape, layers, rule)
    backend = NUMPY if implementation == "numpy" else RUST
    return SequentialArrayNetwork(input_shape, layers, rule, "sequence" if sequence else "multiclass", backend)


def _seeded(network: Any, seed: int) -> Any:
    # the network's own generator, seeded
    pure_python = isinstance(network, BackpropNetworkBase)
    network.rng = default_rng(seed) if pure_python else network.backend.default_rng(seed)
    return network


def _rows(
    input_shape: InputShape, count: int, seed: int, layers: list[LayerSpec] | None = None
) -> list[tuple[tuple[float, ...], Any]]:
    # over an Embedding, token ids and a class per token
    rng = random.Random(seed)
    dimension = 1
    for extent in input_shape:
        dimension *= extent
    if layers and isinstance(layers[0], Embedding):
        output = layers[-1]
        assert isinstance(output, Dense)
        vocabulary, classes = layers[0].vocabulary, output.size
        return [
            (
                tuple(float(rng.randrange(vocabulary)) for _ in range(dimension)),
                tuple(rng.randrange(classes) for _ in range(dimension)),
            )
            for _ in range(count)
        ]
    return [(tuple(rng.random() for _ in range(dimension)), rng.randrange(3)) for _ in range(count)]


def _state_bits(network: Any) -> Any:
    state = network.optimizer.state()
    return [bits(network.snapshot()), state.t, bits(state.layers)]


def _train(network: Any, rows: list[tuple[tuple[float, ...], Any]]) -> None:
    # batches, then single examples: both of the optimizer's paths, apply and step_single
    network.learn_batch(0.1, rows[:4])
    network.learn_batch(0.1, rows[4:])
    for state, category in rows[:3]:
        network.learn(0.1, state, category)


def assert_a_restored_checkpoint_resumes_by_bits(
    implementation: str, input_shape: InputShape, layers: list[LayerSpec], rule: UpdateRule, across_workers: bool
) -> Any:
    """
    A network trains on, from a checkpoint (pickled as lists when across_workers), and another
    network restored from it trains as far; returns the first once their states agree by bits.
    """
    rows = _rows(input_shape, 8, seed=1, layers=layers)

    trained = _seeded(_network(implementation, input_shape, layers, rule), 2)
    trained.randomize()
    _train(trained, rows)
    checkpoint = trained.checkpoint()
    if across_workers:
        checkpoint = pickle.loads(pickle.dumps(_picklable_checkpoint(checkpoint)))
    _train(trained, rows)

    resumed = _seeded(  # other weights and another generator, both overwritten by the checkpoint
        _network(implementation, input_shape, layers, rule), 3
    )
    resumed.randomize()
    resumed.restore_checkpoint(checkpoint)
    _train(resumed, rows)

    assert _state_bits(resumed) == _state_bits(trained)
    return trained


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize(
    "architecture",
    [DENSE, CONV, DROPOUT, SEQUENCE, ATTENTION_DROPOUT],
    ids=["dense", "conv", "dropout", "sequence", "attention dropout"],
)
@pytest.mark.parametrize("across_workers", [False, True], ids=["in_memory", "as_lists"])
def test_a_restored_checkpoint_resumes_training_by_bits(
    implementation: str, rule: UpdateRule, architecture: tuple[InputShape, list[LayerSpec]], across_workers: bool
):
    trained = assert_a_restored_checkpoint_resumes_by_bits(implementation, *architecture, rule, across_workers)
    assert trained.optimizer.t == 2 * 5


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("rule", [Momentum(0.9), Adam()], ids=lambda rule: type(rule).__name__)
def test_a_checkpoint_is_a_copy(implementation: str, rule: UpdateRule):
    input_shape, layers = DENSE
    rows = _rows(input_shape, 8, seed=1)
    input_shape, layers = DROPOUT
    rows = _rows(input_shape, 8, seed=1)
    network = _seeded(_network(implementation, input_shape, layers, rule), 2)
    network.randomize()
    _train(network, rows)

    def held() -> list[Any]:
        return [bits(checkpoint.weights), checkpoint.optimizer.t, bits(checkpoint.optimizer.layers), checkpoint.rng]

    checkpoint = network.checkpoint()
    taken = held()
    _train(network, rows)  # steps the network's weights, state and generator, not the checkpoint's
    network.restore_checkpoint(checkpoint)
    _train(network, rows)  # nor, after restoring it, the network's steps

    assert held() == taken


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_restoring_a_checkpoint_rewinds_the_generator_in_place(implementation: str):
    # the dropout layers hold the network's generator, so it's rewound, not replaced
    input_shape, layers = DROPOUT
    network = _seeded(_network(implementation, input_shape, layers, Adam()), 2)
    generator = network.rng
    checkpoint = network.checkpoint()
    first = [generator.random() if implementation == "python" else generator.random(4).tolist() for _ in range(2)]

    network.restore_checkpoint(checkpoint)

    assert network.rng is generator
    assert [
        generator.random() if implementation == "python" else generator.random(4).tolist() for _ in range(2)
    ] == first


def test_a_checkpoint_before_the_first_step_restores_fresh_state():
    input_shape, layers = DENSE
    rows = _rows(input_shape, 8, seed=1)
    network = _network("numpy", input_shape, layers, Adam())
    checkpoint = network.checkpoint()
    assert checkpoint.optimizer.t == 0 and checkpoint.optimizer.layers == {}

    _train(network, rows)
    network.restore_checkpoint(checkpoint)
    assert network.optimizer.t == 0 and network.optimizer.state().layers == {}


@pytest.mark.parametrize(
    "implementation, architecture",
    [("numpy", DENSE), ("python", SEQUENCE), ("numpy", SEQUENCE), ("rust", SEQUENCE)],
    ids=["numpy-dense", "python-sequence", "numpy-sequence", "rust-sequence"],
)
def test_the_pocket_restores_the_best_epochs_optimizer_state(
    implementation: str, architecture: tuple[InputShape, list[LayerSpec]]
):
    # seeded so the best epoch isn't the last of 8: the student ends at the best epoch's weights
    # and Adam's t and moments, as though training had stopped there; a sequence network's
    # accuracy is per token
    input_shape, layers = architecture
    rows = _rows(input_shape, 40, seed=11, layers=layers)
    batch_size, epochs, batches_per_epoch = 8, 8, 5

    def student() -> Any:
        network = _seeded(_network(implementation, input_shape, layers, Adam()), 11)
        network.randomize()
        return network

    pocketed = student()
    result = train_backprop_network_mini_batch(
        pocketed, rows, batch_size, learning_rate=0.5, epochs=epochs, rng=random.Random(11)
    )
    assert result.diagnostic is not None
    best_epoch_index = result.diagnostic.best_epoch_index
    assert 0 <= best_epoch_index < epochs - 1, f"this case must pocket an earlier epoch; got {best_epoch_index}"
    assert pocketed.optimizer.t == (best_epoch_index + 1) * batches_per_epoch

    # the same run stopped after the best epoch: the same draws, so the same state
    stopped = student()
    train_backprop_network_mini_batch(
        stopped, rows, batch_size, learning_rate=0.5, epochs=best_epoch_index + 1, rng=random.Random(11)
    )
    assert _state_bits(pocketed) == _state_bits(stopped)
