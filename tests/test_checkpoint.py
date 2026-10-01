"""
checkpoint()/restore_checkpoint() (model/checkpoint.py) and the optimizers' state()/load_state():
a restored checkpoint resumes training by bits, for every update rule in all three
implementations, dense and conv, also after crossing a worker boundary as nested lists
(ensemble_train._picklable_checkpoint); a checkpoint is a copy; and train.py's pocket restores the
best epoch's optimizer state along with its weights.
"""

import pickle
import random
from typing import Any

import pytest

from indrajala_ml.ensemble_train import _picklable_checkpoint  # pyright: ignore[reportPrivateUsage]
from indrajala_ml.model.array_backend import NUMPY, RUST
from indrajala_ml.model.layer_specs import Conv, Dense, InputShape, LayerSpec, Pool
from indrajala_ml.model.sequential_array_network import SequentialArrayNetwork
from indrajala_ml.model.sequential_backprop_network import SequentialMultiClassBackpropClassifierNetwork
from indrajala_ml.model.update_rules import SGD, Adam, Momentum, UpdateRule, WeightDecay
from indrajala_ml.seeding import seed_everything
from indrajala_ml.train import train_backprop_network_mini_batch

RULES: list[UpdateRule] = [SGD(), Momentum(0.9), Adam(), WeightDecay(0.01)]
IMPLEMENTATIONS = ["python", "numpy", "rust"]

DENSE: tuple[InputShape, list[LayerSpec]] = ((4,), [Dense(5), Dense(3, output=True)])
# a pool layer between weighted ones: the optimizers key state by layer index, and skip the pool
CONV: tuple[InputShape, list[LayerSpec]] = ((6, 6, 1), [Conv(2, 3), Pool(2), Dense(3, output=True)])


def _network(implementation: str, input_shape: InputShape, layers: list[LayerSpec], rule: UpdateRule) -> Any:
    if implementation == "python":
        return SequentialMultiClassBackpropClassifierNetwork(input_shape, layers, rule)
    return SequentialArrayNetwork(input_shape, layers, rule, backend=NUMPY if implementation == "numpy" else RUST)


def _seeded(network: Any, seed: int) -> Any:
    # an array network's own generator, seeded; a pure-Python one draws from random
    # (seed_everything) until the RNG generators workplan's stage 4
    if not isinstance(network, SequentialMultiClassBackpropClassifierNetwork):
        network.rng = network.backend.default_rng(seed)
    return network


def _rows(input_shape: InputShape, count: int, seed: int) -> list[tuple[tuple[float, ...], int]]:
    rng = random.Random(seed)
    dimension = 1
    for extent in input_shape:
        dimension *= extent
    return [(tuple(rng.random() for _ in range(dimension)), rng.randrange(3)) for _ in range(count)]


def _bits(value: Any) -> Any:
    # every float as float.hex, through arrays, lists, tuples, dicts and checkpoints: compares bits
    to_list = getattr(value, "tolist", None)
    if to_list is not None:
        return _bits(to_list())
    if isinstance(value, (list, tuple)):
        return [_bits(item) for item in value]  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, dict):
        return {key: _bits(item) for key, item in value.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, float):
        return value.hex()
    return value


def _state_bits(network: Any) -> Any:
    state = network.optimizer.state()
    return [_bits(network.snapshot()), state.t, _bits(state.layers)]


def _train(network: Any, rows: list[tuple[tuple[float, ...], int]]) -> None:
    # batches, then single examples: both of the optimizer's paths, apply and step_single
    network.learn_batch(0.1, rows[:4])
    network.learn_batch(0.1, rows[4:])
    for state, category in rows[:3]:
        network.learn(0.1, state, category)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("rule", RULES, ids=lambda rule: type(rule).__name__)
@pytest.mark.parametrize("architecture", [DENSE, CONV], ids=["dense", "conv"])
@pytest.mark.parametrize("across_workers", [False, True], ids=["in_memory", "as_lists"])
def test_a_restored_checkpoint_resumes_training_by_bits(
    implementation: str, rule: UpdateRule, architecture: tuple[InputShape, list[LayerSpec]], across_workers: bool
):
    input_shape, layers = architecture
    rows = _rows(input_shape, 8, seed=1)

    seed_everything(2)
    trained = _seeded(_network(implementation, input_shape, layers, rule), 2)
    trained.randomize()
    _train(trained, rows)
    checkpoint = trained.checkpoint()
    if across_workers:
        checkpoint = pickle.loads(pickle.dumps(_picklable_checkpoint(checkpoint)))
    _train(trained, rows)

    seed_everything(3)  # other weights, overwritten by the checkpoint
    resumed = _seeded(_network(implementation, input_shape, layers, rule), 3)
    resumed.randomize()
    resumed.restore_checkpoint(checkpoint)
    _train(resumed, rows)

    assert _state_bits(resumed) == _state_bits(trained)
    assert trained.optimizer.t == 2 * 5


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("rule", [Momentum(0.9), Adam()], ids=lambda rule: type(rule).__name__)
def test_a_checkpoint_is_a_copy(implementation: str, rule: UpdateRule):
    input_shape, layers = DENSE
    rows = _rows(input_shape, 8, seed=1)
    seed_everything(2)
    network = _seeded(_network(implementation, input_shape, layers, rule), 2)
    network.randomize()
    _train(network, rows)

    checkpoint = network.checkpoint()
    taken = [_bits(checkpoint.weights), checkpoint.optimizer.t, _bits(checkpoint.optimizer.layers)]
    _train(network, rows)  # steps the network's weights and state, not the checkpoint's
    network.restore_checkpoint(checkpoint)
    _train(network, rows)  # nor, after restoring it, the network's steps

    assert [_bits(checkpoint.weights), checkpoint.optimizer.t, _bits(checkpoint.optimizer.layers)] == taken


def test_a_checkpoint_before_the_first_step_restores_fresh_state():
    input_shape, layers = DENSE
    rows = _rows(input_shape, 8, seed=1)
    network = _network("numpy", input_shape, layers, Adam())
    checkpoint = network.checkpoint()
    assert checkpoint.optimizer.t == 0 and checkpoint.optimizer.layers == {}

    _train(network, rows)
    network.restore_checkpoint(checkpoint)
    assert network.optimizer.t == 0 and network.optimizer.state().layers == {}


def test_the_pocket_restores_the_best_epochs_optimizer_state():
    # seeded so the best epoch (index 1) isn't the last of 8: the student ends at epoch 1's weights
    # and Adam's t and moments, as though training had stopped there
    input_shape, layers = DENSE
    rows = _rows(input_shape, 40, seed=11)
    batch_size, epochs, batches_per_epoch = 8, 8, 5

    def student() -> Any:
        network = _seeded(_network("numpy", input_shape, layers, Adam()), 11)
        network.randomize()
        return network

    seed_everything(11)
    pocketed = student()
    result = train_backprop_network_mini_batch(pocketed, rows, batch_size, learning_rate=0.5, epochs=epochs)
    assert result.diagnostic is not None
    best_epoch_index = result.diagnostic.best_epoch_index
    assert 0 <= best_epoch_index < epochs - 1, f"this case must pocket an earlier epoch; got {best_epoch_index}"
    assert pocketed.optimizer.t == (best_epoch_index + 1) * batches_per_epoch

    # the same run stopped after the best epoch: the same draws, so the same state
    seed_everything(11)
    stopped = student()
    train_backprop_network_mini_batch(stopped, rows, batch_size, learning_rate=0.5, epochs=best_epoch_index + 1)
    assert _state_bits(pocketed) == _state_bits(stopped)
