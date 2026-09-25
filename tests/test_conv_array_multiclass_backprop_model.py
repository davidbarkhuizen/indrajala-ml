import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from indrajala_ml.digits_data import load_digits_dataset, split_train_test
from indrajala_ml.model.array_layer import ArrayLayer
from indrajala_ml.model.conv_array_layer import ConvArrayLayer
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.conv_multiclass_backprop_classifier_network import ConvMultiClassBackpropClassifierNetwork
from indrajala_ml.model.conv_rust_array_layer import ConvRustArrayLayer
from indrajala_ml.model.conv_rust_array_multiclass_backprop_classifier_network import (
    ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.conv_vectorized_multiclass_backprop_classifier_network import (
    ConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.max_pool_array_layer import MaxPoolArrayLayer
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.max_pool_rust_array_layer import MaxPoolRustArrayLayer
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from indrajala_ml.multiclass_evaluate import accuracy
from indrajala_ml.train import train_linear_classifier_network
from tests.helpers import (
    Backend,
    assert_conv_array_network_weights_match,
    copy_conv_network_weights_into_array_network,
    matching_conv_array_backprop_networks,
    matching_conv_numpy_rust_networks,
    weighted,
)

CLASS_COUNT = 10

NetworkCls = (
    type[ConvVectorizedMultiClassBackpropClassifierNetwork] | type[ConvRustArrayMultiClassBackpropClassifierNetwork]
)
NETWORK_CLS: dict[str, NetworkCls] = {
    "numpy": ConvVectorizedMultiClassBackpropClassifierNetwork,
    "rust": ConvRustArrayMultiClassBackpropClassifierNetwork,
}
LAYER_CLS = {
    "numpy": (ConvArrayLayer, MaxPoolArrayLayer, ArrayLayer),
    "rust": (ConvRustArrayLayer, MaxPoolRustArrayLayer, RustArrayLayer),
}

# every architecture the step-by-step parity gates below run over, on 8x8 inputs
ARCHITECTURES = {
    "one_conv": ([ConvSpec(3, 4)], [8]),
    "conv_conv_stride_2": ([ConvSpec(3, 3), ConvSpec(2, 4, stride=2)], [8]),  # multi-channel 2nd layer
    "conv_pool_conv": ([ConvSpec(3, 4), PoolSpec(2), ConvSpec(2, 6)], [8]),
    "conv_overlapping_pool": ([ConvSpec(3, 3), PoolSpec(2, stride=1)], [8, 6]),
    "conv_conv_conv": ([ConvSpec(2, 2), ConvSpec(3, 3), ConvSpec(2, 2, stride=2)], [8]),
}
POOLED = [ConvSpec(3, 4), PoolSpec(2), ConvSpec(2, 6)]
OVERLAPPING_POOL_STRIDED = [ConvSpec(3, 4), PoolSpec(2, stride=1), ConvSpec(2, 6, stride=2)]

# over these runs both backends stay within 1.4e-15 of the pure-Python network in weights and
# 3.3e-16 in probabilities
WEIGHT_ATOL = 1e-13
PROBABILITY_ATOL = 1e-14


@pytest.fixture
def network_cls(backend: Backend) -> NetworkCls:
    return NETWORK_CLS[backend.name]


def _digits_rows() -> list[tuple[tuple[float, ...], int]]:
    # real UCI digits: [0, 1] pixels with many exact zeros, so ReLU zeros and pooling ties
    # actually occur in the parity runs rather than only in the layer-level tie tests
    return load_digits_dataset()[:120]


def _matching_networks(rng: random.Random, architecture: str, backend: Backend):
    conv_specs, dense_layer_sizes = ARCHITECTURES[architecture]
    return matching_conv_array_backprop_networks(
        rng, 8, 8, conv_specs, dense_layer_sizes, CLASS_COUNT, NETWORK_CLS[backend.name], backend.owned
    )


def _as_lists(snapshot: Sequence[tuple[Any, ...]]) -> list[list[Any]]:
    return [[array.tolist() for array in entry] for entry in snapshot]


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_predict_probabilities_and_classify_state_match_across_a_sweep(backend: Backend, architecture: str):

    rng = random.Random(0)
    node_network, array_network = _matching_networks(rng, architecture, backend)

    states = [state for state, _label in _digits_rows()[:30]]
    states += [tuple(rng.uniform(0.0, 1.0) for _ in range(64)) for _ in range(20)]
    for state in states:
        np.testing.assert_allclose(
            array_network.predict_probabilities(state),
            node_network.predict_probabilities(state),
            rtol=0,
            atol=PROBABILITY_ATOL,
        )
        assert array_network.classify_state(state) == node_network.classify_state(state)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_learn_matches_after_every_step_not_just_at_the_end(backend: Backend, architecture: str):

    rng = random.Random(1)
    node_network, array_network = _matching_networks(rng, architecture, backend)

    for state, label in _digits_rows()[:60]:
        node_network.learn(0.5, state, label)
        array_network.learn(0.5, state, label)
        assert_conv_array_network_weights_match(node_network, array_network, rtol=0, atol=WEIGHT_ATOL)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_learn_batch_matches_after_every_batch_not_just_at_the_end(backend: Backend, architecture: str):

    rng = random.Random(2)
    node_network, array_network = _matching_networks(rng, architecture, backend)
    rows = _digits_rows()

    for start in range(0, len(rows), 8):
        batch = rows[start : start + 8]
        node_network.learn_batch(0.5, batch)
        array_network.learn_batch(0.5, batch)
        assert_conv_array_network_weights_match(node_network, array_network, rtol=0, atol=WEIGHT_ATOL)


def test_the_parity_runs_really_exercise_relu_zeros_and_pooling_ties(backend: Backend):

    # guards the claim the parity tests above rest on: on real digits rows, some conv outputs
    # are exactly zero and some pooling windows hold a tied maximum. These network-level runs
    # would still pass with last-occurrence tie-breaking (a tie after a ReLU is between exact
    # zeros, whose gradient the mask zeroes), so first-occurrence parity is pinned at the layer
    # level (tests/test_max_pool_array_layer.py)
    rng = random.Random(0)
    _node_network, array_network = _matching_networks(rng, "conv_pool_conv", backend)
    X = backend.owned([list(state) for state, _label in _digits_rows()])
    # Any: the network's own layers, chained within one backend, which a union can't express
    layers: list[Any] = list(array_network.layers)
    conv, pool = layers[0], layers[1]
    assert isinstance(conv, (ConvArrayLayer, ConvRustArrayLayer))
    layers[2].forward_batch(pool.forward_batch(conv.forward_batch(X)))

    A = np.array(conv.A.tolist())
    assert np.any(A == 0.0)
    windows = A.reshape(len(A), 4, 3, 2, 3, 2).transpose(0, 1, 2, 4, 3, 5)
    window_values = windows.reshape(len(A), 4, 3, 3, 4)
    tied = (window_values == window_values.max(axis=-1, keepdims=True)).sum(axis=-1) > 1
    assert np.any(tied)


def test_reproduces_the_pinned_pure_python_uci_digits_result(network_cls: NetworkCls):

    # test_conv_multiclass_backprop_model.py pins best_training_accuracy 0.9875 at epoch index 10
    # and test accuracy 0.925 for the pure-Python network. Same seeded initial weights (through a
    # numpy network, whose snapshot restores into either backend), same training call
    random.seed(0)

    dataset = load_digits_dataset()
    subset = dataset[:200]
    train_data, test_data = split_train_test(subset, test_fraction=0.2, seed=1)

    reference = ConvMultiClassBackpropClassifierNetwork.randomized(
        input_height=8, input_width=8, conv_specs=[ConvSpec(3, 4)], dense_layer_sizes=[16], class_count=10
    )
    bridge = ConvVectorizedMultiClassBackpropClassifierNetwork(8, 8, [ConvSpec(3, 4)], [16], class_count=10)
    copy_conv_network_weights_into_array_network(reference, bridge)
    student = network_cls(8, 8, [ConvSpec(3, 4)], [16], class_count=10)
    student.restore(bridge.snapshot())

    result = train_linear_classifier_network(student, train_data, learning_rate=0.5, epochs=15)

    diagnostic = result.diagnostic
    assert diagnostic.best_training_accuracy == 0.9875
    assert diagnostic.best_epoch_index == 10
    assert diagnostic.plateaued is True
    assert diagnostic.converged is False
    assert diagnostic.still_improving is False

    assert accuracy(student, test_data) == 0.925


def test_construction_shape_chains_conv_and_pool_layers(backend: Backend, network_cls: NetworkCls):

    conv_cls, pool_cls, dense_cls = LAYER_CLS[backend.name]
    network = network_cls(8, 8, POOLED, [8], class_count=10)
    first, pool, last = network.conv_layers

    assert network.dimension == 64
    assert isinstance(first, conv_cls) and isinstance(pool, pool_cls) and isinstance(last, conv_cls)
    assert (first.out_height, first.out_width, first.channel_count) == (6, 6, 4)
    assert (pool.out_height, pool.out_width, pool.channel_count) == (3, 3, 4)
    assert (last.input_channels, last.out_height, last.out_width, last.channel_count) == (4, 2, 2, 6)
    assert np.array(last.W.tolist()).shape == (6, 2 * 2 * 4)

    dense, output = network.layers[3:]
    assert isinstance(dense, dense_cls) and np.array(dense.W.tolist()).shape == (8, 2 * 2 * 6)
    assert output is network.output_layer and np.array(weighted(output).W.tolist()).shape == (10, 8)
    assert network.layers == network.conv_layers + [dense, output]


@pytest.mark.parametrize(
    "arguments",
    [
        (8, 8, [], [8], 10),  # no conv layers
        (8, 8, [PoolSpec(2)], [8], 10),  # only pooling
        (8, 8, [ConvSpec(3, 4)], [], 10),  # no dense layer
        (8, 8, [ConvSpec(3, 4)], [0], 10),  # empty dense layer
        (8, 8, [ConvSpec(3, 4)], [8], 1),  # class_count
        (8, 8, [ConvSpec(9, 4)], [8], 10),  # kernel larger than the input
    ],
)
def test_construction_rejects_invalid_arguments(
    network_cls: NetworkCls, arguments: tuple[int, int, list[ConvSpec | PoolSpec], list[int], int]
):

    with pytest.raises(AssertionError):
        network_cls(*arguments)


def test_randomized_breaks_symmetry_and_builds_a_usable_network(network_cls: NetworkCls):

    network = network_cls.randomized(8, 8, POOLED, [8], class_count=10)
    first, _pool, last = network.conv_layers

    for layer in (first, last):
        assert isinstance(layer, (ConvArrayLayer, ConvRustArrayLayer))
        W = np.array(layer.W.tolist())
        assert len({tuple(row) for row in W}) == layer.channel_count
        # fan-in-aware: every draw within 1/sqrt(input_channels * kernel_size**2)
        limit = 1.0 / np.sqrt(layer.fan_in)
        assert np.all(np.abs(W) <= limit) and np.all(np.abs(layer.b.tolist()) <= limit)
    assert np.all(np.abs(weighted(network.layers[3]).W.tolist()) <= 1.0 / np.sqrt(last.size))
    assert np.all(np.abs(network.output_layer.W.tolist()) <= 1.0 / np.sqrt(8))

    state = tuple(0.01 * i for i in range(64))
    probabilities = network.predict_probabilities(state)
    assert len(probabilities) == 10
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert 0 <= network.classify_state(state) < 10


def test_snapshot_has_an_empty_pool_entry_and_restore_round_trips(network_cls: NetworkCls):

    network = network_cls.randomized(8, 8, POOLED, [8], class_count=10)
    before = network.snapshot()
    before_lists = _as_lists(before)
    assert before[1] == ()

    # whole-snapshot comparisons: pa.uniform can't be seeded, and ~0.5% of initialisations leave
    # the first conv layer's W alone unchanged by one example's step (measured over 2000); the
    # output bias always moves (its gradient is p - one_hot)
    for state, label in _digits_rows()[:5]:
        network.learn(0.5, state, label)
    assert _as_lists(network.snapshot()) != before_lists

    network.restore(before)
    assert _as_lists(network.snapshot()) == before_lists
    # restore copies: training afterwards leaves the snapshot itself untouched
    network.learn(0.5, *_digits_rows()[0])
    assert _as_lists(network.snapshot()) != before_lists
    assert _as_lists(before) == before_lists

    with pytest.raises(AssertionError):
        network.restore([before[0], before[0], *before[2:]])


@pytest.mark.parametrize("conv_specs", [[ConvSpec(3, 4)], OVERLAPPING_POOL_STRIDED])
def test_save_load_round_trips_specs_weights_and_predictions(
    network_cls: NetworkCls, tmp_path: Path, conv_specs: list[ConvSpec | PoolSpec]
):

    network = network_cls.randomized(8, 8, conv_specs, [8], class_count=10)
    path = str(tmp_path / "conv_model.json")
    network.save(path)
    loaded = network_cls.load(path)

    assert loaded.conv_specs == network.conv_specs
    assert (loaded.input_height, loaded.input_width, loaded.dense_layer_sizes, loaded.class_count) == (8, 8, [8], 10)
    assert _as_lists(loaded.snapshot()) == _as_lists(network.snapshot())
    for state, _label in _digits_rows()[:10]:
        assert loaded.predict_probabilities(state) == network.predict_probabilities(state)


@pytest.mark.parametrize("save_backend", ["numpy", "rust"])
def test_a_model_saved_by_either_backend_loads_into_the_other_with_the_same_predictions(
    tmp_path: Path, save_backend: str
):

    rng = random.Random(3)
    numpy_network, rust_network = matching_conv_numpy_rust_networks(
        rng, 8, 8, OVERLAPPING_POOL_STRIDED, [8], CLASS_COUNT
    )
    for state, label in _digits_rows()[:20]:
        numpy_network.learn(0.5, state, label)
        rust_network.learn(0.5, state, label)

    path = str(tmp_path / "conv_model.json")
    if save_backend == "numpy":
        saved, loaded = numpy_network, ConvRustArrayMultiClassBackpropClassifierNetwork
    else:
        saved, loaded = rust_network, ConvVectorizedMultiClassBackpropClassifierNetwork
    saved.save(path)
    other = loaded.load(path)

    assert isinstance(other, loaded) and other.conv_specs == saved.conv_specs
    # weights round-trip exactly through JSON; predictions agree to the backends' reduction order
    assert _as_lists(other.snapshot()) == _as_lists(saved.snapshot())
    for state, _label in _digits_rows()[20:60]:
        np.testing.assert_allclose(
            other.predict_probabilities(state), saved.predict_probabilities(state), rtol=1e-12, atol=1e-14
        )
        assert other.classify_state(state) == saved.classify_state(state)
