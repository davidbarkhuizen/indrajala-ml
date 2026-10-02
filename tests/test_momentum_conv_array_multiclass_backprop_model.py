import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from indrajala_ml.digits_data import load_digits_dataset
from indrajala_ml.model.array_layer import ArrayLayer
from indrajala_ml.model.conv_array_layer import ConvArrayLayer
from indrajala_ml.model.conv_layer import ConvSpec
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
from indrajala_ml.model.momentum_conv_rust_array_multiclass_backprop_classifier_network import (
    MomentumConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.momentum_conv_vectorized_multiclass_backprop_classifier_network import (
    MomentumConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.rust_array_layer import RustArrayLayer
from indrajala_ml.model.specs.layer_specs import Dense, LayerSpec
from indrajala_ml.model.specs.update_rules import SGD, Momentum
from tests.array_network_contract import SEQUENTIAL_CLS, assert_sequential_matches_preset
from tests.helpers import (
    Backend,
    assert_conv_array_network_weights_match,
    conv_reference,
    matching_conv_numpy_rust_networks,
)
from tests.test_conv_array_multiclass_backprop_model import ARCHITECTURES, OVERLAPPING_POOL_STRIDED, WEIGHT_ATOL

CLASS_COUNT = 10
MOMENTUM = 0.9
LEARNING_RATE = 0.5
# not a power of two, so g / B rounds, and the last batch of the 120 rows is short
BATCH_SIZE = 7

NetworkCls = (
    type[MomentumConvVectorizedMultiClassBackpropClassifierNetwork]
    | type[MomentumConvRustArrayMultiClassBackpropClassifierNetwork]
)
NETWORK_CLS: dict[str, NetworkCls] = {
    "numpy": MomentumConvVectorizedMultiClassBackpropClassifierNetwork,
    "rust": MomentumConvRustArrayMultiClassBackpropClassifierNetwork,
}
PLAIN_NETWORK_CLS = {
    "numpy": ConvVectorizedMultiClassBackpropClassifierNetwork,
    "rust": ConvRustArrayMultiClassBackpropClassifierNetwork,
}
# (conv, pool, dense) layer classes
LAYER_CLS = {
    "numpy": (ConvArrayLayer, MaxPoolArrayLayer, ArrayLayer),
    "rust": (ConvRustArrayLayer, MaxPoolRustArrayLayer, RustArrayLayer),
}


def _digits_rows() -> list[tuple[tuple[float, ...], int]]:
    # real UCI digits, as the conv parity tests use: ReLU zeros and pooling ties occur
    return load_digits_dataset()[:120]


def _batches() -> list[list[tuple[tuple[float, ...], int]]]:
    rows = _digits_rows()
    return [rows[start : start + BATCH_SIZE] for start in range(0, len(rows), BATCH_SIZE)]


def _as_lists(snapshot: Sequence[tuple[Any, ...]]) -> list[list[Any]]:
    return [[array.tolist() for array in entry] for entry in snapshot]


def _specs(architecture: str) -> list[LayerSpec]:
    # the architecture as layer specs, which its pure-Python reference and sequential network share
    conv_specs, dense_layer_sizes = ARCHITECTURES[architecture]
    return [*conv_specs, *(Dense(size) for size in dense_layer_sizes), Dense(CLASS_COUNT, output=True)]


def _with_reference(architecture: str, backend: Backend, momentum: float) -> tuple[Any, Any]:
    # the pure-Python network of the same specs under Momentum, and this backend's momentum conv
    # network, with the same injected weights
    conv_specs, dense_layer_sizes = ARCHITECTURES[architecture]
    network = NETWORK_CLS[backend.name](8, 8, conv_specs, dense_layer_sizes, CLASS_COUNT, momentum=momentum)
    reference = conv_reference(
        random.Random(1), network, (8, 8, 1), _specs(architecture), Momentum(momentum), backend.owned
    )
    return reference, network


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_learn_matches_the_pure_python_reference_after_every_step(backend: Backend, architecture: str):

    # over these runs both backends stay within 1.3e-14 of the reference in weights
    reference, network = _with_reference(architecture, backend, MOMENTUM)

    for state, label in _digits_rows()[:60]:
        reference.learn(LEARNING_RATE, state, label)
        network.learn(LEARNING_RATE, state, label)
        assert_conv_array_network_weights_match(reference, network, rtol=0, atol=WEIGHT_ATOL)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_learn_batch_matches_the_pure_python_reference_after_every_batch(backend: Backend, architecture: str):

    reference, network = _with_reference(architecture, backend, MOMENTUM)

    for batch in _batches():
        reference.learn_batch(LEARNING_RATE, batch)
        network.learn_batch(LEARNING_RATE, batch)
        assert_conv_array_network_weights_match(reference, network, rtol=0, atol=WEIGHT_ATOL)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_numpy_and_rust_match_after_every_batch(architecture: str):

    # after 3a and 3b the backends' update rules agree bit for bit, so any difference is BLAS
    # reduction order
    conv_specs, dense_layer_sizes = ARCHITECTURES[architecture]
    plain_numpy, plain_rust = matching_conv_numpy_rust_networks(
        random.Random(2), 8, 8, conv_specs, dense_layer_sizes, CLASS_COUNT
    )
    numpy_network = MomentumConvVectorizedMultiClassBackpropClassifierNetwork(
        8, 8, conv_specs, dense_layer_sizes, CLASS_COUNT, momentum=MOMENTUM
    )
    rust_network = MomentumConvRustArrayMultiClassBackpropClassifierNetwork(
        8, 8, conv_specs, dense_layer_sizes, CLASS_COUNT, momentum=MOMENTUM
    )
    numpy_network.restore(plain_numpy.snapshot())
    rust_network.restore(plain_rust.snapshot())

    for batch in _batches():
        numpy_network.learn_batch(LEARNING_RATE, batch)
        rust_network.learn_batch(LEARNING_RATE, batch)
        for numpy_entry, rust_entry in zip(_as_lists(numpy_network.snapshot()), _as_lists(rust_network.snapshot())):
            for numpy_array, rust_array in zip(numpy_entry, rust_entry):
                np.testing.assert_allclose(rust_array, numpy_array, rtol=0, atol=WEIGHT_ATOL)


def _zero_momentum_and_plain(backend: Backend) -> tuple[Any, Any]:
    conv_specs, dense_layer_sizes = ARCHITECTURES["conv_pool_conv"]
    plain = PLAIN_NETWORK_CLS[backend.name](8, 8, conv_specs, dense_layer_sizes, CLASS_COUNT)
    # the reference is discarded: this is only its weight injection into plain
    conv_reference(random.Random(3), plain, (8, 8, 1), _specs("conv_pool_conv"), SGD(), backend.owned)
    network = NETWORK_CLS[backend.name](8, 8, conv_specs, dense_layer_sizes, CLASS_COUNT, momentum=0.0)
    network.restore(plain.snapshot())
    return plain, network


def test_zero_momentum_is_bit_identical_to_the_plain_conv_network_through_learn(backend: Backend):

    plain, network = _zero_momentum_and_plain(backend)

    for state, label in _digits_rows()[:30]:
        plain.learn(LEARNING_RATE, state, label)
        network.learn(LEARNING_RATE, state, label)
        assert _as_lists(network.snapshot()) == _as_lists(plain.snapshot())


def test_zero_momentum_is_bit_identical_to_the_plain_conv_network_through_learn_batch(backend: Backend):

    # at momentum 0.0 eq. (9) is exactly SGD's w - lr * (g / B), also where g / B rounds
    plain, network = _zero_momentum_and_plain(backend)

    for batch in _batches():
        plain.learn_batch(LEARNING_RATE, batch)
        network.learn_batch(LEARNING_RATE, batch)
        assert _as_lists(network.snapshot()) == _as_lists(plain.snapshot())


def test_the_optimizer_steps_every_layer_by_momentum(backend: Backend):

    conv_cls, pool_cls, dense_cls = LAYER_CLS[backend.name]
    network = NETWORK_CLS[backend.name](8, 8, OVERLAPPING_POOL_STRIDED, [8, 6], CLASS_COUNT, momentum=0.7)
    first, pool, last = network.conv_layers

    for layer in (first, last):
        assert type(layer) is conv_cls
    assert type(pool) is pool_cls
    for layer in network.layers[3:]:
        assert type(layer) is dense_cls
    assert network.optimizer.rule == Momentum(0.7)


def test_save_and_load_round_trip_keeps_momentum(backend: Backend, tmp_path: Path):

    network_cls = NETWORK_CLS[backend.name]
    network = network_cls.randomized(8, 8, OVERLAPPING_POOL_STRIDED, [8], CLASS_COUNT, momentum=0.7)
    path = str(tmp_path / "momentum_conv.json")
    network.save(path)
    loaded = network_cls.load(path)

    saved = json.loads(Path(path).read_text())
    assert saved["preset"]["arguments"]["momentum"] == 0.7
    assert saved["update_rule"] == {"rule": "momentum", "momentum": 0.7}
    assert loaded.momentum == 0.7 and loaded.conv_specs == network.conv_specs
    assert _as_lists(loaded.snapshot()) == _as_lists(network.snapshot())
    # the loaded layers train with the saved momentum: two batches match a fresh 0.7 network's
    reference = network_cls(8, 8, OVERLAPPING_POOL_STRIDED, [8], CLASS_COUNT, momentum=0.7)
    reference.restore(loaded.snapshot())
    for batch in _batches()[:2]:
        loaded.learn_batch(LEARNING_RATE, batch)
        reference.learn_batch(LEARNING_RATE, batch)
    assert _as_lists(loaded.snapshot()) == _as_lists(reference.snapshot())


@pytest.mark.parametrize("save_backend", ["numpy", "rust"])
def test_a_model_saved_by_either_backend_loads_into_the_other_with_momentum(tmp_path: Path, save_backend: str):

    numpy_network, rust_network = matching_conv_numpy_rust_networks(
        random.Random(4), 8, 8, OVERLAPPING_POOL_STRIDED, [8], CLASS_COUNT
    )
    networks = {
        name: NETWORK_CLS[name](8, 8, OVERLAPPING_POOL_STRIDED, [8], CLASS_COUNT, momentum=0.7) for name in NETWORK_CLS
    }
    networks["numpy"].restore(numpy_network.snapshot())
    networks["rust"].restore(rust_network.snapshot())
    for batch in _batches()[:3]:
        for network in networks.values():
            network.learn_batch(LEARNING_RATE, batch)

    saved = networks[save_backend]
    load_cls = NETWORK_CLS["rust" if save_backend == "numpy" else "numpy"]
    path = str(tmp_path / "momentum_conv.json")
    saved.save(path)
    other = load_cls.load(path)

    assert isinstance(other, load_cls) and other.momentum == 0.7 and other.conv_specs == saved.conv_specs
    # weights round-trip exactly through JSON; predictions agree to the backends' reduction order
    assert _as_lists(other.snapshot()) == _as_lists(saved.snapshot())
    for state, _label in _digits_rows()[:20]:
        np.testing.assert_allclose(
            other.predict_probabilities(state), saved.predict_probabilities(state), rtol=1e-12, atol=1e-14
        )


def test_momentum_is_required(backend: Backend):

    with pytest.raises(TypeError):
        NETWORK_CLS[backend.name](8, 8, [ConvSpec(3, 4), PoolSpec(2)], [8], CLASS_COUNT)  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_the_sequential_network_of_its_layer_specs_matches_it_by_bits(backend: Backend, architecture: str):
    conv_specs, dense_layer_sizes = ARCHITECTURES[architecture]
    preset = NETWORK_CLS[backend.name](8, 8, conv_specs, dense_layer_sizes, CLASS_COUNT, momentum=MOMENTUM)
    sequential = SEQUENTIAL_CLS["multiclass"][backend.name]((8, 8, 1), _specs(architecture), Momentum(MOMENTUM))

    def example(rng: random.Random) -> tuple[tuple[float, ...], int]:
        return tuple(rng.random() for _ in range(64)), rng.randrange(CLASS_COUNT)

    assert_sequential_matches_preset(preset, sequential, backend, example, 0.5)
