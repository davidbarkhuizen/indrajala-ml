"""
Seeded initialisation is identical across backends: randomized(..., seed=s) builds bit-identical
numpy and Rust networks, for every array network class. Each network owns its generator, numpy's
default_rng on numpy and the crate's Generator on Rust, which is numpy's bit for bit
(rust/tests/test_random_pcg64_parity.py), and both random_layers compute limit = 1/sqrt(fan_in)
the same way, so the weights match bit for bit, not within a tolerance. A learn_batch then draws
the same dropout masks and leaves both generators in the same state. 1 / n ** 0.5 is 1 ULP off 1 / np.sqrt(n) at fan-in 5579 (and math.sqrt never is, 1 to
99,999), so the Rust limit uses math.sqrt; 2921, whose n ** 0.5 alone differs, is kept as a control.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model.layers.array.array_backend import NumpyBackend, RustBackend
from indrajala_ml.model.layers.python.conv_layer import ConvSpec
from indrajala_ml.model.layers.python.max_pool_layer import PoolSpec
from indrajala_ml.model.networks.array_network_base import ArrayNetworkBase
from indrajala_ml.model.networks.numpy.numpy_array_network_base import NumpyArrayNetworkBase
from indrajala_ml.model.networks.rust.rust_array_network_base import RustArrayNetworkBase
from indrajala_ml.model.specs.layer_specs import Dense
from indrajala_ml.model.specs.update_rules import SGD
from indrajala_ml.pcg64 import SeedSequence
from tests.array_network_contract import snapshot_bits
from tests.helpers import all_subclasses, model_modules

model_modules()

SIDE = 6
CLASS_COUNT = 3
CONV_SPECS = [ConvSpec(3, 2), PoolSpec(2), ConvSpec(2, 3)]

# (numpy class name, arguments after the shape arguments); the Rust class is its counterpart
MULTICLASS: dict[str, tuple[Any, ...]] = {
    "VectorizedMultiClassBackpropClassifierNetwork": (),
    "AdamVectorizedMultiClassBackpropClassifierNetwork": (),
    "CrossEntropyVectorizedMultiClassBackpropClassifierNetwork": (),
    "DropoutVectorizedMultiClassBackpropClassifierNetwork": (0.3,),
    "L2VectorizedMultiClassBackpropClassifierNetwork": (0.01,),
    "MomentumVectorizedMultiClassBackpropClassifierNetwork": (0.9,),
    "ReLUVectorizedMultiClassBackpropClassifierNetwork": (),
    "SoftmaxVectorizedMultiClassBackpropClassifierNetwork": (),
}
CONV: dict[str, tuple[Any, ...]] = {
    "ConvVectorizedMultiClassBackpropClassifierNetwork": (),
    "MomentumConvVectorizedMultiClassBackpropClassifierNetwork": (0.9,),
    "AdamConvVectorizedMultiClassBackpropClassifierNetwork": (),
    "L2ConvVectorizedMultiClassBackpropClassifierNetwork": (0.01,),
    "ReLUConvVectorizedMultiClassBackpropClassifierNetwork": (),
    "DropoutConvVectorizedMultiClassBackpropClassifierNetwork": (0.3,),
    "CrossEntropyConvVectorizedMultiClassBackpropClassifierNetwork": (),
    "SoftmaxConvVectorizedMultiClassBackpropClassifierNetwork": (),
}
# (numpy class name, its keyword-only hyperparameters)
SINGLE_OUTPUT: dict[str, dict[str, float]] = {
    "ArrayBackpropClassifierNetwork": {},
    "AdamArrayBackpropClassifierNetwork": {},
    "CrossEntropyArrayBackpropClassifierNetwork": {},
    "DropoutArrayBackpropClassifierNetwork": {"drop_probability": 0.3},
    "L2ArrayBackpropClassifierNetwork": {"l2_lambda": 0.01},
    "MomentumArrayBackpropClassifierNetwork": {"momentum": 0.9},
    "ReLUArrayBackpropClassifierNetwork": {},
}
# (numpy class name, its constructor's arguments): a conv front end, then every dense kind
SEQUENTIAL: dict[str, tuple[Any, ...]] = {
    "SequentialVectorizedMultiClassBackpropClassifierNetwork": (
        (SIDE, SIDE, 1),
        [*CONV_SPECS, Dense(5, activation="relu"), Dense(4, dropout=0.3), Dense(CLASS_COUNT, output=True)],
        SGD(),
    ),
    "SequentialArrayBackpropClassifierNetwork": ((9,), [Dense(4), Dense(1, output=True)], SGD()),
}


def _classes(base: type[Any]) -> dict[str, type[Any]]:
    return {cls.__name__: cls for cls in all_subclasses(base)}


NUMPY_CLASSES = _classes(NumpyArrayNetworkBase)
RUST_CLASSES = _classes(RustArrayNetworkBase)


def rust_counterpart(numpy_name: str) -> type[Any]:
    if "Vectorized" in numpy_name:
        return RUST_CLASSES[numpy_name.replace("Vectorized", "RustArray")]
    return RUST_CLASSES[numpy_name.replace("ArrayBackprop", "RustArrayBackprop")]


def test_every_array_network_class_is_covered():
    covered = {*MULTICLASS, *CONV, *SINGLE_OUTPUT, *SEQUENTIAL}
    assert covered == set(NUMPY_CLASSES)
    assert {rust_counterpart(name) for name in covered} == set(RUST_CLASSES.values())
    assert set(NUMPY_CLASSES.values()) | set(RUST_CLASSES.values()) == set(all_subclasses(ArrayNetworkBase)) - {
        NumpyArrayNetworkBase,
        RustArrayNetworkBase,
    }


def assert_seeded_randomized_identical(build: Callable[[type[Any], int], Any], numpy_name: str, seed: int) -> None:
    numpy_network = build(NUMPY_CLASSES[numpy_name], seed)
    rust_network = build(rust_counterpart(numpy_name), seed)
    assert snapshot_bits(rust_network) == snapshot_bits(numpy_network)
    assert rust_network.rng.state == numpy_network.rng.bit_generator.state

    # a training step draws the same masks from both generators, and as many draws
    width = numpy_network.input_shape[0] if len(numpy_network.input_shape) == 1 else SIDE * SIDE
    rows = _rows(width, numpy_network)
    numpy_network.learn_batch(0.5, rows)
    rust_network.learn_batch(0.5, rows)
    assert _mask_bits(rust_network) == _mask_bits(numpy_network)
    assert rust_network.rng.state == numpy_network.rng.bit_generator.state


def _rows(width: int, network: Any) -> list[tuple[tuple[float, ...], Any]]:
    rng = random.Random(width)
    single_output = getattr(network, "class_count", None) is None
    return [
        (tuple(rng.uniform(0.0, 1.0) for _ in range(width)), rng.random() if single_output else i % CLASS_COUNT)
        for i in range(4)
    ]


def _mask_bits(network: Any) -> list[Any]:
    return [np.array(layer._mask_batch.tolist()).tobytes() for layer in network.layers if hasattr(layer, "_mask_batch")]


SEEDS = [0, 1, 42, 2**32 - 1]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("numpy_name", MULTICLASS)
def test_multiclass_randomized_is_identical_after_the_same_seed(numpy_name: str, seed: int):
    args = MULTICLASS[numpy_name]
    assert_seeded_randomized_identical(
        lambda cls, s: cls.randomized([7, 5], 12, CLASS_COUNT, *args, seed=s), numpy_name, seed
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("numpy_name", CONV)
def test_conv_randomized_is_identical_after_the_same_seed(numpy_name: str, seed: int):
    args = CONV[numpy_name]
    assert_seeded_randomized_identical(
        lambda cls, s: cls.randomized(SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT, *args, seed=s), numpy_name, seed
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("numpy_name", SINGLE_OUTPUT)
def test_single_output_randomized_is_identical_after_the_same_seed(numpy_name: str, seed: int):
    hyperparameters = SINGLE_OUTPUT[numpy_name]
    assert_seeded_randomized_identical(
        lambda cls, s: cls.randomized([4], 9, seed=s, **hyperparameters), numpy_name, seed
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("numpy_name", SEQUENTIAL)
def test_sequential_randomized_is_identical_after_the_same_seed(numpy_name: str, seed: int):
    assert_seeded_randomized_identical(lambda cls, s: cls.randomized(*SEQUENTIAL[numpy_name], seed=s), numpy_name, seed)


@pytest.mark.parametrize("fan_in", [2921, 5579])
def test_randomized_is_identical_at_the_fan_ins_where_power_and_sqrt_differ(fan_in: int):
    assert_seeded_randomized_identical(
        lambda cls, s: cls.randomized([2], fan_in, CLASS_COUNT, seed=s),
        "VectorizedMultiClassBackpropClassifierNetwork",
        7,
    )


def test_randomize_draws_from_the_network_generator_and_no_global_state():
    # a network's draws move neither np.random nor the crate's global stream, nor another
    # network's generator
    np.random.seed(5)
    pa.seed(5)
    expected = np.random.random(10)
    np.random.seed(5)
    first = NUMPY_CLASSES["DropoutVectorizedMultiClassBackpropClassifierNetwork"].randomized([7], 12, 3, 0.3, seed=1)
    rust = rust_counterpart("DropoutVectorizedMultiClassBackpropClassifierNetwork").randomized([7], 12, 3, 0.3, seed=1)
    second = NUMPY_CLASSES["DropoutVectorizedMultiClassBackpropClassifierNetwork"].randomized([7], 12, 3, 0.3, seed=1)
    first.learn_batch(0.5, _rows(12, first))
    rust.learn_batch(0.5, _rows(12, rust))
    assert np.random.random(10).tobytes() == expected.tobytes()
    assert np.array(pa.random(10).tolist()).tobytes() == expected.tobytes()
    third = NUMPY_CLASSES["DropoutVectorizedMultiClassBackpropClassifierNetwork"].randomized([7], 12, 3, 0.3, seed=1)
    assert snapshot_bits(second) == snapshot_bits(third)
    assert second.rng.bit_generator.state == third.rng.bit_generator.state


@pytest.mark.parametrize("backend", [NumpyBackend, RustBackend], ids=["numpy", "rust"])
def test_a_network_without_a_seed_draws_from_os_entropy(backend: Any):
    name = "VectorizedMultiClassBackpropClassifierNetwork"
    cls = NUMPY_CLASSES[name] if backend is NumpyBackend else rust_counterpart(name)
    assert snapshot_bits(cls.randomized([7], 12, 3)) != snapshot_bits(cls.randomized([7], 12, 3))


def test_a_spawned_seed_sequence_from_any_implementation_seeds_both_backends_alike():
    # an ensemble's sub-network seeds (ensemble_train.py): numpy's, the crate's and pcg64's
    # SeedSequence children with the same entropy and spawn key give the same generator
    expected = np.random.default_rng(np.random.SeedSequence(9).spawn(3)[2]).bit_generator.state
    for child in [np.random.SeedSequence(9).spawn(3)[2], pa.SeedSequence(9).spawn(3)[2], SeedSequence(9).spawn(3)[2]]:
        assert NumpyBackend.default_rng(child).bit_generator.state == expected
        assert RustBackend.default_rng(child).state == expected


def test_randomized_takes_a_generator_and_shares_it():
    rng = NumpyBackend.default_rng(3)
    network = NUMPY_CLASSES["DropoutVectorizedMultiClassBackpropClassifierNetwork"].randomized([7], 12, 3, 0.3, rng=rng)
    assert network.rng is rng
    assert all(layer.rng is rng for layer in network.layers if hasattr(layer, "set_rng"))
    seeded = NUMPY_CLASSES["DropoutVectorizedMultiClassBackpropClassifierNetwork"].randomized([7], 12, 3, 0.3, seed=3)
    assert snapshot_bits(network) == snapshot_bits(seeded)


def test_assigning_rng_reaches_the_dropout_layers():
    network = rust_counterpart("DropoutVectorizedMultiClassBackpropClassifierNetwork")([7, 5], 12, 3, 0.3)
    rng = RustBackend.default_rng(4)
    network.rng = rng
    assert [layer.rng is rng for layer in network.layers if hasattr(layer, "set_rng")] == [True, True]
