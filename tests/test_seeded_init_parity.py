"""
Seeded initialisation is identical across backends: after backend.seed(s), randomized() builds
bit-identical numpy and Rust networks, for every array network class. The crate's RNG is numpy's
np.random in a separate state (rust/tests/test_random_numpy_parity.py), and both random_layers
compute limit = 1/sqrt(fan_in) the same way, so the weights match bit for bit, not within a
tolerance. 1 / n ** 0.5 is 1 ULP off 1 / np.sqrt(n) at fan-in 5579 (and math.sqrt never is, 1 to
99,999), so the Rust limit uses math.sqrt; 2921, whose n ** 0.5 alone differs, is kept as a control.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import indrajala_ml.model
from indrajala_ml.model.array_backend import NumpyBackend, RustBackend
from indrajala_ml.model.array_network_base import ArrayNetworkBase
from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.numpy_array_network_base import NumpyArrayNetworkBase
from indrajala_ml.model.rust_array_network_base import RustArrayNetworkBase
from tests.helpers import all_subclasses

for _module in pkgutil.iter_modules(indrajala_ml.model.__path__):
    importlib.import_module(f"indrajala_ml.model.{_module.name}")

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
}
SINGLE_OUTPUT = ["ArrayBackpropClassifierNetwork", "CrossEntropyArrayBackpropClassifierNetwork"]


def _classes(base: type[Any]) -> dict[str, type[Any]]:
    return {cls.__name__: cls for cls in all_subclasses(base)}


NUMPY_CLASSES = _classes(NumpyArrayNetworkBase)
RUST_CLASSES = _classes(RustArrayNetworkBase)


def rust_counterpart(numpy_name: str) -> type[Any]:
    if "Vectorized" in numpy_name:
        return RUST_CLASSES[numpy_name.replace("Vectorized", "RustArray")]
    return RUST_CLASSES[numpy_name.replace("ArrayBackprop", "RustArrayBackprop")]


def test_every_array_network_class_is_covered():
    covered = {*MULTICLASS, *CONV, *SINGLE_OUTPUT}
    assert covered == set(NUMPY_CLASSES)
    assert {rust_counterpart(name) for name in covered} == set(RUST_CLASSES.values())
    assert set(NUMPY_CLASSES.values()) | set(RUST_CLASSES.values()) == set(all_subclasses(ArrayNetworkBase)) - {
        NumpyArrayNetworkBase,
        RustArrayNetworkBase,
    }


def snapshot_bits(network: Any) -> list[list[bytes]]:
    return [[np.asarray(array.tolist(), dtype=np.float64).tobytes() for array in entry] for entry in network.snapshot()]


def assert_seeded_randomized_identical(build: Callable[[type[Any]], Any], numpy_name: str, seed: int) -> None:
    NumpyBackend.seed(seed)
    numpy_network = build(NUMPY_CLASSES[numpy_name])
    RustBackend.seed(seed)
    rust_network = build(rust_counterpart(numpy_name))
    assert snapshot_bits(rust_network) == snapshot_bits(numpy_network)


SEEDS = [0, 1, 42, 2**32 - 1]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("numpy_name", MULTICLASS)
def test_multiclass_randomized_is_identical_after_the_same_seed(numpy_name: str, seed: int):
    args = MULTICLASS[numpy_name]
    assert_seeded_randomized_identical(lambda cls: cls.randomized([7, 5], 12, CLASS_COUNT, *args), numpy_name, seed)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("numpy_name", CONV)
def test_conv_randomized_is_identical_after_the_same_seed(numpy_name: str, seed: int):
    args = CONV[numpy_name]
    assert_seeded_randomized_identical(
        lambda cls: cls.randomized(SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT, *args), numpy_name, seed
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("numpy_name", SINGLE_OUTPUT)
def test_single_output_randomized_is_identical_after_the_same_seed(numpy_name: str, seed: int):
    assert_seeded_randomized_identical(lambda cls: cls.randomized([4], 9), numpy_name, seed)


@pytest.mark.parametrize("fan_in", [2921, 5579])
def test_randomized_is_identical_at_the_fan_ins_where_power_and_sqrt_differ(fan_in: int):
    assert_seeded_randomized_identical(
        lambda cls: cls.randomized([2], fan_in, CLASS_COUNT),
        "VectorizedMultiClassBackpropClassifierNetwork",
        7,
    )


def test_backend_seeds_are_separate_states():
    # seeding one backend never moves the other's stream
    NumpyBackend.seed(5)
    expected = np.random.random(10)
    NumpyBackend.seed(5)
    RustBackend.seed(6)
    RustBackend.random_layer(3, 4)
    assert np.random.random(10).tobytes() == expected.tobytes()
