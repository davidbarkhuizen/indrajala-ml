"""
Seeded initialisation is identical across backends: randomized(..., seed=s) builds bit-identical
numpy and Rust networks, for every array network class. Each network owns its generator, numpy's
default_rng on numpy and the crate's Generator on Rust, which is numpy's bit for bit
(rust/tests/test_random_pcg64_parity.py), and both random_layers compute limit = 1/sqrt(fan_in)
the same way, so the weights match bit for bit, not within a tolerance. A learn_batch then draws
the same dropout masks and leaves both generators in the same state. 1 / n ** 0.5 is 1 ULP off 1 / np.sqrt(n) at fan-in 5579 (and math.sqrt never is, 1 to
99,999), so the Rust limit uses math.sqrt; 2921, whose n ** 0.5 alone differs, is kept as a control.

Pure Python draws from the same PCG64 stream (indrajala_ml.pcg64) in numpy's order, so every
pure-Python network with an array twin starts from a seed with numpy's weights by bits, and its
generator in numpy's state (the RNG draw-order workplan, D2 and D4). The bounds-width networks
(BackpropClassifierNetwork's presets) have no array twin.
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
from indrajala_ml.model.networks.python.backprop_network_base import BackpropNetworkBase
from indrajala_ml.model.networks.rust.rust_array_network_base import RustArrayNetworkBase
from indrajala_ml.model.specs.layer_specs import Attention, Dense, Embedding, LayerNorm, Position, Residual
from indrajala_ml.model.specs.update_rules import SGD
from indrajala_ml.pcg64 import SeedSequence, generator_state
from tests.array_network_contract import snapshot_bits
from tests.helpers import all_subclasses, model_modules
from tests.model.networks.test_attention_python_network import as_array_snapshot

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
    # a causal transformer over 5 ids of a vocabulary of 7: every sequence layer draws
    "SequentialSequenceArrayNetwork": (
        (5,),
        [
            Embedding(7, 6),
            Position(),
            Residual((LayerNorm(), Attention(heads=2, causal=True))),
            Residual((LayerNorm(), Dense(8, activation="relu"), Dense(6, activation="linear", bias=True))),
            LayerNorm(),
            Dense(7, output=True, activation="softmax", loss="cross_entropy"),
        ],
        SGD(),
    ),
}


def _classes(base: type[Any]) -> dict[str, type[Any]]:
    return {cls.__name__: cls for cls in all_subclasses(base)}


NUMPY_CLASSES = _classes(NumpyArrayNetworkBase)
RUST_CLASSES = _classes(RustArrayNetworkBase)


def rust_counterpart(numpy_name: str) -> type[Any]:
    if "Vectorized" in numpy_name:
        return RUST_CLASSES[numpy_name.replace("Vectorized", "RustArray")]
    if "ArrayBackprop" in numpy_name:
        return RUST_CLASSES[numpy_name.replace("ArrayBackprop", "RustArrayBackprop")]
    return RUST_CLASSES[numpy_name.replace("ArrayNetwork", "RustArrayNetwork")]


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
    if network.format2_shape == "sequence":
        # token ids, and a class per token
        classes = network.class_count
        return [
            (
                tuple(float(rng.randrange(classes)) for _ in range(width)),
                tuple(rng.randrange(classes) for _ in range(width)),
            )
            for _ in range(4)
        ]
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


# pure Python: (pure-Python class name, numpy twin's name); each is built with its twin's arguments,
# the multiclass and single-output ones with input bounds after the dimension
PYTHON_MULTICLASS = {
    "MultiClassBackpropClassifierNetwork": "VectorizedMultiClassBackpropClassifierNetwork",
    **{
        f"{prefix}MultiClassBackpropClassifierNetwork": f"{twin}VectorizedMultiClassBackpropClassifierNetwork"
        for prefix, twin in [
            ("Adam", "Adam"),
            ("CrossEntropy", "CrossEntropy"),
            ("Dropout", "Dropout"),
            ("L2Regularized", "L2"),
            ("Momentum", "Momentum"),
            ("ReLU", "ReLU"),
            ("Softmax", "Softmax"),
        ]
    },
}
PYTHON_CONV = {
    python.replace("MultiClass", "ConvMultiClass"): numpy.replace("Vectorized", "ConvVectorized")
    for python, numpy in PYTHON_MULTICLASS.items()
}
PYTHON_SEQUENTIAL = {
    "SequentialMultiClassBackpropClassifierNetwork": "SequentialVectorizedMultiClassBackpropClassifierNetwork",
    "SequentialBackpropClassifierNetwork": "SequentialArrayBackpropClassifierNetwork",
    "SequentialSequenceBackpropNetwork": "SequentialSequenceArrayNetwork",
}
PYTHON_SINGLE_OUTPUT = {"FanInAwareBackpropClassifierNetwork": "ArrayBackpropClassifierNetwork"}
# bounds-width randomize (BackpropClassifierNetwork.randomize), which no array network draws
PYTHON_WITHOUT_TWIN = {
    "BackpropClassifierNetwork",
    "AdamBackpropClassifierNetwork",
    "BinaryCrossEntropyBackpropClassifierNetwork",
    "DropoutBackpropClassifierNetwork",
    "L2RegularizedBackpropClassifierNetwork",
    "MomentumBackpropClassifierNetwork",
    "ReLUBackpropClassifierNetwork",
}
PYTHON_CLASSES = _classes(BackpropNetworkBase)


def test_every_pure_python_network_class_is_covered():
    twins = {**PYTHON_MULTICLASS, **PYTHON_CONV, **PYTHON_SEQUENTIAL, **PYTHON_SINGLE_OUTPUT}
    assert set(twins) | PYTHON_WITHOUT_TWIN == set(PYTHON_CLASSES)
    assert set(twins.values()) <= set(NUMPY_CLASSES)


def assert_pure_python_seeded_like_numpy(python: Any, numpy_network: Any) -> None:
    expected = [
        [np.asarray(array, dtype=np.float64).tobytes() for array in entry] for entry in numpy_network.snapshot()
    ]
    actual = [
        [np.asarray(values, dtype=np.float64).tobytes() for values in entry] for entry in as_array_snapshot(python)
    ]
    assert actual == expected
    assert generator_state(python.rng) == numpy_network.rng.bit_generator.state


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("python_name", PYTHON_MULTICLASS)
def test_pure_python_multiclass_randomized_is_numpys_after_the_same_seed(python_name: str, seed: int):
    numpy_name = PYTHON_MULTICLASS[python_name]
    args = MULTICLASS[numpy_name]
    python = PYTHON_CLASSES[python_name].randomized([7, 5], 12, [(0.0, 1.0)] * 12, CLASS_COUNT, *args, seed=seed)
    numpy_network = NUMPY_CLASSES[numpy_name].randomized([7, 5], 12, CLASS_COUNT, *args, seed=seed)
    assert_pure_python_seeded_like_numpy(python, numpy_network)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("python_name", PYTHON_CONV)
def test_pure_python_conv_randomized_is_numpys_after_the_same_seed(python_name: str, seed: int):
    numpy_name = PYTHON_CONV[python_name]
    args = CONV[numpy_name]
    python = PYTHON_CLASSES[python_name].randomized(SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT, *args, seed=seed)
    numpy_network = NUMPY_CLASSES[numpy_name].randomized(SIDE, SIDE, CONV_SPECS, [5], CLASS_COUNT, *args, seed=seed)
    assert_pure_python_seeded_like_numpy(python, numpy_network)


@pytest.mark.parametrize("seed", SEEDS)
def test_pure_python_fan_in_aware_single_output_randomized_is_numpys_after_the_same_seed(seed: int):
    python = PYTHON_CLASSES["FanInAwareBackpropClassifierNetwork"].randomized([4], 9, [(0.0, 1.0)] * 9, seed=seed)
    numpy_network = NUMPY_CLASSES["ArrayBackpropClassifierNetwork"].randomized([4], 9, seed=seed)
    assert_pure_python_seeded_like_numpy(python, numpy_network)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("python_name", PYTHON_SEQUENTIAL)
def test_pure_python_sequential_randomized_is_numpys_after_the_same_seed(python_name: str, seed: int):
    numpy_name = PYTHON_SEQUENTIAL[python_name]
    python = PYTHON_CLASSES[python_name].randomized(*SEQUENTIAL[numpy_name], seed=seed)
    numpy_network = NUMPY_CLASSES[numpy_name].randomized(*SEQUENTIAL[numpy_name], seed=seed)
    assert_pure_python_seeded_like_numpy(python, numpy_network)


def _record_python_masks(python: Any) -> list[list[list[float]]]:
    # each dropout layer's mask per forward pass, in call order: a training batch's (batch, size)
    # mask per layer, as numpy's _mask_batch, when the layer runs every example before the next layer
    masks: list[list[list[float]]] = []
    for layer in python._generator_layers:
        rows: list[list[float]] = []
        masks.append(rows)

        def forward(layer: Any = layer, rows: list[list[float]] = rows, inner: Any = layer.forward) -> None:
            inner()
            rows.append([1.0 if node._kept else 0.0 for node in layer.nodes])

        layer.forward = forward
    return masks


def assert_pure_python_trains_like_numpy(python: Any, numpy_network: Any) -> None:
    # a seeded training batch draws numpy's masks, layer by layer (the RNG draw-order workplan, D3),
    # and leaves the generator in numpy's state
    assert_pure_python_seeded_like_numpy(python, numpy_network)
    width = numpy_network.input_shape[0] if len(numpy_network.input_shape) == 1 else SIDE * SIDE
    rows = _rows(width, numpy_network)
    masks = _record_python_masks(python)
    python.learn_batch(0.5, rows)
    numpy_network.learn_batch(0.5, rows)
    assert [np.array(mask).tobytes() for mask in masks] == _mask_bits(numpy_network)
    assert generator_state(python.rng) == numpy_network.rng.bit_generator.state


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("layer_sizes", [[7], [7, 5], [6, 5, 4]], ids=["one", "two", "three"])
def test_pure_python_dropout_masks_are_numpys_after_the_same_seed(layer_sizes: list[int], seed: int):
    # every hidden layer drops out: one, two and three dropout layers
    python = PYTHON_CLASSES["DropoutMultiClassBackpropClassifierNetwork"].randomized(
        layer_sizes, 12, [(0.0, 1.0)] * 12, CLASS_COUNT, 0.3, seed=seed
    )
    numpy_network = NUMPY_CLASSES["DropoutVectorizedMultiClassBackpropClassifierNetwork"].randomized(
        layer_sizes, 12, CLASS_COUNT, 0.3, seed=seed
    )
    assert_pure_python_trains_like_numpy(python, numpy_network)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "python_name",
    ["DropoutConvMultiClassBackpropClassifierNetwork", "SequentialMultiClassBackpropClassifierNetwork"],
)
def test_pure_python_conv_dropout_masks_are_numpys_after_the_same_seed(python_name: str, seed: int):
    # a dropout layer behind conv and pool layers (the preset, and a Sequential network's specs)
    if python_name in PYTHON_CONV:
        numpy_name = PYTHON_CONV[python_name]
        args: tuple[Any, ...] = (SIDE, SIDE, CONV_SPECS, [5, 4], CLASS_COUNT, 0.3)
    else:
        numpy_name = PYTHON_SEQUENTIAL[python_name]
        args = SEQUENTIAL[numpy_name]
    python = PYTHON_CLASSES[python_name].randomized(*args, seed=seed)
    numpy_network = NUMPY_CLASSES[numpy_name].randomized(*args, seed=seed)
    assert_pure_python_trains_like_numpy(python, numpy_network)
