"""
The tests every array network shares, numpy and Rust: parity with its pure-Python reference network
(predictions, and every step of learn and learn_batch), randomized, snapshot and save/load round
trips, and argument validation.

Both the reference and the bit-level check come from the network's equivalent, the layer specs and
update rule the test file declares: the reference is the pure-Python sequential network of them
(sequential_backprop_network.py), and the array sequential network of them
(sequential_array_network.py), built generically, must match the network by bits at every step
(stages 3 and 4 of the composable-layers workplan).

A test file declares an ArrayNetworkSpec and adds the generated tests to its module, which keeps
each test's usual name and id (test_x[numpy], test_x[rust]):

    globals().update(multiclass_network_tests(SPEC))    # or single_output_network_tests(SPEC),
                                                        # or conv_network_tests(SPEC)
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.sequential_array_network import (
    SequentialArrayBackpropClassifierNetwork,
    SequentialRustArrayBackpropClassifierNetwork,
    SequentialRustArrayMultiClassBackpropClassifierNetwork,
    SequentialVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.layer_specs import LayerSpec
from indrajala_ml.model.specs.update_rules import UpdateRule
from tests.helpers import (
    Backend,
    approx,
    assert_array_network_save_load_round_trip,
    assert_array_network_snapshot_restore_round_trip,
    assert_array_network_weights_match,
    assert_conv_array_network_weights_match,
    assert_single_output_array_network_save_load_round_trip,
    assert_single_output_array_network_snapshot_restore_round_trip,
    conv_reference,
    dense_reference,
)

DIMENSION = 6
LAYER_SIZES = [5]
CLASS_COUNT = 3
# the conv networks' input and architecture: an overlapping pool and a strided conv after it
CONV_SIDE = 8
CONV_SPECS: list[ConvSpec | PoolSpec] = [ConvSpec(3, 4), PoolSpec(2, stride=1), ConvSpec(2, 3, stride=2)]
CONV_DENSE_LAYER_SIZES = [6]

# a network class, pure-Python or array, is Any here: the specs cover every array network class, whose
# constructors take different hyperparameters
TestFunction = Callable[..., None]


@dataclass(frozen=True)
class ArrayNetworkSpec:
    # backend name -> network class
    network_cls: dict[str, Any]
    # the network as layer specs and an update rule, written out independently of its class:
    # equivalent(layer_sizes, output_size, *hyperparameters) -> (specs, rule). Its pure-Python
    # reference and its sequential counterpart are both built from them. A conv network's specs
    # are its dense layers', after its conv specs
    equivalent: Callable[..., tuple[list[LayerSpec], UpdateRule]]
    # the constructor's hyperparameters, in order (e.g. {"momentum": 0.5}): positional after
    # class_count in a multiclass or conv network, keyword-only after input_bounds in a
    # single-output one
    hyperparameters: dict[str, float] = field(default_factory=dict[str, float])
    learning_rate: float = 0.1
    learn_steps: int = 30
    learn_batches: int = 15
    # False when training can't be compared with the reference (the pure-Python dropout layers
    # draw their masks from Python's random): predictions are compared at eval only, and there are
    # no learn tests
    parity_in_training: bool = True
    # multiclass and conv only
    probabilities_sum_to_one: bool = False
    # the save/load round-trip test for the hyperparameters: its name suffix and the values saved
    saved_hyperparameters_test: str | None = None
    saved_hyperparameters: dict[str, float] = field(default_factory=dict[str, float])
    # hyperparameter values the constructor must reject
    invalid_hyperparameters: dict[str, float] = field(default_factory=dict[str, float])


SEQUENTIAL_CLS = {
    "multiclass": {
        "numpy": SequentialVectorizedMultiClassBackpropClassifierNetwork,
        "rust": SequentialRustArrayMultiClassBackpropClassifierNetwork,
    },
    "single_output": {
        "numpy": SequentialArrayBackpropClassifierNetwork,
        "rust": SequentialRustArrayBackpropClassifierNetwork,
    },
}


def snapshot_bits(network: Any) -> list[list[bytes]]:
    return [[np.asarray(array.tolist(), dtype=np.float64).tobytes() for array in entry] for entry in network.snapshot()]


def assert_sequential_matches_preset(
    preset: Any,
    sequential: Any,
    backend: Backend,
    examples: Callable[[random.Random], tuple[tuple[float, ...], Any]],
    learning_rate: float,
    steps: int = 10,
    batches: int = 5,
) -> None:
    """
    sequential, a generically built network, is preset by bits: the same layer classes, then,
    from preset's randomized weights, the same weights after every learn and learn_batch step and
    the same outputs. Both networks' generators are reseeded before each step, so dropout draws
    the same masks.
    """
    assert [type(layer) for layer in sequential.layers] == [type(layer) for layer in preset.layers]
    assert sequential.optimizer.rule == preset.optimizer.rule

    preset.rng = backend.default_rng(0)
    preset.randomize()
    sequential.restore(preset.snapshot())
    assert snapshot_bits(sequential) == snapshot_bits(preset)

    rng = random.Random(4)
    for step in range(steps):
        state, category = examples(rng)
        for network in (preset, sequential):
            network.rng = backend.default_rng(step)
            network.learn(learning_rate, state, category)
        assert snapshot_bits(sequential) == snapshot_bits(preset), f"after learn step {step}"

    for step in range(batches):
        batch = [examples(rng) for _ in range(8)]
        for network in (preset, sequential):
            network.rng = backend.default_rng(steps + step)
            network.learn_batch(learning_rate, batch)
        assert snapshot_bits(sequential) == snapshot_bits(preset), f"after learn_batch step {step}"

    state, _category = examples(rng)
    assert sequential.classify_state(state) == preset.classify_state(state)


def _test_registry() -> tuple[dict[str, TestFunction], Callable[[str], Callable[[TestFunction], TestFunction]]]:
    tests: dict[str, TestFunction] = {}

    def test(name: str) -> Callable[[TestFunction], TestFunction]:
        def register(function: TestFunction) -> TestFunction:
            function.__name__ = name
            tests[name] = function
            return function

        return register

    return tests, test


def multiclass_network_tests(spec: ArrayNetworkSpec) -> dict[str, TestFunction]:
    hyperparameters = tuple(spec.hyperparameters.values())
    eval_suffix = "" if spec.parity_in_training else "_at_eval_mode"
    tests, test = _test_registry()

    def matching_networks(rng: random.Random, backend: Backend) -> tuple[Any, Any]:
        array_network = spec.network_cls[backend.name](LAYER_SIZES, DIMENSION, CLASS_COUNT, *hyperparameters)
        specs, rule = spec.equivalent(LAYER_SIZES, CLASS_COUNT, *hyperparameters)
        return dense_reference(rng, array_network, specs, rule, backend.owned, DIMENSION), array_network

    def randomized(backend: Backend) -> Any:
        return spec.network_cls[backend.name].randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT, *hyperparameters)

    @test(f"test_predict_probabilities{eval_suffix}_matches_across_a_random_sweep")
    def _(backend: Backend) -> None:
        rng = random.Random(0)
        node_network, array_network = matching_networks(rng, backend)

        for _ in range(50):
            state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
            expected = node_network.predict_probabilities(state)
            actual = array_network.predict_probabilities(state)
            assert actual == approx(expected, rel=1e-9, abs=1e-12)
            if spec.probabilities_sum_to_one:
                assert sum(actual) == approx(1.0)

    @test(f"test_classify_state{eval_suffix}_matches_across_a_random_sweep")
    def _(backend: Backend) -> None:
        rng = random.Random(1)
        node_network, array_network = matching_networks(rng, backend)

        for _ in range(50):
            state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
            assert array_network.classify_state(state) == node_network.classify_state(state)

    if spec.parity_in_training:

        @test("test_learn_matches_after_every_step_not_just_at_the_end")
        def _(backend: Backend) -> None:
            # checked after every step, so one wrong step can't be averaged away by the rest (and
            # a stateful update, e.g. Adam's m/v/t, shows a mistake only across steps)
            rng = random.Random(2)
            node_network, array_network = matching_networks(rng, backend)

            for _ in range(spec.learn_steps):
                state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
                category = rng.randrange(CLASS_COUNT)

                node_network.learn(spec.learning_rate, state, category)
                array_network.learn(spec.learning_rate, state, category)

                assert_array_network_weights_match(node_network, array_network)

        @test("test_learn_batch_matches_after_every_batch_not_just_at_the_end")
        def _(backend: Backend) -> None:
            rng = random.Random(3)
            node_network, array_network = matching_networks(rng, backend)
            batch_size = 8

            for _ in range(spec.learn_batches):
                batch = [
                    (tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION)), rng.randrange(CLASS_COUNT))
                    for _ in range(batch_size)
                ]

                node_network.learn_batch(spec.learning_rate, batch)
                array_network.learn_batch(spec.learning_rate, batch)

                assert_array_network_weights_match(node_network, array_network)

    @test("test_randomized_builds_a_usable_network")
    def _(backend: Backend) -> None:
        network = randomized(backend)
        state = tuple(0.1 * i for i in range(DIMENSION))

        probabilities = network.predict_probabilities(state)
        assert len(probabilities) == CLASS_COUNT
        assert all(0.0 <= p <= 1.0 for p in probabilities)
        if spec.probabilities_sum_to_one:
            assert sum(probabilities) == approx(1.0)
        assert 0 <= network.classify_state(state) < CLASS_COUNT

    @test("test_snapshot_restore_round_trips_weights")
    def _(backend: Backend) -> None:
        assert_array_network_snapshot_restore_round_trip(
            spec.network_cls[backend.name], LAYER_SIZES, DIMENSION, CLASS_COUNT, *hyperparameters
        )

    @test("test_save_load_round_trips_weights_and_predictions")
    def _(backend: Backend, tmp_path: Path) -> None:
        network = randomized(backend)
        state = tuple(0.1 * i for i in range(DIMENSION))

        network_cls = spec.network_cls[backend.name]
        assert_array_network_save_load_round_trip(network, network_cls.load, tmp_path, "model.json", state)

    if spec.saved_hyperparameters_test is not None:

        @test(f"test_save_load_round_trips_the_{spec.saved_hyperparameters_test}")
        def _(backend: Backend, tmp_path: Path) -> None:
            network_cls = spec.network_cls[backend.name]
            network = network_cls.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT, **spec.saved_hyperparameters)
            path = str(tmp_path / "hyperparameters.json")
            network.save(path)

            loaded = network_cls.load(path)
            for name, value in spec.saved_hyperparameters.items():
                assert getattr(loaded, name) == value

    @test("test_construction_rejects_invalid_arguments")
    def _(backend: Backend) -> None:
        network_cls = spec.network_cls[backend.name]

        with pytest.raises(AssertionError):
            network_cls([], DIMENSION, CLASS_COUNT, *hyperparameters)

        with pytest.raises(AssertionError):
            network_cls([0], DIMENSION, CLASS_COUNT, *hyperparameters)

        with pytest.raises(AssertionError):
            network_cls(LAYER_SIZES, DIMENSION, 1, *hyperparameters)

        for name, value in spec.invalid_hyperparameters.items():
            with pytest.raises(AssertionError):
                network_cls(LAYER_SIZES, DIMENSION, CLASS_COUNT, **{**spec.hyperparameters, name: value})

    @test("test_learn_batch_rejects_an_empty_batch")
    def _(backend: Backend) -> None:
        network = randomized(backend)
        with pytest.raises(AssertionError):
            network.learn_batch(0.1, [])

    @test("test_the_sequential_network_of_its_layer_specs_matches_it_by_bits")
    def _(backend: Backend) -> None:
        preset = spec.network_cls[backend.name](LAYER_SIZES, DIMENSION, CLASS_COUNT, *hyperparameters)
        specs, rule = spec.equivalent(LAYER_SIZES, CLASS_COUNT, *hyperparameters)
        sequential: Any = SEQUENTIAL_CLS["multiclass"][backend.name]((DIMENSION,), specs, rule)

        def example(rng: random.Random) -> tuple[tuple[float, ...], int]:
            return tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION)), rng.randrange(CLASS_COUNT)

        assert_sequential_matches_preset(preset, sequential, backend, example, spec.learning_rate)
        state = tuple(0.1 * i for i in range(DIMENSION))
        assert sequential.predict_probabilities(state) == preset.predict_probabilities(state)

    return tests


def single_output_network_tests(spec: ArrayNetworkSpec) -> dict[str, TestFunction]:
    hyperparameters = spec.hyperparameters
    eval_suffix = "" if spec.parity_in_training else "_at_eval_mode"
    tests, test = _test_registry()

    def matching_networks(rng: random.Random, backend: Backend) -> tuple[Any, Any]:
        array_network = spec.network_cls[backend.name](LAYER_SIZES, DIMENSION, **hyperparameters)
        specs, rule = spec.equivalent(LAYER_SIZES, 1, *hyperparameters.values())
        reference = dense_reference(rng, array_network, specs, rule, backend.owned, DIMENSION, "single_output")
        return reference, array_network

    def randomized(backend: Backend) -> Any:
        return spec.network_cls[backend.name].randomized(LAYER_SIZES, DIMENSION, **hyperparameters)

    @test(f"test_predict_probability{eval_suffix}_matches_across_a_random_sweep")
    def _(backend: Backend) -> None:
        rng = random.Random(0)
        node_network, array_network = matching_networks(rng, backend)

        for _ in range(50):
            state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
            expected = node_network.predict_probability(state)
            actual = array_network.predict_probability(state)
            assert actual == approx(expected, rel=1e-9, abs=1e-12)

    @test(f"test_classify_state{eval_suffix}_matches_across_a_random_sweep")
    def _(backend: Backend) -> None:
        rng = random.Random(1)
        node_network, array_network = matching_networks(rng, backend)

        for _ in range(50):
            state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
            assert array_network.classify_state(state) == node_network.classify_state(state)

    if spec.parity_in_training:

        @test("test_learn_matches_after_every_step_not_just_at_the_end")
        def _(backend: Backend) -> None:
            rng = random.Random(2)
            node_network, array_network = matching_networks(rng, backend)

            for _ in range(spec.learn_steps):
                state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
                category = float(rng.randrange(2))

                node_network.learn(spec.learning_rate, state, category)
                array_network.learn(spec.learning_rate, state, category)

                assert_array_network_weights_match(node_network, array_network)

        @test("test_learn_batch_matches_after_every_batch_not_just_at_the_end")
        def _(backend: Backend) -> None:
            rng = random.Random(3)
            node_network, array_network = matching_networks(rng, backend)
            batch_size = 8

            for _ in range(spec.learn_batches):
                batch = [
                    (tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION)), float(rng.randrange(2)))
                    for _ in range(batch_size)
                ]

                node_network.learn_batch(spec.learning_rate, batch)
                array_network.learn_batch(spec.learning_rate, batch)

                assert_array_network_weights_match(node_network, array_network)

    @test("test_randomized_builds_a_usable_network")
    def _(backend: Backend) -> None:
        network = randomized(backend)
        state = tuple(0.1 * i for i in range(DIMENSION))

        assert 0.0 <= network.predict_probability(state) <= 1.0
        assert network.classify_state(state) in (0.0, 1.0)

    @test("test_randomized_accepts_and_discards_input_bounds_for_duck_type_compatibility")
    def _(backend: Backend) -> None:
        # ensemble_train.py calls classifier_cls.randomized(layer_sizes, dimension, input_bounds)
        network_cls = spec.network_cls[backend.name]
        network = network_cls.randomized(LAYER_SIZES, DIMENSION, [(-1.0, 1.0)] * DIMENSION, **hyperparameters)
        state = tuple(0.1 * i for i in range(DIMENSION))
        assert 0.0 <= network.predict_probability(state) <= 1.0

    @test("test_snapshot_restore_round_trips_weights")
    def _(backend: Backend) -> None:
        assert_single_output_array_network_snapshot_restore_round_trip(
            spec.network_cls[backend.name], LAYER_SIZES, DIMENSION, **hyperparameters
        )

    @test("test_save_load_round_trips_weights_and_predictions")
    def _(backend: Backend, tmp_path: Path) -> None:
        network = randomized(backend)
        state = tuple(0.1 * i for i in range(DIMENSION))

        network_cls = spec.network_cls[backend.name]
        assert_single_output_array_network_save_load_round_trip(
            network, network_cls.load, tmp_path, "model.json", state
        )

    if spec.saved_hyperparameters_test is not None:

        @test(f"test_save_load_round_trips_the_{spec.saved_hyperparameters_test}")
        def _(backend: Backend, tmp_path: Path) -> None:
            network_cls = spec.network_cls[backend.name]
            network = network_cls.randomized(LAYER_SIZES, DIMENSION, **spec.saved_hyperparameters)
            path = str(tmp_path / "hyperparameters.json")
            network.save(path)

            loaded = network_cls.load(path)
            for name, value in spec.saved_hyperparameters.items():
                assert getattr(loaded, name) == value

    @test("test_construction_rejects_invalid_arguments")
    def _(backend: Backend) -> None:
        network_cls = spec.network_cls[backend.name]

        with pytest.raises(AssertionError):
            network_cls([], DIMENSION, **hyperparameters)

        with pytest.raises(AssertionError):
            network_cls([0], DIMENSION, **hyperparameters)

        for name, value in spec.invalid_hyperparameters.items():
            with pytest.raises(AssertionError):
                network_cls(LAYER_SIZES, DIMENSION, **{**hyperparameters, name: value})

    @test("test_learn_batch_rejects_an_empty_batch")
    def _(backend: Backend) -> None:
        network = randomized(backend)
        with pytest.raises(AssertionError):
            network.learn_batch(0.1, [])

    @test("test_the_sequential_network_of_its_layer_specs_matches_it_by_bits")
    def _(backend: Backend) -> None:
        preset = spec.network_cls[backend.name](LAYER_SIZES, DIMENSION, **hyperparameters)
        specs, rule = spec.equivalent(LAYER_SIZES, 1, *hyperparameters.values())
        sequential: Any = SEQUENTIAL_CLS["single_output"][backend.name]((DIMENSION,), specs, rule)

        def example(rng: random.Random) -> tuple[tuple[float, ...], float]:
            return tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION)), float(rng.randrange(2))

        assert_sequential_matches_preset(preset, sequential, backend, example, spec.learning_rate)
        state = tuple(0.1 * i for i in range(DIMENSION))
        assert sequential.predict_probability(state) == preset.predict_probability(state)

    return tests


def conv_network_tests(spec: ArrayNetworkSpec) -> dict[str, TestFunction]:
    hyperparameters = tuple(spec.hyperparameters.values())
    input_size = CONV_SIDE * CONV_SIDE
    tests, test = _test_registry()

    def build(backend: Backend, *arguments: float, **keywords: float) -> Any:
        return spec.network_cls[backend.name](
            CONV_SIDE, CONV_SIDE, CONV_SPECS, CONV_DENSE_LAYER_SIZES, CLASS_COUNT, *arguments, **keywords
        )

    def equivalent() -> tuple[list[LayerSpec], UpdateRule]:
        dense_specs, rule = spec.equivalent(CONV_DENSE_LAYER_SIZES, CLASS_COUNT, *hyperparameters)
        return [*CONV_SPECS, *dense_specs], rule

    def matching_networks(rng: random.Random, backend: Backend) -> tuple[Any, Any]:
        array_network = build(backend, *hyperparameters)
        specs, rule = equivalent()
        reference = conv_reference(rng, array_network, (CONV_SIDE, CONV_SIDE, 1), specs, rule, backend.owned)
        return reference, array_network

    def example(rng: random.Random) -> tuple[tuple[float, ...], int]:
        return tuple(rng.random() for _ in range(input_size)), rng.randrange(CLASS_COUNT)

    @test("test_predict_probabilities_matches_the_pure_python_reference")
    def _(backend: Backend) -> None:
        rng = random.Random(0)
        reference, array_network = matching_networks(rng, backend)

        for _ in range(20):
            state, _category = example(rng)
            actual = array_network.predict_probabilities(state)
            assert actual == approx(reference.predict_probabilities(state), rel=1e-9, abs=1e-12)
            if spec.probabilities_sum_to_one:
                assert sum(actual) == approx(1.0)

    if spec.parity_in_training:

        @test("test_learn_matches_the_pure_python_reference_after_every_step")
        def _(backend: Backend) -> None:
            rng = random.Random(1)
            reference, array_network = matching_networks(rng, backend)

            for _ in range(spec.learn_steps):
                state, category = example(rng)
                reference.learn(spec.learning_rate, state, category)
                array_network.learn(spec.learning_rate, state, category)
                assert_conv_array_network_weights_match(reference, array_network)

        @test("test_learn_batch_matches_the_pure_python_reference_after_every_batch")
        def _(backend: Backend) -> None:
            rng = random.Random(2)
            reference, array_network = matching_networks(rng, backend)

            for _ in range(spec.learn_batches):
                batch = [example(rng) for _ in range(7)]
                reference.learn_batch(spec.learning_rate, batch)
                array_network.learn_batch(spec.learning_rate, batch)
                assert_conv_array_network_weights_match(reference, array_network)

    @test("test_the_optimizer_applies_its_rule")
    def _(backend: Backend) -> None:
        assert build(backend, *hyperparameters).optimizer.rule == equivalent()[1]

    @test("test_save_load_round_trips_weights_and_predictions")
    def _(backend: Backend, tmp_path: Path) -> None:
        network_cls = spec.network_cls[backend.name]
        network = network_cls.randomized(
            CONV_SIDE, CONV_SIDE, CONV_SPECS, CONV_DENSE_LAYER_SIZES, CLASS_COUNT, *hyperparameters
        )
        path = str(tmp_path / "model.json")
        network.save(path)

        loaded = network_cls.load(path)
        assert snapshot_bits(loaded) == snapshot_bits(network)
        assert loaded.conv_specs == network.conv_specs
        state, _category = example(random.Random(3))
        assert loaded.predict_probabilities(state) == network.predict_probabilities(state)

    if spec.saved_hyperparameters_test is not None:

        @test(f"test_save_load_round_trips_the_{spec.saved_hyperparameters_test}")
        def _(backend: Backend, tmp_path: Path) -> None:
            network_cls = spec.network_cls[backend.name]
            network = network_cls.randomized(
                CONV_SIDE, CONV_SIDE, CONV_SPECS, CONV_DENSE_LAYER_SIZES, CLASS_COUNT, **spec.saved_hyperparameters
            )
            path = str(tmp_path / "hyperparameters.json")
            network.save(path)

            loaded = network_cls.load(path)
            for name, value in spec.saved_hyperparameters.items():
                assert getattr(loaded, name) == value
            assert loaded.optimizer.rule == network.optimizer.rule

    if spec.invalid_hyperparameters:

        @test("test_construction_rejects_invalid_hyperparameters")
        def _(backend: Backend) -> None:
            for name, value in spec.invalid_hyperparameters.items():
                with pytest.raises(AssertionError):
                    build(backend, **{**spec.hyperparameters, name: value})

    @test("test_the_sequential_network_of_its_layer_specs_matches_it_by_bits")
    def _(backend: Backend) -> None:
        preset = build(backend, *hyperparameters)
        specs, rule = equivalent()
        sequential: Any = SEQUENTIAL_CLS["multiclass"][backend.name]((CONV_SIDE, CONV_SIDE, 1), specs, rule)

        assert_sequential_matches_preset(preset, sequential, backend, example, spec.learning_rate)
        state, _category = example(random.Random(5))
        assert sequential.predict_probabilities(state) == preset.predict_probabilities(state)

    return tests
