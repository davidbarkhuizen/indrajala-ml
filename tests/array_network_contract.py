"""
The tests every multiclass array network shares, numpy and Rust: parity with its per-node
reference network (predictions, and every step of learn and learn_batch), randomized, snapshot
and save/load round trips, and argument validation.

A test file declares an ArrayNetworkSpec and adds the generated tests to its module, which keeps
each test's usual name and id (test_x[numpy], test_x[rust]):

    globals().update(multiclass_network_tests(SPEC))
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import pytest

from tests.helpers import (
    assert_array_network_save_load_round_trip,
    assert_array_network_snapshot_restore_round_trip,
    assert_array_network_weights_match,
)

DIMENSION = 6
LAYER_SIZES = [5]
CLASS_COUNT = 3


@dataclass(frozen=True)
class ArrayNetworkSpec:
    # backend name -> network class
    network_cls: dict[str, type]
    # a matching_*_array_backprop_networks helper: builds the per-node reference and the array
    # network with identical weights
    matching: Callable
    # the constructor's hyperparameters after class_count, in order (e.g. {"momentum": 0.5})
    hyperparameters: dict = field(default_factory=dict)
    learning_rate: float = 0.1
    learn_steps: int = 30
    learn_batches: int = 15
    # False when training can't be compared with the reference (dropout's masks come from
    # unrelated RNGs): predictions are compared at eval only, and there are no learn tests
    parity_in_training: bool = True
    probabilities_sum_to_one: bool = False
    # the save/load round-trip test for the hyperparameters: its name suffix and the values saved
    saved_hyperparameters_test: str | None = None
    saved_hyperparameters: dict = field(default_factory=dict)
    # hyperparameter values the constructor must reject
    invalid_hyperparameters: dict = field(default_factory=dict)


def multiclass_network_tests(spec: ArrayNetworkSpec) -> dict[str, Callable]:
    hyperparameters = tuple(spec.hyperparameters.values())
    eval_suffix = "" if spec.parity_in_training else "_at_eval_mode"
    tests: dict[str, Callable] = {}

    def test(name: str):
        def register(function: Callable) -> Callable:
            function.__name__ = name
            tests[name] = function
            return function

        return register

    def matching_networks(rng: random.Random, backend):
        return spec.matching(
            rng, spec.network_cls[backend.name], backend.owned, LAYER_SIZES, DIMENSION, CLASS_COUNT, *hyperparameters
        )

    def randomized(backend):
        return spec.network_cls[backend.name].randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT, *hyperparameters)

    @test(f"test_predict_probabilities{eval_suffix}_matches_across_a_random_sweep")
    def _(backend):
        rng = random.Random(0)
        node_network, array_network = matching_networks(rng, backend)

        for _ in range(50):
            state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
            expected = node_network.predict_probabilities(state)
            actual = array_network.predict_probabilities(state)
            assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12)
            if spec.probabilities_sum_to_one:
                assert sum(actual) == pytest.approx(1.0)

    @test(f"test_classify_state{eval_suffix}_matches_across_a_random_sweep")
    def _(backend):
        rng = random.Random(1)
        node_network, array_network = matching_networks(rng, backend)

        for _ in range(50):
            state = tuple(rng.uniform(-10.0, 10.0) for _ in range(DIMENSION))
            assert array_network.classify_state(state) == node_network.classify_state(state)

    if spec.parity_in_training:

        @test("test_learn_matches_after_every_step_not_just_at_the_end")
        def _(backend):
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
        def _(backend):
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
    def _(backend):
        network = randomized(backend)
        state = tuple(0.1 * i for i in range(DIMENSION))

        probabilities = network.predict_probabilities(state)
        assert len(probabilities) == CLASS_COUNT
        assert all(0.0 <= p <= 1.0 for p in probabilities)
        if spec.probabilities_sum_to_one:
            assert sum(probabilities) == pytest.approx(1.0)
        assert 0 <= network.classify_state(state) < CLASS_COUNT

    @test("test_snapshot_restore_round_trips_weights")
    def _(backend):
        assert_array_network_snapshot_restore_round_trip(
            spec.network_cls[backend.name], LAYER_SIZES, DIMENSION, CLASS_COUNT, *hyperparameters
        )

    @test("test_save_load_round_trips_weights_and_predictions")
    def _(backend, tmp_path):
        network = randomized(backend)
        state = tuple(0.1 * i for i in range(DIMENSION))

        network_cls = spec.network_cls[backend.name]
        assert_array_network_save_load_round_trip(network, network_cls.load, tmp_path, "model.json", state)

    if spec.saved_hyperparameters_test is not None:

        @test(f"test_save_load_round_trips_the_{spec.saved_hyperparameters_test}")
        def _(backend, tmp_path):
            network_cls = spec.network_cls[backend.name]
            network = network_cls.randomized(LAYER_SIZES, DIMENSION, CLASS_COUNT, **spec.saved_hyperparameters)
            path = str(tmp_path / "hyperparameters.json")
            network.save(path)

            loaded = network_cls.load(path)
            for name, value in spec.saved_hyperparameters.items():
                assert getattr(loaded, name) == value

    @test("test_construction_rejects_invalid_arguments")
    def _(backend):
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
    def _(backend):
        network = randomized(backend)
        with pytest.raises(AssertionError):
            network.learn_batch(0.1, [])

    return tests
