import math
import pickle
import random
from pathlib import Path
from typing import Any

import pytest

from indrajala_ml.ensemble_train import (
    _available_memory_bytes,
    _estimate_bytes_per_example,
    _select_worker_count,
    build_balanced_binary_dataset,
    select_balanced_indices,
    train_ensemble_parallel,
    train_ensemble_parallel_from_indices,
    train_ensemble_serial_from_indices,
)
from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.array_backprop_classifier_network import ArrayBackpropClassifierNetwork
from indrajala_ml.model.ensemble_array_backprop_classifier_network import EnsembleArrayBackpropClassifierNetwork
from indrajala_ml.model.ensemble_rust_array_backprop_classifier_network import (
    EnsembleRustArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.fan_in_aware_backprop_classifier_network import FanInAwareBackpropClassifierNetwork
from indrajala_ml.model.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork
from tests.helpers import approx


def _synthetic_dataset(counts: dict[int, int]) -> list[tuple[tuple[float, ...], int]]:
    # each example's state is a distinct singleton tuple, so identity is easy to trace
    dataset: list[tuple[tuple[float, ...], int]] = []
    for label, count in counts.items():
        for i in range(count):
            dataset.append(((float(label), float(i)), label))
    return dataset


def write_test_records(path: str, records: list[tuple[tuple[float, ...], int]]) -> None:
    # a stand-in on-disk dataset; module-level so its loader can be sent to worker processes
    with open(path, "wb") as f:
        pickle.dump(records, f)


def load_test_records_at_indices(path: str, indices: list[int]) -> list[tuple[tuple[float, ...], int]]:
    with open(path, "rb") as f:
        records = pickle.load(f)
    return [records[index] for index in indices]


def test_positives_are_exactly_every_example_of_the_target_label():

    dataset = _synthetic_dataset({0: 6, 1: 10, 2: 10, 3: 10})

    result = build_balanced_binary_dataset(dataset, target_label=0, class_count=4, rng=random.Random(0))

    positives = [state for state, category in result if category == 1.0]
    assert len(positives) == 6
    assert set(positives) == {(0.0, float(i)) for i in range(6)}


def test_negatives_are_evenly_stratified_across_other_classes():

    # 6 positives, 3 other classes, each with plenty available - divides evenly, so this
    # should be exactly 2 from each other class, not just 6 total drawn from a pooled sample
    dataset = _synthetic_dataset({0: 6, 1: 10, 2: 10, 3: 10})

    result = build_balanced_binary_dataset(dataset, target_label=0, class_count=4, rng=random.Random(0))

    negatives = [state for state, category in result if category == 0.0]
    assert len(negatives) == 6
    for other_label in (1, 2, 3):
        count_from_label = sum(1 for state in negatives if state[0] == float(other_label))
        assert count_from_label == 2


def test_negative_remainder_is_spread_across_the_first_few_classes():

    # 7 positives, 3 other classes: base_count=2, remainder=1 - the first class (sorted order)
    # gets 3, the other two get 2 each, totaling 7
    dataset = _synthetic_dataset({0: 7, 1: 10, 2: 10, 3: 10})

    result = build_balanced_binary_dataset(dataset, target_label=0, class_count=4, rng=random.Random(0))

    negatives = [state for state, category in result if category == 0.0]
    counts_by_label = {
        other_label: sum(1 for state in negatives if state[0] == float(other_label)) for other_label in (1, 2, 3)
    }
    assert sorted(counts_by_label.values()) == [2, 2, 3]
    assert sum(counts_by_label.values()) == 7


def test_negative_sampling_is_capped_by_availability():

    # class 1 has 1 example, not the 3 (9 positives / 3 classes) it would give: it gives 1,
    # without raising or padding, and the negatives fall short of 9
    dataset = _synthetic_dataset({0: 9, 1: 1, 2: 10, 3: 10})

    result = build_balanced_binary_dataset(dataset, target_label=0, class_count=4, rng=random.Random(0))

    negatives = [state for state, category in result if category == 0.0]
    count_from_label_1 = sum(1 for state in negatives if state[0] == 1.0)
    assert count_from_label_1 == 1
    assert len(negatives) == 1 + 3 + 3  # short of the full 9 by class 1's 2-example shortfall


def test_result_is_shuffled_and_labels_are_recoded_to_floats():

    dataset = _synthetic_dataset({0: 6, 1: 10, 2: 10, 3: 10})

    result = build_balanced_binary_dataset(dataset, target_label=0, class_count=4, rng=random.Random(0))

    assert len(result) == 12
    assert all(category in (0.0, 1.0) for _, category in result)
    # not simply "all positives first, then all negatives" - a real (if probabilistic) shuffle
    # check: the first half isn't all one category
    categories_in_order = [category for _, category in result]
    assert len(set(categories_in_order[:6])) == 2


def test_reproducible_under_a_fixed_seed():

    dataset = _synthetic_dataset({0: 6, 1: 10, 2: 10, 3: 10})

    result_a = build_balanced_binary_dataset(dataset, target_label=0, class_count=4, rng=random.Random(42))
    result_b = build_balanced_binary_dataset(dataset, target_label=0, class_count=4, rng=random.Random(42))

    assert result_a == result_b


def test_rejects_an_out_of_range_target_label():

    dataset = _synthetic_dataset({0: 6, 1: 10, 2: 10, 3: 10})

    with pytest.raises(AssertionError):
        build_balanced_binary_dataset(dataset, target_label=4, class_count=4, rng=random.Random(0))


def test_build_balanced_binary_dataset_matches_select_balanced_indices():

    dataset = _synthetic_dataset({0: 6, 1: 10, 2: 10, 3: 10})
    labels = [label for _, label in dataset]

    index_result = select_balanced_indices(labels, target_label=0, class_count=4, rng=random.Random(0))
    expected = [(dataset[index][0], category) for index, category in index_result]

    direct_result = build_balanced_binary_dataset(dataset, target_label=0, class_count=4, rng=random.Random(0))

    assert direct_result == expected


def test_select_balanced_indices_positives_are_exactly_every_index_of_the_target_label():

    labels = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]  # label 0 at indices 0, 4, 8, 12

    result = select_balanced_indices(labels, target_label=0, class_count=4, rng=random.Random(0))

    positive_indices = {index for index, category in result if category == 1.0}
    assert positive_indices == {0, 4, 8, 12}


def test_select_balanced_indices_rejects_an_out_of_range_target_label():

    labels = [0, 1, 2, 3]

    with pytest.raises(AssertionError):
        select_balanced_indices(labels, target_label=4, class_count=4, rng=random.Random(0))


def _synthetic_multiclass_dataset() -> list[tuple[tuple[float, float], int]]:
    # three well-separated 2D clusters: learnable, unlike the label-only dataset above
    centers = {0: (-5.0, -5.0), 1: (5.0, 5.0), 2: (5.0, -5.0)}
    rng = random.Random(1)
    dataset: list[tuple[tuple[float, float], int]] = []
    for label, (cx, cy) in centers.items():
        for _ in range(20):
            dataset.append(((cx + rng.uniform(-1.0, 1.0), cy + rng.uniform(-1.0, 1.0)), label))
    return dataset


def _train_synthetic(
    dataset: list[tuple[tuple[float, float], int]],
    bounds: list[tuple[float, float]],
    *,
    epochs: int,
    seed: int | None,
):
    # the fixed class_count/layer_sizes/dimension/learning_rate/worker_count every test in this
    # file trains the synthetic 3-cluster dataset with - only epochs/seed actually vary per test
    return train_ensemble_parallel(
        dataset,
        class_count=3,
        layer_sizes=[4],
        dimension=2,
        input_bounds=bounds,
        learning_rate=0.5,
        epochs=epochs,
        worker_count=2,
        seed=seed,
    )


def _train_synthetic_from_indices(
    path: str,
    labels: list[int],
    bounds: list[tuple[float, float]],
    *,
    epochs: int,
    seed: int | None,
):
    # the index-based counterpart to _train_synthetic, same fixed parameter set
    return train_ensemble_parallel_from_indices(
        path,
        load_test_records_at_indices,
        labels,
        class_count=3,
        layer_sizes=[4],
        dimension=2,
        input_bounds=bounds,
        learning_rate=0.5,
        epochs=epochs,
        worker_count=2,
        seed=seed,
    )


def test_train_ensemble_parallel_produces_a_working_ensemble():

    dataset = _synthetic_multiclass_dataset()
    bounds = square_bounds(10.0)

    ensemble, diagnostics = _train_synthetic(dataset, bounds, epochs=5, seed=0)

    assert ensemble.class_count == 3
    assert set(diagnostics.keys()) == {0, 1, 2}

    # the ensemble should classify each cluster's own center correctly - a low bar, but a real
    # end-to-end correctness check, not just "it ran without crashing"
    assert ensemble.classify_state((-5.0, -5.0)) == 0
    assert ensemble.classify_state((5.0, 5.0)) == 1
    assert ensemble.classify_state((5.0, -5.0)) == 2


def test_train_ensemble_parallel_respects_a_custom_classifier_cls():

    # classifier_cls must reach the worker processes. epochs=0 leaves the initial weights, and
    # only the fan-in-aware scheme bounds every layer, output included, by 1/sqrt(fan_in) (the
    # default draws the output layer from uniform(-1.0, 1.0))
    dataset = _synthetic_multiclass_dataset()
    bounds = square_bounds(10.0)

    ensemble, _ = train_ensemble_parallel(
        dataset,
        class_count=3,
        layer_sizes=[4],
        dimension=2,
        input_bounds=bounds,
        learning_rate=0.5,
        epochs=0,
        worker_count=2,
        seed=0,
        classifier_cls=FanInAwareBackpropClassifierNetwork,
    )

    limit = 1.0 / math.sqrt(2)
    for classifier in ensemble.classifiers:
        assert isinstance(classifier, FanInAwareBackpropClassifierNetwork)
        for node in classifier.output_layer.nodes:
            assert all(-limit <= w <= limit for w in node.input_node_weights)


def test_train_ensemble_parallel_gives_each_worker_independent_initial_weights():

    # epochs=0 leaves each sub-network untrained, so only the seeding can make them differ. With
    # epochs=1, a mutant giving every worker the same seed passed: training on different data
    # hid the identical initial weights
    dataset = _synthetic_multiclass_dataset()
    bounds = square_bounds(10.0)

    ensemble, _ = _train_synthetic(dataset, bounds, epochs=0, seed=0)

    weight_sets = [
        tuple(classifier.hidden_layers[0].nodes[0].input_node_weights) for classifier in ensemble.classifiers
    ]
    assert len(set(weight_sets)) == len(weight_sets)


def test_train_ensemble_parallel_is_reproducible_under_a_fixed_seed():

    dataset = _synthetic_multiclass_dataset()
    bounds = square_bounds(10.0)

    ensemble_a, _ = _train_synthetic(dataset, bounds, epochs=3, seed=7)
    ensemble_b, _ = _train_synthetic(dataset, bounds, epochs=3, seed=7)

    assert ensemble_a.snapshot() == ensemble_b.snapshot()


def test_train_ensemble_parallel_accepts_array_backed_classifier_cls():

    # the result is always the per-node EnsembleBackpropClassifierNetwork, which classifies
    # array classifiers fine but can't save them; EnsembleArrayBackpropClassifierNetwork
    # rewraps .classifiers for that
    dataset = _synthetic_multiclass_dataset()
    bounds = square_bounds(10.0)

    result, _ = train_ensemble_parallel(
        dataset,
        class_count=3,
        layer_sizes=[4],
        dimension=2,
        input_bounds=bounds,
        learning_rate=0.5,
        epochs=5,
        worker_count=2,
        seed=0,
        classifier_cls=ArrayBackpropClassifierNetwork,
    )

    assert all(isinstance(classifier, ArrayBackpropClassifierNetwork) for classifier in result.classifiers)
    assert result.classify_state((-5.0, -5.0)) == 0
    assert result.classify_state((5.0, 5.0)) == 1
    assert result.classify_state((5.0, -5.0)) == 2

    ensemble = EnsembleArrayBackpropClassifierNetwork(result.classifiers)
    assert ensemble.classify_state((-5.0, -5.0)) == 0


def test_train_ensemble_parallel_accepts_rust_array_backed_classifier_cls():

    # indrajala_math_rust.Array can't be pickled: _picklable_snapshot sends plain lists back
    # from the worker, and the Rust network's restore() accepts them
    dataset = _synthetic_multiclass_dataset()
    bounds = square_bounds(10.0)

    result, _ = train_ensemble_parallel(
        dataset,
        class_count=3,
        layer_sizes=[4],
        dimension=2,
        input_bounds=bounds,
        learning_rate=0.5,
        epochs=5,
        worker_count=2,
        seed=0,
        classifier_cls=RustArrayBackpropClassifierNetwork,
    )

    assert all(isinstance(classifier, RustArrayBackpropClassifierNetwork) for classifier in result.classifiers)
    assert result.classify_state((-5.0, -5.0)) == 0
    assert result.classify_state((5.0, 5.0)) == 1
    assert result.classify_state((5.0, -5.0)) == 2

    ensemble = EnsembleRustArrayBackpropClassifierNetwork(result.classifiers)
    assert ensemble.classify_state((-5.0, -5.0)) == 0


def test_train_ensemble_parallel_from_indices_produces_a_working_ensemble(tmp_path: Path):

    dataset = _synthetic_multiclass_dataset()
    labels = [label for _, label in dataset]
    path = str(tmp_path / "records.pkl")
    write_test_records(path, dataset)
    bounds = square_bounds(10.0)

    ensemble, diagnostics = _train_synthetic_from_indices(path, labels, bounds, epochs=5, seed=0)

    assert ensemble.class_count == 3
    assert set(diagnostics.keys()) == {0, 1, 2}
    assert ensemble.classify_state((-5.0, -5.0)) == 0
    assert ensemble.classify_state((5.0, 5.0)) == 1
    assert ensemble.classify_state((5.0, -5.0)) == 2


def test_train_ensemble_parallel_from_indices_matches_the_fully_decoded_path(tmp_path: Path):

    # same selection and per-worker seeds; only how each worker gets its examples differs
    dataset = _synthetic_multiclass_dataset()
    labels = [label for _, label in dataset]
    path = str(tmp_path / "records.pkl")
    write_test_records(path, dataset)
    bounds = square_bounds(10.0)

    indexed_ensemble, _ = _train_synthetic_from_indices(path, labels, bounds, epochs=3, seed=7)
    direct_ensemble, _ = _train_synthetic(dataset, bounds, epochs=3, seed=7)

    assert indexed_ensemble.snapshot() == direct_ensemble.snapshot()


def test_train_ensemble_serial_from_indices_produces_a_working_ensemble(tmp_path: Path):

    dataset = _synthetic_multiclass_dataset()
    labels = [label for _, label in dataset]
    path = str(tmp_path / "records.pkl")
    write_test_records(path, dataset)
    bounds = square_bounds(10.0)

    ensemble, diagnostics = train_ensemble_serial_from_indices(
        path,
        load_test_records_at_indices,
        labels,
        class_count=3,
        layer_sizes=[4],
        dimension=2,
        input_bounds=bounds,
        learning_rate=0.5,
        epochs=5,
        seed=0,
    )

    assert ensemble.class_count == 3
    assert set(diagnostics.keys()) == {0, 1, 2}
    assert ensemble.classify_state((-5.0, -5.0)) == 0
    assert ensemble.classify_state((5.0, 5.0)) == 1
    assert ensemble.classify_state((5.0, -5.0)) == 2


def test_train_ensemble_serial_from_indices_matches_the_parallel_path(tmp_path: Path):

    # no Pool, but the same selection and per-class seeds (one random.Random(seed), in class
    # order)
    dataset = _synthetic_multiclass_dataset()
    labels = [label for _, label in dataset]
    path = str(tmp_path / "records.pkl")
    write_test_records(path, dataset)
    bounds = square_bounds(10.0)

    serial_ensemble, _ = train_ensemble_serial_from_indices(
        path,
        load_test_records_at_indices,
        labels,
        class_count=3,
        layer_sizes=[4],
        dimension=2,
        input_bounds=bounds,
        learning_rate=0.5,
        epochs=3,
        seed=7,
    )
    parallel_ensemble, _ = _train_synthetic_from_indices(path, labels, bounds, epochs=3, seed=7)

    assert serial_ensemble.snapshot() == parallel_ensemble.snapshot()


def test_train_ensemble_serial_from_indices_accepts_array_and_rust_backed_classifier_cls(tmp_path: Path):

    # each backend's serial result must equal its parallel one
    dataset = _synthetic_multiclass_dataset()
    labels = [label for _, label in dataset]
    path = str(tmp_path / "records.pkl")
    write_test_records(path, dataset)
    bounds = square_bounds(10.0)

    for classifier_cls, ensemble_cls in [
        (ArrayBackpropClassifierNetwork, EnsembleArrayBackpropClassifierNetwork),
        (RustArrayBackpropClassifierNetwork, EnsembleRustArrayBackpropClassifierNetwork),
    ]:
        result, diagnostics = train_ensemble_serial_from_indices(
            path,
            load_test_records_at_indices,
            labels,
            class_count=3,
            layer_sizes=[4],
            dimension=2,
            input_bounds=bounds,
            learning_rate=0.5,
            epochs=5,
            seed=0,
            classifier_cls=classifier_cls,
        )

        assert set(diagnostics.keys()) == {0, 1, 2}
        ensemble = ensemble_cls(result.classifiers)
        assert ensemble.classify_state((-5.0, -5.0)) == 0
        assert ensemble.classify_state((5.0, 5.0)) == 1
        assert ensemble.classify_state((5.0, -5.0)) == 2


def test_available_memory_bytes_returns_a_real_positive_value_on_linux():

    # this dev environment is Linux (see cli's install_os_packages), so /proc/meminfo should
    # genuinely exist and parse to a plausible value, not just "doesn't crash"
    available = _available_memory_bytes()

    assert available is not None
    assert available > 0


def test_estimate_bytes_per_example_matches_a_direct_pickle_measurement():

    import pickle

    dataset = [((float(i), float(i) * 2), i % 3) for i in range(200)]

    estimated = _estimate_bytes_per_example(dataset, sample_size=50)

    direct = len(pickle.dumps(dataset[:50])) / 50
    assert estimated == approx(direct)


def test_estimate_bytes_per_example_handles_a_dataset_smaller_than_the_sample_size():

    dataset = [((1.0, 2.0), 0), ((3.0, 4.0), 1)]

    estimated = _estimate_bytes_per_example(dataset, sample_size=50)

    assert estimated > 0


def test_estimate_bytes_per_example_rejects_an_empty_dataset():

    with pytest.raises(AssertionError):
        _estimate_bytes_per_example([], sample_size=50)


def test_select_worker_count_is_limited_by_available_memory(monkeypatch: pytest.MonkeyPatch):

    # half of 100MB for workers at 1000 examples * 1000 bytes * 2.0 safety = 2MB each: 25
    # workers. cpu_count and class_count are set high enough not to bind
    monkeypatch.setattr("indrajala_ml.ensemble_train._available_memory_bytes", lambda: 100_000_000)
    monkeypatch.setattr("os.cpu_count", lambda: 64)

    worker_count = _select_worker_count(
        class_count=64, estimated_examples_per_classifier=1000, bytes_per_example=1000.0, requested_worker_count=None
    )

    assert worker_count == 25


def test_select_worker_count_is_limited_by_cpu_count_when_memory_is_abundant(monkeypatch: pytest.MonkeyPatch):

    monkeypatch.setattr("indrajala_ml.ensemble_train._available_memory_bytes", lambda: 10_000_000_000)
    monkeypatch.setattr("os.cpu_count", lambda: 4)

    worker_count = _select_worker_count(
        class_count=64, estimated_examples_per_classifier=1000, bytes_per_example=1000.0, requested_worker_count=None
    )

    assert worker_count == 4


def test_select_worker_count_is_limited_by_class_count(monkeypatch: pytest.MonkeyPatch):

    monkeypatch.setattr("indrajala_ml.ensemble_train._available_memory_bytes", lambda: 10_000_000_000)
    monkeypatch.setattr("os.cpu_count", lambda: 64)

    worker_count = _select_worker_count(
        class_count=3, estimated_examples_per_classifier=1000, bytes_per_example=1000.0, requested_worker_count=None
    )

    assert worker_count == 3


def test_select_worker_count_respects_an_explicit_lower_request(monkeypatch: pytest.MonkeyPatch):

    monkeypatch.setattr("indrajala_ml.ensemble_train._available_memory_bytes", lambda: 10_000_000_000)
    monkeypatch.setattr("os.cpu_count", lambda: 64)

    worker_count = _select_worker_count(
        class_count=64, estimated_examples_per_classifier=1000, bytes_per_example=1000.0, requested_worker_count=2
    )

    assert worker_count == 2


def test_select_worker_count_never_goes_below_one_even_under_severe_memory_pressure(monkeypatch: pytest.MonkeyPatch):

    monkeypatch.setattr("indrajala_ml.ensemble_train._available_memory_bytes", lambda: 1)
    monkeypatch.setattr("os.cpu_count", lambda: 64)

    worker_count = _select_worker_count(
        class_count=64,
        estimated_examples_per_classifier=1_000_000,
        bytes_per_example=1000.0,
        requested_worker_count=None,
    )

    assert worker_count == 1


def test_select_worker_count_falls_back_to_cpu_and_class_count_when_memory_is_undetectable(
    monkeypatch: pytest.MonkeyPatch,
):

    monkeypatch.setattr("indrajala_ml.ensemble_train._available_memory_bytes", lambda: None)
    monkeypatch.setattr("os.cpu_count", lambda: 4)

    worker_count = _select_worker_count(
        class_count=64, estimated_examples_per_classifier=1000, bytes_per_example=1000.0, requested_worker_count=None
    )

    assert worker_count == 4


@pytest.mark.parametrize(
    "classifier_cls", [ArrayBackpropClassifierNetwork, RustArrayBackpropClassifierNetwork], ids=["numpy", "rust"]
)
def test_array_classifiers_get_independent_reproducible_initial_weights(classifier_cls: Any):

    # the array classifiers draw from np.random or the crate's RNG, not random: forked workers
    # inherit those states, so a worker that reseeds only random builds identical sub-networks
    dataset = _synthetic_multiclass_dataset()
    runs = [
        [
            repr([[array.tolist() for array in entry] for entry in classifier.snapshot()])
            for classifier in train_ensemble_parallel(
                dataset,
                class_count=3,
                layer_sizes=[4],
                dimension=2,
                input_bounds=square_bounds(10.0),
                learning_rate=0.5,
                epochs=0,
                worker_count=3,
                seed=seed,
                classifier_cls=classifier_cls,
            )[0].classifiers
        ]
        for seed in (7, 7, 8)
    ]
    assert len(set(runs[0])) == 3
    assert runs[0] == runs[1]
    assert runs[0] != runs[2]
