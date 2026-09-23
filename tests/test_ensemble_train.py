import math
import pickle
import random

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


def _synthetic_dataset(counts: dict[int, int]) -> list[tuple[tuple[float, ...], int]]:
    # each example's state is a distinct singleton tuple, so identity is easy to trace
    dataset = []
    for label, count in counts.items():
        for i in range(count):
            dataset.append(((float(label), float(i)), label))
    return dataset


def write_test_records(path: str, records: list[tuple[tuple[float, ...], int]]) -> None:
    # a minimal, picklable-file-backed stand-in for a real on-disk dataset format (like
    # mnist_data.py's binary format) - module-level, not a closure, so the matching loader below
    # can be passed to multiprocessing workers
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

    # class 1 only has 1 example available, far fewer than the 3 that would otherwise be drawn
    # from it (9 positives / 3 other classes = 3 each) - it should contribute just that 1, not
    # raise or pad, and the total negative count falls correspondingly short of 9
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

    # build_balanced_binary_dataset is a thin wrapper around select_balanced_indices - this
    # confirms the two produce identical output: the same states/categories come out, just
    # looked up from the index-based result rather than computed a second, independent way
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
    # three well-separated 2D clusters - small, but genuinely learnable (unlike the label-only
    # synthetic dataset above, these states actually carry signal), so training this for real
    # through the full multiprocessing.Pool path is meaningful, not just mechanically exercised
    centers = {0: (-5.0, -5.0), 1: (5.0, 5.0), 2: (5.0, -5.0)}
    rng = random.Random(1)
    dataset = []
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

    # confirms classifier_cls actually reaches the forked worker processes, not just the main
    # one - epochs=0 leaves every sub-network's weights exactly at randomize()'s own output, so
    # the resulting weight magnitude cleanly distinguishes FanInAwareBackpropClassifierNetwork's
    # scheme (limit=1/sqrt(dimension)=1/sqrt(2)=0.707) from the default BackpropClassifierNetwork
    # scheme this test's dimension/bounds would otherwise produce (2.0/half_width=2.0/10.0=0.2 for
    # the first hidden layer, but unbounded beyond it - the fan-in-aware ceiling is the
    # unambiguous signal here since it applies uniformly to every layer, including the output
    # layer, where the default scheme instead draws from a fixed uniform(-1.0, 1.0))
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

    # epochs=0: train_linear_classifier_network's loop body never runs, so each returned
    # sub-network is left exactly at its randomized(), untrained state (confirmed directly -
    # its snapshot is unchanged from before the call) - isolates "are initial weights actually
    # independent per worker" from "training on different data made them diverge anyway",
    # which a nonzero epoch count can't distinguish (confirmed by mutation testing: hardcoding
    # every worker's seed to the same value still passed a version of this test using epochs=1,
    # since post-training divergence from each sub-network's different binary dataset masked
    # the identical initial weights)
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

    # ArrayBackpropClassifierNetwork.randomized's accepted-and-discarded input_bounds parameter
    # (see its own docstring) lets it plug into the existing multiprocessing.Pool path
    # completely unchanged - numpy arrays pickle fine across a worker boundary natively (unlike
    # indrajala_math_rust.Array - see
    # test_train_ensemble_parallel_accepts_rust_array_backed_classifier_cls below for that case).
    #
    # train_ensemble_parallel's own _collect_ensemble_results always wraps the trained
    # classifiers in EnsembleBackpropClassifierNetwork (the per-node wrapper), regardless of
    # classifier_cls - fine for predict_probabilities/classify_state (duck-typed, calls
    # classifier.predict_probability only), but that wrapper's own save() reaches into
    # hidden_layers/input_bounds, which ArrayBackpropClassifierNetwork has neither of. The real
    # ensemble wrapper for this backend is EnsembleArrayBackpropClassifierNetwork - built here by
    # rewrapping .classifiers, not by changing train_ensemble_parallel's own return type.
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

    # indrajala_math_rust.Array does not support pickling (confirmed directly: pickle.dumps
    # raises TypeError), which would otherwise make this the one classifier_cls that can never
    # train through a multiprocessing.Pool worker boundary. Closed by
    # ensemble_train._picklable_snapshot (converts a worker's returned snapshot to plain,
    # always-picklable lists before it crosses the process boundary) paired with
    # RustArrayBackpropClassifierNetwork.restore()'s own tolerance for receiving plain lists as
    # well as pa.Array - no new public training function needed, the existing
    # train_ensemble_parallel/train_ensemble_parallel_from_indices machinery already works
    # unchanged once both sides of that boundary agree on a picklable representation.
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


def test_train_ensemble_parallel_from_indices_produces_a_working_ensemble(tmp_path):

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


def test_train_ensemble_parallel_from_indices_matches_the_fully_decoded_path(tmp_path):

    # same data, same seed, same everything - the index-based path should train to exactly the
    # same result as the fully-decoded path, since select_balanced_indices' choices and
    # per-worker seeding are identical either way, only *how* each worker gets its examples
    # differs
    dataset = _synthetic_multiclass_dataset()
    labels = [label for _, label in dataset]
    path = str(tmp_path / "records.pkl")
    write_test_records(path, dataset)
    bounds = square_bounds(10.0)

    indexed_ensemble, _ = _train_synthetic_from_indices(path, labels, bounds, epochs=3, seed=7)
    direct_ensemble, _ = _train_synthetic(dataset, bounds, epochs=3, seed=7)

    assert indexed_ensemble.snapshot() == direct_ensemble.snapshot()


def test_train_ensemble_serial_from_indices_produces_a_working_ensemble(tmp_path):

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


def test_train_ensemble_serial_from_indices_matches_the_parallel_path(tmp_path):

    # no multiprocessing.Pool at all, but select_balanced_indices' choices and per-class seeding
    # are identical either way (both draw from one random.Random(seed) in class order) - so the
    # serial path should train to exactly the same result as the parallel one
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


def test_train_ensemble_serial_from_indices_accepts_array_and_rust_backed_classifier_cls(tmp_path):

    # the serial path this function provides for both backends - ArrayBackpropClassifierNetwork
    # (numpy) and RustArrayBackpropClassifierNetwork alike (see _picklable_snapshot for why Rust
    # can also use the parallel path, not just this one) - as a comparison point against each
    # backend's own parallel-path result.
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
    assert estimated == pytest.approx(direct)


def test_estimate_bytes_per_example_handles_a_dataset_smaller_than_the_sample_size():

    dataset = [((1.0, 2.0), 0), ((3.0, 4.0), 1)]

    estimated = _estimate_bytes_per_example(dataset, sample_size=50)

    assert estimated > 0


def test_estimate_bytes_per_example_rejects_an_empty_dataset():

    with pytest.raises(AssertionError):
        _estimate_bytes_per_example([], sample_size=50)


def test_select_worker_count_is_limited_by_available_memory(monkeypatch):

    # 100MB available, budget half of it (50MB) for workers, each estimated at
    # 1000 examples * 1000 bytes/example * 2.0x safety multiplier = 2,000,000 bytes/worker ->
    # floor(50_000_000 / 2_000_000) = 25 workers by memory alone - but cpu_count/class_count
    # below are set high enough not to bind, isolating the memory limit specifically
    monkeypatch.setattr("indrajala_ml.ensemble_train._available_memory_bytes", lambda: 100_000_000)
    monkeypatch.setattr("os.cpu_count", lambda: 64)

    worker_count = _select_worker_count(
        class_count=64, estimated_examples_per_classifier=1000, bytes_per_example=1000.0, requested_worker_count=None
    )

    assert worker_count == 25


def test_select_worker_count_is_limited_by_cpu_count_when_memory_is_abundant(monkeypatch):

    monkeypatch.setattr("indrajala_ml.ensemble_train._available_memory_bytes", lambda: 10_000_000_000)
    monkeypatch.setattr("os.cpu_count", lambda: 4)

    worker_count = _select_worker_count(
        class_count=64, estimated_examples_per_classifier=1000, bytes_per_example=1000.0, requested_worker_count=None
    )

    assert worker_count == 4


def test_select_worker_count_is_limited_by_class_count(monkeypatch):

    monkeypatch.setattr("indrajala_ml.ensemble_train._available_memory_bytes", lambda: 10_000_000_000)
    monkeypatch.setattr("os.cpu_count", lambda: 64)

    worker_count = _select_worker_count(
        class_count=3, estimated_examples_per_classifier=1000, bytes_per_example=1000.0, requested_worker_count=None
    )

    assert worker_count == 3


def test_select_worker_count_respects_an_explicit_lower_request(monkeypatch):

    monkeypatch.setattr("indrajala_ml.ensemble_train._available_memory_bytes", lambda: 10_000_000_000)
    monkeypatch.setattr("os.cpu_count", lambda: 64)

    worker_count = _select_worker_count(
        class_count=64, estimated_examples_per_classifier=1000, bytes_per_example=1000.0, requested_worker_count=2
    )

    assert worker_count == 2


def test_select_worker_count_never_goes_below_one_even_under_severe_memory_pressure(monkeypatch):

    monkeypatch.setattr("indrajala_ml.ensemble_train._available_memory_bytes", lambda: 1)
    monkeypatch.setattr("os.cpu_count", lambda: 64)

    worker_count = _select_worker_count(
        class_count=64, estimated_examples_per_classifier=1_000_000, bytes_per_example=1000.0, requested_worker_count=None
    )

    assert worker_count == 1


def test_select_worker_count_falls_back_to_cpu_and_class_count_when_memory_is_undetectable(monkeypatch):

    monkeypatch.setattr("indrajala_ml.ensemble_train._available_memory_bytes", lambda: None)
    monkeypatch.setattr("os.cpu_count", lambda: 4)

    worker_count = _select_worker_count(
        class_count=64, estimated_examples_per_classifier=1000, bytes_per_example=1000.0, requested_worker_count=None
    )

    assert worker_count == 4
