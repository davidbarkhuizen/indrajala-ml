import math
import pickle
import random
from pathlib import Path
from typing import Any

import pytest

from indrajala_ml.geometry import square_bounds
from indrajala_ml.model.ensembles.ensemble_array_backprop_classifier_network import (
    EnsembleArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.ensembles.ensemble_rust_array_backprop_classifier_network import (
    EnsembleRustArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.networks.numpy.array_backprop_classifier_network import ArrayBackpropClassifierNetwork
from indrajala_ml.model.networks.python.fan_in_aware_backprop_classifier_network import (
    FanInAwareBackpropClassifierNetwork,
)
from indrajala_ml.model.networks.rust.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork
from indrajala_ml.training.ensemble_train import (
    train_ensemble_parallel,
    train_ensemble_parallel_from_indices,
    train_ensemble_serial_from_indices,
)


def write_test_records(path: str, records: list[tuple[tuple[float, ...], int]]) -> None:
    # a stand-in on-disk dataset; module-level so its loader can be sent to worker processes
    with open(path, "wb") as f:
        pickle.dump(records, f)


def load_test_records_at_indices(path: str, indices: list[int]) -> list[tuple[tuple[float, ...], int]]:
    with open(path, "rb") as f:
        records = pickle.load(f)
    return [records[index] for index in indices]


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

    # indrajala_math_rust.Array can't be pickled: _picklable_checkpoint sends plain lists back
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
    # the optimizers' state crosses the Pool boundary in the checkpoint, with the weights
    serial_steps = [classifier.optimizer.t for classifier in serial_ensemble.classifiers]
    assert [classifier.optimizer.t for classifier in parallel_ensemble.classifiers] == serial_steps
    assert all(t > 0 for t in serial_steps)


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
