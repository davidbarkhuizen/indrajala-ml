import dataclasses
import multiprocessing
import multiprocessing.pool
import pickle
import random
from collections.abc import Callable, Iterable
from typing import Any, cast

from indrajala_ml.model.ensembles.ensemble_backprop_classifier_network import EnsembleBackpropClassifierNetwork
from indrajala_ml.model.networks.python.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.protocols.classifier_protocols import BinaryClassifier, BinaryClassifierClass
from indrajala_ml.pcg64 import SeedSequence
from indrajala_ml.training.balanced_dataset import build_balanced_binary_dataset, select_balanced_indices
from indrajala_ml.training.train import train_linear_classifier_network
from indrajala_ml.training.training_diagnostics import TrainingDiagnostic
from indrajala_ml.training.worker_sizing import estimate_bytes_per_example, select_worker_count

RecordLoader = Callable[[str, list[int]], list[tuple[tuple[float, ...], int]]]


def _picklable_checkpoint(checkpoint: object) -> object:
    """
    A classifier's checkpoint() as nested lists, which cross a multiprocessing.Pool boundary for
    any backend: indrajala_math_rust.Array doesn't pickle. Recurses through lists, tuples, dicts
    and dataclasses (model/persistence/checkpoint.py), calling .tolist() on each array leaf (numpy or Rust);
    per-node weights and state are already lists and pass through. The collecting side's
    restore_checkpoint() accepts lists.
    """
    to_list = getattr(checkpoint, "tolist", None)
    if to_list is not None:
        return to_list()
    if isinstance(checkpoint, (list, tuple)):
        sequence = cast("list[object] | tuple[object, ...]", checkpoint)
        return type(sequence)(_picklable_checkpoint(item) for item in sequence)
    if isinstance(checkpoint, dict):
        mapping = cast("dict[object, object]", checkpoint)
        return {key: _picklable_checkpoint(value) for key, value in mapping.items()}
    if dataclasses.is_dataclass(checkpoint) and not isinstance(checkpoint, type):
        return dataclasses.replace(
            checkpoint,
            **{
                field.name: _picklable_checkpoint(getattr(checkpoint, field.name))
                for field in dataclasses.fields(checkpoint)
            },
        )
    return checkpoint


def _train_classifier_on_binary_dataset[ClassifierT: BinaryClassifier](
    label: int,
    binary_dataset: list[tuple[tuple[float, ...], float]],
    layer_sizes: list[int],
    dimension: int,
    input_bounds: list[tuple[float, float]],
    learning_rate: float,
    epochs: int,
    seed: int | None,
    network_seed: SeedSequence,
    classifier_cls: BinaryClassifierClass[ClassifierT],
) -> tuple[int, object, TrainingDiagnostic]:
    """
    The training both Pool workers share, given a binary dataset: one class's classifier_cls, with
    no state shared with any other worker.

    The classifier owns its generator, seeded from network_seed, this class's child of the
    ensemble's SeedSequence, so the sub-networks' streams are independent and one ensemble seed
    reproduces them all. The trainer's random.Random is seeded from seed, this job's, and from OS
    entropy when it is None (the RNG generators workplan, D6, D9).
    """

    student = classifier_cls.randomized(layer_sizes, dimension, input_bounds, seed=network_seed)
    result = train_linear_classifier_network(
        student, binary_dataset, learning_rate=learning_rate, epochs=epochs, rng=random.Random(seed)
    )

    return label, _picklable_checkpoint(student.checkpoint()), result.diagnostic


def _train_one_classifier(
    args: tuple[
        int,
        list[tuple[tuple[float, ...], float]],
        list[int],
        int,
        list[tuple[float, float]],
        float,
        int,
        int | None,
        SeedSequence,
        BinaryClassifierClass[BinaryClassifier],
    ],
) -> tuple[int, object, TrainingDiagnostic]:
    """
    The Pool worker for a decoded dataset (module-level, so it pickles).
    """

    (
        label,
        binary_dataset,
        layer_sizes,
        dimension,
        input_bounds,
        learning_rate,
        epochs,
        seed,
        network_seed,
        classifier_cls,
    ) = args

    return _train_classifier_on_binary_dataset(
        label,
        binary_dataset,
        layer_sizes,
        dimension,
        input_bounds,
        learning_rate,
        epochs,
        seed,
        network_seed,
        classifier_cls,
    )


def _train_one_indexed_classifier(
    args: tuple[
        int,
        str,
        RecordLoader,
        list[tuple[int, float]],
        list[int],
        int,
        list[tuple[float, float]],
        float,
        int,
        int | None,
        SeedSequence,
        BinaryClassifierClass[BinaryClassifier],
    ],
) -> tuple[int, object, TrainingDiagnostic]:
    """
    The Pool worker for datasets too large to send decoded: it gets a path, a record_loader (e.g.
    mnist_data.load_mnist_records_at_indices) and the (index, category) pairs
    select_balanced_indices chose, and loads only its own examples.
    """

    (
        label,
        path,
        record_loader,
        index_category_pairs,
        layer_sizes,
        dimension,
        input_bounds,
        learning_rate,
        epochs,
        seed,
        network_seed,
        classifier_cls,
    ) = args

    # index_category_pairs already arrives shuffled (select_balanced_indices' own last step) -
    # loading records in that same order, via zip below, needs no further shuffling here
    indices = [index for index, _ in index_category_pairs]
    categories_in_order = [category for _, category in index_category_pairs]
    records = record_loader(path, indices)
    binary_dataset = [(state, category) for (state, _label), category in zip(records, categories_in_order)]

    return _train_classifier_on_binary_dataset(
        label,
        binary_dataset,
        layer_sizes,
        dimension,
        input_bounds,
        learning_rate,
        epochs,
        seed,
        network_seed,
        classifier_cls,
    )


def _assemble_ensemble_from_results[ClassifierT: BinaryClassifier](
    results: list[tuple[int, object, TrainingDiagnostic]],
    layer_sizes: list[int],
    dimension: int,
    input_bounds: list[tuple[float, float]],
    classifier_cls: BinaryClassifierClass[ClassifierT],
) -> tuple[EnsembleBackpropClassifierNetwork[ClassifierT], dict[int, TrainingDiagnostic]]:
    """
    The tail of every ensemble trainer, parallel or serial: sorts the (label, checkpoint,
    diagnostic) results into label order, rebuilds each classifier_cls from its checkpoint
    (restore_checkpoint() sets the weights and the optimizer's state, so the class's randomize()
    doesn't matter) and assembles the ensemble.
    """

    results = sorted(results, key=lambda result: result[0])

    classifiers: list[ClassifierT] = []
    diagnostics: dict[int, TrainingDiagnostic] = {}
    for label, checkpoint, diagnostic in results:
        student = classifier_cls(layer_sizes, dimension, input_bounds)
        student.restore_checkpoint(checkpoint)
        classifiers.append(student)
        diagnostics[label] = diagnostic

    return EnsembleBackpropClassifierNetwork(classifiers), diagnostics


def _collect_ensemble_results[ClassifierT: BinaryClassifier](
    pool: multiprocessing.pool.Pool,
    worker_fn: Callable[..., tuple[int, object, TrainingDiagnostic]],
    jobs: Iterable[Any],
    layer_sizes: list[int],
    dimension: int,
    input_bounds: list[tuple[float, float]],
    classifier_cls: BinaryClassifierClass[ClassifierT],
) -> tuple[EnsembleBackpropClassifierNetwork[ClassifierT], dict[int, TrainingDiagnostic]]:
    """
    Runs worker_fn over jobs with pool.imap and hands the results to
    _assemble_ensemble_from_results.
    """

    results = list(pool.imap(worker_fn, jobs))
    return _assemble_ensemble_from_results(results, layer_sizes, dimension, input_bounds, classifier_cls)


def train_ensemble_parallel[ClassifierT: BinaryClassifier = BackpropClassifierNetwork](
    dataset: list[tuple[tuple[float, ...], int]],
    class_count: int,
    layer_sizes: list[int],
    dimension: int,
    input_bounds: list[tuple[float, float]],
    learning_rate: float,
    epochs: int,
    worker_count: int | None = None,
    seed: int | None = None,
    classifier_cls: BinaryClassifierClass[ClassifierT] = BackpropClassifierNetwork,
) -> tuple[EnsembleBackpropClassifierNetwork[ClassifierT], dict[int, TrainingDiagnostic]]:
    """
    Trains one classifier_cls per class on a multiprocessing.Pool. Nothing is synchronized between
    them, so the only communication is dispatch and collection.

    classifier_cls defaults to BackpropClassifierNetwork and takes any class with its constructor
    and randomized() signature, e.g. FanInAwareBackpropClassifierNetwork, whose initialization
    matters at MNIST scale.

    worker_count is capped by select_worker_count, by CPUs and by estimated memory: a worker
    unpickling its own dataset copy shares nothing with the parent.

    dataset must be decoded in memory, fine for small datasets (UCI digits, the tests' synthetic
    data). MNIST decoded in every process costs several GB (47 million boxed floats); use
    train_ensemble_parallel_from_indices.

    seed makes the run reproducible: it seeds one random.Random for every class's stratified
    sampling (in class order) and each job's worker seed, and a SeedSequence whose spawned
    children seed the array classifiers' generators, one per class.
    """

    rng = random.Random(seed)
    network_seeds = SeedSequence(seed).spawn(class_count)

    positive_counts = [sum(1 for _, label in dataset if label == target) for target in range(class_count)]
    estimated_examples_per_classifier = 2 * max(positive_counts)
    bytes_per_example = estimate_bytes_per_example(dataset)
    actual_worker_count = select_worker_count(
        class_count, estimated_examples_per_classifier, bytes_per_example, worker_count
    )

    def jobs():
        for label in range(class_count):
            binary_dataset = build_balanced_binary_dataset(dataset, label, class_count, rng)
            job_seed = rng.randrange(2**31) if seed is not None else None
            yield (
                label,
                binary_dataset,
                layer_sizes,
                dimension,
                input_bounds,
                learning_rate,
                epochs,
                job_seed,
                network_seeds[label],
                classifier_cls,
            )

    with multiprocessing.Pool(actual_worker_count) as pool:
        return _collect_ensemble_results(
            pool, _train_one_classifier, jobs(), layer_sizes, dimension, input_bounds, classifier_cls
        )


def train_ensemble_parallel_from_indices[ClassifierT: BinaryClassifier = BackpropClassifierNetwork](
    path: str,
    record_loader: RecordLoader,
    labels: list[int],
    class_count: int,
    layer_sizes: list[int],
    dimension: int,
    input_bounds: list[tuple[float, float]],
    learning_rate: float,
    epochs: int,
    worker_count: int | None = None,
    seed: int | None = None,
    classifier_cls: BinaryClassifierClass[ClassifierT] = BackpropClassifierNetwork,
) -> tuple[EnsembleBackpropClassifierNetwork[ClassifierT], dict[int, TrainingDiagnostic]]:
    """
    train_ensemble_parallel for large datasets: takes a path, a record_loader that loads examples by
    index from it (e.g. mnist_data.load_mnist_records_at_indices), and the labels (e.g.
    mnist_data.load_mnist_labels). select_balanced_indices works from the labels, and each worker
    loads only its own examples, so no process holds the decoded dataset.

    On MNIST (one class's ~11846-example set, one worker) the decode-and-ship path peaked at ~2.25
    GB and this one at ~374 MB.

    record_loader must be module-level, so it pickles. classifier_cls is as in
    train_ensemble_parallel; demo_mnist_ensemble_recognition.py uses this path.
    """

    rng = random.Random(seed)
    network_seeds = SeedSequence(seed).spawn(class_count)

    positive_counts = [labels.count(target) for target in range(class_count)]
    estimated_examples_per_classifier = 2 * max(positive_counts)
    sample = record_loader(path, list(range(min(50, len(labels)))))
    bytes_per_example = len(pickle.dumps(sample)) / len(sample)
    actual_worker_count = select_worker_count(
        class_count, estimated_examples_per_classifier, bytes_per_example, worker_count
    )

    def jobs():
        for label in range(class_count):
            index_category_pairs = select_balanced_indices(labels, label, class_count, rng)
            job_seed = rng.randrange(2**31) if seed is not None else None
            yield (
                label,
                path,
                record_loader,
                index_category_pairs,
                layer_sizes,
                dimension,
                input_bounds,
                learning_rate,
                epochs,
                job_seed,
                network_seeds[label],
                classifier_cls,
            )

    with multiprocessing.Pool(actual_worker_count) as pool:
        return _collect_ensemble_results(
            pool, _train_one_indexed_classifier, jobs(), layer_sizes, dimension, input_bounds, classifier_cls
        )


def train_ensemble_serial_from_indices[ClassifierT: BinaryClassifier = BackpropClassifierNetwork](
    path: str,
    record_loader: RecordLoader,
    labels: list[int],
    class_count: int,
    layer_sizes: list[int],
    dimension: int,
    input_bounds: list[tuple[float, float]],
    learning_rate: float,
    epochs: int,
    seed: int | None = None,
    classifier_cls: BinaryClassifierClass[ClassifierT] = BackpropClassifierNetwork,
) -> tuple[EnsembleBackpropClassifierNetwork[ClassifierT], dict[int, TrainingDiagnostic]]:
    """
    train_ensemble_parallel_from_indices in this process, one class after another, calling
    _train_one_indexed_classifier directly. For the array networks, training is fast enough that
    serial can match or beat the Pool's dispatch and collection overhead. seed works as there.
    """

    rng = random.Random(seed)
    network_seeds = SeedSequence(seed).spawn(class_count)

    results: list[tuple[int, object, TrainingDiagnostic]] = []
    for label in range(class_count):
        index_category_pairs = select_balanced_indices(labels, label, class_count, rng)
        job_seed = rng.randrange(2**31) if seed is not None else None
        results.append(
            _train_one_indexed_classifier(
                (
                    label,
                    path,
                    record_loader,
                    index_category_pairs,
                    layer_sizes,
                    dimension,
                    input_bounds,
                    learning_rate,
                    epochs,
                    job_seed,
                    network_seeds[label],
                    classifier_cls,
                )
            )
        )

    return _assemble_ensemble_from_results(results, layer_sizes, dimension, input_bounds, classifier_cls)
