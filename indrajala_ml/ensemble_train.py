import multiprocessing
import multiprocessing.pool
import os
import pickle
import random
from collections.abc import Callable, Iterable

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.ensemble_backprop_classifier_network import EnsembleBackpropClassifierNetwork
from indrajala_ml.train import TrainingDiagnostic, train_linear_classifier_network

RecordLoader = Callable[[str, list[int]], list[tuple[tuple[float, ...], int]]]

# don't plan to spend more than this fraction of currently-available memory on worker datasets -
# leaves headroom for the main process, the OS, and everything else already running
MEMORY_SAFETY_FRACTION = 0.5

# a dataset unpickled in a worker shares nothing with the parent's (in one process a balanced
# subset only references the full dataset's tuples; across processes every float is copied and
# rebuilt), so the pickled-size estimate is multiplied to stay above the real per-worker
# footprint
WORKER_MEMORY_SAFETY_MULTIPLIER = 2.0


def select_balanced_indices(
    labels: list[int],
    target_label: int,
    class_count: int,
    rng: random.Random,
) -> list[tuple[int, float]]:
    """
    The stratified sampling behind build_balanced_binary_dataset, over labels only: every index
    labeled target_label (as 1.0), plus about len(positives) // (class_count - 1) from each other
    class, as many as it has (as 0.0), not a pooled sample that would skew classes with different
    counts. Shuffled.

    It needs only labels, so for large datasets (MNIST) train_ensemble_parallel_from_indices lets
    each worker load just its own selected examples, and no process decodes the whole dataset.
    """

    assert class_count >= 2, f"class_count must be at least 2; got {class_count}"
    assert 0 <= target_label < class_count, f"target_label must be in [0, {class_count}); got {target_label}"

    positive_indices = [index for index, label in enumerate(labels) if label == target_label]

    by_other_label: dict[int, list[int]] = {label: [] for label in range(class_count) if label != target_label}
    for index, label in enumerate(labels):
        if label != target_label:
            by_other_label[label].append(index)

    other_labels = sorted(by_other_label)
    target_negative_count = len(positive_indices)
    base_count, remainder = divmod(target_negative_count, len(other_labels))

    index_category_pairs: list[tuple[int, float]] = [(index, 1.0) for index in positive_indices]
    for position, label in enumerate(other_labels):
        # spread the remainder (from integer division) across the first few classes, so the
        # total negative count matches target_negative_count as closely as availability allows
        desired = base_count + (1 if position < remainder else 0)
        available = by_other_label[label]
        sampled = rng.sample(available, min(desired, len(available)))
        index_category_pairs.extend((index, 0.0) for index in sampled)

    rng.shuffle(index_category_pairs)
    return index_category_pairs


def build_balanced_binary_dataset(
    dataset: list[tuple[tuple[float, ...], int]],
    target_label: int,
    class_count: int,
    rng: random.Random,
) -> list[tuple[tuple[float, ...], float]]:
    """
    One sub-network's "is this target_label?" training set, from a dataset decoded in memory (UCI
    digits, small datasets). The stratification is select_balanced_indices; large datasets use
    train_ensemble_parallel_from_indices.
    """

    labels = [label for _, label in dataset]
    index_category_pairs = select_balanced_indices(labels, target_label, class_count, rng)
    return [(dataset[index][0], category) for index, category in index_category_pairs]


def _picklable_snapshot(snapshot):
    """
    A classifier's snapshot() as nested lists, which cross a multiprocessing.Pool boundary for any
    backend: indrajala_math_rust.Array doesn't pickle. Recurses through lists and tuples, calling
    .tolist() on each array leaf (numpy or Rust); per-node snapshots are already lists and pass
    through. The collecting side's restore() accepts lists.
    """
    to_list = getattr(snapshot, "tolist", None)
    if to_list is not None:
        return to_list()
    if isinstance(snapshot, (list, tuple)):
        return type(snapshot)(_picklable_snapshot(item) for item in snapshot)
    return snapshot


def _train_classifier_on_binary_dataset(
    label: int,
    binary_dataset: list[tuple[tuple[float, ...], float]],
    layer_sizes: list[int],
    dimension: int,
    input_bounds: list[tuple[float, float]],
    learning_rate: float,
    epochs: int,
    seed: int | None,
    classifier_cls: type[BackpropClassifierNetwork],
) -> tuple[int, list[list[tuple[list[float], float]]], TrainingDiagnostic]:
    """
    The training both Pool workers share, given a binary dataset: one class's classifier_cls, with
    no state shared with any other worker.

    Seeds this process's random state first: forked workers can share the parent's random state,
    which would give sub-networks correlated or identical initial weights. seed=None reseeds from
    the OS, independent per process but not reproducible.
    """

    random.seed(seed)
    student = classifier_cls.randomized(layer_sizes, dimension, input_bounds)
    result = train_linear_classifier_network(student, binary_dataset, learning_rate=learning_rate, epochs=epochs)

    return label, _picklable_snapshot(student.snapshot()), result.diagnostic


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
        type[BackpropClassifierNetwork],
    ],
) -> tuple[int, list[list[tuple[list[float], float]]], TrainingDiagnostic]:
    """
    The Pool worker for a decoded dataset (module-level, so it pickles).
    """

    label, binary_dataset, layer_sizes, dimension, input_bounds, learning_rate, epochs, seed, classifier_cls = args

    return _train_classifier_on_binary_dataset(
        label, binary_dataset, layer_sizes, dimension, input_bounds, learning_rate, epochs, seed, classifier_cls
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
        type[BackpropClassifierNetwork],
    ],
) -> tuple[int, list[list[tuple[list[float], float]]], TrainingDiagnostic]:
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
        classifier_cls,
    ) = args

    # index_category_pairs already arrives shuffled (select_balanced_indices' own last step) -
    # loading records in that same order, via zip below, needs no further shuffling here
    indices = [index for index, _ in index_category_pairs]
    categories_in_order = [category for _, category in index_category_pairs]
    records = record_loader(path, indices)
    binary_dataset = [(state, category) for (state, _label), category in zip(records, categories_in_order)]

    return _train_classifier_on_binary_dataset(
        label, binary_dataset, layer_sizes, dimension, input_bounds, learning_rate, epochs, seed, classifier_cls
    )


def _available_memory_bytes() -> int | None:
    """
    MemAvailable from /proc/meminfo, the kernel's estimate of memory usable without swapping
    (unlike "free", which excludes reclaimable cache), or None when unavailable, in which case the
    worker count is limited by cores only.
    """

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass

    return None


def _estimate_bytes_per_example(dataset: list[tuple[tuple[float, ...], int]], sample_size: int = 50) -> float:
    """
    The pickled size of one example, from a 50-example sample: what a worker receives over IPC,
    measured on the data at hand.
    """

    sample = dataset[: min(sample_size, len(dataset))]
    assert sample, "dataset must not be empty"
    return len(pickle.dumps(sample)) / len(sample)


def _select_worker_count(
    class_count: int,
    estimated_examples_per_classifier: int,
    bytes_per_example: float,
    requested_worker_count: int | None,
) -> int:
    """
    The smallest of the requested count, the CPUs, the classes, and a memory limit: 8 workers each
    unpickling a ~10k-example dataset exhausted RAM and swapped well before the CPUs were busy.
    """

    limits = [os.cpu_count() or 1, class_count]
    if requested_worker_count is not None:
        limits.append(requested_worker_count)

    available = _available_memory_bytes()
    estimated_worker_bytes = estimated_examples_per_classifier * bytes_per_example * WORKER_MEMORY_SAFETY_MULTIPLIER
    if available is not None and estimated_worker_bytes > 0:
        memory_limit = int(available * MEMORY_SAFETY_FRACTION / estimated_worker_bytes)
        limits.append(memory_limit)

    return max(1, min(limits))


def _assemble_ensemble_from_results(
    results: list[tuple[int, list[list[tuple[list[float], float]]], TrainingDiagnostic]],
    layer_sizes: list[int],
    dimension: int,
    input_bounds: list[tuple[float, float]],
    classifier_cls: type[BackpropClassifierNetwork],
) -> tuple[EnsembleBackpropClassifierNetwork, dict[int, TrainingDiagnostic]]:
    """
    The tail of every ensemble trainer, parallel or serial: sorts the (label, snapshot, diagnostic)
    results into label order, rebuilds each classifier_cls from its snapshot (restore() sets the
    weights, so the class's randomize() doesn't matter) and assembles the ensemble.
    """

    results = sorted(results, key=lambda result: result[0])

    classifiers = []
    diagnostics: dict[int, TrainingDiagnostic] = {}
    for label, snapshot, diagnostic in results:
        student = classifier_cls(layer_sizes, dimension, input_bounds)
        student.restore(snapshot)
        classifiers.append(student)
        diagnostics[label] = diagnostic

    return EnsembleBackpropClassifierNetwork(classifiers), diagnostics


def _collect_ensemble_results(
    pool: multiprocessing.pool.Pool,
    worker_fn: Callable[..., tuple[int, list[list[tuple[list[float], float]]], TrainingDiagnostic]],
    jobs: Iterable,
    layer_sizes: list[int],
    dimension: int,
    input_bounds: list[tuple[float, float]],
    classifier_cls: type[BackpropClassifierNetwork],
) -> tuple[EnsembleBackpropClassifierNetwork, dict[int, TrainingDiagnostic]]:
    """
    Runs worker_fn over jobs with pool.imap and hands the results to
    _assemble_ensemble_from_results.
    """

    results = list(pool.imap(worker_fn, jobs))
    return _assemble_ensemble_from_results(results, layer_sizes, dimension, input_bounds, classifier_cls)


def train_ensemble_parallel(
    dataset: list[tuple[tuple[float, ...], int]],
    class_count: int,
    layer_sizes: list[int],
    dimension: int,
    input_bounds: list[tuple[float, float]],
    learning_rate: float,
    epochs: int,
    worker_count: int | None = None,
    seed: int | None = None,
    classifier_cls: type[BackpropClassifierNetwork] = BackpropClassifierNetwork,
) -> tuple[EnsembleBackpropClassifierNetwork, dict[int, TrainingDiagnostic]]:
    """
    Trains one classifier_cls per class on a multiprocessing.Pool. Nothing is synchronized between
    them, so the only communication is dispatch and collection.

    classifier_cls defaults to BackpropClassifierNetwork and takes any class with its constructor
    and randomized() signature, e.g. FanInAwareBackpropClassifierNetwork, whose initialization
    matters at MNIST scale.

    worker_count is capped by _select_worker_count, by CPUs and by estimated memory: a worker
    unpickling its own dataset copy shares nothing with the parent.

    dataset must be decoded in memory, fine for small datasets (UCI digits, the tests' synthetic
    data). MNIST decoded in every process costs several GB (47 million boxed floats); use
    train_ensemble_parallel_from_indices.

    seed makes the run reproducible: it seeds one random.Random for every class's stratified
    sampling (in class order) and each job's worker seed.
    """

    rng = random.Random(seed)

    positive_counts = [sum(1 for _, label in dataset if label == target) for target in range(class_count)]
    estimated_examples_per_classifier = 2 * max(positive_counts)
    bytes_per_example = _estimate_bytes_per_example(dataset)
    actual_worker_count = _select_worker_count(
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
                classifier_cls,
            )

    with multiprocessing.Pool(actual_worker_count) as pool:
        return _collect_ensemble_results(
            pool, _train_one_classifier, jobs(), layer_sizes, dimension, input_bounds, classifier_cls
        )


def train_ensemble_parallel_from_indices(
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
    classifier_cls: type[BackpropClassifierNetwork] = BackpropClassifierNetwork,
) -> tuple[EnsembleBackpropClassifierNetwork, dict[int, TrainingDiagnostic]]:
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

    positive_counts = [labels.count(target) for target in range(class_count)]
    estimated_examples_per_classifier = 2 * max(positive_counts)
    sample = record_loader(path, list(range(min(50, len(labels)))))
    bytes_per_example = len(pickle.dumps(sample)) / len(sample)
    actual_worker_count = _select_worker_count(
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
                classifier_cls,
            )

    with multiprocessing.Pool(actual_worker_count) as pool:
        return _collect_ensemble_results(
            pool, _train_one_indexed_classifier, jobs(), layer_sizes, dimension, input_bounds, classifier_cls
        )


def train_ensemble_serial_from_indices(
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
    classifier_cls: type[BackpropClassifierNetwork] = BackpropClassifierNetwork,
) -> tuple[EnsembleBackpropClassifierNetwork, dict[int, TrainingDiagnostic]]:
    """
    train_ensemble_parallel_from_indices in this process, one class after another, calling
    _train_one_indexed_classifier directly. For the array networks, training is fast enough that
    serial can match or beat the Pool's dispatch and collection overhead. seed works as there.
    """

    rng = random.Random(seed)

    results = []
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
                    classifier_cls,
                )
            )
        )

    return _assemble_ensemble_from_results(results, layer_sizes, dimension, input_bounds, classifier_cls)
