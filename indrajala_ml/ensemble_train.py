import multiprocessing
import multiprocessing.pool
import os
import pickle
import random
from typing import Callable, Iterable

from indrajala_ml.model.backprop_classifier_network import BackpropClassifierNetwork
from indrajala_ml.model.ensemble_backprop_classifier_network import EnsembleBackpropClassifierNetwork
from indrajala_ml.train import TrainingDiagnostic, train_linear_classifier_network

RecordLoader = Callable[[str, list[int]], list[tuple[tuple[float, ...], int]]]

# don't plan to spend more than this fraction of currently-available memory on worker datasets -
# leaves headroom for the main process, the OS, and everything else already running
MEMORY_SAFETY_FRACTION = 0.5

# a freshly unpickled dataset in a worker process gets none of the reference-sharing a
# same-process copy would (measured directly: building a balanced binary dataset in the same
# process that already holds the full dataset costs next to nothing extra, since each example's
# state tuple is a shared reference, not a copy - but pickling it across a process boundary for
# multiprocessing.Pool serializes the real float data every time, and unpickling reconstructs
# fresh objects with normal-per-object overhead) - this multiplies the pickled-size estimate to
# stay safely above the real per-worker footprint rather than under it
WORKER_MEMORY_SAFETY_MULTIPLIER = 2.0


def select_balanced_indices(
    labels: list[int],
    target_label: int,
    class_count: int,
    rng: random.Random,
) -> list[tuple[int, float]]:
    """
    The core stratified-sampling logic behind build_balanced_binary_dataset, operating on cheap
    label-only data (indices + int labels) rather than full decoded examples - every index
    labeled target_label (recoded 1.0), plus a genuinely stratified sample of the other classes'
    indices - as close to len(positives) // (class_count - 1) from *each* other class as that
    class has available (recoded 0.0), not a pooled random sample over all non-target indices
    (which would silently over/under-represent classes whose real counts differ from each
    other). Shuffled before returning.

    Deliberately index-based, not example-based: for a large, high-dimensional dataset (e.g.
    real MNIST), deciding *which* examples belong in a class's balanced set doesn't require ever
    decoding the examples themselves - only their labels. See
    train_ensemble_parallel_from_indices, which uses this to let each worker load just its own
    selected examples directly, without any process needing the full dataset decoded in memory
    at once.
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
    Builds the "is this class target_label?" binary training set for one sub-network of an
    EnsembleBackpropClassifierNetwork, from a dataset already fully decoded in memory - fine for
    small-to-medium datasets (this codebase's own UCI digits demo, most non-MNIST uses). See
    select_balanced_indices for the actual stratification logic (shared with the index-based
    path large datasets use instead), and train_ensemble_parallel_from_indices for that path.
    """

    labels = [label for _, label in dataset]
    index_category_pairs = select_balanced_indices(labels, target_label, class_count, rng)
    return [(dataset[index][0], category) for index, category in index_category_pairs]


def _picklable_snapshot(snapshot):
    """
    Converts a classifier's own snapshot() output into a form guaranteed picklable across a
    multiprocessing.Pool worker boundary, regardless of classifier_cls's backend - a no-op for
    a per-node classifier's snapshot (already plain lists/tuples/floats) and for a numpy-backed
    one (ndarrays already pickle natively; converting them here too is harmless, not required),
    but the fix that actually matters: indrajala_math_rust.Array (RustArrayBackpropClassifierNetwork's
    own backend) does not support pickling at all (confirmed directly - pickle.dumps raises
    TypeError), so without this, a Rust-backed classifier_cls could never train through this
    module's multiprocessing.Pool path. Recurses through nested lists/tuples so it works
    uniformly across every snapshot shape this module ever sees (per-node's nested
    list-of-layers-of-node-tuples, or an array backend's flat list of (W, b) pairs), calling
    each array-like leaf's own .tolist() (both numpy ndarrays and pa.Array support it) rather
    than assuming any one shape. The corresponding reconstruction happens on the collecting
    side: RustArrayBackpropClassifierNetwork.restore() accepts plain lists as well as pa.Array,
    wrapping via pa.Array(...) when needed, the same pattern its own load() already used.
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
    The actual training work shared by both multiprocessing.Pool worker functions below
    (_train_one_classifier, _train_one_indexed_classifier), once each has its own binary_dataset
    in hand: trains one class's binary classifier_cls (a BackpropClassifierNetwork, or a
    subclass - e.g. FanInAwareBackpropClassifierNetwork, see train_ensemble_parallel_from_indices's
    own classifier_cls parameter) completely independently - no state is shared with any other
    worker, which is what makes this genuinely (not just approximately) parallelizable.

    Explicitly seeds this process's own random state before building anything: fork-based
    multiprocessing workers are not guaranteed to diverge from each other's global random state
    on their own before their first random call, so relying on incidental post-fork divergence
    would risk correlated (or even identical) initial weights across sub-networks. seed=None
    still calls random.seed(None), which reseeds from the OS's own entropy source independently
    per process - safe, just not reproducible.
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
    The multiprocessing.Pool worker - a plain module-level function, required for picklability -
    for the fully-decoded-dataset path. See _train_classifier_on_binary_dataset for the actual
    training work.
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
    The index-based counterpart to _train_one_classifier, for datasets too large to pass a
    fully-decoded binary_dataset through multiprocessing IPC without exhausting memory. Instead
    of receiving already-decoded examples,
    this receives a path, a record_loader function (e.g.
    mnist_data.load_mnist_records_at_indices), and the (index, category) pairs
    select_balanced_indices already chose - and loads only its own examples, directly, itself.
    No process (main or any other worker) ever needs the full dataset decoded in memory at once.
    See _train_classifier_on_binary_dataset for the actual training work, including the same
    explicit per-process random seeding _train_one_classifier uses, for the same reason.
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
    Best-effort available-memory detection via Linux's /proc/meminfo MemAvailable - the
    kernel's own estimate of memory available for new allocations without swapping (not just
    "free", which excludes reclaimable cache and undercounts what's actually usable). Returns
    None when unavailable (e.g. non-Linux, or the file's shape ever changes) so callers can fall
    back to a core-count-only worker limit rather than fail outright - this repo already assumes
    Linux elsewhere (see cli's install_os_packages), so no portability fallback beyond that.
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
    Empirically estimates the pickled (i.e. what a worker actually has to receive and
    deserialize over multiprocessing IPC) size of one training example, from a small real
    sample - cheap (a 50-example pickle is near-instant) and self-calibrating to the actual
    dimension/data at hand, rather than a hardcoded bytes-per-float constant that could drift.
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
    Caps the worker pool at whichever is smallest: the requested count (if any), the number of
    CPUs, the number of classes (no benefit spawning more workers than there are jobs), and a
    memory-based limit - since a data-parallel job like this one is just as likely to be
    memory-bound as CPU-bound (measured directly: 8 concurrent workers each deserializing their
    own ~10k-plus-example dataset copy over IPC exhausted this machine's RAM and drove it into
    heavy swapping, well before CPU was the bottleneck).
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
    The shared tail of every ensemble-training entry point in this module (parallel and serial
    alike), once each has its own (label, snapshot, diagnostic) results list ready, however it
    was produced (a multiprocessing.Pool's own pool.imap, or an in-process loop - see
    train_ensemble_serial_from_indices): sorts back into label order, rebuilds each label's
    classifier_cls instance from the weight snapshot its training run returned (harmless if
    classifier_cls's own randomize() differs from whatever training actually used - restore()
    only ever sets weights/bias directly, and inference (classify_state/predict_probability)
    reads only the same weights via the identical sigmoid forward pass every
    BackpropClassifierNetwork subclass shares, regardless of which randomize() built the
    now-overwritten initial weights), and assembles the final EnsembleBackpropClassifierNetwork.
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
    The shared tail of both train_ensemble_parallel and train_ensemble_parallel_from_indices,
    once each has its own pool, worker function, and jobs iterable ready: dispatches worker_fn
    over jobs via pool.imap, then hands the results to _assemble_ensemble_from_results.
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
    Trains one classifier_cls instance per class completely independently - dispatched across
    a multiprocessing.Pool, since nothing needs to be synchronized between them (unlike a
    data-parallel weight-averaging approach, this has no communication cost beyond the one-time
    dispatch and final collection).

    classifier_cls defaults to BackpropClassifierNetwork (every existing caller's behavior is
    unchanged) but accepts any subclass with a matching constructor/randomized() signature - e.g.
    FanInAwareBackpropClassifierNetwork, whose init scheme measurably matters at real MNIST scale
    in a way BackpropClassifierNetwork's own default scheme, tuned for 1-2D geometric problems,
    does not.

    worker_count is capped by _select_worker_count using both CPU count *and* an estimate of
    available memory, not cores alone - a worker deserializing its own dataset copy over IPC
    gets none of the reference-sharing a same-process copy would.

    This function expects dataset to already be fully decoded in memory, which is fine for
    small-to-medium data (this codebase's own UCI digits demo, the synthetic datasets its own
    tests use) - for something MNIST-sized, decoding every example up front, in every process
    that touches it, costs several GB (not because of any particular library - 47 million
    individual boxed Python float objects is simply a lot of memory, however they got there).
    See train_ensemble_parallel_from_indices for that case.

    seed, when given, makes the whole run reproducible: it seeds a single random.Random used for
    every dataset's stratified sampling (in class order, so the sequence is deterministic) and
    to derive each job's own per-worker seed - not the same rng instance as any worker's (those
    run in separate processes with their own random module state).
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
    The large-dataset counterpart to train_ensemble_parallel: instead of a fully-decoded
    dataset, takes a path, a record_loader function (e.g. mnist_data.load_mnist_records_at_indices)
    that loads specific examples by index directly from that path, and the cheap labels-only
    list (e.g. mnist_data.load_mnist_labels) needed to decide which examples belong to which
    class's balanced set. No process - not the main one, not any worker - ever needs the full
    dataset decoded in memory at once: select_balanced_indices works from labels alone, and each
    worker loads only its own chosen examples, via record_loader, itself.

    Measured directly, on real MNIST data (a single class's ~11846-example balanced set, one
    worker): the naive fully-decoded-then-shipped-via-IPC approach train_ensemble_parallel uses
    peaked at ~2.25GB; this index-based approach peaked at ~374MB for the same work.

    record_loader must be a plain, module-level function (not a closure or lambda) for
    multiprocessing picklability, same as every worker function in this module. classifier_cls -
    see train_ensemble_parallel's own docstring - is the same extension point, used by
    demo_mnist_ensemble_recognition.py (real MNIST is exactly the case this path exists for).
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
    The single-process counterpart to train_ensemble_parallel_from_indices: trains every class's
    binary classifier_cls sequentially, in this process, with no multiprocessing.Pool at all -
    reusing _train_one_indexed_classifier directly (the identical per-class work every
    multiprocessing worker already does, just called synchronously here instead of dispatched
    across a process boundary).

    A second training path worth comparing against train_ensemble_parallel_from_indices for
    every classifier_cls, numpy- and Rust-backed alike (see _picklable_snapshot and
    RustArrayBackpropClassifierNetwork.restore()'s own tolerance for why the Rust-backed path
    can use either training function, not just this one): on the array/Rust-backed paths,
    per-classifier training is fast enough that training all class_count classifiers serially can
    match or beat the parallel path's own multiprocessing dispatch/collection overhead.

    Same reproducibility contract as train_ensemble_parallel_from_indices: seed, when given,
    seeds one random.Random used for every class's stratified sampling (in class order) and to
    derive each class's own training seed.
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
