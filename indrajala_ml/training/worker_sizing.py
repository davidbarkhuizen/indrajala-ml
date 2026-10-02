import os
import pickle

# don't plan to spend more than this fraction of currently-available memory on worker datasets -
# leaves headroom for the main process, the OS, and everything else already running
MEMORY_SAFETY_FRACTION = 0.5

# a dataset unpickled in a worker shares nothing with the parent's (in one process a balanced
# subset only references the full dataset's tuples; across processes every float is copied and
# rebuilt), so the pickled-size estimate is multiplied to stay above the real per-worker
# footprint
WORKER_MEMORY_SAFETY_MULTIPLIER = 2.0


def available_memory_bytes() -> int | None:
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
    except OSError, ValueError, IndexError:
        pass

    return None


def estimate_bytes_per_example(dataset: list[tuple[tuple[float, ...], int]], sample_size: int = 50) -> float:
    """
    The pickled size of one example, from a 50-example sample: what a worker receives over IPC,
    measured on the data at hand.
    """

    sample = dataset[: min(sample_size, len(dataset))]
    assert sample, "dataset must not be empty"
    return len(pickle.dumps(sample)) / len(sample)


def select_worker_count(
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

    available = available_memory_bytes()
    estimated_worker_bytes = estimated_examples_per_classifier * bytes_per_example * WORKER_MEMORY_SAFETY_MULTIPLIER
    if available is not None and estimated_worker_bytes > 0:
        memory_limit = int(available * MEMORY_SAFETY_FRACTION / estimated_worker_bytes)
        limits.append(memory_limit)

    return max(1, min(limits))
