import random


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
