import random

import numpy as np

from indrajala_ml.ensemble_train import select_balanced_indices
from indrajala_ml.mnist_data import load_mnist_labels, load_mnist_records_at_indices

MNIST_DIGIT_COUNT = 10


class BenchmarkProxy:
    """
    A fixed, class-balanced MNIST-digit proxy dataset for a sweep, built once and shared by every
    seed.

    train_x/test_x are (n, 784) float64 pixels in [0.0, 1.0], as load_mnist_dataset_as_array. With
    one digit, train_y/test_y are float 1.0/0.0 (that digit or not), the single-output networks'
    target; with several, int categories in [0, len(digits)), each the digit's position in the
    digits list, not its value.
    """

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> None:
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y


def build_mnist_digit_proxy(
    path: str,
    digits: list[int],
    examples_per_class: int,
    split_fraction: float = 0.8,
    seed: int | None = None,
) -> BenchmarkProxy:
    """
    A fixed MNIST-digit proxy: examples_per_class balanced examples per requested digit, decoding
    only the selected records, split into train and test halves that keep each class's balance.

    One digit is the binary "this digit vs every other" task, sampled by
    ensemble_train.select_balanced_indices (digits[0] is 1.0, a stratified sample of the rest 0.0).
    Several digits are an N-way task over exactly those digits, sampled per digit by
    _multiclass_index_category_pairs.
    """

    assert len(digits) >= 1, f"digits needs at least one target digit; got {digits}"
    assert len(set(digits)) == len(digits), f"digits must be unique; got {digits}"
    assert examples_per_class >= 1, f"examples_per_class must be positive; got {examples_per_class}"
    assert 0.0 < split_fraction < 1.0, f"split_fraction must be in (0.0, 1.0); got {split_fraction}"

    rng = random.Random(seed)
    labels = load_mnist_labels(path)

    if len(digits) == 1:
        index_category_pairs = _binary_index_category_pairs(labels, digits[0], examples_per_class, rng)
        category_dtype = np.float64
    else:
        index_category_pairs = _multiclass_index_category_pairs(labels, digits, examples_per_class, rng)
        category_dtype = np.int64

    train_pairs, test_pairs = _stratified_split(index_category_pairs, split_fraction, rng)
    train_x, train_y = _decode_split(path, train_pairs, category_dtype)
    test_x, test_y = _decode_split(path, test_pairs, category_dtype)
    return BenchmarkProxy(train_x, train_y, test_x, test_y)


def _binary_index_category_pairs(
    labels: list[int], target_digit: int, examples_per_class: int, rng: random.Random
) -> list[tuple[int, float]]:
    pairs = select_balanced_indices(labels, target_digit, MNIST_DIGIT_COUNT, rng)
    positives = [pair for pair in pairs if pair[1] == 1.0]
    negatives = [pair for pair in pairs if pair[1] == 0.0]

    assert len(positives) >= examples_per_class, (
        f"digit {target_digit} has only {len(positives)} examples; need {examples_per_class}"
    )
    assert len(negatives) >= examples_per_class, (
        f"the other digits together have only {len(negatives)} examples; need {examples_per_class}"
    )

    sampled = rng.sample(positives, examples_per_class) + rng.sample(negatives, examples_per_class)
    rng.shuffle(sampled)
    return sampled


def _multiclass_index_category_pairs(
    labels: list[int], digits: list[int], examples_per_class: int, rng: random.Random
) -> list[tuple[int, float]]:
    for digit in digits:
        assert 0 <= digit < MNIST_DIGIT_COUNT, f"digit must be in [0, {MNIST_DIGIT_COUNT}); got {digit}"

    indices_by_digit: dict[int, list[int]] = {digit: [] for digit in digits}
    for index, label in enumerate(labels):
        if label in indices_by_digit:
            indices_by_digit[label].append(index)

    pairs: list[tuple[int, float]] = []
    for category, digit in enumerate(digits):
        available = indices_by_digit[digit]
        assert len(available) >= examples_per_class, (
            f"digit {digit} has only {len(available)} examples; need {examples_per_class}"
        )
        pairs.extend((index, float(category)) for index in rng.sample(available, examples_per_class))

    rng.shuffle(pairs)
    return pairs


def _stratified_split(
    index_category_pairs: list[tuple[int, float]], split_fraction: float, rng: random.Random
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    pairs_by_category: dict[float, list[tuple[int, float]]] = {}
    for pair in index_category_pairs:
        pairs_by_category.setdefault(pair[1], []).append(pair)

    train_pairs: list[tuple[int, float]] = []
    test_pairs: list[tuple[int, float]] = []
    for category in sorted(pairs_by_category):
        group = pairs_by_category[category]
        rng.shuffle(group)
        split_index = round(len(group) * split_fraction)
        train_pairs.extend(group[:split_index])
        test_pairs.extend(group[split_index:])

    rng.shuffle(train_pairs)
    rng.shuffle(test_pairs)
    return train_pairs, test_pairs


def _decode_split(path: str, pairs: list[tuple[int, float]], category_dtype: type) -> tuple[np.ndarray, np.ndarray]:
    # records come back in the order of the indices, so they zip against the recoded categories
    # (the proxy replaces the raw MNIST labels)
    indices = [index for index, _ in pairs]
    categories = [category for _, category in pairs]
    records = load_mnist_records_at_indices(path, indices)

    x = np.array([state for state, _ in records], dtype=np.float64)
    y = np.array(categories, dtype=category_dtype)
    return x, y
