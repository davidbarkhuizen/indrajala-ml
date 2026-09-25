import random


def split_train_test(
    data: list[tuple[tuple[float, ...], int]],
    test_fraction: float = 0.2,
    seed: int | None = None,
) -> tuple[list[tuple[tuple[float, ...], int]], list[tuple[tuple[float, ...], int]]]:
    """
    Shuffles a copy of data and splits it into (train, test), for the fixed datasets (UCI digits,
    Iris); the synthetic targets are resampled instead.
    """

    assert 0.0 < test_fraction < 1.0, f"test_fraction must be strictly between 0 and 1; got {test_fraction}"

    shuffled = list(data)
    rng = random.Random(seed) if seed is not None else random
    rng.shuffle(shuffled)

    test_size = round(len(shuffled) * test_fraction)
    test_data = shuffled[:test_size]
    train_data = shuffled[test_size:]

    return train_data, test_data
