import pytest

from indrajala_ml.dataset_utils import split_train_test


def _synthetic_dataset(size: int) -> list[tuple[tuple[float, ...], int]]:
    return [((float(i),), i % 3) for i in range(size)]


def test_split_train_test_sizes_and_no_overlap():

    dataset = _synthetic_dataset(100)

    train, test = split_train_test(dataset, test_fraction=0.2, seed=0)

    assert len(train) + len(test) == len(dataset)
    assert len(test) == round(len(dataset) * 0.2)

    train_states = set(state for state, _ in train)
    test_states = set(state for state, _ in test)
    assert train_states.isdisjoint(test_states)


def test_split_train_test_is_reproducible_under_a_fixed_seed():

    dataset = _synthetic_dataset(100)

    train_a, test_a = split_train_test(dataset, test_fraction=0.2, seed=42)
    train_b, test_b = split_train_test(dataset, test_fraction=0.2, seed=42)

    assert train_a == train_b
    assert test_a == test_b


def test_split_train_test_rejects_a_non_fractional_test_fraction():

    dataset = _synthetic_dataset(100)

    with pytest.raises(AssertionError):
        split_train_test(dataset, test_fraction=0.0)

    with pytest.raises(AssertionError):
        split_train_test(dataset, test_fraction=1.0)
