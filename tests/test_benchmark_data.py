import numpy as np
import pytest

from indrajala_ml.benchmark_data import build_mnist_digit_proxy

TRAIN_PATH = "data/mnist/mnist-train.bin"


def test_binary_proxy_shape_and_balance():

    proxy = build_mnist_digit_proxy(TRAIN_PATH, digits=[3], examples_per_class=20, seed=0)

    assert proxy.train_x.shape == (32, 28 * 28)
    assert proxy.test_x.shape == (8, 28 * 28)
    assert proxy.train_y.dtype == np.float64
    assert set(np.unique(proxy.train_y)) == {0.0, 1.0}
    assert (proxy.train_y == 1.0).sum() == 16
    assert (proxy.train_y == 0.0).sum() == 16
    assert (proxy.test_y == 1.0).sum() == 4
    assert (proxy.test_y == 0.0).sum() == 4
    assert np.all((proxy.train_x >= 0.0) & (proxy.train_x <= 1.0))


def test_multiclass_proxy_shape_and_balance():

    proxy = build_mnist_digit_proxy(TRAIN_PATH, digits=[3, 7, 1], examples_per_class=20, seed=0)

    assert proxy.train_x.shape == (48, 28 * 28)
    assert proxy.test_x.shape == (12, 28 * 28)
    assert proxy.train_y.dtype == np.int64
    assert set(np.unique(proxy.train_y)) == {0, 1, 2}
    for category in (0, 1, 2):
        assert (proxy.train_y == category).sum() == 16
        assert (proxy.test_y == category).sum() == 4


def test_train_test_split_matches_requested_fraction():

    proxy = build_mnist_digit_proxy(TRAIN_PATH, digits=[5], examples_per_class=50, split_fraction=0.6, seed=1)

    assert (proxy.train_y == 1.0).sum() == 30
    assert (proxy.train_y == 0.0).sum() == 30
    assert (proxy.test_y == 1.0).sum() == 20
    assert (proxy.test_y == 0.0).sum() == 20


def test_same_seed_reproduces_the_identical_proxy():

    first = build_mnist_digit_proxy(TRAIN_PATH, digits=[2], examples_per_class=20, seed=42)
    second = build_mnist_digit_proxy(TRAIN_PATH, digits=[2], examples_per_class=20, seed=42)

    assert np.array_equal(first.train_x, second.train_x)
    assert np.array_equal(first.train_y, second.train_y)
    assert np.array_equal(first.test_x, second.test_x)
    assert np.array_equal(first.test_y, second.test_y)


def test_different_seeds_sample_different_examples():

    first = build_mnist_digit_proxy(TRAIN_PATH, digits=[2], examples_per_class=20, seed=0)
    second = build_mnist_digit_proxy(TRAIN_PATH, digits=[2], examples_per_class=20, seed=1)

    assert not np.array_equal(first.train_x, second.train_x)


def test_duplicate_digits_are_rejected():

    with pytest.raises(AssertionError):
        build_mnist_digit_proxy(TRAIN_PATH, digits=[3, 3], examples_per_class=20, seed=0)
