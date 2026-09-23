import numpy as np
import pytest

from indrajala_ml.mnist_data import (
    IMAGE_SIZE,
    RECORD_SIZE,
    convert_parquet_to_binary,
    load_mnist_dataset,
    load_mnist_dataset_as_array,
    load_mnist_labels,
    load_mnist_records_at_indices,
)


def test_convert_parquet_to_binary_produces_a_correctly_shaped_file(tmp_path):

    binary_path = str(tmp_path / "converted.bin")

    convert_parquet_to_binary("data/mnist/mnist-train.parquet", binary_path, limit=5)

    import os

    assert os.path.getsize(binary_path) == 5 * RECORD_SIZE

    dataset = load_mnist_dataset(binary_path)
    labels = [label for _, label in dataset]
    # the standard MNIST training set's well-known first-few labels - confirms the conversion
    # (parquet -> PNG decode -> flat binary) round-trips real data correctly, not just the
    # right byte count
    assert labels == [5, 0, 4, 1, 9]


def test_load_mnist_dataset_shape_and_normalization():

    dataset = load_mnist_dataset("data/mnist/mnist-train.bin", limit=20)

    assert len(dataset) == 20
    assert all(len(state) == IMAGE_SIZE * IMAGE_SIZE for state, _ in dataset)
    assert all(0.0 <= value <= 1.0 for state, _ in dataset for value in state)
    assert all(0 <= label <= 9 for _, label in dataset)


def test_load_mnist_dataset_decodes_a_known_real_sample_correctly():

    # the standard MNIST training set's well-known first-few labels (5, 0, 4, 1, 9, ...) - a
    # sanity check this is genuinely the real dataset, not corrupted or reordered
    dataset = load_mnist_dataset("data/mnist/mnist-train.bin", limit=5)
    labels = [label for _, label in dataset]
    assert labels == [5, 0, 4, 1, 9]

    # specific pixel values from the first image, read directly off the decoded PNG and
    # independently cross-checked (not re-derived from this loader) - chosen to cover every PNG
    # scanline filter type this image's IDAT stream actually uses (confirmed directly: row 0 is
    # None, row 6 is Paeth, row 7 is Sub, row 11 is Up), not just coordinates that happen to
    # land on unfiltered rows
    state0, _ = dataset[0]

    def pixel(row: int, col: int) -> float:
        return state0[row * IMAGE_SIZE + col]

    assert pixel(0, 0) == pytest.approx(0 / 255)  # filter type 0 (None)
    assert pixel(6, 10) == pytest.approx(94 / 255)  # filter type 4 (Paeth)
    assert pixel(7, 20) == pytest.approx(82 / 255)  # filter type 1 (Sub)
    assert pixel(11, 11) == pytest.approx(139 / 255)  # filter type 2 (Up)
    assert pixel(11, 13) == pytest.approx(190 / 255)  # filter type 2 (Up)


def test_load_mnist_dataset_test_split_shape():

    dataset = load_mnist_dataset("data/mnist/mnist-test.bin", limit=10)

    assert len(dataset) == 10
    assert all(len(state) == IMAGE_SIZE * IMAGE_SIZE for state, _ in dataset)


def test_load_mnist_dataset_without_limit_reads_the_full_file():

    # the test split is the smaller of the two bundled files (10000 rows) - still exercises
    # "no limit" without paying the much larger training file's full decode cost
    dataset = load_mnist_dataset("data/mnist/mnist-test.bin")

    assert len(dataset) == 10000


def test_load_mnist_dataset_rejects_a_file_whose_size_is_not_a_record_multiple(tmp_path):

    bad_path = str(tmp_path / "truncated.bin")
    with open(bad_path, "wb") as f:
        f.write(b"\x00" * (RECORD_SIZE + 1))

    with pytest.raises(AssertionError):
        load_mnist_dataset(bad_path)


def test_load_mnist_labels_matches_load_mnist_dataset_labels():

    dataset = load_mnist_dataset("data/mnist/mnist-train.bin", limit=50)
    expected_labels = [label for _, label in dataset]

    labels = load_mnist_labels("data/mnist/mnist-train.bin")

    assert labels[:50] == expected_labels
    assert len(labels) == 60000


def test_load_mnist_records_at_indices_matches_load_mnist_dataset():

    full = load_mnist_dataset("data/mnist/mnist-train.bin", limit=20)
    indices = [1, 5, 10, 15]

    records = load_mnist_records_at_indices("data/mnist/mnist-train.bin", indices)

    assert records == [full[index] for index in indices]


def test_load_mnist_records_at_indices_respects_a_non_sorted_index_order():

    full = load_mnist_dataset("data/mnist/mnist-train.bin", limit=20)
    indices = [15, 1, 10, 5]

    records = load_mnist_records_at_indices("data/mnist/mnist-train.bin", indices)

    assert records == [full[index] for index in indices]


def test_load_mnist_dataset_as_array_matches_load_mnist_dataset_pixels():

    dataset = load_mnist_dataset("data/mnist/mnist-train.bin", limit=50)
    expected = np.array([state for state, _label in dataset])

    actual = load_mnist_dataset_as_array("data/mnist/mnist-train.bin", limit=50)

    assert actual.shape == (50, IMAGE_SIZE * IMAGE_SIZE)
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_load_mnist_dataset_as_array_without_limit_reads_the_full_file():

    actual = load_mnist_dataset_as_array("data/mnist/mnist-test.bin")

    assert actual.shape == (10000, IMAGE_SIZE * IMAGE_SIZE)


def test_load_mnist_dataset_as_array_rejects_a_file_whose_size_is_not_a_record_multiple(tmp_path):

    bad_path = str(tmp_path / "truncated.bin")
    with open(bad_path, "wb") as f:
        f.write(b"\x00" * (RECORD_SIZE + 1))

    with pytest.raises(AssertionError):
        load_mnist_dataset_as_array(bad_path)
