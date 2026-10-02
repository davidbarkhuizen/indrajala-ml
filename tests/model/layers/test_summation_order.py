"""
The summation order of numpy's reductions at batch norm's shapes, against a sequential loop and
the crate's sums (the batch-norm workplan, stage 0, Pitfalls). The numpy batch-norm layers must sum
in the crate's order, a left fold in row order, so that numpy and Rust agree by bits. These pin
which numpy forms do:

- a dense layer's per-feature sums over the batch, X.sum(axis=0) on a (batch, features) array,
  are sequential when there are two or more features, but pairwise for a single feature once the
  batch reaches 8 rows (the column is contiguous);
- a conv layer's per-channel sums over the batch and positions, D.sum(axis=(0, 2)) on the
  (batch, channels, positions) view, are not sequential, and neither is the conv layer's grad_b
  today (crate: conv_accumulate_gradient_batch);
- np.cumsum along the summed axis, last element, is sequential at every shape.

Comparisons are with np.array_equal, which takes 0.0 and -0.0 as equal: a loop from 0.0 and a
fold from the first value differ only in the sign of an all-zero sum.
"""

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model.layers.numpy.array_layer import FloatArray

BATCH_SIZES = [2, 3, 4, 5, 7, 8, 9, 16, 32, 64, 100, 128, 129, 256, 512]
FEATURE_COUNTS = [1, 2, 3, 30]
# (input height, width, channels, kernel size): positions P of 4, 16, 36 and 64
CONV_GEOMETRIES = [(4, 4, 1, 3), (6, 6, 1, 3), (8, 8, 2, 3), (10, 10, 1, 3)]
CHANNEL_COUNTS = [1, 2, 8]


def _sequential(values: list[float]) -> float:
    # the crate's order: a left fold from 0.0 (sum_axis0 per column; conv grad_b per channel)
    total = 0.0
    for value in values:
        total += value
    return total


def _sequential_columns(X: FloatArray) -> FloatArray:
    return np.array([_sequential(X[:, column].tolist()) for column in range(X.shape[1])])


def _draw(*shape: int) -> FloatArray:
    return np.random.default_rng(sum(shape)).standard_normal(shape)


def _dense(feature_counts: list[int]) -> list[tuple[int, int]]:
    return [(batch_size, features) for batch_size in BATCH_SIZES for features in feature_counts]


@pytest.mark.parametrize("batch_size, features", _dense(FEATURE_COUNTS))
def test_the_crate_sums_rows_sequentially(batch_size: int, features: int):
    X = _draw(batch_size, features)

    assert np.array_equal(np.array(pa.sum_axis0(pa.Array(X.tolist())).tolist()), _sequential_columns(X))


@pytest.mark.parametrize("batch_size, features", _dense([f for f in FEATURE_COUNTS if f >= 2]))
def test_numpy_sums_rows_sequentially_across_two_or_more_features(batch_size: int, features: int):
    X = _draw(batch_size, features)

    assert np.array_equal(X.sum(axis=0), _sequential_columns(X))


def test_numpy_sums_a_single_feature_pairwise_from_eight_rows():
    # sequential below 8 values, numpy's pairwise block size; past it, not in the crate's order
    matches = {
        batch_size: np.array_equal((X := _draw(batch_size, 1)).sum(axis=0), _sequential_columns(X))
        for batch_size in BATCH_SIZES
    }

    assert all(matches[batch_size] for batch_size in BATCH_SIZES if batch_size < 8)
    assert not all(matches.values())


@pytest.mark.parametrize("batch_size, features", _dense(FEATURE_COUNTS))
def test_cumsum_sums_rows_sequentially(batch_size: int, features: int):
    X = _draw(batch_size, features)

    assert np.array_equal(np.cumsum(X, axis=0)[-1], _sequential_columns(X))


def _conv_cases() -> list[tuple[tuple[int, int, int, int], int, int]]:
    return [
        (geometry, channels, batch_size)
        for geometry in CONV_GEOMETRIES
        for channels in CHANNEL_COUNTS
        for batch_size in (2, 3, 8, 32)
    ]


def _per_channel(D: FloatArray, channels: int) -> FloatArray:
    # each channel's values in the crate's order: example-major, then position, a (batch,
    # channels * positions) row being channel-major
    batch_size = D.shape[0]
    by_channel = D.reshape(batch_size, channels, -1)
    return np.array([_sequential(by_channel[:, channel, :].ravel().tolist()) for channel in range(channels)])


@pytest.mark.parametrize("geometry, channels, batch_size", _conv_cases())
def test_the_crate_sums_a_conv_channel_example_major(
    geometry: tuple[int, int, int, int], channels: int, batch_size: int
):
    conv = pa.ConvGeometry(*geometry, 1)
    D = _draw(batch_size, channels * conv.positions)
    cols = pa.Array.zeros((batch_size * conv.positions, conv.fan_in))

    _grad_W, grad_b = pa.conv_accumulate_gradient_batch(
        pa.Array(D.tolist()), cols, pa.Array.zeros((channels, conv.fan_in)), pa.Array.zeros(channels), conv
    )

    assert np.array_equal(np.array(grad_b.tolist()), _per_channel(D, channels))


def test_numpy_conv_channel_sums_are_not_the_crates():
    # ConvArrayLayer's grad_b form, D.sum(axis=(0, 2)), so a conv batch-norm layer can't use it
    matches = [
        np.array_equal(
            (D := _draw(batch_size, channels * pa.ConvGeometry(*geometry, 1).positions))
            .reshape(batch_size, channels, -1)
            .sum(axis=(0, 2)),
            _per_channel(D, channels),
        )
        for geometry, channels, batch_size in _conv_cases()
    ]

    # it agrees with the crate on some draws, by chance, but not on all
    assert not all(matches)


@pytest.mark.parametrize("geometry, channels, batch_size", _conv_cases())
def test_cumsum_sums_a_conv_channel_example_major(geometry: tuple[int, int, int, int], channels: int, batch_size: int):
    positions = pa.ConvGeometry(*geometry, 1).positions
    D = _draw(batch_size, channels * positions)
    # (channels, batch * positions), each row in the crate's order
    by_channel = D.reshape(batch_size, channels, positions).transpose(1, 0, 2).reshape(channels, -1)

    assert np.array_equal(np.cumsum(by_channel, axis=1)[:, -1], _per_channel(D, channels))
