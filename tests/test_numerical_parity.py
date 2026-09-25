"""
Not new functionality - a single, consolidated sweep through every operation in this crate's
numpy interface subset, checked against real numpy (three-way, against a pure-Python reference
too, where one exists independent of numpy itself) in one auditable file, rather than something
someone has to reassemble from each operation's own scattered test file to answer "has the whole
subset actually been checked."

Every operation below already has its own dedicated, more detailed test file (test_array_basics,
test_transpose_and_slicing, test_array_ops, test_ufuncs_exp, test_linalg, test_ufuncs_argmax,
test_random_uniform, test_mnist_decode, test_tolist_roundtrip) - this file's job is breadth in one
place, not depth.
"""

import random

import numpy as np
import pytest
from indrajala_math_rust import Array, argmax, decode_mnist_pixels, exp, outer, seed, sum_axis0, uniform

from indrajala_ml.mnist_data import RECORD_SIZE, load_mnist_dataset_as_array
from tests.helpers import approx, rust_to_numpy

MNIST_TEST_PATH = "data/mnist/mnist-test.bin"

SEEDS = range(50)


@pytest.mark.parametrize("seed", SEEDS)
def test_full_subset_sweep_against_numpy(seed: int):
    rng = random.Random(seed)

    def random_vector(n: int):
        return [rng.uniform(-4.0, 4.0) for _ in range(n)]

    def random_matrix(rows: int, cols: int):
        return [random_vector(cols) for _ in range(rows)]

    # construction, zeros, shape
    vector_data = random_vector(6)
    matrix_data = random_matrix(4, 5)
    v, m = Array(vector_data), Array(matrix_data)
    np_v, np_m = np.array(vector_data), np.array(matrix_data)
    assert v.shape == np_v.shape
    assert m.shape == np_m.shape
    assert rust_to_numpy(Array.zeros(6)) == approx(np.zeros(6))
    assert rust_to_numpy(Array.zeros((3, 3))) == approx(np.zeros((3, 3)))

    # transpose
    assert rust_to_numpy(m.T) == approx(np_m.T)
    assert rust_to_numpy(v.T) == approx(np_v.T)

    # single-element read/write, both index shapes
    v_copy = v.copy()
    v_copy[2] = 99.0
    assert v_copy[2] == 99.0 and v[2] != 99.0
    m_copy = m.copy()
    m_copy[1, 3] = 99.0
    assert m_copy[1, 3] == 99.0 and m[1, 3] != 99.0

    # slicing
    assert rust_to_numpy(m[:, :-1]) == approx(np_m[:, :-1])

    # elementwise + - * /, same-shape and broadcast, and scalar operands
    other_vector = Array(random_vector(6))
    np_other_vector = rust_to_numpy(other_vector)
    assert rust_to_numpy(v + other_vector) == approx(np_v + np_other_vector)
    assert rust_to_numpy(v - other_vector) == approx(np_v - np_other_vector)
    assert rust_to_numpy(v * other_vector) == approx(np_v * np_other_vector)
    assert rust_to_numpy(v / other_vector) == approx(np_v / np_other_vector)

    row_vector = Array(random_vector(5))
    np_row_vector = rust_to_numpy(row_vector)
    assert rust_to_numpy(m + row_vector) == approx(np_m + np_row_vector)

    assert rust_to_numpy(0.5 * v) == approx(0.5 * np_v)
    assert rust_to_numpy(v / 4) == approx(np_v / 4)
    assert rust_to_numpy(1.0 - v) == approx(1.0 - np_v)

    # in-place accumulate
    accum = Array.zeros(6)
    accum += v
    assert rust_to_numpy(accum) == approx(np_v)
    accum -= other_vector
    assert rust_to_numpy(accum) == approx(np_v - np_other_vector)

    # exp
    assert rust_to_numpy(exp(v)) == approx(np.exp(np_v))

    # matmul: matrix@vector, vector@matrix, matrix@matrix
    square_ish = Array(random_matrix(5, 6))
    np_square_ish = rust_to_numpy(square_ish)
    assert rust_to_numpy(square_ish @ v) == approx(np_square_ish @ np_v)
    assert rust_to_numpy(v @ square_ish.T) == approx(np_v @ np_square_ish.T)
    other_matrix = Array(random_matrix(4, 3))
    assert rust_to_numpy(m.T @ other_matrix) == approx(np_m.T @ rust_to_numpy(other_matrix))

    # outer
    assert rust_to_numpy(outer(v, other_vector)) == approx(np.outer(np_v, np_other_vector))

    # sum_axis0
    assert rust_to_numpy(sum_axis0(m)) == approx(np_m.sum(axis=0))

    # argmax
    assert argmax(v) == int(np.argmax(np_v))

    # tolist round-trip
    assert v.tolist() == np_v.tolist()
    assert m.tolist() == np_m.tolist()
    assert Array(v.tolist()).tolist() == v.tolist()

    # reshape
    flat = Array(random_vector(12))
    reshaped = flat.reshape((3, 4))
    assert rust_to_numpy(reshaped) == approx(np.array(flat.tolist()).reshape(3, 4))


def test_uniform_matches_numpy_bit_for_bit_after_the_same_seed():
    # the crate's RNG is numpy's np.random in a separate state (rust/tests/test_random_numpy_parity.py)
    np.random.seed(2000)
    seed(2000)
    assert uniform(-1.0, 1.0, 2000).tolist() == np.random.uniform(-1.0, 1.0, 2000).tolist()


def test_mnist_decode_matches_the_real_reference_implementation():
    with open(MNIST_TEST_PATH, "rb") as f:
        raw = f.read(RECORD_SIZE * 20)
    actual = decode_mnist_pixels(raw, RECORD_SIZE)
    expected = load_mnist_dataset_as_array(MNIST_TEST_PATH, limit=20)
    assert rust_to_numpy(actual) == approx(expected, abs=1e-15)
