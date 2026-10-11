"""
Rotary position embedding's angles (the rotary-positions workplan, D4), numpy-free, so the pure-Python,
numpy and Rust layers share one copy: cos and sin aren't correctly rounded, and numpy's, Rust's and
Python's math may differ in the last bit, so each implementation reads this one table rather than
computing its own.
"""

from __future__ import annotations

import math


def rotary_table(positions: int, key_size: int, base: float) -> tuple[list[list[float]], list[list[float]]]:
    """
    cos and sin of t * theta_j, (positions, key_size / 2) each, for positions t from 0 and each pair j
    of a head's features (j with j + key_size / 2, rotate half; D3), theta_j = base ** (-2j / key_size).
    Generation extends it past a window's tokens by asking for more positions; the rows already there
    don't change.
    """
    assert key_size >= 2 and key_size % 2 == 0, f"a rotary head size is even; got {key_size}"
    assert positions >= 1, f"a rotary table covers at least one position; got {positions}"
    assert base > 0.0, f"a rotary_base is positive; got {base}"
    thetas = [base ** (-2 * j / key_size) for j in range(key_size // 2)]
    angles = [[t * theta for theta in thetas] for t in range(positions)]
    return [[math.cos(a) for a in row] for row in angles], [[math.sin(a) for a in row] for row in angles]
