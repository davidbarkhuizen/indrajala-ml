"""
indrajala_ml/studies/inference.py against published values (Fisher's tea-tasting table, the
control's and D4's figures, Wilson's interval) and independent oracles: Mann-Whitney's exact p by
enumerating every split, Cochran-Armitage over two groups as Pearson's chi-square.
"""

import itertools
import math
from collections.abc import Sequence

import pytest

from indrajala_ml.studies import inference as st


def test_wilson_interval():
    assert st.wilson(8, 10) == pytest.approx((0.490162, 0.943318), abs=1e-6)
    assert st.wilson(0, 10) == pytest.approx((0.0, 0.277533), abs=1e-6) and st.wilson(0, 10)[0] == 0.0
    assert st.wilson(20, 20)[1] == 1.0 and st.wilson(0, 0) == (0.0, 1.0)
    low, high = st.wilson(10, 20)
    assert low == pytest.approx(1 - high)


def test_fisher_exact_two_sided():
    assert st.fisher_exact(3, 1, 1, 3) == pytest.approx(0.485714, abs=1e-6)  # the lady tasting tea
    assert st.fisher_exact(10, 0, 8, 2) == pytest.approx(0.473684, abs=1e-6)  # the control at k 4
    assert st.fisher_exact(2, 3, 4, 1) == pytest.approx(0.523810, abs=1e-6)  # the control at k 1
    assert st.fisher_exact(16, 4, 8, 12) == pytest.approx(0.0225, abs=1e-4)  # D4: 0.8 against 0.4 at 20
    assert st.fisher_exact(15, 5, 10, 10) == pytest.approx(0.19, abs=0.005)  # D4: 0.75 against 0.5
    assert st.fisher_exact(5, 5, 5, 5) == 1.0 and st.fisher_exact(0, 0, 0, 0) == 1.0
    # the same test whichever way the table is read
    assert st.fisher_exact(4, 16, 12, 8) == st.fisher_exact(16, 4, 8, 12) == st.fisher_exact(16, 8, 4, 12)


def _enumerated_p(first: Sequence[float], second: Sequence[float]) -> float:
    """Mann-Whitney's two-sided exact p by every split of the pooled values."""
    pooled = [*first, *second]

    def u(chosen: tuple[int, ...]) -> float:
        rest = [pooled[i] for i in range(len(pooled)) if i not in chosen]
        return sum(1.0 if pooled[i] > y else 0.5 if pooled[i] == y else 0.0 for i in chosen for y in rest)

    observed = u(tuple(range(len(first))))
    splits = [u(chosen) for chosen in itertools.combinations(range(len(pooled)), len(first))]
    below = sum(x <= observed for x in splits) / len(splits)
    above = sum(x >= observed for x in splits) / len(splits)
    return min(1.0, 2 * min(below, above))


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ([1, 2, 3], [4, 5, 6]),
        ([0.39, 1.05, 0.26, 0.52, 0.71], [6.4, 4.8, 12.3, 0.6, 7.1, 5.5]),
        ([3, 9, 1, 7], [2, 8, 4, 6, 5]),
        ([10.0], [1.0, 2.0, 3.0]),
    ],
)
def test_mann_whitney_exact_against_enumeration(first: Sequence[float], second: Sequence[float]):
    result = st.mann_whitney(first, second)
    assert result.exact and result.p == pytest.approx(_enumerated_p(first, second), abs=1e-12)
    assert st.mann_whitney(second, first).u == len(first) * len(second) - result.u
    assert st.mann_whitney(second, first).p == pytest.approx(result.p)


def test_mann_whitney_with_ties_is_near_the_enumeration():
    first, second = [1, 2, 2, 3, 5, 5, 6], [2, 4, 5, 7, 8, 8, 9, 9]
    result = st.mann_whitney(first, second)
    assert not result.exact and result.u == 10.0  # 0 + 2 x 0.5 + 1 + 2 x 2.5 + 3
    assert result.p == pytest.approx(_enumerated_p(first, second), abs=0.02)
    assert st.mann_whitney([1, 1], [1, 1, 1]).p == 1.0


def test_cochran_armitage_over_two_groups_is_pearsons_chi_square():
    a, b, c, d = 16, 4, 8, 12  # successes and failures of two groups
    chi_square = (a + b + c + d) * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    trend = st.cochran_armitage([a, c], [a + b, c + d], [0, 1])
    assert trend.z**2 == pytest.approx(chi_square) and trend.z < 0
    assert trend.p == pytest.approx(math.erfc(math.sqrt(chi_square / 2)))


def test_cochran_armitage_trend():
    successes, trials, scores = [2, 3, 5, 8], [10, 10, 10, 10], [3, 4, 8, 16]
    trend = st.cochran_armitage(successes, trials, scores)
    assert trend.z > 0 and trend.p < 0.01
    # a linear change of the scores leaves the test alone; reversing them flips the sign
    assert st.cochran_armitage(successes, trials, [2 * t + 5 for t in scores]).z == pytest.approx(trend.z)
    assert st.cochran_armitage(successes, trials, [-t for t in scores]).z == pytest.approx(-trend.z)
    assert st.cochran_armitage([0, 0], [5, 5], [1, 2]) == st.Trend(0.0, 1.0)
    assert st.cochran_armitage([3, 3, 3], [10, 10, 10], [1, 2, 3]).z == 0.0


def test_holm_adjusts_step_down():
    assert st.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert st.holm([0.5, 0.9]) == [1.0, 1.0] and st.holm([]) == []
