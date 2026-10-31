"""The sampler's filters and draw (the generation workplan, D4, stage 2), checked by hand."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from indrajala_ml.generation.sampling import Sampler, argmax, draw, temperature, top_k, top_p
from indrajala_ml.pcg64 import default_rng


def test_argmax_takes_the_first_most_likely_id() -> None:
    assert argmax([0.2, 0.4, 0.4]) == 1


def test_temperature_one_is_the_identity() -> None:
    p = [0.1, 0.2, 0.7]
    assert temperature(1.0)(p) == p


def test_temperature_zero_is_the_argmax_as_a_one_hot() -> None:
    assert temperature(0.0)([0.3, 0.35, 0.35]) == [0.0, 1.0, 0.0]


def test_temperature_is_a_softmax_of_log_p_over_t() -> None:
    p = [0.1, 0.3, 0.6]
    t = 0.5
    # p ** (1 / t), renormalized: the same distribution, by hand
    powered = [p_i**2 for p_i in p]
    expected = [v / sum(powered) for v in powered]
    assert temperature(t)(p) == pytest.approx(expected, rel=1e-15)


def test_temperature_keeps_impossible_ids_impossible() -> None:
    assert temperature(2.0)([0.0, 0.5, 0.5])[0] == 0.0


def test_top_k_keeps_the_k_most_likely_ties_by_id() -> None:
    # 0.3 at ids 1 and 2: the lower id is kept
    assert top_k(2)([0.1, 0.3, 0.3, 0.3]) == [0.0, 0.5, 0.5, 0.0]


def test_top_k_at_the_vocabulary_renormalizes_only() -> None:
    p = [0.25, 0.25, 0.5]
    assert top_k(3)(p) == p


def test_top_p_keeps_the_smallest_set_reaching_the_mass() -> None:
    p = [0.1, 0.5, 0.15, 0.25]
    # descending 0.5 (id 1), 0.25 (id 3): 0.75 reaches 0.7
    assert top_p(0.7)(p) == [0.0, 0.5 / 0.75, 0.0, 0.25 / 0.75]
    # exactly the mass: the set stops there
    assert top_p(0.75)(p) == [0.0, 0.5 / 0.75, 0.0, 0.25 / 0.75]
    assert top_p(0.76)(p)[2] > 0.0


def test_top_p_keeps_at_least_the_most_likely() -> None:
    assert top_p(0.01)([0.2, 0.8]) == [0.0, 1.0]


def test_draw_is_the_first_id_whose_cumulative_sum_exceeds_u() -> None:
    p = [0.25, 0.0, 0.25, 0.5]
    assert draw(p, 0.0) == 0
    assert draw(p, 0.2499) == 0
    # 0.25 does not exceed 0.25: the next id with mass
    assert draw(p, 0.25) == 2
    assert draw(p, 0.75) == 3
    assert draw(p, 0.999) == 3


def test_draw_past_a_rounded_total_takes_the_last_id_with_mass() -> None:
    assert draw([0.3, 0.3, 0.0], 0.99) == 1


def test_draw_never_takes_an_id_without_mass() -> None:
    rng = default_rng(5)
    p = [0.0, 0.4, 0.0, 0.6, 0.0]
    assert {draw(p, rng.random()) for _ in range(200)} == {1, 3}


def test_the_sampler_chains_temperature_top_k_then_top_p() -> None:
    sampler = Sampler(temperature=0.5, top_k=3, top_p=0.9)
    p = [0.05, 0.1, 0.15, 0.3, 0.4]
    expected = top_p(0.9)(top_k(3)(temperature(0.5)(p)))
    assert sampler.filtered(p) == expected
    assert len(sampler.filters()) == 3


def test_the_default_sampler_draws_from_p_itself() -> None:
    assert Sampler().filters() == []
    assert Sampler(top_p=1.0).filters() == []


def test_one_seed_samples_the_same_ids() -> None:
    sampler = Sampler(temperature=0.8, top_k=4)
    p = [0.1, 0.2, 0.3, 0.25, 0.15]

    def ids(seed: int) -> list[int]:
        rng = default_rng(seed)
        return [sampler(p, rng) for _ in range(50)]

    assert ids(3) == ids(3)
    assert ids(3) != ids(4)


def test_the_draws_are_numpys_stream() -> None:
    # pure Python's generator and numpy's default_rng give the same uniforms, so the same ids
    p = [0.1, 0.2, 0.3, 0.4]
    ours, numpys = default_rng(11), np.random.default_rng(11)
    assert [Sampler()(p, ours) for _ in range(20)] == [Sampler()(p, numpys) for _ in range(20)]


def test_greedy_takes_one_draw_per_token() -> None:
    rng, reference = default_rng(2), default_rng(2)
    Sampler(temperature=0.0)([0.5, 0.5], rng)
    reference.random()
    assert rng.random() == reference.random()


def test_the_draws_follow_the_distribution() -> None:
    rng = default_rng(0)
    p = [0.2, 0.5, 0.3]
    counts = [0, 0, 0]
    n = 20000
    for _ in range(n):
        counts[Sampler()(p, rng)] += 1
    for count, p_i in zip(counts, p, strict=True):
        assert abs(count / n - p_i) < 4 * math.sqrt(p_i * (1 - p_i) / n)


@pytest.mark.parametrize("bad", [lambda: temperature(-0.1), lambda: top_k(0), lambda: top_p(0.0), lambda: top_p(1.5)])
def test_out_of_range_settings_are_refused(bad: Callable[[], object]) -> None:
    with pytest.raises(AssertionError):
        bad()
