import json
import math

import pytest

from indrajala_ml.training.lr_schedule import (
    Constant,
    CosineDecay,
    LinearDecay,
    Warmup,
    schedule_from_json,
    schedule_of,
    schedule_to_json,
)


def _old_linear_warmup(target_rate: float, warmup_steps: int, step: int) -> float:
    # the linear_warmup callable Warmup(steps, Constant(rate)) replaced, verbatim (D6)
    return target_rate * min((step + 1) / warmup_steps, 1.0)


@pytest.mark.parametrize(("rate", "steps"), [(64.0, 10), (0.5, 1), (0.1, 469), (12.0, 59), (0.003, 7)])
def test_warmup_of_a_constant_gives_linear_warmups_rate_bits_at_every_step(rate: float, steps: int):
    schedule = Warmup(steps, Constant(rate))
    for step in range(3 * steps + 2):
        assert schedule(step).hex() == _old_linear_warmup(rate, steps, step).hex()


def test_warmup_ramps_from_the_rate_over_steps_to_the_rate():
    schedule = Warmup(10, Constant(64.0))
    assert [schedule(step) for step in (0, 8, 9, 10, 100)] == [6.4, 64.0 * 9 / 10, 64.0, 64.0, 64.0]


def test_warmup_counts_the_schedule_after_it_from_its_own_end():
    decay = LinearDecay(4.0, 4, 0.0)
    schedule = Warmup(2, decay)
    assert [schedule(step) for step in range(8)] == [2.0, 4.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0]


def test_cosine_decay_follows_sgdrs_form_then_holds_the_floor():
    schedule = CosineDecay(1.0, 4, 0.1)
    assert schedule(0) == 1.0
    assert schedule(2) == 0.1 + 0.9 * (1.0 + math.cos(math.pi * 2 / 4)) / 2.0
    assert schedule(2) == pytest.approx(0.55)
    assert schedule(4) == 0.1
    assert schedule(1000) == 0.1
    rates = [schedule(step) for step in range(5)]
    assert rates == sorted(rates, reverse=True)


def test_linear_decay_falls_linearly_then_holds_the_floor():
    schedule = LinearDecay(2.0, 4, 1.0)
    assert [schedule(step) for step in range(6)] == [2.0, 1.75, 1.5, 1.25, 1.0, 1.0]


def test_the_production_schedule_peaks_after_warmup_and_ends_at_a_tenth():
    schedule = Warmup(5, CosineDecay(0.01, 95, 0.001))
    assert schedule(0) == pytest.approx(0.002)
    assert schedule(4) == schedule(5) == CosineDecay(0.01, 95, 0.001)(0)
    assert schedule(5) == pytest.approx(0.01)  # D3's grouping: floor + (rate - floor) * 1, by rounding
    assert schedule(100) == 0.001


def test_schedules_are_data():
    assert Warmup(3, CosineDecay(1.0, 5, 0.1)) == Warmup(3, CosineDecay(1.0, 5, 0.1))
    assert Warmup(3, CosineDecay(1.0, 5, 0.1)) != Warmup(3, CosineDecay(1.0, 5, 0.2))
    with pytest.raises(AttributeError):
        Constant(1.0).rate = 2.0  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize(
    "make",
    [
        lambda: Constant(-1.0),
        lambda: Constant(math.nan),
        lambda: Warmup(0, Constant(1.0)),
        lambda: CosineDecay(1.0, 0, 0.1),
        lambda: CosineDecay(1.0, 5, 2.0),
        lambda: LinearDecay(1.0, 5, -0.1),
    ],
)
def test_a_schedule_that_cannot_be_trained_is_refused(make: object):
    with pytest.raises(ValueError):
        make()  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize(
    "schedule",
    [Constant(0.1), Warmup(3, Constant(0.3)), Warmup(7, CosineDecay(1e-3, 93, 1e-4)), LinearDecay(2.0, 9, 0.0)],
)
def test_a_schedule_round_trips_through_json_by_bits(schedule: object):
    assert isinstance(schedule, Constant | Warmup | CosineDecay | LinearDecay)
    loaded = schedule_from_json(json.loads(json.dumps(schedule_to_json(schedule))))
    assert loaded == schedule
    assert [loaded(step).hex() for step in range(120)] == [schedule(step).hex() for step in range(120)]


def test_an_unknown_schedule_is_refused():
    with pytest.raises(ValueError, match="'step'"):
        schedule_from_json({"schedule": "step"})


def test_schedule_of_takes_a_float_as_constant_and_leaves_a_callable_unrecorded():
    assert schedule_of(0.5) == Constant(0.5)
    assert schedule_of(Warmup(2, Constant(0.5))) == Warmup(2, Constant(0.5))
    assert schedule_of(lambda step: 0.5) is None
