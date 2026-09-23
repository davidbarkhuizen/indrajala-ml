import pytest

from indrajala_ml.lr_schedule import linear_warmup


def test_linear_warmup_first_step_is_target_rate_over_warmup_steps():
    schedule = linear_warmup(target_rate=64.0, warmup_steps=10)
    assert schedule(0) == pytest.approx(64.0 / 10)


def test_linear_warmup_ramps_linearly_before_the_last_warmup_step():
    schedule = linear_warmup(target_rate=64.0, warmup_steps=10)
    assert schedule(8) == pytest.approx(64.0 * 9 / 10)


def test_linear_warmup_reaches_target_rate_exactly_at_warmup_steps_minus_one():
    schedule = linear_warmup(target_rate=64.0, warmup_steps=10)
    assert schedule(9) == pytest.approx(64.0)


def test_linear_warmup_holds_at_target_rate_past_warmup_steps():
    schedule = linear_warmup(target_rate=64.0, warmup_steps=10)
    assert schedule(10) == pytest.approx(64.0)
    assert schedule(100) == pytest.approx(64.0)


def test_linear_warmup_of_one_step_is_target_rate_from_the_start():
    schedule = linear_warmup(target_rate=0.5, warmup_steps=1)
    assert schedule(0) == pytest.approx(0.5)
    assert schedule(1) == pytest.approx(0.5)
