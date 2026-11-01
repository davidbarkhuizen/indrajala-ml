"""
erf in all three implementations (the FFN activations workplan, D2): the pure-Python port of
fdlibm's s_erf.c, numpy's elementwise port and the crate's give the same bits wherever their exp
does, so each test gives them one exp (the crate's, correctly rounded, or math.exp) and compares by
bits; each is within one ulp of math.erf. The crate's accuracy against an exact reference is its own
suite's (rust/tests/test_erf.py).
"""

import math
import random
import struct

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model.layers.numpy import erf as numpy_erf
from indrajala_ml.model.layers.python import erf as python_erf
from tests.helpers import exp_by_crate, exp_by_math

# fdlibm's range bounds on |x| (2^-28, 0.84375, 1.25, 1/0.35, 6) and the smallest normal, as bits
BOUNDS = [
    0x3E30000000000000, 0x3FEB000000000000, 0x3FF4000000000000, 0x4006DB6E00000000, 0x4018000000000000,
    0x0010000000000000,
]  # fmt: skip


def _double(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _bits(values: list[float]) -> list[int]:
    return [struct.unpack("<Q", struct.pack("<d", v))[0] for v in values]


def _inputs() -> list[float]:
    rng = random.Random(20261101)
    values = [rng.uniform(-7.0, 7.0) for _ in range(4000)]
    values += [math.copysign(10.0 ** rng.uniform(-320.0, 1.5), rng.choice((-1.0, 1.0))) for _ in range(1000)]
    for bound in BOUNDS:
        for b in range(bound - 3, bound + 4):
            values += [_double(b), -_double(b)]
    return [*values, 0.0, -0.0, 5e-324, -5e-324, 27.0, -27.0, 1e300, -1e300, math.inf, -math.inf]


def _crate_scalar_exp(x: float) -> float:
    return pa.exp(pa.Array([x])).tolist()[0]


def test_three_implementations_agree_by_bits_on_the_crates_exp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(python_erf, "exp", _crate_scalar_exp)
    monkeypatch.setattr(numpy_erf, "exp", exp_by_crate)
    values = _inputs()
    crate = pa.erf(pa.Array(values)).tolist()
    python = [python_erf.erf(x) for x in values]
    numpy = numpy_erf.erf(np.array(values)).tolist()
    assert _bits(python) == _bits(crate)
    assert _bits(numpy) == _bits(crate)


def test_python_and_numpy_agree_by_bits_on_math_exp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(numpy_erf, "exp", exp_by_math)
    values = _inputs()
    assert _bits(numpy_erf.erf(np.array(values)).tolist()) == _bits([python_erf.erf(x) for x in values])


def test_nan() -> None:
    assert math.isnan(python_erf.erf(math.nan))
    assert np.isnan(numpy_erf.erf(np.array([math.nan, 1.0])))[0]
    assert math.isnan(pa.erf(pa.Array([math.nan])).tolist()[0])


@pytest.mark.parametrize("implementation", ["python", "numpy"])
def test_within_one_ulp_of_math_erf(implementation: str) -> None:
    values = _inputs()
    got = (
        [python_erf.erf(x) for x in values] if implementation == "python" else numpy_erf.erf(np.array(values)).tolist()
    )
    for x, y in zip(values, got, strict=True):
        assert abs(y - math.erf(x)) <= math.ulp(math.erf(x)), f"erf({x!r}) = {y!r}, math.erf {math.erf(x)!r}"


def test_numpy_keeps_the_shape() -> None:
    values = np.linspace(-3.0, 3.0, 24).reshape(2, 3, 4)
    got = numpy_erf.erf(values)
    assert got.shape == (2, 3, 4)
    assert _bits(got.reshape(-1).tolist()) == _bits(numpy_erf.erf(values.reshape(-1)).tolist())
