"""
GELU, GELU-tanh and SiLU, and their derivatives, in all three implementations (the FFN activations
workplan, D2, D3, D9; README, Activations): pure Python's activations.py, numpy's and the crate's
(rust/core/src/activation.rs) take the same steps in the same grouping, through their own exp and
erf, so each test gives them one exp (the crate's, correctly rounded, or math.exp) and compares by
bits. The crate's accuracy against an exact reference is its own suite's
(rust/tests/test_activation_ops.py).
"""

import math
import random
import struct

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model.layers.numpy import activations as numpy_activations
from indrajala_ml.model.layers.numpy import erf as numpy_erf
from indrajala_ml.model.layers.python import activations as python_activations
from indrajala_ml.model.layers.python import erf as python_erf
from indrajala_ml.model.layers.python.activations import Smooth
from tests.helpers import exp_by_crate, exp_by_math

SMOOTH: list[Smooth] = ["gelu", "gelu_tanh", "silu"]


def _bits(values: list[float]) -> list[int]:
    return [struct.unpack("<Q", struct.pack("<d", v))[0] for v in values]


def _inputs() -> list[float]:
    # erf's ranges (|z| / sqrt 2 past 0.84375, 1.25, 2.857 and 6), tanh's saturation, tiny values
    rng = random.Random(20261102)
    values = [rng.uniform(-12.0, 12.0) for _ in range(3000)]
    values += [math.copysign(10.0 ** rng.uniform(-300.0, 2.5), rng.choice((-1.0, 1.0))) for _ in range(500)]
    return [*values, 0.0, -0.0, 5e-324, -5e-324, 1.0, -1.0, 40.0, -40.0, 1e6, -1e6]


def _crate_scalar_exp(x: float) -> float:
    return pa.exp(pa.Array([x])).tolist()[0]


def _crate(activation: Smooth, values: list[float]) -> tuple[list[float], list[float]]:
    # act(z) through the fused forward on z alone (1 * z + 0 is z, but for -0.0), and act'(z) as
    # the mask of a downstream of ones
    rows = pa.Array([[v] for v in values])
    a = pa.layer_activation_forward_batch(pa.Array([[1.0]]), rows, pa.Array([0.0]), activation)[0]
    derivative = pa.array_activation_mask(pa.Array([1.0] * len(values)), pa.Array(values), activation)
    return np.array(a.tolist()).reshape(-1).tolist(), derivative.tolist()


def _python(activation: Smooth, values: list[float]) -> tuple[list[float], list[float]]:
    return (
        [python_activations.activate(activation, z) for z in values],
        [python_activations.derivative(activation, z) for z in values],
    )


def _numpy(activation: Smooth, values: list[float]) -> tuple[list[float], list[float]]:
    z = np.array(values)
    return numpy_activations.activate(activation, z).tolist(), numpy_activations.derivative(activation, z).tolist()


@pytest.mark.parametrize("activation", SMOOTH)
def test_three_implementations_agree_by_bits_on_the_crates_exp(activation: Smooth, monkeypatch: pytest.MonkeyPatch):
    for module in (python_activations, python_erf):
        monkeypatch.setattr(module, "exp", _crate_scalar_exp)
    for module in (numpy_activations, numpy_erf):
        monkeypatch.setattr(module, "exp", exp_by_crate)
    values = _inputs()
    # the crate's forward makes -0.0 into 0.0 (1 * -0.0 + 0.0), so its values are compared from z = 1 * z + 0
    crate_values, crate_derivatives = _crate(activation, values)
    python_values, python_derivatives = _python(activation, [z + 0.0 for z in values])
    numpy_values, numpy_derivatives = _numpy(activation, [z + 0.0 for z in values])
    assert _bits(python_values) == _bits(crate_values)
    assert _bits(numpy_values) == _bits(crate_values)
    _, python_derivatives = _python(activation, values)
    _, numpy_derivatives = _numpy(activation, values)
    assert _bits(python_derivatives) == _bits(crate_derivatives)
    assert _bits(numpy_derivatives) == _bits(crate_derivatives)


@pytest.mark.parametrize("activation", SMOOTH)
def test_python_and_numpy_agree_by_bits_on_math_exp(activation: Smooth, monkeypatch: pytest.MonkeyPatch):
    for module in (numpy_activations, numpy_erf):
        monkeypatch.setattr(module, "exp", exp_by_math)
    # math.exp raises past exp(709), where GELU-tanh's tanh saturates (|z| above about 20); SiLU's
    # sigmoid catches it
    values = [v for v in _inputs() if abs(v) < 20.0]
    python_values, python_derivatives = _python(activation, values)
    numpy_values, numpy_derivatives = _numpy(activation, values)
    assert _bits(numpy_values) == _bits(python_values)
    assert _bits(numpy_derivatives) == _bits(python_derivatives)


@pytest.mark.parametrize("activation", SMOOTH)
def test_the_derivative_is_the_values_slope(activation: Smooth):
    # central differences of each implementation's value against its derivative, away from 0's
    # cancellation; the exact functions are the crate suite's
    step = 1e-6
    values = [v for v in _inputs() if 1e-3 < abs(v) < 8.0][:400]
    for implementation in (_python, _numpy, _crate):
        _, derivatives = implementation(activation, values)
        above, _ = implementation(activation, [z + step for z in values])
        below, _ = implementation(activation, [z - step for z in values])
        for z, d, up, down in zip(values, derivatives, above, below, strict=True):
            assert (up - down) / (2 * step) == pytest.approx(d, abs=1e-7), (implementation.__name__, z)


def test_gelu_tanh_approximates_gelu():
    # GPT-2's form is within about 2e-4 of the exact GELU on [-6, 6] (Hendrycks & Gimpel 2016)
    z = np.linspace(-6.0, 6.0, 1201)
    gap = np.abs(numpy_activations.gelu_tanh(z) - numpy_activations.gelu(z)).max()
    assert 0.0 < gap < 5e-4


def test_numpy_keeps_the_shape():
    z = np.linspace(-3.0, 3.0, 24).reshape(2, 3, 4)
    for activation in SMOOTH:
        assert numpy_activations.activate(activation, z).shape == (2, 3, 4)
        assert numpy_activations.derivative(activation, z).shape == (2, 3, 4)
