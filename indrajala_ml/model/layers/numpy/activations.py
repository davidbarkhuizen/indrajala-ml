"""
The transcendental activations over arrays, through this module's exp (the FFN activations
workplan, D2, D9): the cell's tanh, 1 - 2 / (exp(2z) + 1) (the window-recurrent adder workplan,
W15), GELU, exact through erf (erf.py) and GPT-2's tanh form, and SiLU, with their derivatives, in the
pure-Python port's expressions and grouping (python/activations.py, which writes them out) and the
crate's. numpy's + - * / are correctly rounded and never contracted, so the bits are the scalar
port's wherever exp's are. exp is np.exp, which tests replace with another implementation's (it
isn't correctly rounded); erf goes through erf.py's own.
"""

from __future__ import annotations

import numpy as np

from indrajala_ml.model.layers.numpy import erf
from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.layers.python.activations import INV_SQRT_2PI, SQRT1_2, C, K, Smooth

exp = np.exp


def tanh(z: FloatArray) -> FloatArray:
    """
    1 - 2 / (exp(2z) + 1) (the window-recurrent adder workplan, W15): through exp, as sigmoid, so
    the three implementations agree where their exp does. A large z overflows exp(2z) to inf and
    gives exactly 1; a large negative z gives exactly -1.
    """
    with np.errstate(over="ignore"):
        return 1.0 - 2.0 / (exp(2.0 * z) + 1.0)


def gelu(z: FloatArray) -> FloatArray:
    return (0.5 * z) * (1.0 + erf.erf(z * SQRT1_2))


def gelu_derivative(z: FloatArray) -> FloatArray:
    return 0.5 * (1.0 + erf.erf(z * SQRT1_2)) + z * (exp(-0.5 * (z * z)) * INV_SQRT_2PI)


def _gelu_tanh_th(z: FloatArray) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore"):
        return tanh(C * (z + K * ((z * z) * z)))


def gelu_tanh(z: FloatArray) -> FloatArray:
    return (0.5 * z) * (1.0 + _gelu_tanh_th(z))


def gelu_tanh_derivative(z: FloatArray) -> FloatArray:
    th = _gelu_tanh_th(z)
    with np.errstate(over="ignore", invalid="ignore"):
        return 0.5 * (1.0 + th) + (0.5 * z) * ((1.0 - th * th) * (C * (1.0 + (3.0 * K) * (z * z))))


def sigmoid(z: FloatArray) -> FloatArray:
    """1 / (1 + exp(-z)) through this module's exp; array_layer.sigmoid's form."""
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + exp(-z))


def silu(z: FloatArray) -> FloatArray:
    return z * sigmoid(z)


def silu_derivative(z: FloatArray) -> FloatArray:
    s = sigmoid(z)
    return s * (1.0 + z * (1.0 - s))


def activate(activation: Smooth, z: FloatArray) -> FloatArray:
    if activation == "silu":
        return silu(z)
    return gelu(z) if activation == "gelu" else gelu_tanh(z)


def derivative(activation: Smooth, z: FloatArray) -> FloatArray:
    if activation == "silu":
        return silu_derivative(z)
    return gelu_derivative(z) if activation == "gelu" else gelu_tanh_derivative(z)
