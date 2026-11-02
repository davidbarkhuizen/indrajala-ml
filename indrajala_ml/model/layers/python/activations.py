"""
The transcendental activations in pure Python, through this module's exp (the FFN activations
workplan, D2, D9): the cell's tanh, 1 - 2 / (exp(2z) + 1) (the window-recurrent adder workplan,
W15), and GELU, exact through erf (erf.py) and GPT-2's tanh form, with their derivatives. The
expressions and their grouping are the crate's (rust/core/src/activation.rs) and numpy's
(numpy/activations.py), so the three agree by bits wherever their exp does (README, Activations):

    gelu(z)       = (0.5 * z) * (1.0 + erf(z * SQRT1_2))
    gelu'(z)      = 0.5 * (1.0 + erf(z * SQRT1_2)) + z * (exp(-0.5 * (z * z)) * INV_SQRT_2PI)
    gelu_tanh(z)  = (0.5 * z) * (1.0 + th),  th = tanh(C * (z + K * ((z * z) * z)))
    gelu_tanh'(z) = 0.5 * (1.0 + th) + (0.5 * z) * ((1.0 - th * th) * (C * (1.0 + (3.0 * K) * (z * z))))

exp is math.exp, which tests may replace (erf goes through erf.py's own).
"""

from __future__ import annotations

import math
from typing import Literal

from indrajala_ml.model.layers.python import erf

exp = math.exp

# a hidden layer's activation whose derivative reads its pre-activation z
Gelu = Literal["gelu", "gelu_tanh"]
GELUS: tuple[Gelu, ...] = ("gelu", "gelu_tanh")
# a hidden dense layer's fused activation, the crate's family (D9)
HiddenActivation = Literal["relu", "gelu", "gelu_tanh"]
# a token-wise dense layer's (the layer-norm and attention workplan, D4; D5 here)
TokenActivation = Literal["relu", "gelu", "gelu_tanh", "linear"]

SQRT1_2 = 0.7071067811865476  # the double nearest 1/sqrt(2)
INV_SQRT_2PI = 0.3989422804014327  # the double nearest 1/sqrt(2 pi)
C = 0.7978845608028654  # the double nearest sqrt(2/pi)
K = 0.044715  # GPT-2's cubic coefficient


def tanh(z: float) -> float:
    """1 - 2 / (exp(2z) + 1) (W15), numpy's form: exactly 1.0 where exp(2z) overflows."""
    try:
        return 1.0 - 2.0 / (exp(2.0 * z) + 1.0)
    except OverflowError:
        return 1.0


def gelu(z: float) -> float:
    return (0.5 * z) * (1.0 + erf.erf(z * SQRT1_2))


def gelu_derivative(z: float) -> float:
    return 0.5 * (1.0 + erf.erf(z * SQRT1_2)) + z * (exp(-0.5 * (z * z)) * INV_SQRT_2PI)


def _gelu_tanh_th(z: float) -> float:
    return tanh(C * (z + K * ((z * z) * z)))


def gelu_tanh(z: float) -> float:
    return (0.5 * z) * (1.0 + _gelu_tanh_th(z))


def gelu_tanh_derivative(z: float) -> float:
    th = _gelu_tanh_th(z)
    return 0.5 * (1.0 + th) + (0.5 * z) * ((1.0 - th * th) * (C * (1.0 + (3.0 * K) * (z * z))))


def activate(activation: Gelu, z: float) -> float:
    return gelu(z) if activation == "gelu" else gelu_tanh(z)


def derivative(activation: Gelu, z: float) -> float:
    return gelu_derivative(z) if activation == "gelu" else gelu_tanh_derivative(z)
