"""
erf elementwise (the FFN activations workplan, D2): the pure-Python port of fdlibm's s_erf.c
(indrajala_ml/model/layers/python/erf.py, which holds the constants and the original's notice), the
same operations in the same grouping on each range's elements, chosen by the high word of |x|. numpy's
+ - * / are correctly rounded and never contracted, so the bits are the scalar port's wherever exp's
are. exp is this module's, np.exp, which tests replace with another implementation's (it isn't
correctly rounded).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.layers.python import erf as scalar
from indrajala_ml.model.layers.python.erf import EFX, EFX8, ERX, PA, PP, QA, QQ, RA, RB, SA, SB, TINY

exp = np.exp


def erf(values: FloatArray) -> FloatArray:
    """erf of each value, any shape: scalar.erf's bits, value by value, given the same exp."""
    x = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    out = np.empty_like(x)
    bits = x.view(np.uint64)
    hx = (bits >> np.uint64(32)).astype(np.uint32).view(np.int32)
    ix = hx & np.int32(0x7FFF_FFFF)
    negative = hx < 0

    special = ix >= scalar.SPECIAL
    with np.errstate(divide="ignore", invalid="ignore"):
        out[special] = np.where(negative[special], -1.0, 1.0) + 1.0 / x[special]

    subnormal = ix < scalar.SUBNORMAL
    v = x[subnormal]
    out[subnormal] = 0.125 * (8.0 * v + EFX8 * v)

    tiny = (ix >= scalar.SUBNORMAL) & (ix < scalar.TINY_X)
    v = x[tiny]
    out[tiny] = v + EFX * v

    small = (ix >= scalar.TINY_X) & (ix < scalar.SMALL)
    v = x[small]
    z = v * v
    r = PP[0] + z * (PP[1] + z * (PP[2] + z * (PP[3] + z * PP[4])))
    s = 1.0 + z * (QQ[0] + z * (QQ[1] + z * (QQ[2] + z * (QQ[3] + z * QQ[4]))))
    out[small] = v + v * (r / s)

    middle = (ix >= scalar.SMALL) & (ix < scalar.MIDDLE)
    v = x[middle]
    s = np.abs(v) - 1.0
    p = PA[0] + s * (PA[1] + s * (PA[2] + s * (PA[3] + s * (PA[4] + s * (PA[5] + s * PA[6])))))
    q = 1.0 + s * (QA[0] + s * (QA[1] + s * (QA[2] + s * (QA[3] + s * (QA[4] + s * QA[5])))))
    out[middle] = np.where(negative[middle], -ERX - p / q, ERX + p / q)

    saturated = (ix >= scalar.SATURATED) & ~special
    out[saturated] = np.where(negative[saturated], TINY - 1.0, 1.0 - TINY)

    near = (ix >= scalar.MIDDLE) & (ix < scalar.NEAR)
    v = np.abs(x[near])
    s = 1.0 / (v * v)
    r = RA[0] + s * (RA[1] + s * (RA[2] + s * (RA[3] + s * (RA[4] + s * (RA[5] + s * (RA[6] + s * RA[7]))))))
    big_s = 1.0 + s * (
        SA[0] + s * (SA[1] + s * (SA[2] + s * (SA[3] + s * (SA[4] + s * (SA[5] + s * (SA[6] + s * SA[7]))))))
    )
    out[near] = _tail(v, r, big_s, negative[near])

    far = (ix >= scalar.NEAR) & (ix < scalar.SATURATED)
    v = np.abs(x[far])
    s = 1.0 / (v * v)
    r = RB[0] + s * (RB[1] + s * (RB[2] + s * (RB[3] + s * (RB[4] + s * (RB[5] + s * RB[6])))))
    big_s = 1.0 + s * (SB[0] + s * (SB[1] + s * (SB[2] + s * (SB[3] + s * (SB[4] + s * (SB[5] + s * SB[6]))))))
    out[far] = _tail(v, r, big_s, negative[far])

    return out.reshape(np.shape(values))


def _tail(v: FloatArray, r: FloatArray, big_s: FloatArray, negative: npt.NDArray[np.bool_]) -> FloatArray:
    """1 - erfc(|x|) with the sign, erfc(|x|) = exp(-z^2 - 0.5625) exp((z - x)(z + x) + R/S) / |x|."""
    z = (v.view(np.uint64) & np.uint64(0xFFFF_FFFF_0000_0000)).view(np.float64)
    e = exp(-z * z - 0.5625) * exp((z - v) * (z + v) + r / big_s)
    return np.where(negative, e / v - 1.0, 1.0 - e / v)
