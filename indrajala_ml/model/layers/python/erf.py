"""
The error function, erf(x) = 2 / sqrt(pi) * the integral of exp(-t^2) from 0 to x (the FFN
activations workplan, D2): GELU's x * (1 + erf(x / sqrt 2)) / 2.

A port of fdlibm's s_erf.c (Sun, 1993; netlib's fdlibm 5.3, s_erf.c 1.3), its erf half, step for
step, as the crate's core/src/erf.rs and numpy's erf.py: four ranges of |x|, each a rational
approximation, the last two through exp twice. Apart from exp its operations are + - * /, abs and
one exact truncation (the low 32 bits of |x| cleared), each correctly rounded, so the three agree by
bits wherever their exp does. exp is this module's, math.exp, which tests may replace with another
implementation's (the crate's is correctly rounded; math.exp and np.exp need not be). Within one
ulp of the exact erf; not correctly rounded. The constants are the C source's, from the bits its
comments give, shared with numpy's port. Numpy-free.

The original's notice:

    Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.

    Developed at SunSoft, a Sun Microsystems, Inc. business.
    Permission to use, copy, modify, and distribute this
    software is freely granted, provided that this notice
    is preserved.
"""

from __future__ import annotations

import math
import struct


def _double(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def high_word(x: float) -> int:
    """fdlibm's __HI(x): the top 32 bits of x, signed as a C int."""
    hi = struct.unpack("<Q", struct.pack("<d", x))[0] >> 32
    return hi - (1 << 32) if hi >= 1 << 31 else hi


def low_word_cleared(x: float) -> float:
    """x with its low 32 bits zeroed (fdlibm's __LO(z) = 0): exact."""
    return _double(struct.unpack("<Q", struct.pack("<d", x))[0] & 0xFFFF_FFFF_0000_0000)


exp = math.exp

TINY = 1e-300
ERX = _double(0x3FEB_0AC1_6000_0000)  # c = (float)0.84506291151, erf(1) in single precision

# erf on [0, 0.84375]: x + x * P(x^2) / Q(x^2)
EFX = _double(0x3FC0_6EBA_8214_DB69)
EFX8 = _double(0x3FF0_6EBA_8214_DB69)
PP = tuple(
    _double(b)
    for b in (
        0x3FC0_6EBA_8214_DB68,
        0xBFD4_CD7D_691C_B913,
        0xBF9D_2A51_DBD7_194F,
        0xBF77_A291_2366_68E4,
        0xBEF8_EAD6_1200_16AC,
    )
)
QQ = tuple(
    _double(b)
    for b in (
        0x3FD9_7779_CDDA_DC09,
        0x3FB0_A54C_5536_CEBA,
        0x3F74_D022_C4D3_6B0F,
        0x3F21_5DC9_221C_1A10,
        0xBED0_9C43_42A2_6120,
    )
)

# erf on [0.84375, 1.25]: c + P1(s) / Q1(s), s = |x| - 1
PA = tuple(
    _double(b)
    for b in (
        0xBF63_59B8_BEF7_7538, 0x3FDA_8D00_AD92_B34D, 0xBFD7_D240_FBB8_C3F1, 0x3FD4_5FCA_8051_20E4,
        0xBFBC_6398_3D3E_28EC, 0x3FA2_2A36_5997_95EB, 0xBF61_BF38_0A96_073F,
    )
)  # fmt: skip
QA = tuple(
    _double(b)
    for b in (
        0x3FBB_3E66_18EE_E323, 0x3FE1_4AF0_92EB_6F33, 0x3FB2_635C_D99F_E9A7, 0x3FC0_2660_E763_351F,
        0x3F8B_EDC2_6B51_DD1C, 0x3F88_8B54_5735_151D,
    )
)  # fmt: skip

# erfc on [1.25, 1/0.35]: R1(s) / S1(s), s = 1 / x^2
RA = tuple(
    _double(b)
    for b in (
        0xBF84_3412_600D_6435, 0xBFE6_3416_E4BA_7360, 0xC025_1E04_41B0_E726, 0xC04F_300A_E4CB_A38D,
        0xC064_4CB1_8428_2266, 0xC067_135C_EBCC_ABB2, 0xC054_5265_57E4_D2F2, 0xC023_A0EF_C69A_C25C,
    )
)  # fmt: skip
SA = tuple(
    _double(b)
    for b in (
        0x4033_A6B9_BD70_7687, 0x4061_350C_526A_E721, 0x407B_290D_D58A_1A71, 0x4084_2B19_21EC_2868,
        0x407A_D021_5770_0314, 0x405B_28A3_EE48_AE2C, 0x401A_47EF_8E48_4A93, 0xBFAE_EFF2_EE74_9A62,
    )
)  # fmt: skip

# erfc on [1/0.35, 6]: R2(s) / S2(s), s = 1 / x^2
RB = tuple(
    _double(b)
    for b in (
        0xBF84_3412_39E8_6F4A, 0xBFE9_93BA_70C2_85DE, 0xC031_C209_555F_995A, 0xC064_145D_43C5_ED98,
        0xC083_EC88_1375_F228, 0xC090_0461_6A2E_5992, 0xC07E_384E_9BDC_383F,
    )
)  # fmt: skip
SB = tuple(
    _double(b)
    for b in (
        0x403E_568B_261D_5190, 0x4074_5CAE_221B_9F0A, 0x4098_02EB_189D_5118, 0x40A8_FFB7_688C_246A,
        0x40A3_F219_CEDF_3BE6, 0x407D_A874_E79F_E763, 0xC036_70E2_4271_2D62,
    )
)  # fmt: skip

# the ranges' bounds on the high word of |x|
SPECIAL = 0x7FF0_0000  # inf or NaN
SMALL = 0x3FEB_0000  # 0.84375
TINY_X = 0x3E30_0000  # 2^-28
SUBNORMAL = 0x0080_0000
MIDDLE = 0x3FF4_0000  # 1.25
SATURATED = 0x4018_0000  # 6
NEAR = 0x4006_DB6E  # 1/0.35


def erf(x: float) -> float:
    """erf(x), fdlibm's: odd, erf(+-inf) = +-1, erf(NaN) NaN, within one ulp of the exact value."""
    hx = high_word(x)
    ix = hx & 0x7FFF_FFFF
    if ix >= SPECIAL:
        # erf(+-inf) = +-1, erf(NaN) = NaN
        return float(1 - (2 if hx < 0 else 0)) + 1.0 / x
    if ix < SMALL:
        if ix < TINY_X:
            if ix < SUBNORMAL:
                return 0.125 * (8.0 * x + EFX8 * x)
            return x + EFX * x
        z = x * x
        r = PP[0] + z * (PP[1] + z * (PP[2] + z * (PP[3] + z * PP[4])))
        s = 1.0 + z * (QQ[0] + z * (QQ[1] + z * (QQ[2] + z * (QQ[3] + z * QQ[4]))))
        return x + x * (r / s)
    if ix < MIDDLE:
        s = abs(x) - 1.0
        p = PA[0] + s * (PA[1] + s * (PA[2] + s * (PA[3] + s * (PA[4] + s * (PA[5] + s * PA[6])))))
        q = 1.0 + s * (QA[0] + s * (QA[1] + s * (QA[2] + s * (QA[3] + s * (QA[4] + s * QA[5])))))
        return ERX + p / q if hx >= 0 else -ERX - p / q
    if ix >= SATURATED:
        return 1.0 - TINY if hx >= 0 else TINY - 1.0
    x = abs(x)
    s = 1.0 / (x * x)
    if ix < NEAR:
        r = RA[0] + s * (RA[1] + s * (RA[2] + s * (RA[3] + s * (RA[4] + s * (RA[5] + s * (RA[6] + s * RA[7]))))))
        big_s = 1.0 + s * (
            SA[0] + s * (SA[1] + s * (SA[2] + s * (SA[3] + s * (SA[4] + s * (SA[5] + s * (SA[6] + s * SA[7]))))))
        )
    else:
        r = RB[0] + s * (RB[1] + s * (RB[2] + s * (RB[3] + s * (RB[4] + s * (RB[5] + s * RB[6])))))
        big_s = 1.0 + s * (SB[0] + s * (SB[1] + s * (SB[2] + s * (SB[3] + s * (SB[4] + s * (SB[5] + s * SB[6]))))))
    z = low_word_cleared(x)
    r = exp(-z * z - 0.5625) * exp((z - x) * (z + x) + r / big_s)
    return 1.0 - r / x if hx >= 0 else r / x - 1.0
