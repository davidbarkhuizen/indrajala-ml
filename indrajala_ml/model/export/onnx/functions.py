"""
Elementwise functions as ONNX ops (the export-and-hub workplan, stage 3): the activations our
layers apply (model/layers/numpy/activations.py) in the graph's dtype.

- sigmoid, ReLU and softmax are ONNX's own ops, which onnxruntime's CPU provider
  runs in float64 and float32.
- **erf, and so exact GELU, is decomposed** (D10): onnxruntime has no float64 Erf or exact Gelu
  kernel. The decomposition is fdlibm's s_erf.c, whose numpy port is
  model/layers/numpy/erf.py: each range's rational approximation, evaluated on every element and
  chosen by |x|. It differs from the port only where the port truncates z to its high word in the
  tails, a trick for the last bits that the graph leaves out (the check's tolerance is 1e-12).
  One graph serves both dtypes.
- GPT-2's GELU is written out with Tanh, as onnx's reference Gelu(approximate="tanh") is 5e-10
  from ours in float64.
"""

from __future__ import annotations

import struct

from indrajala_ml.model.export.onnx.graph import Graph
from indrajala_ml.model.layers.python import erf as fdlibm
from indrajala_ml.model.layers.python.activations import SQRT1_2, C, K


def _high_word(word: int) -> float:
    # the double whose high 32 bits are word and low 32 bits zero: s_erf.c's range bounds
    return struct.unpack(">d", struct.pack(">Q", word << 32))[0]


SMALL = _high_word(fdlibm.SMALL)
MIDDLE = _high_word(fdlibm.MIDDLE)
NEAR = _high_word(fdlibm.NEAR)
SATURATED = _high_word(fdlibm.SATURATED)


def _horner(g: Graph, s: str, coefficients: tuple[float, ...], one_first: bool) -> str:
    # c0 + s (c1 + s (c2 + ...)), or 1 + s (c0 + s (c1 + ...)) with one_first, as erf.py groups them
    acc = g.constant(coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        acc = g.op("Add", [g.constant(coefficient), g.op("Mul", [s, acc])])
    if one_first:
        acc = g.op("Add", [g.constant(1.0), g.op("Mul", [s, acc])])
    return acc


def erf(g: Graph, x: str) -> str:
    """erf(x), elementwise: s_erf.c's ranges, each chosen where |x| falls."""
    # every range is evaluated on every element, each on its input clipped to its range, so no
    # unchosen range overflows
    v = g.op("Abs", [x])
    sign = g.op("Sign", [x])
    # |x| < 0.84375: x + x r/s, r and s in z = x^2 (TINY_X's x + EFX x agrees within an ulp there)
    near_zero = g.op("Clip", [x, g.constant(-SMALL), g.constant(SMALL)])
    z = g.op("Mul", [near_zero, near_zero])
    small = g.op(
        "Add",
        [
            near_zero,
            g.op(
                "Mul",
                [near_zero, g.op("Div", [_horner(g, z, fdlibm.PP, False), _horner(g, z, fdlibm.QQ, True)])],
            ),
        ],
    )
    # 0.84375 <= |x| < 1.25: sign (erx + p/q), p and q in s = |x| - 1
    s = g.op("Sub", [g.op("Clip", [v, g.constant(SMALL), g.constant(MIDDLE)]), g.constant(1.0)])
    middle = g.op(
        "Mul",
        [
            sign,
            g.op(
                "Add",
                [
                    g.constant(fdlibm.ERX),
                    g.op("Div", [_horner(g, s, fdlibm.PA, False), _horner(g, s, fdlibm.QA, True)]),
                ],
            ),
        ],
    )
    # 1.25 <= |x| < 6: sign (1 - erfc(|x|)), erfc = exp(-x^2 - 0.5625 + R/S) / |x|, R and S in 1/x^2,
    # one table below 1/0.35 and another above
    w = g.op("Clip", [v, g.constant(MIDDLE), g.constant(SATURATED)])
    inverse = g.op("Div", [g.constant(1.0), g.op("Mul", [w, w])])
    below = g.op("Less", [w, g.constant(NEAR)])
    r = g.op("Where", [below, _horner(g, inverse, fdlibm.RA, False), _horner(g, inverse, fdlibm.RB, False)])
    big_s = g.op("Where", [below, _horner(g, inverse, fdlibm.SA, True), _horner(g, inverse, fdlibm.SB, True)])
    exponent = g.op(
        "Add",
        [g.op("Sub", [g.op("Neg", [g.op("Mul", [w, w])]), g.constant(0.5625)]), g.op("Div", [r, big_s])],
    )
    erfc = g.op("Div", [g.op("Exp", [exponent]), w])
    tail = g.op("Mul", [sign, g.op("Sub", [g.constant(1.0), erfc])])
    # |x| >= 6: sign (1 - tiny), which rounds to sign
    out = g.op("Where", [g.op("Less", [v, g.constant(SATURATED)]), tail, sign])
    out = g.op("Where", [g.op("Less", [v, g.constant(MIDDLE)]), middle, out])
    return g.op("Where", [g.op("Less", [v, g.constant(SMALL)]), small, out], base="erf")


def gelu(g: Graph, z: str) -> str:
    """Exact GELU, 0.5 z (1 + erf(z / sqrt 2)), as activations.gelu."""
    half = g.op("Mul", [g.constant(0.5), z])
    return g.op("Mul", [half, g.op("Add", [g.constant(1.0), erf(g, g.op("Mul", [z, g.constant(SQRT1_2)]))])])


def gelu_tanh(g: Graph, z: str) -> str:
    """
    GPT-2's GELU, 0.5 z (1 + tanh(C (z + K z^3))), as activations.gelu_tanh, with Tanh: onnx's
    reference Gelu(approximate="tanh") is 5e-10 from it in float64.
    """
    cube = g.op("Mul", [g.op("Mul", [z, z]), z])
    inner = g.op("Mul", [g.constant(C), g.op("Add", [z, g.op("Mul", [g.constant(K), cube])])])
    half = g.op("Mul", [g.constant(0.5), z])
    return g.op("Mul", [half, g.op("Add", [g.constant(1.0), g.op("Tanh", [inner])])])


def activate(g: Graph, activation: str, z: str) -> str:
    """A hidden or output layer's activation of z."""
    match activation:
        case "sigmoid":
            return g.op("Sigmoid", [z])
        case "relu":
            return g.op("Relu", [z])
        case "gelu":
            return gelu(g, z)
        case "gelu_tanh":
            return gelu_tanh(g, z)
        case "silu":
            return g.op("Mul", [z, g.op("Sigmoid", [z])])
        case "softmax":
            return g.op("Softmax", [z], axis=-1)
        case "linear":
            return z
        case _:
            raise ValueError(f"no ONNX form for the activation {activation!r}")
