"""
Cross-attention in pure Python (the cross-attention workplan, stage 7a): CrossAttentionLayer
against numpy's CrossAttentionArrayLayer with the same parameters, numpy's exp math.exp. The
forward pass, the gradient to the tokens, the memory's gradient and the parameters' gradients
agree within numpy's BLAS rounding. Where no product reaches a value they agree by bits: a pad's
query, which reads no key, has the output bo. Dropout draws numpy's masks from generators in the
same state, N h T S of them. And D13's segment rule, a separator opening its segment, is numpy's.
"""

from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.forward_context import ForwardContext
from indrajala_ml.model.layers.numpy import attention_array_layer
from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.layers.numpy.cross_attention_array_layer import CrossAttentionArrayLayer
from indrajala_ml.model.layers.numpy.segments import segment_context
from indrajala_ml.model.layers.python.cross_attention_layer import CrossAttentionLayer
from indrajala_ml.model.layers.python.segments import segment_rows
from indrajala_ml.model.layers.python.state_layer import StateLayer
from indrajala_ml.pcg64 import default_rng
from tests.helpers import bits, exp_by_math, patching

# numpy's tiled passes' exp as math.exp, the pure-Python layer's
attention_exp = patching(attention_array_layer, "exp", exp_by_math)

N, T, S, D, D_M = 2, 21, 37, 8, 12
QUERIES = np.array([[0] * 3 + [1] * 8 + [2] * 6 + [3] * 4, [1] * 15 + [2] * 6])
KEYS = np.array([[1] * 12 + [2] * 15 + [3] * 10, [0] * 5 + [1] * 14 + [2] * 18])
RTOL, ATOL = 1e-12, 1e-14


class _Next:
    """A next layer giving one example's delta."""

    def __init__(self, delta: list[float]) -> None:
        self.delta = delta

    def downstream_sum(self, own_index: int) -> float:
        return self.delta[own_index]


def layers(
    heads: int = 4, key_value_heads: int | None = 2, dropout: float = 0.0, seed: int = 0
) -> tuple[CrossAttentionArrayLayer, CrossAttentionLayer, StateLayer]:
    key_size = 4
    numpy = CrossAttentionArrayLayer(T, D, S, D_M, heads, key_size, key_value_heads, dropout)
    inputs = StateLayer(T * D, [(0.0, 1.0)] * (T * D))
    python = CrossAttentionLayer(inputs, T, D, S, D_M, heads, key_size, key_value_heads, dropout)
    rng = np.random.default_rng(seed)
    parameters = [rng.uniform(-0.5, 0.5, p.shape) for p in numpy.parameters()]
    numpy.set_parameters(parameters)
    rows: list[tuple[list[float], float]] = []
    for w, b in zip(parameters[0::2], parameters[1::2], strict=True):
        rows += [(list(map(float, w_row)), float(b_i)) for w_row, b_i in zip(w, b, strict=True)]
    python.restore_state(rows)
    return numpy, python, inputs


def inputs(seed: int = 1) -> tuple[FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(N, T * D)), rng.normal(size=(N, S * D_M)), rng.normal(size=(N, T * D))


def numpy_passes(
    layer: CrossAttentionArrayLayer, X: FloatArray, context: ForwardContext, G: FloatArray
) -> dict[str, Any]:
    A = layer.forward_batch(X, context=context)
    layer.reset_gradient_accum()
    layer._backward(G)  # pyright: ignore[reportPrivateUsage]
    layer.accumulate_gradient_batch(X, context=context)
    W = [g.copy() for g in layer.gradients()]
    return {"A": A, "dX": layer.downstream_batch(), "dM": layer.memory_downstream_batch().reshape(N, -1), "W": W}


def python_passes(
    layer: CrossAttentionLayer, state: StateLayer, X: FloatArray, context: ForwardContext, G: FloatArray
) -> dict[str, Any]:
    # example by example, the gradients accumulating over the batch, as a network's loop
    A: list[list[float]] = []
    dX: list[list[float]] = []
    dM: list[list[float]] = []
    for n in range(N):
        example = context.example(n)
        state.update_state(tuple(float(x) for x in X[n]))
        layer.forward(context=example)
        A.append([node.value() for node in layer.nodes])
        layer.compute_hidden_deltas(_Next([float(g) for g in G[n]]), context=example)  # pyright: ignore[reportArgumentType]
        layer.accumulate_gradients(context=example)
        dX.append([layer.downstream_sum(i) for i in range(T * D)])
        dM.append(layer.memory_downstream())
    rows = layer.weight_sets()
    W: list[FloatArray] = []
    for block in (layer.queries, layer.keys, layer.values, layer.outputs):
        W.append(np.array([row.weight_gradient_accum for row in block]))
        W.append(np.array([row.bias_gradient_accum for row in block]))
    assert len(rows) == sum(len(b) for b in (layer.queries, layer.keys, layer.values, layer.outputs))
    return {"A": np.array(A), "dX": np.array(dX), "dM": np.array(dM), "W": W}


def contexts(memory: FloatArray, segmented: bool, keys: bool = True) -> tuple[ForwardContext, ForwardContext]:
    if not segmented:
        return ForwardContext(memory=memory), ForwardContext(memory=memory.tolist())
    numpy = ForwardContext(segments=QUERIES, memory=memory, memory_segments=KEYS if keys else None)
    python = ForwardContext(
        segments=QUERIES.tolist(),
        positions=QUERIES.tolist(),
        counted=QUERIES.tolist(),
        memory=memory.tolist(),
        memory_segments=KEYS.tolist() if keys else None,
    )
    return numpy, python


def both(
    numpy: Any, python: Any, state: StateLayer, segmented: bool, keys: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    X, memory, G = inputs()
    numpy_context, python_context = contexts(memory, segmented, keys)
    return numpy_passes(numpy, X, numpy_context, G), python_passes(python, state, X, python_context, G)


def assert_close(expected: dict[str, Any], got: dict[str, Any]) -> None:
    for name in ("A", "dX", "dM"):
        assert got[name].shape == expected[name].shape, name
        np.testing.assert_allclose(got[name], expected[name], rtol=RTOL, atol=ATOL, err_msg=name)
    for i, (e, g) in enumerate(zip(expected["W"], got["W"], strict=True)):
        np.testing.assert_allclose(g, e, rtol=RTOL, atol=ATOL, err_msg=f"gradient {i}")


@pytest.mark.usefixtures("attention_exp")
@pytest.mark.parametrize("segmented", [False, True], ids=["unsegmented", "segmented"])
@pytest.mark.parametrize("heads,key_value_heads", [(1, None), (4, None), (4, 2), (4, 1)])
def test_the_passes_are_numpys_within_blas_rounding(heads: int, key_value_heads: int | None, segmented: bool):
    numpy, python, state = layers(heads, key_value_heads)
    assert_close(*both(numpy, python, state, segmented))


@pytest.mark.usefixtures("attention_exp")
def test_a_side_without_segments_is_in_segment_1_as_numpy():
    numpy, python, state = layers()
    assert_close(*both(numpy, python, state, True, keys=False))


@pytest.mark.usefixtures("attention_exp")
def test_a_pad_query_reads_no_key_so_its_output_is_bo_by_bits():
    numpy, python, state = layers()
    expected, got = both(numpy, python, state, True)
    A = got["A"].reshape(N * T, D)
    pads = np.flatnonzero(QUERIES.reshape(-1) == 0)
    assert all(bits(A[r]) == bits(numpy.bo) for r in pads)
    assert bits(A[pads]) == bits(expected["A"].reshape(N * T, D)[pads])


@pytest.mark.usefixtures("attention_exp")
def test_dropout_draws_numpys_masks_in_a_batch_and_alone():
    numpy, python, state = layers(dropout=0.3)
    numpy.set_rng(np.random.default_rng(7))
    python.set_rng(default_rng(7))
    numpy.set_training_mode(True)
    python.set_training_mode(True)
    python.begin_batch(N)
    examples: list[int] = list(range(N))
    X, memory, G = inputs()
    numpy_context, python_context = contexts(memory, True)
    expected = numpy_passes(numpy, X, numpy_context, G)
    A: list[list[float]] = []
    for n in examples:
        python.begin_example(n)
        state.update_state(tuple(float(x) for x in X[n]))
        python.forward(context=python_context.example(n))
        A.append([node.value() for node in python.nodes])
    python.end_batch()
    np.testing.assert_allclose(np.array(A), expected["A"], rtol=RTOL, atol=ATOL)
    after = np.random.default_rng(7)
    after.random(N * numpy.heads * T * S)
    assert python.rng.state == after.bit_generator.state


@pytest.mark.parametrize("opens", [False, True])
def test_the_segment_rules_are_numpys(opens: bool):
    # pads, then a separator opening each pair (D13) or ending it (step 11)
    sep, pad = 5, 6
    rows = [[pad, pad, sep, 1, 2, sep, 3, sep, 4, 1], [sep, 1, sep, 2, 2, 3, sep, 4, 4, sep]]
    numpy = segment_context(np.array(rows, dtype=float), sep, pad, opens)
    python = segment_rows([[float(i) for i in row] for row in rows], sep, pad, opens)
    for name in ("segments", "positions", "counted"):
        assert np.asarray(getattr(numpy, name)).tolist() == getattr(python, name), name
    assert (numpy.count, numpy.examples) == (python.count, python.examples)
