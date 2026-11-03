"""
Cross-attention in numpy and the target's segment rule (the cross-attention workplan, stage 4a, D3,
D13): by hand, against finite differences, the memory's padding and other segments changing
nothing, a pad's query reading no key, and the separator opening its segment.
"""

from collections.abc import Callable

import numpy as np
import pytest

from indrajala_ml.data.parallel_data import Pair, pack_pairs
from indrajala_ml.data.text_data import Vocabulary, pad_id, separator_id
from indrajala_ml.model.layers.forward_context import ForwardContext
from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.layers.numpy.cross_attention_array_layer import CrossAttentionArrayLayer
from indrajala_ml.model.layers.numpy.segments import segment_context

N, T, S, D, D_M = 2, 5, 4, 6, 8


def layer(
    heads: int = 2, key_value_heads: int | None = None, block: int = 16, seed: int = 0
) -> CrossAttentionArrayLayer:
    cross = CrossAttentionArrayLayer(T, D, S, D_M, heads=heads, key_value_heads=key_value_heads, block=block)
    rng = np.random.default_rng(seed)
    cross.set_parameters([rng.uniform(-0.5, 0.5, p.shape) for p in cross.parameters()])
    return cross


def context(memory: FloatArray, queries: np.ndarray | None, keys: np.ndarray | None) -> ForwardContext:
    return ForwardContext(segments=queries, memory=memory.reshape(memory.shape[0], -1), memory_segments=keys)


def backward(
    cross: CrossAttentionArrayLayer, X: FloatArray, G: FloatArray
) -> tuple[FloatArray, FloatArray, list[FloatArray]]:
    # the gradients of sum(A * G): to X, to the memory, to each parameter
    cross.reset_gradient_accum()
    cross._backward(G)  # pyright: ignore[reportPrivateUsage]
    cross.accumulate_gradient_batch(X)
    return cross.downstream_batch(), cross.memory_downstream_batch(), [g.copy() for g in cross.gradients()]


QUERIES = np.array([[0, 1, 1, 2, 2], [1, 1, 1, 1, 0]])  # a left pad, pairs 1 and 2; a right pad
KEYS = np.array([[0, 1, 2, 2], [1, 1, 0, 0]])


def test_by_hand_one_memory_token_and_every_weight_one():
    cross = CrossAttentionArrayLayer(T, D, 1, D_M, heads=2)
    cross.set_parameters([np.ones(p.shape) if p.ndim == 2 else np.zeros(p.shape) for p in cross.parameters()])
    rng = np.random.default_rng(1)
    X, memory = rng.normal(size=(N, T * D)), rng.normal(size=(N, D_M))
    A = cross.forward_batch(X, context=context(memory, None, None))
    # one key: P = 1, so H = V = the memory's sum in every feature, and out = h d_k times it
    expected = np.repeat((cross.width * memory.sum(axis=1))[:, np.newaxis], T * D, axis=1)
    np.testing.assert_allclose(A, expected, rtol=1e-12)


def test_the_memory_s_padding_changes_nothing_by_bits():
    rng = np.random.default_rng(2)
    X, memory, G = rng.normal(size=(N, T * D)), rng.normal(size=(N, S, D_M)), rng.normal(size=(N, T * D))
    padded = CrossAttentionArrayLayer(T, D, S + 3, D_M, heads=2)
    plain = layer()
    padded.set_parameters([p.copy() for p in plain.parameters()])
    keys = np.ones((N, S), dtype=np.intp)
    extra = np.concatenate((memory, rng.normal(size=(N, 3, D_M))), axis=1)
    A = plain.forward_batch(X, context=context(memory, np.ones((N, T), dtype=np.intp), keys))
    A_padded = padded.forward_batch(
        X, context=context(extra, np.ones((N, T), dtype=np.intp), np.concatenate((keys, np.zeros((N, 3), np.intp)), 1))
    )
    np.testing.assert_array_equal(A, A_padded)
    dX, dM, grads = backward(plain, X, G)
    dX_padded, dM_padded, grads_padded = backward(padded, X, G)
    np.testing.assert_array_equal(dX, dX_padded)
    np.testing.assert_array_equal(dM.reshape(N, S, D_M), dM_padded.reshape(N, S + 3, D_M)[:, :S])
    assert not dM_padded.reshape(N, S + 3, D_M)[:, S:].any()  # a pad's memory gets no gradient
    for g, g_padded in zip(grads, grads_padded, strict=True):
        np.testing.assert_array_equal(g, g_padded)


def test_a_token_reads_only_its_own_segment_s_memory_and_a_pad_s_query_reads_none():
    cross = layer()
    rng = np.random.default_rng(3)
    X, memory = rng.normal(size=(N, T * D)), rng.normal(size=(N, S, D_M))
    A = cross.forward_batch(X, context=context(memory, QUERIES, KEYS)).reshape(N, T, D)
    changed = memory.copy()
    changed[0, 2:] += 1.0  # example 0's segment 2
    B = cross.forward_batch(X, context=context(changed, QUERIES, KEYS)).reshape(N, T, D)
    np.testing.assert_array_equal(A[0, :3], B[0, :3])  # its pad and segment 1 don't see it
    assert not np.allclose(A[0, 3:], B[0, 3:])
    # a pad's query attends to nothing: H = 0, so its output is the bias bo
    np.testing.assert_array_equal(A[0, 0], cross.bo)
    np.testing.assert_array_equal(A[1, 4], cross.bo)
    P = cross.weights()
    assert not P[0, :, 0].any() and not P[1, :, 4].any()
    np.testing.assert_allclose(P[0, :, 1:].sum(axis=-1), 1.0, rtol=1e-12)
    assert not P[0, :, 1:3, 2:].any() and not P[0, :, 3:, :2].any()  # each segment's own keys only


def numeric(f: Callable[[], float], value: FloatArray, index: tuple[int, ...], eps: float = 1e-6) -> float:
    saved = value[index]
    value[index] = saved + eps
    up = f()
    value[index] = saved - eps
    down = f()
    value[index] = saved
    return (up - down) / (2 * eps)


@pytest.mark.parametrize(("heads", "key_value_heads", "block"), [(2, None, 16), (2, 1, 2), (3, None, 3)])
def test_the_gradients_are_finite_differences(heads: int, key_value_heads: int | None, block: int):
    cross = CrossAttentionArrayLayer(T, 6, S, D_M, heads=heads, key_value_heads=key_value_heads, block=block)
    rng = np.random.default_rng(4)
    cross.set_parameters([rng.uniform(-0.5, 0.5, p.shape) for p in cross.parameters()])
    X, memory, G = rng.normal(size=(N, T * 6)), rng.normal(size=(N, S, D_M)), rng.normal(size=(N, T * 6))

    def loss() -> float:
        return float((cross.forward_batch(X, context=context(memory, QUERIES, KEYS)) * G).sum())

    loss()
    dX, dM, grads = backward(cross, X, G)
    dM = dM.reshape(N, S, D_M)
    checks: list[tuple[float, float]] = []
    for index in [(0, 7), (0, 25), (1, 3), (1, 29)]:
        checks.append((float(dX[index]), numeric(loss, X, index)))
    for index in [(0, 1, 2), (0, 3, 5), (1, 0, 0), (1, 1, 7)]:
        checks.append((float(dM[index]), numeric(loss, memory, index)))
    for parameter, gradient in zip(cross.parameters(), grads, strict=True):
        for flat in rng.choice(parameter.size, size=3, replace=False):
            index = tuple(int(i) for i in np.unravel_index(int(flat), parameter.shape))
            checks.append((float(gradient[index]), numeric(loss, parameter, index)))
    for analytic, finite in checks:
        assert analytic == pytest.approx(finite, rel=1e-6, abs=1e-8)
    # the pads' memory gets nothing
    assert not dM[0, 0].any() and not dM[1, 2:].any()


def test_without_segments_every_token_reads_every_key():
    cross = layer()
    rng = np.random.default_rng(5)
    X, memory = rng.normal(size=(N, T * D)), rng.normal(size=(N, S, D_M))
    A = cross.forward_batch(X, context=context(memory, None, None))
    B = cross.forward_batch(X, context=context(memory, np.ones((N, T), np.intp), np.ones((N, S), np.intp)))
    np.testing.assert_array_equal(A, B)


def test_a_separator_opens_its_segment_on_the_target():
    s, p = 3, 4  # a vocabulary of 3 symbols, then the separator and the pad
    ids = np.array([[p, p, s, 0, 1, s, 2, s], [s, 0, s, 1, 1, s, 2, 2]], dtype=np.float64)
    opened = segment_context(ids, s, p, separator_opens=True)
    np.testing.assert_array_equal(opened.segments, [[0, 0, 1, 1, 1, 2, 2, 3], [1, 1, 2, 2, 2, 3, 3, 3]])
    np.testing.assert_array_equal(opened.positions, [[0, 1, 0, 1, 2, 0, 1, 0], [0, 1, 0, 1, 2, 0, 1, 2]])
    np.testing.assert_array_equal(opened.counted, [[0, 0, 1, 1, 1, 1, 1, 1], [1] * 8])
    ended = segment_context(ids, s, p)
    np.testing.assert_array_equal(ended.segments, [[0, 0, 1, 2, 2, 2, 3, 3], [1, 2, 2, 3, 3, 3, 4, 4]])
    # text before a row's first separator joins its first segment
    np.testing.assert_array_equal(
        segment_context(np.array([[0, 1, s, 2]], np.float64), s, p, separator_opens=True).segments, [[1, 1, 1, 1]]
    )


def test_a_packed_window_s_pairs_are_the_same_segments_on_both_sides():
    source, target = Vocabulary.of("αβγ"), Vocabulary.of("abc")
    pairs = [
        Pair("w", str(k), "train", grc, eng) for k, (grc, eng) in enumerate([("αβ", "ab"), ("γ", "cca"), ("β", "b")])
    ]
    [window] = pack_pairs(pairs, source, target, window=12)
    enc = segment_context(np.array([window.source], np.float64), separator_id(source), pad_id(source))
    dec = segment_context(
        np.array([window.decoder_input], np.float64), separator_id(target), pad_id(target), separator_opens=True
    )
    for k, pair in enumerate(pairs, start=1):
        read = [
            source.symbols[i] for i, g in zip(window.source, enc.segments[0], strict=True) if g == k and i < len(source)
        ]
        written = [
            target.symbols[i] for i, g in zip(window.labels, dec.segments[0], strict=True) if g == k and i < len(target)
        ]
        assert "".join(read) == pair.grc and "".join(written) == pair.eng
