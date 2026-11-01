"""
Tiled attention in numpy (the context-length workplan, D3, D12): the keys in blocks, an online softmax
per query row, nothing (N, h, T, T) kept.

- Against the plain softmax and its backward within rounding, at block sizes that split the keys
  unevenly, causal or not, with segments (pads included) and with a band.
- By bits against the recurrences written out by hand per row, the scores the tiles' own: m' =
  max(m, the block's max), l = l exp(m - m') + the fold of e, H = H exp(m - m') + e V, then H / l;
  and the backward's D = the fold of dH H over d_k.
- Skipping a wholly masked block: within rounding of every block computed (numpy's products'
  order of summation may follow their shapes; the crate's and pure Python's folds are exact).
- Dropout draws every weight once, key block by key block, so the generator ends where one (N, h,
  T, T) draw leaves it, and the backward pass draws the same masks again.
- Memory is linear in T: at a fixed token budget the passes' peak doesn't grow with T.
"""

from __future__ import annotations

import math
import tracemalloc
from typing import Any

import numpy as np
import pytest

from indrajala_ml.model.layers.numpy import attention_array_layer
from indrajala_ml.model.layers.numpy.attention_array_layer import (
    AttentionArrayLayer,
    Masking,
    key_blocks,
    _scores,  # pyright: ignore[reportPrivateUsage]
    tiled_backward,
    tiled_forward,
)
from tests.helpers import bits, exp_by_math, patching

math_exp = patching(attention_array_layer, "exp", exp_by_math)

N, HEADS, T, D_K = 2, 3, 11, 4
SCALE = math.sqrt(D_K)
# two examples' segments: a row starting with pads, then three paragraphs; one paragraph, then two
SEGMENTS = np.array([[0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3], [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]], dtype=np.intp)


def _qkv(seed: int = 0) -> tuple[Any, Any, Any]:
    rng = np.random.default_rng(seed)
    return tuple(rng.uniform(-1.5, 1.5, (N, HEADS, T, D_K)) for _ in range(3))  # pyright: ignore[reportReturnType]


def _allowed(masking: Masking) -> Any:
    # (N or 1, 1, T, T): the mask the tiles build, written whole
    t = np.arange(T)[:, np.newaxis] + masking.offset
    u = np.arange(T)[np.newaxis, :]
    allowed = np.ones((1, T, T), dtype=bool)
    if masking.causal:
        allowed = allowed & (u <= t)
    if masking.band is not None:
        allowed = allowed & (u > t - masking.band)
    if masking.query_segments is not None:
        assert masking.key_segments is not None
        q, k = masking.query_segments[:, :, np.newaxis], masking.key_segments[:, np.newaxis, :]
        allowed = allowed & (((q == k) & (k != 0)) | (u == t))
    return allowed[:, np.newaxis]


def _reference(Q: Any, K: Any, V: Any, dH: Any, masking: Masking) -> tuple[Any, ...]:
    # the softmax whole and its backward (the README's expressions before tiling)
    S = np.where(_allowed(masking), (Q @ K.swapaxes(-1, -2)) / SCALE, -np.inf)
    e = np.exp(S - S.max(axis=-1, keepdims=True))
    P = e / e.sum(axis=-1, keepdims=True)
    H = P @ V
    dP = dH @ V.swapaxes(-1, -2)
    dS = P * (dP - (dP * P).sum(axis=-1, keepdims=True))
    return H, (dS @ K) / SCALE, (dS.swapaxes(-1, -2) @ Q) / SCALE, P.swapaxes(-1, -2) @ dH


MASKINGS = {
    "unmasked": Masking(causal=False),
    "causal": Masking(causal=True),
    "causal, segments": Masking(causal=True, query_segments=SEGMENTS, key_segments=SEGMENTS),
    "segments": Masking(causal=False, query_segments=SEGMENTS, key_segments=SEGMENTS),
    "causal, band of 4": Masking(causal=True, band=4),
}


@pytest.mark.parametrize("block", [1, 2, 3, 5, 11, 64])
@pytest.mark.parametrize("name", MASKINGS)
def test_tiles_are_the_softmax_and_its_backward_within_rounding(name: str, block: int):
    masking = MASKINGS[name]
    Q, K, V = _qkv()
    dH = np.random.default_rng(1).uniform(-1.0, 1.0, (N, HEADS, T, D_K))
    H, m, l = tiled_forward(Q, K, V, SCALE, masking, block, None, 1.0)
    grads = tiled_backward(dH, (Q, K, V, H, m, l), SCALE, masking, block, None, 1.0)
    expected = _reference(Q, K, V, dH, masking)
    for actual, wanted in zip((H, *grads), expected, strict=True):
        np.testing.assert_allclose(actual, wanted, rtol=1e-12, atol=1e-14)


@pytest.mark.usefixtures("math_exp")
@pytest.mark.parametrize("block", [2, 3, 4])
@pytest.mark.parametrize("name", ["causal", "causal, segments"])
def test_the_online_softmax_is_its_recurrences_by_bits(name: str, block: int):
    masking = MASKINGS[name]
    Q, K, V = _qkv(2)
    H, m, l = tiled_forward(Q, K, V, SCALE, masking, block, None, 1.0)
    allowed = np.broadcast_to(_allowed(masking), (N, HEADS, T, T))
    # the scores are the tiles' own products (S is a BLAS product, not a fold): each block's rows
    # as the tile computes them
    tiles: dict[int, tuple[int, Any]] = {}
    for k0 in range(0, T, block):
        tile = masking.tile(T, k0, min(T, k0 + block))
        if tile is not None:
            r0, r1, mask = tile
            tiles[k0] = (r0, _scores(Q[:, :, r0:r1], K[:, :, k0 : min(T, k0 + block)], SCALE, mask))
    for n in range(N):
        for i in range(HEADS):
            for t in range(T):
                row_m, row_l = -math.inf, 0.0
                row_h = np.zeros(D_K)
                for k0, (r0, S_tile) in tiles.items():
                    k1 = min(T, k0 + block)
                    keys = [u for u in range(k0, k1) if allowed[n, i, t, u]]
                    if not keys:
                        continue  # a block wholly masked for this row changes nothing
                    S = [float(S_tile[n, i, t - r0, u - k0]) for u in keys]
                    new_m = max(row_m, *S)
                    alpha = math.exp(row_m - new_m)
                    e = [math.exp(x - new_m) for x in S]
                    fold = 0.0
                    for x in e:
                        fold += x
                    row_l = row_l * alpha + fold
                    row_h = row_h * alpha + np.array(e) @ V[n, i, keys]
                    row_m = new_m
                assert (m[n, i, t, 0], l[n, i, t, 0]) == (row_m, row_l)
                np.testing.assert_allclose(H[n, i, t], row_h / row_l, rtol=1e-13, atol=1e-15)


def test_the_backwards_d_is_the_fold_of_dh_h_over_d_k():
    # dS = P (dP - D): with dP = 0 (V zero) dS = -P D, so dQ's rows carry D exactly
    Q, K, _V = _qkv(3)
    V = np.zeros_like(Q)
    dH = np.random.default_rng(4).uniform(-1.0, 1.0, (N, HEADS, T, D_K))
    H = np.random.default_rng(5).uniform(-1.0, 1.0, (N, HEADS, T, D_K))  # D reads H as given
    masking = Masking(causal=True)
    _, m, l = tiled_forward(Q, K, V, SCALE, masking, 64, None, 1.0)
    dQ, _dK, _dV = tiled_backward(dH, (Q, K, V, H, m, l), SCALE, masking, 64, None, 1.0)
    D = np.zeros((N, HEADS, T, 1))
    for j in range(D_K):
        D[..., 0] += dH[..., j] * H[..., j]
    # the first query attends only to itself: P = 1, so dQ_0 = -D_0 K_0 / scale
    assert bits(dQ[:, :, 0]) == bits((-D[:, :, 0] * K[:, :, 0]) / SCALE)


class _NoSkip(Masking):
    """A masking whose every block is computed, its whole rows, the masked scores -inf."""

    def tile(self, queries: int, k0: int, k1: int) -> tuple[int, int, Any] | None:
        rows = np.arange(queries)[:, np.newaxis] + self.offset
        keys = np.arange(k0, k1)[np.newaxis, :]
        allowed = np.ones((1, queries, k1 - k0), dtype=bool)
        if self.causal:
            allowed = allowed & (keys <= rows)
        if self.query_segments is not None:
            assert self.key_segments is not None
            q = self.query_segments[:, :, np.newaxis]
            k = self.key_segments[:, np.newaxis, k0:k1]
            allowed = allowed & (((q == k) & (k != 0)) | (keys == rows))
        return 0, queries, allowed[:, np.newaxis]


@pytest.mark.parametrize("block", [1, 2, 3, 4])
def test_skipping_wholly_masked_blocks_gives_the_same_bits_as_computing_them(block: int):
    Q, K, V = _qkv(6)
    dH = np.random.default_rng(7).uniform(-1.0, 1.0, (N, HEADS, T, D_K))
    skipping = Masking(causal=True, query_segments=SEGMENTS, key_segments=SEGMENTS)
    computing = _NoSkip(causal=True, query_segments=SEGMENTS, key_segments=SEGMENTS)
    outputs: list[list[Any]] = []
    for masking in (skipping, computing):
        H, m, l = tiled_forward(Q, K, V, SCALE, masking, block, None, 1.0)
        outputs.append([H, m, l, *tiled_backward(dH, (Q, K, V, H, m, l), SCALE, masking, block, None, 1.0)])
    # within rounding: numpy's products are BLAS's, whose order of summation may follow a product's
    # shape, and skipping changes the shapes (a block's rows; dK's and dV's sums over them). A
    # sequential fold, the crate's or pure Python's, adds a skipped block's zeros exactly, so there
    # skipping is exact by bits (their tests)
    for skipped, computed in zip(outputs[0], outputs[1], strict=True):
        np.testing.assert_allclose(skipped, computed, rtol=1e-13, atol=1e-15)


def test_a_block_no_query_reaches_is_skipped_and_a_causal_one_starts_at_its_first_key():
    masking = Masking(causal=True, query_segments=SEGMENTS, key_segments=SEGMENTS)
    # keys 2..3: example 0's paragraph 1 (rows 2..4) and example 1's paragraph 1 (rows 2..5)
    tile = masking.tile(T, 2, 4)
    assert tile is not None and tile[:2] == (2, 6)
    # without segments, a causal block's rows start at its first key and run to the end
    assert Masking(causal=True).tile(T, 4, 6)[:2] == (4, T)  # pyright: ignore[reportOptionalSubscript]
    # a pad's key, attended only by itself: keys 0..1 reach rows 0..1 (example 0's pads' diagonals)
    # and example 1's paragraph 1 (rows 0..5)
    assert masking.tile(T, 0, 2)[:2] == (0, 6)  # pyright: ignore[reportOptionalSubscript]


@pytest.mark.parametrize("block", [64, 4])
def test_dropout_draws_every_weight_once_and_the_backward_draws_the_same_masks(block: int):
    layer = AttentionArrayLayer(T, HEADS * D_K, HEADS, causal=True, dropout=0.25, block=block)
    rng = np.random.default_rng(8)
    layer.set_parameters([rng.uniform(-0.5, 0.5, p.shape) for p in layer.parameters()])
    layer.set_rng(np.random.default_rng(9))
    layer.set_training_mode(True)
    X = rng.uniform(-1.0, 1.0, (N, T * HEADS * D_K))
    layer.forward_batch(X)
    after = np.random.default_rng(9)
    after.random((N, HEADS, T, T))
    assert layer.rng.bit_generator.state == after.bit_generator.state
    # the backward with the masks drawn again: the gradient against finite differences of the same
    # masked forward (the masks are fixed by the saved state)
    M = layer.dropout_mask()
    assert M is not None
    delta = rng.uniform(-1.0, 1.0, X.shape)
    layer._backward(delta)  # pyright: ignore[reportPrivateUsage]
    dX = layer.downstream_batch()
    assert layer.rng.bit_generator.state == after.bit_generator.state  # the backward draws from a copy

    def loss(x: Any) -> float:
        layer.set_rng(np.random.default_rng(9))
        return float((layer.forward_batch(x) * delta).sum())

    eps = 1e-6
    for index in [(0, 0), (1, 5), (0, T * HEADS * D_K - 1)]:
        plus, minus = X.copy(), X.copy()
        plus[index] += eps
        minus[index] -= eps
        assert dX[index] == pytest.approx((loss(plus) - loss(minus)) / (2 * eps), rel=1e-6, abs=1e-8)


def _peak_mb(tokens: int) -> float:
    examples = 4096 // tokens
    layer = AttentionArrayLayer(tokens, 16, 2, causal=True, dropout=0.1)
    rng = np.random.default_rng(10)
    layer.set_parameters([rng.uniform(-0.5, 0.5, p.shape) for p in layer.parameters()])
    layer.set_rng(np.random.default_rng(11))
    layer.set_training_mode(True)
    X = rng.uniform(-1.0, 1.0, (examples, tokens * 16))
    delta = rng.uniform(-1.0, 1.0, X.shape)
    tracemalloc.start()
    layer.forward_batch(X)
    layer._backward(delta)  # pyright: ignore[reportPrivateUsage]
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 2**20


def test_memory_is_linear_in_t_at_a_fixed_token_budget():
    # quadratic attention's peak would grow 4x from T = 256 to 1,024 at the same tokens; tiled, the
    # tiles are (N, h, T, b), so the peak stays about level
    assert _peak_mb(1024) < 1.5 * _peak_mb(256)


def test_key_blocks_cut_at_multiples_of_the_block_from_key_zeros_column():
    assert key_blocks(11, 4) == [(0, 4), (4, 8), (8, 11)]
    # a cache whose first key is column 2: cut at columns 4 and 8 (indices 2 and 6)
    assert key_blocks(11, 4, origin=2) == [(0, 2), (2, 6), (6, 10), (10, 11)]
    assert key_blocks(8, 4, origin=4) == [(0, 4), (4, 8)]
    assert key_blocks(3, 4, origin=5) == [(0, 3)]


def test_a_cache_cut_from_its_first_column_is_the_full_pass_over_every_column():
    # the last query against keys 3.. (origin 3) is the full pass's last row with keys 0..2 masked
    Q, K, V = _qkv(12)
    band = Masking(causal=True, band=T - 3)
    full, _, _ = tiled_forward(Q, K, V, SCALE, band, 4, None, 1.0)
    one = Masking(causal=True, offset=T - 4)
    last, _, _ = tiled_forward(Q[:, :, -1:], K[:, :, 3:], V[:, :, 3:], SCALE, one, 4, None, 1.0, origin=3)
    np.testing.assert_allclose(last[:, :, 0], full[:, :, -1], rtol=1e-13, atol=1e-15)
