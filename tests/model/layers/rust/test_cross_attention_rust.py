"""
Cross-attention on Rust (the cross-attention workplan, stage 6): CrossAttentionRustArrayLayer
against numpy's CrossAttentionArrayLayer with the same parameters, numpy's exp the crate's. The
forward pass, the gradient to the tokens, the memory's gradient and the parameters' gradients agree
within the products' rounding (the crate's products are not BLAS's). Where no product reaches a
value they agree by bits: a pad's query, which reads no key, has the output bo and zero query
gradients. Dropout draws numpy's masks from generators in the same state. One example is a batch of
one. The crate's ops against their tiled composition are the crate's own tests
(tests/test_cross_attention_ops.py there).
"""

from typing import Any

import indrajala_math_rust as pa
import numpy as np
import pytest

from indrajala_ml.model.layers.forward_context import ForwardContext
from indrajala_ml.model.layers.numpy import attention_array_layer
from indrajala_ml.model.layers.numpy.array_layer import FloatArray
from indrajala_ml.model.layers.numpy.cross_attention_array_layer import CrossAttentionArrayLayer
from indrajala_ml.model.layers.rust.cross_attention_rust_array_layer import CrossAttentionRustArrayLayer
from tests.helpers import bits, exp_by_crate, patching, rust_to_numpy

# numpy's tiled passes' exp as the crate's
attention_exp = patching(attention_array_layer, "exp", exp_by_crate)

N, T, S, D, D_M = 2, 21, 37, 8, 12
QUERIES = np.array([[0] * 3 + [1] * 8 + [2] * 6 + [3] * 4, [1] * 15 + [2] * 6])
KEYS = np.array([[1] * 12 + [2] * 15 + [3] * 10, [0] * 5 + [1] * 14 + [2] * 18])
RTOL, ATOL = 1e-12, 1e-14


def layers(
    heads: int = 4, key_value_heads: int | None = 2, dropout: float = 0.0, seed: int = 0
) -> tuple[CrossAttentionArrayLayer, CrossAttentionRustArrayLayer]:
    key_size = 4
    numpy = CrossAttentionArrayLayer(T, D, S, D_M, heads, key_size, key_value_heads, dropout)
    rust = CrossAttentionRustArrayLayer(T, D, S, D_M, heads, key_size, key_value_heads, dropout)
    rng = np.random.default_rng(seed)
    parameters = [rng.uniform(-0.5, 0.5, p.shape) for p in numpy.parameters()]
    numpy.set_parameters(parameters)
    rust.set_parameters([pa.Array(p.tolist()) for p in parameters])
    return numpy, rust


def contexts(memory: FloatArray, segmented: bool) -> tuple[ForwardContext, ForwardContext]:
    if not segmented:
        return ForwardContext(memory=memory), ForwardContext(memory=pa.Array(memory.tolist()))
    numpy = ForwardContext(segments=QUERIES, memory=memory, memory_segments=KEYS)
    rust = ForwardContext(
        segments=pa.Array(QUERIES.astype(float).tolist()),
        memory=pa.Array(memory.tolist()),
        memory_segments=pa.Array(KEYS.astype(float).tolist()),
    )
    return numpy, rust


class _Next:
    """A next layer giving the delta, on either backend."""

    def __init__(self, delta: Any) -> None:
        self.delta = delta

    def downstream_batch(self, *, context: Any = None) -> Any:
        return self.delta


def passes(layer: Any, X: Any, memory_context: ForwardContext, G: Any) -> dict[str, Any]:
    A = layer.forward_batch(X, context=memory_context)
    layer.reset_gradient_accum()
    layer.compute_hidden_delta_batch(_Next(G), context=memory_context)
    layer.accumulate_gradient_batch(X, context=memory_context)
    return {
        "A": A,
        "dX": layer.downstream_batch(context=memory_context),
        "dM": layer.memory_downstream_batch(),
        "gradients": list(layer.gradients()),
    }


def as_numpy(value: Any) -> FloatArray:
    if isinstance(value, pa.Array):
        return rust_to_numpy(value)
    numpy: FloatArray = value
    return numpy


def inputs(seed: int = 1) -> tuple[FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(N, T * D)), rng.normal(size=(N, S * D_M)), rng.normal(size=(N, T * D))


def both(numpy: Any, rust: Any, segmented: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    X, memory, G = inputs()
    numpy_context, rust_context = contexts(memory, segmented)
    expected = passes(numpy, X, numpy_context, G)
    got = passes(rust, pa.Array(X.tolist()), rust_context, pa.Array(G.tolist()))
    return expected, got


def assert_close(expected: dict[str, Any], got: dict[str, Any]) -> None:
    for name in ("A", "dX", "dM"):
        e, g = as_numpy(expected[name]), as_numpy(got[name])
        assert g.shape == e.shape, name
        np.testing.assert_allclose(g, e, rtol=RTOL, atol=ATOL, err_msg=name)
    for i, (e, g) in enumerate(zip(expected["gradients"], got["gradients"], strict=True)):
        np.testing.assert_allclose(as_numpy(g), e, rtol=RTOL, atol=ATOL, err_msg=f"gradient {i}")


@pytest.mark.usefixtures("attention_exp")
@pytest.mark.parametrize("segmented", [False, True], ids=["unsegmented", "segmented"])
@pytest.mark.parametrize("heads,key_value_heads", [(1, None), (4, None), (4, 2), (4, 1)])
def test_the_passes_are_numpys_within_the_products_rounding(heads: int, key_value_heads: int | None, segmented: bool):
    numpy, rust = layers(heads, key_value_heads)
    assert_close(*both(numpy, rust, segmented))


@pytest.mark.usefixtures("attention_exp")
def test_a_pad_query_reads_no_key_so_its_output_is_bo_and_its_query_gradient_zero_by_bits():
    numpy, rust = layers()
    expected, got = both(numpy, rust, True)
    A = rust_to_numpy(got["A"]).reshape(N * T, D)
    pads = np.flatnonzero(QUERIES.reshape(-1) == 0)
    bo = rust_to_numpy(rust.bo)
    assert all(bits(A[r]) == bits(bo) for r in pads)
    assert bits(A[pads]) == bits(as_numpy(expected["A"]).reshape(N * T, D)[pads])
    dQ = rust_to_numpy(rust._dQ)  # pyright: ignore[reportPrivateUsage]
    assert np.all(dQ[pads] == 0.0)


@pytest.mark.usefixtures("attention_exp")
def test_one_side_without_segments_puts_its_every_token_in_segment_1_as_numpy():
    numpy, rust = layers()
    X, memory, G = inputs(2)
    numpy_context = ForwardContext(segments=QUERIES, memory=memory)
    rust_context = ForwardContext(segments=pa.Array(QUERIES.astype(float).tolist()), memory=pa.Array(memory.tolist()))
    assert_close(
        passes(numpy, X, numpy_context, G), passes(rust, pa.Array(X.tolist()), rust_context, pa.Array(G.tolist()))
    )


@pytest.mark.usefixtures("attention_exp")
def test_dropout_draws_numpys_masks_from_generators_in_the_same_state():
    numpy, rust = layers(dropout=0.3)
    numpy.set_rng(np.random.default_rng(7))
    rust.set_rng(pa.default_rng(7))
    numpy.set_training_mode(True)
    rust.set_training_mode(True)
    assert_close(*both(numpy, rust, True))
    # the same number of draws: N h T S
    after = np.random.default_rng(7)
    after.random(N * numpy.heads * T * S)
    assert rust.rng.state == after.bit_generator.state


@pytest.mark.usefixtures("attention_exp")
def test_one_example_is_a_batch_of_one():
    _, rust = layers()
    X, memory, _ = inputs(3)
    context = ForwardContext(memory=pa.Array(memory[:1].tolist()))
    batch = rust_to_numpy(rust.forward_batch(pa.Array(X[:1].tolist()), context=context))
    one = rust_to_numpy(rust.forward(pa.Array(X[0].tolist()), context=context))
    assert bits(one) == bits(batch[0])
