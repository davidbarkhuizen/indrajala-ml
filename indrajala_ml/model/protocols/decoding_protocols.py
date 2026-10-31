"""
The decoding protocol of a sequence model's token layers (the generation workplan, D2): each layer
decodes the tokens a Decoder (indrajala_ml/generation/decoder.py) gives it, the prompt's first, then
one at a time, on its own cache.

- start_decoding() returns a new cache for one decode, or None for a layer that keeps nothing; a
  layer that can't decode refuses there (a non-causal attention, whose earlier outputs would depend
  on later tokens).
- prefill(X, cache) runs the prompt's L tokens, at positions 0 to L - 1, L at most the window's
  tokens T: X is their (L, d) rows (an Embedding's, their L ids), and the result their (L, d')
  rows.
- step(x, cache, position) runs one token at position, after the tokens before it: x is its (1, d)
  row (an Embedding's, its one id).

A token-wise layer (embedding, layer norm, token-wise dense and softmax, the residual fork and add,
dropout, which is the identity in inference) computes each token from that token alone, so its
prefill and step are one forward on the rows, decode_rows (RowDecoding), and it keeps nothing. A
layer without these methods (TokenMean, Patches, a flat layer) can't decode, and a network refuses
at decoder().
"""

from __future__ import annotations

from typing import Any, Protocol


class DecodingLayer[R](Protocol):
    def start_decoding(self) -> Any: ...

    def prefill(self, X: R, cache: Any, /) -> R: ...

    def step(self, x: R, cache: Any, position: int, /) -> R: ...


class RowDecoding[R]:
    """A token-wise layer's decoding: prefill and step its forward on the new tokens' rows, no cache."""

    def start_decoding(self) -> None:
        return None

    def decode_rows(self, R: R) -> R:
        raise NotImplementedError

    def prefill(self, X: R, cache: Any) -> R:
        return self.decode_rows(X)

    def step(self, x: R, cache: Any, position: int) -> R:
        return self.decode_rows(x)


def refuse_undecodable(layers: list[Any]) -> None:
    """Refuses unless every layer decodes (has the protocol's methods)."""
    for index, layer in enumerate(layers):
        if not all(hasattr(layer, name) for name in ("start_decoding", "prefill", "step")):
            raise ValueError(
                f"layer {index}, a {type(layer).__name__}, can't decode one token at a time (the generation "
                "workplan, D2): only a sequence model's causal token layers decode"
            )


def refuse_non_causal_decoding(causal: bool) -> None:
    """An attention's start_decoding: a non-causal one's earlier outputs depend on later tokens."""
    if not causal:
        raise ValueError(
            "a non-causal Attention can't decode one token at a time (the generation workplan, D2): each token's "
            "output reads the tokens after it"
        )
