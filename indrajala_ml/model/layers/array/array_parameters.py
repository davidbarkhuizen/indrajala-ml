# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, W, which strict mode takes for constants)
"""
The optimizer's accessors (array_protocols.TrainedArrayLayer) for a layer whose parameters are W and
b, every dense and conv layer, for a normalization layer's gamma and beta, and for attention's four
projections, on both backends.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar


class WeightAndBias[A]:
    """(W, b), with W decayed under WeightDecay and b not."""

    decayed: ClassVar[tuple[bool, ...]] = (True, False)

    W: A
    b: A
    grad_W: A
    grad_b: A

    def parameters(self) -> tuple[A, ...]:
        return self.W, self.b

    def gradients(self) -> tuple[A, ...]:
        return self.grad_W, self.grad_b

    def set_parameters(self, parameters: Sequence[A]) -> None:
        self.W, self.b = parameters


class GammaAndBeta[A]:
    """
    A normalization layer's (gamma, beta), neither decayed (the batch-norm workplan, D7): batch
    norm's, and layer norm's (the layer-norm and attention workplan).
    """

    decayed: ClassVar[tuple[bool, ...]] = (False, False)

    gamma: A
    beta: A
    grad_gamma: A
    grad_beta: A

    def parameters(self) -> tuple[A, ...]:
        return self.gamma, self.beta

    def gradients(self) -> tuple[A, ...]:
        return self.grad_gamma, self.grad_beta

    def set_parameters(self, parameters: Sequence[A]) -> None:
        self.gamma, self.beta = parameters


class AttentionProjections[A]:
    """
    Attention's (Wq, bq, Wk, bk, Wv, bv, Wo, bo), each drawn as a dense layer's (W, b) in turn, the
    weights decayed and the biases not (the multi-head attention workplan, D2): Wq, Wk and Wv are
    (heads * key_size, d), their rows in head blocks, and Wo (d, heads * key_size), its columns in
    the same blocks. The order is also the draw, save and step order.
    """

    decayed: ClassVar[tuple[bool, ...]] = (True, False) * 4

    tokens: int
    features: int
    heads: int
    key_size: int
    width: int
    size: int
    input_size: int
    Wq: A
    bq: A
    Wk: A
    bk: A
    Wv: A
    bv: A
    Wo: A
    bo: A
    grad_Wq: A
    grad_bq: A
    grad_Wk: A
    grad_bk: A
    grad_Wv: A
    grad_bv: A
    grad_Wo: A
    grad_bo: A

    @property
    def projection_shapes(self) -> tuple[tuple[int, int], ...]:
        # the projections' (rows, fan_in), each drawn as a dense layer's (W, b), in order
        return ((self.width, self.features),) * 3 + ((self.features, self.width),)

    def _set_up(self, tokens: int, features: int, heads: int, key_size: int | None) -> None:
        # tokens tokens of features features, in heads heads of key_size features (features /
        # heads when None); the parameters and gradients zero
        self.tokens = tokens
        self.features = features
        self.heads = heads
        self.key_size = features // heads if key_size is None else key_size
        self.width = heads * self.key_size
        self.size = tokens * features
        self.input_size = self.size
        self.set_parameters(self._zeros())
        self.reset_gradient_accum()

    def _zeros(self) -> list[A]:
        # a zero array per parameter, in order: the backend's
        raise NotImplementedError

    def parameters(self) -> tuple[A, ...]:
        return self.Wq, self.bq, self.Wk, self.bk, self.Wv, self.bv, self.Wo, self.bo

    def gradients(self) -> tuple[A, ...]:
        return (
            self.grad_Wq,
            self.grad_bq,
            self.grad_Wk,
            self.grad_bk,
            self.grad_Wv,
            self.grad_bv,
            self.grad_Wo,
            self.grad_bo,
        )

    def set_parameters(self, parameters: Sequence[A]) -> None:
        self.Wq, self.bq, self.Wk, self.bk, self.Wv, self.bv, self.Wo, self.bo = parameters

    def _set_gradients(self, gradients: Sequence[A]) -> None:
        (
            self.grad_Wq,
            self.grad_bq,
            self.grad_Wk,
            self.grad_bk,
            self.grad_Wv,
            self.grad_bv,
            self.grad_Wo,
            self.grad_bo,
        ) = gradients

    def reset_gradient_accum(self) -> None:
        self._set_gradients(self._zeros())


class RunningAverages[A]:
    """A batch-norm layer's running averages (running_mean, running_var), which the network saves
    but no optimizer trains."""

    running_mean: A
    running_var: A

    def running_state(self) -> tuple[A, ...]:
        return self.running_mean, self.running_var

    def set_running_state(self, state: Sequence[A]) -> None:
        self.running_mean, self.running_var = state
