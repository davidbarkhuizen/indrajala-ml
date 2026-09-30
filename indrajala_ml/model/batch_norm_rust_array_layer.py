# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, X, A, which strict mode takes for constants)
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import indrajala_math_rust as pa

from indrajala_ml.model.layer_specs import refuse_single_example


class BatchNormRustArrayLayer:
    """
    BatchNormArrayLayer on the Rust backend, each method one Rust call (the crate's batch_norm.rs),
    which computes the README's expressions (Batch normalization) as the numpy layer does: given
    the same inputs, the same bits. The hidden delta is the dense layers' fused call on the next
    layer's W and delta, as for RustArrayLayer and ReLURustArrayLayer.

    training is set by set_training_mode. The backward pass reads _was_training, training as
    forward_batch saw it, since the network switches training off before the backward pass.
    """

    decayed: ClassVar[tuple[bool, ...]] = (False, False)

    def __init__(
        self,
        size: int,
        activation: Literal["sigmoid", "relu"],
        epsilon: float,
        running_rate: float,
        positions: int = 1,
    ) -> None:
        # a conv layer's channels (BatchNormArrayLayer's positions) are stage 4c
        assert positions == 1, "conv batch norm on the rust backend is stage 4c of docs/batch-norm-workplan.md"
        self.size = size
        self.input_size = size
        self.activation = activation
        self.epsilon = epsilon
        self.running_rate = running_rate
        self.training = False
        self._was_training = False

        # D5's initialization: nothing drawn
        self.gamma = pa.Array([1.0] * size)
        self.beta = pa.Array.zeros(size)
        self.running_mean = pa.Array.zeros(size)
        self.running_var = pa.Array([1.0] * size)

        self.grad_gamma = pa.Array.zeros(size)
        self.grad_beta = pa.Array.zeros(size)

    def parameters(self) -> tuple[pa.Array, ...]:
        return self.gamma, self.beta

    def gradients(self) -> tuple[pa.Array, ...]:
        return self.grad_gamma, self.grad_beta

    def set_parameters(self, parameters: Sequence[pa.Array]) -> None:
        self.gamma, self.beta = parameters

    def running_state(self) -> tuple[pa.Array, ...]:
        return self.running_mean, self.running_var

    def set_running_state(self, state: Sequence[pa.Array]) -> None:
        self.running_mean, self.running_var = state

    def set_training_mode(self, training: bool) -> None:
        self.training = training

    def _inference(self, X: pa.Array) -> pa.Array:
        return pa.batch_norm_forward(
            X, self.gamma, self.beta, self.running_mean, self.running_var, self.epsilon, self.activation
        )

    def forward(self, x: pa.Array) -> pa.Array:
        # classify_state's single-example forward pass: inference only
        if self.training:
            refuse_single_example(self)
        self.a = self._inference(x)
        return self.a

    def forward_batch(self, X: pa.Array) -> pa.Array:
        self._was_training = self.training
        if not self.training:
            self.A = self._inference(X)
            return self.A

        if X.shape[0] < 2:
            refuse_single_example(self)
        (
            self.A,
            self._xhat,
            self._d,
            self._var,
            self._std,
            self.running_mean,
            self.running_var,
        ) = pa.batch_norm_forward_batch(
            X,
            self.gamma,
            self.beta,
            self.running_mean,
            self.running_var,
            self.epsilon,
            self.running_rate,
            self.activation,
        )
        return self.A

    def compute_output_delta(self, reference: pa.Array) -> None:
        raise NotImplementedError("a batch-norm layer is hidden")

    def compute_output_delta_batch(self, reference_batch: pa.Array) -> None:
        raise NotImplementedError("a batch-norm layer is hidden")

    def compute_hidden_delta(self, next_layer: Any) -> None:
        refuse_single_example(self)

    def compute_hidden_delta_batch(self, next_layer: Any) -> None:
        # dl/dy: the downstream times the activation's derivative
        assert self._was_training, "the backward pass needs a training forward pass's batch statistics"
        if self.activation == "sigmoid":
            self.delta_batch = pa.layer_hidden_delta_batch(next_layer.W, next_layer.delta_batch, self.A)
        else:
            self.delta_batch = pa.layer_relu_hidden_delta_batch(next_layer.W, next_layer.delta_batch, self.A)

    def downstream(self) -> pa.Array:
        refuse_single_example(self)

    def downstream_batch(self) -> pa.Array:
        # dl/dx, the linear layer's delta: the paper's § 3 chain rule, term by term
        return pa.batch_norm_downstream_batch(self.delta_batch, self.gamma, self._d, self._var, self._std, self.epsilon)

    def accumulate_gradient(self, input_activation: pa.Array) -> None:
        refuse_single_example(self)

    def accumulate_gradient_batch(self, input_activation_batch: pa.Array) -> None:
        self.grad_gamma, self.grad_beta = pa.batch_norm_accumulate_gradient_batch(
            self.delta_batch, self._xhat, self.grad_gamma, self.grad_beta
        )

    def reset_gradient_accum(self) -> None:
        self.grad_gamma = pa.Array.zeros(self.size)
        self.grad_beta = pa.Array.zeros(self.size)
