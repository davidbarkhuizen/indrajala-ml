# pyright: reportConstantRedefinition=false
# (matrices are named as in the literature, X, A, which strict mode takes for constants)
from __future__ import annotations

from typing import Any, Literal

import indrajala_math_rust as pa

from indrajala_ml.model.array_parameters import GammaAndBeta, RunningAverages
from indrajala_ml.model.layer_specs import ghost_groups, refuse_single_example
from indrajala_ml.model.residual_rust_array_layer import ForkRustArrayLayer


class BatchNormRustArrayLayer(GammaAndBeta[pa.Array], RunningAverages[pa.Array]):
    """
    BatchNormArrayLayer on the Rust backend, each method one Rust call (the crate's batch_norm.rs),
    which computes the README's expressions (Batch normalization) as the numpy layer does: given
    the same inputs, the same bits.

    After a conv layer, positions is its out_height * out_width, and the size values are size //
    positions channels in the conv layer's channel-major layout, each normalized over the batch and
    every position, as BatchNormArrayLayer's (N * positions, channels) view: the crate's ops index
    that layout in place, in the view's row order. Only ReLU follows a conv layer
    (validate_layer_specs).

    A sigmoid's hidden delta is ArrayLayer's fused call on the next layer's W and delta: only a
    dense layer or a residual block's fork follows it (the fused skip op before a fork). A ReLU's masks the next layer's downstream, which may be a dense, conv
    or pool layer's (a conv layer's output can be one position, so positions doesn't tell);
    after a dense layer that is the fused layer_relu_hidden_delta_batch's bits.

    training is set by set_training_mode. The backward pass reads _was_training, training as
    forward_batch saw it, since the network switches training off before the backward pass.

    With a group_size, the ops normalize each ghost group (layer_specs.ghost_groups, D6) as a
    batch of its own; _var and _std are then per group per channel, group-major.
    """

    def __init__(
        self,
        size: int,
        activation: Literal["sigmoid", "relu"],
        epsilon: float,
        running_rate: float,
        positions: int = 1,
        group_size: int | None = None,
    ) -> None:
        assert positions >= 1 and size % positions == 0, f"{size} values aren't {positions} positions per channel"
        assert positions == 1 or activation == "relu", "only ReLU follows a conv layer (validate_layer_specs)"
        self.size = size
        self.input_size = size
        self.positions = positions
        channels = size // positions
        self.activation = activation
        self.epsilon = epsilon
        self.running_rate = running_rate
        self.group_size = group_size
        self.training = False
        self._was_training = False

        # D5's initialization: nothing drawn
        self.gamma = pa.Array([1.0] * channels)
        self.beta = pa.Array.zeros(channels)
        self.running_mean = pa.Array.zeros(channels)
        self.running_var = pa.Array([1.0] * channels)

        self.grad_gamma = pa.Array.zeros(channels)
        self.grad_beta = pa.Array.zeros(channels)

    def set_training_mode(self, training: bool) -> None:
        self.training = training

    def _inference(self, X: pa.Array) -> pa.Array:
        return pa.batch_norm_forward(
            X, self.gamma, self.beta, self.running_mean, self.running_var, self.epsilon, self.activation, self.positions
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
        # the ops refuse a last group of one too, but this is every implementation's message
        ghost_groups(X.shape[0], self.group_size)
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
            self.positions,
            self.group_size,
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
        if self.activation == "sigmoid" and isinstance(next_layer, ForkRustArrayLayer):
            # before a residual block: the fused skip op (D8, RustArrayLayer.compute_hidden_delta)
            body = next_layer.body_first
            self.delta_batch = pa.layer_hidden_delta_skip_batch(
                body.W, body.delta_batch, next_layer.add.delta_batch, self.A
            )
        elif self.activation == "sigmoid":
            self.delta_batch = pa.layer_hidden_delta_batch(next_layer.W, next_layer.delta_batch, self.A)
        else:
            self.delta_batch = pa.array_relu_mask(next_layer.downstream_batch(), self.A)

    def downstream(self) -> pa.Array:
        refuse_single_example(self)

    def downstream_batch(self) -> pa.Array:
        # dl/dx, the linear layer's delta: the paper's § 3 chain rule, term by term
        return pa.batch_norm_downstream_batch(
            self.delta_batch, self.gamma, self._d, self._var, self._std, self.epsilon, self.positions, self.group_size
        )

    def accumulate_gradient(self, input_activation: pa.Array) -> None:
        refuse_single_example(self)

    def accumulate_gradient_batch(self, input_activation_batch: pa.Array) -> None:
        self.grad_gamma, self.grad_beta = pa.batch_norm_accumulate_gradient_batch(
            self.delta_batch, self._xhat, self.grad_gamma, self.grad_beta, self.positions
        )

    def reset_gradient_accum(self) -> None:
        self.grad_gamma = pa.Array.zeros(self.size // self.positions)
        self.grad_beta = pa.Array.zeros(self.size // self.positions)
