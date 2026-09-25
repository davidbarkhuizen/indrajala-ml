from __future__ import annotations

import indrajala_math_rust as pa

from indrajala_ml.model.rust_array_layer import RustArrayLayer


class DropoutRustArrayLayer(RustArrayLayer):
    """
    The Rust-matmul-backed counterpart to DropoutArrayLayer. Same inverted-dropout formulas, but forward/forward_batch/
    compute_hidden_delta/compute_hidden_delta_batch are each a single fused Rust call
    (`layer_dropout_forward`/`layer_dropout_forward_batch`/`layer_dropout_hidden_delta`/
    `layer_dropout_hidden_delta_batch`, `fused.rs`, built on the Rust core's new
    `bernoulli_mask`/`draw_bernoulli_mask` RNG primitive, `random.rs`) instead of a numpy
    expression - mirroring how `RustArrayLayer` itself relates to `ArrayLayer`.
    `accumulate_gradient`/`apply_accumulated_gradient` are inherited unchanged from
    `RustArrayLayer`.

    Unlike `uniform()` (`randomize()`'s own RNG source, drawn once at construction time), this
    mask is drawn fresh on every training-time forward pass - the same "no numpy-seed-compatible
    RNG to match" caveat this crate's `uniform()` already carries applies here too, sharper here
    since the mask *is* the mechanism under test, not incidental to it: parity against the
    numpy-backed sibling can only be checked statistically at training=True, exactly at
    training=False.

    drop_probability is a required constructor argument, no default - the same posture
    DropoutArrayLayer already has.
    """

    hyperparameters = ("drop_probability",)

    def __init__(self, size: int, input_size: int, drop_probability: float) -> None:
        super().__init__(size, input_size)
        assert 0.0 <= drop_probability < 1.0, f"drop_probability must be in [0.0, 1.0); got {drop_probability}"
        self._drop_probability = drop_probability
        self._keep_probability = 1.0 - drop_probability
        self.training = False

    def set_training_mode(self, training: bool) -> None:
        self.training = training

    def forward(self, x: "pa.Array") -> "pa.Array":
        self.a, self._mask, self._base_activation = pa.layer_dropout_forward(
            self.W, x, self.b, self._drop_probability, self.training
        )
        self._was_training = self.training
        return self.a

    def forward_batch(self, X: "pa.Array") -> "pa.Array":
        self.A, self._mask_batch, self._base_activation_batch = pa.layer_dropout_forward_batch(
            self.W, X, self.b, self._drop_probability, self.training
        )
        self._was_training = self.training
        return self.A

    def compute_hidden_delta(self, next_layer: "RustArrayLayer") -> None:
        self.delta = pa.layer_dropout_hidden_delta(
            next_layer.W,
            next_layer.delta,
            self._base_activation,
            self._mask,
            self._keep_probability,
            self._was_training,
        )

    def compute_hidden_delta_batch(self, next_layer: "RustArrayLayer") -> None:
        self.delta_batch = pa.layer_dropout_hidden_delta_batch(
            next_layer.W,
            next_layer.delta_batch,
            self._base_activation_batch,
            self._mask_batch,
            self._keep_probability,
            self._was_training,
        )
