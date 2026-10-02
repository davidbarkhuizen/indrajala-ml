from __future__ import annotations

from collections.abc import Sequence

from indrajala_ml.model.conv_layer import ConvSpec
from indrajala_ml.model.conv_rust_array_multiclass_backprop_classifier_network import (
    ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.max_pool_layer import PoolSpec
from indrajala_ml.model.update_rules import DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON, Adam


class AdamConvRustArrayMultiClassBackpropClassifierNetwork(ConvRustArrayMultiClassBackpropClassifierNetwork):
    """
    AdamConvVectorizedMultiClassBackpropClassifierNetwork on the Rust backend: the optimizer's Adam
    rule for the conv, dense and output layers.
    """

    hyperparameters = ("beta1", "beta2", "epsilon")

    def __init__(
        self,
        input_height: int,
        input_width: int,
        conv_specs: Sequence[ConvSpec | PoolSpec],
        dense_layer_sizes: list[int],
        class_count: int,
        beta1: float = DEFAULT_BETA1,
        beta2: float = DEFAULT_BETA2,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        super().__init__(input_height, input_width, conv_specs, dense_layer_sizes, class_count)

    def _update_rule(self) -> Adam:
        return Adam(self.beta1, self.beta2, self.epsilon)
