from indrajala_ml.model.adam_conv_rust_array_multiclass_backprop_classifier_network import (
    AdamConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.adam_conv_vectorized_multiclass_backprop_classifier_network import (
    AdamConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.layer_specs import Dense, LayerSpec
from indrajala_ml.model.specs.update_rules import Adam, UpdateRule
from tests.array_network_contract import ArrayNetworkSpec, conv_network_tests


def _equivalent(
    sizes: list[int], output: int, beta1: float, beta2: float, epsilon: float
) -> tuple[list[LayerSpec], UpdateRule]:
    # this network's dense layers as layer specs, and its update rule
    return (
        [*(Dense(size) for size in sizes), Dense(output, output=True)],
        Adam(beta1, beta2, epsilon),
    )


SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": AdamConvVectorizedMultiClassBackpropClassifierNetwork,
        "rust": AdamConvRustArrayMultiClassBackpropClassifierNetwork,
    },
    equivalent=_equivalent,
    hyperparameters={"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
    learning_rate=0.01,
    saved_hyperparameters_test="adam_hyperparameters",
    saved_hyperparameters={"beta1": 0.8, "beta2": 0.99, "epsilon": 1e-6},
)

globals().update(conv_network_tests(SPEC))
