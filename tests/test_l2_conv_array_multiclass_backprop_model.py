from indrajala_ml.model.l2_conv_rust_array_multiclass_backprop_classifier_network import (
    L2ConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.l2_conv_vectorized_multiclass_backprop_classifier_network import (
    L2ConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.layer_specs import Dense, LayerSpec
from indrajala_ml.model.update_rules import UpdateRule, WeightDecay
from tests.array_network_contract import ArrayNetworkSpec, conv_network_tests


def _equivalent(sizes: list[int], output: int, l2_lambda: float) -> tuple[list[LayerSpec], UpdateRule]:
    # this network's dense layers as layer specs, and its update rule
    return (
        [*(Dense(size) for size in sizes), Dense(output, output=True)],
        WeightDecay(l2_lambda),
    )


SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": L2ConvVectorizedMultiClassBackpropClassifierNetwork,
        "rust": L2ConvRustArrayMultiClassBackpropClassifierNetwork,
    },
    equivalent=_equivalent,
    hyperparameters={"l2_lambda": 0.05},
    learning_rate=0.5,
    saved_hyperparameters_test="l2_lambda",
    saved_hyperparameters={"l2_lambda": 0.02},
)

globals().update(conv_network_tests(SPEC))
