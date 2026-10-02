from indrajala_ml.model.layer_specs import Dense, LayerSpec
from indrajala_ml.model.softmax_conv_rust_array_multiclass_backprop_classifier_network import (
    SoftmaxConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.softmax_conv_vectorized_multiclass_backprop_classifier_network import (
    SoftmaxConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.update_rules import SGD, UpdateRule
from tests.array_network_contract import ArrayNetworkSpec, conv_network_tests


def _equivalent(sizes: list[int], output: int) -> tuple[list[LayerSpec], UpdateRule]:
    # this network's dense layers as layer specs, and its update rule
    return (
        [*(Dense(size) for size in sizes), Dense(output, output=True, activation="softmax", loss="cross_entropy")],
        SGD(),
    )


SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": SoftmaxConvVectorizedMultiClassBackpropClassifierNetwork,
        "rust": SoftmaxConvRustArrayMultiClassBackpropClassifierNetwork,
    },
    equivalent=_equivalent,
    learning_rate=0.01,
    probabilities_sum_to_one=True,
)

globals().update(conv_network_tests(SPEC))
