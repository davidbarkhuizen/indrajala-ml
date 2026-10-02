from indrajala_ml.model.relu_conv_rust_array_multiclass_backprop_classifier_network import (
    ReLUConvRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.relu_conv_vectorized_multiclass_backprop_classifier_network import (
    ReLUConvVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.layer_specs import Dense, LayerSpec
from indrajala_ml.model.specs.update_rules import SGD, UpdateRule
from tests.array_network_contract import ArrayNetworkSpec, conv_network_tests


def _equivalent(sizes: list[int], output: int) -> tuple[list[LayerSpec], UpdateRule]:
    # this network's dense layers as layer specs, and its update rule
    return (
        [*(Dense(size, activation="relu") for size in sizes), Dense(output, output=True)],
        SGD(),
    )


SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": ReLUConvVectorizedMultiClassBackpropClassifierNetwork,
        "rust": ReLUConvRustArrayMultiClassBackpropClassifierNetwork,
    },
    equivalent=_equivalent,
    learning_rate=0.01,
)

globals().update(conv_network_tests(SPEC))
