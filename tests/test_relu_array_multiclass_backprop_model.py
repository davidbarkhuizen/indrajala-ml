from indrajala_ml.model.layer_specs import Dense, LayerSpec
from indrajala_ml.model.relu_rust_array_multiclass_backprop_classifier_network import (
    ReLURustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.relu_vectorized_multiclass_backprop_classifier_network import (
    ReLUVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.update_rules import SGD, UpdateRule
from tests.array_network_contract import ArrayNetworkSpec, multiclass_network_tests
from tests.helpers import matching_relu_array_backprop_networks


def _equivalent(sizes: list[int], output: int) -> tuple[list[LayerSpec], UpdateRule]:
    # this network as layer specs and an update rule
    return (
        [*(Dense(size, activation="relu") for size in sizes), Dense(output, output=True)],
        SGD(),
    )


SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": ReLUVectorizedMultiClassBackpropClassifierNetwork,
        "rust": ReLURustArrayMultiClassBackpropClassifierNetwork,
    },
    matching=matching_relu_array_backprop_networks,
    equivalent=_equivalent,
    learning_rate=0.01,
)

globals().update(multiclass_network_tests(SPEC))
