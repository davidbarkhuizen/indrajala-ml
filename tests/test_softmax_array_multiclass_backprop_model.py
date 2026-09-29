from indrajala_ml.model.layer_specs import Dense, LayerSpec
from indrajala_ml.model.softmax_rust_array_multiclass_backprop_classifier_network import (
    SoftmaxRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.softmax_vectorized_multiclass_backprop_classifier_network import (
    SoftmaxVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.update_rules import SGD, UpdateRule
from tests.array_network_contract import ArrayNetworkSpec, multiclass_network_tests
from tests.helpers import matching_softmax_array_backprop_networks


def _equivalent(sizes: list[int], output: int) -> tuple[list[LayerSpec], UpdateRule]:
    # this network as layer specs and an update rule
    return (
        [*(Dense(size) for size in sizes), Dense(output, output=True, activation="softmax", loss="cross_entropy")],
        SGD(),
    )


SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": SoftmaxVectorizedMultiClassBackpropClassifierNetwork,
        "rust": SoftmaxRustArrayMultiClassBackpropClassifierNetwork,
    },
    matching=matching_softmax_array_backprop_networks,
    equivalent=_equivalent,
    learning_rate=0.01,
    probabilities_sum_to_one=True,
)

globals().update(multiclass_network_tests(SPEC))
