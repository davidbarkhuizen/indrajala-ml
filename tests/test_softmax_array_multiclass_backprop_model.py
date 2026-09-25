from indrajala_ml.model.softmax_vectorized_multiclass_backprop_classifier_network import (
    SoftmaxVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.softmax_rust_array_multiclass_backprop_classifier_network import (
    SoftmaxRustArrayMultiClassBackpropClassifierNetwork,
)
from tests.array_network_contract import ArrayNetworkSpec, multiclass_network_tests
from tests.helpers import matching_softmax_array_backprop_networks

SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": SoftmaxVectorizedMultiClassBackpropClassifierNetwork,
        "rust": SoftmaxRustArrayMultiClassBackpropClassifierNetwork,
    },
    matching=matching_softmax_array_backprop_networks,
    learning_rate=0.01,
    probabilities_sum_to_one=True,
)

globals().update(multiclass_network_tests(SPEC))
