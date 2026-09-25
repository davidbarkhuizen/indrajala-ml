from indrajala_ml.model.cross_entropy_vectorized_multiclass_backprop_classifier_network import (
    CrossEntropyVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.cross_entropy_rust_array_multiclass_backprop_classifier_network import (
    CrossEntropyRustArrayMultiClassBackpropClassifierNetwork,
)
from tests.array_network_contract import ArrayNetworkSpec, multiclass_network_tests
from tests.helpers import matching_cross_entropy_multiclass_array_backprop_networks

SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": CrossEntropyVectorizedMultiClassBackpropClassifierNetwork,
        "rust": CrossEntropyRustArrayMultiClassBackpropClassifierNetwork,
    },
    matching=matching_cross_entropy_multiclass_array_backprop_networks,
    learning_rate=0.01,
)

globals().update(multiclass_network_tests(SPEC))
