from indrajala_ml.model.relu_vectorized_multiclass_backprop_classifier_network import (
    ReLUVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.relu_rust_array_multiclass_backprop_classifier_network import (
    ReLURustArrayMultiClassBackpropClassifierNetwork,
)
from tests.array_network_contract import ArrayNetworkSpec, multiclass_network_tests
from tests.helpers import matching_relu_array_backprop_networks

SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": ReLUVectorizedMultiClassBackpropClassifierNetwork,
        "rust": ReLURustArrayMultiClassBackpropClassifierNetwork,
    },
    matching=matching_relu_array_backprop_networks,
    learning_rate=0.01,
)

globals().update(multiclass_network_tests(SPEC))
