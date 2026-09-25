from indrajala_ml.model.l2_rust_array_multiclass_backprop_classifier_network import (
    L2RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.l2_vectorized_multiclass_backprop_classifier_network import (
    L2VectorizedMultiClassBackpropClassifierNetwork,
)
from tests.array_network_contract import ArrayNetworkSpec, multiclass_network_tests
from tests.helpers import matching_l2_array_backprop_networks

SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": L2VectorizedMultiClassBackpropClassifierNetwork,
        "rust": L2RustArrayMultiClassBackpropClassifierNetwork,
    },
    matching=matching_l2_array_backprop_networks,
    hyperparameters={"l2_lambda": 0.05},
    saved_hyperparameters_test="l2_lambda",
    saved_hyperparameters={"l2_lambda": 0.02},
)

globals().update(multiclass_network_tests(SPEC))
