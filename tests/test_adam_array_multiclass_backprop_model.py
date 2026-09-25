from indrajala_ml.model.adam_vectorized_multiclass_backprop_classifier_network import (
    AdamVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.adam_rust_array_multiclass_backprop_classifier_network import (
    AdamRustArrayMultiClassBackpropClassifierNetwork,
)
from tests.array_network_contract import ArrayNetworkSpec, multiclass_network_tests
from tests.helpers import matching_adam_array_backprop_networks

SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": AdamVectorizedMultiClassBackpropClassifierNetwork,
        "rust": AdamRustArrayMultiClassBackpropClassifierNetwork,
    },
    matching=matching_adam_array_backprop_networks,
    hyperparameters={"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
    saved_hyperparameters_test="adam_hyperparameters",
    saved_hyperparameters={"beta1": 0.8, "beta2": 0.99, "epsilon": 1e-6},
)

globals().update(multiclass_network_tests(SPEC))
