from indrajala_ml.model.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)
from tests.array_network_contract import ArrayNetworkSpec, multiclass_network_tests
from tests.helpers import matching_array_backprop_networks

SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": VectorizedMultiClassBackpropClassifierNetwork,
        "rust": RustArrayMultiClassBackpropClassifierNetwork,
    },
    matching=matching_array_backprop_networks,
    learning_rate=0.3,
    learn_steps=100,
    learn_batches=20,
)

globals().update(multiclass_network_tests(SPEC))
