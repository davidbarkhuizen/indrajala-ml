from indrajala_ml.model.cross_entropy_array_backprop_classifier_network import (
    CrossEntropyArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.cross_entropy_rust_array_backprop_classifier_network import (
    CrossEntropyRustArrayBackpropClassifierNetwork,
)
from tests.array_network_contract import ArrayNetworkSpec, single_output_network_tests
from tests.helpers import matching_cross_entropy_array_backprop_networks

SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": CrossEntropyArrayBackpropClassifierNetwork,
        "rust": CrossEntropyRustArrayBackpropClassifierNetwork,
    },
    matching=matching_cross_entropy_array_backprop_networks,
    learning_rate=0.1,
    learn_steps=100,
    learn_batches=20,
)

globals().update(single_output_network_tests(SPEC))
