from indrajala_ml.model.array_backprop_classifier_network import ArrayBackpropClassifierNetwork
from indrajala_ml.model.rust_array_backprop_classifier_network import RustArrayBackpropClassifierNetwork
from tests.array_network_contract import ArrayNetworkSpec, single_output_network_tests
from tests.helpers import matching_single_output_array_backprop_networks

SPEC = ArrayNetworkSpec(
    network_cls={"numpy": ArrayBackpropClassifierNetwork, "rust": RustArrayBackpropClassifierNetwork},
    matching=matching_single_output_array_backprop_networks,
    learning_rate=0.3,
    learn_steps=100,
    learn_batches=20,
)

globals().update(single_output_network_tests(SPEC))
