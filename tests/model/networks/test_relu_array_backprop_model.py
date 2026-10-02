from indrajala_ml.model.networks.numpy.relu_array_backprop_classifier_network import ReLUArrayBackpropClassifierNetwork
from indrajala_ml.model.networks.rust.relu_rust_array_backprop_classifier_network import (
    ReLURustArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.layer_specs import Dense, LayerSpec
from indrajala_ml.model.specs.update_rules import SGD, UpdateRule
from tests.array_network_contract import ArrayNetworkSpec, single_output_network_tests


def _equivalent(sizes: list[int], output: int) -> tuple[list[LayerSpec], UpdateRule]:
    # this network as layer specs and an update rule
    return (
        [*(Dense(size, activation="relu") for size in sizes), Dense(output, output=True)],
        SGD(),
    )


SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": ReLUArrayBackpropClassifierNetwork,
        "rust": ReLURustArrayBackpropClassifierNetwork,
    },
    equivalent=_equivalent,
    learning_rate=0.01,
)

globals().update(single_output_network_tests(SPEC))
