from indrajala_ml.model.layer_specs import Dense, LayerSpec
from indrajala_ml.model.momentum_array_backprop_classifier_network import MomentumArrayBackpropClassifierNetwork
from indrajala_ml.model.momentum_rust_array_backprop_classifier_network import (
    MomentumRustArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.update_rules import Momentum, UpdateRule
from tests.array_network_contract import ArrayNetworkSpec, single_output_network_tests


def _equivalent(sizes: list[int], output: int, momentum: float) -> tuple[list[LayerSpec], UpdateRule]:
    # this network as layer specs and an update rule
    return (
        [*(Dense(size) for size in sizes), Dense(output, output=True)],
        Momentum(momentum),
    )


SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": MomentumArrayBackpropClassifierNetwork,
        "rust": MomentumRustArrayBackpropClassifierNetwork,
    },
    equivalent=_equivalent,
    hyperparameters={"momentum": 0.5},
    saved_hyperparameters_test="momentum_coefficient",
    saved_hyperparameters={"momentum": 0.7},
)

globals().update(single_output_network_tests(SPEC))
