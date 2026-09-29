from indrajala_ml.model.layer_specs import Dense, LayerSpec
from indrajala_ml.model.momentum_rust_array_multiclass_backprop_classifier_network import (
    MomentumRustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.momentum_vectorized_multiclass_backprop_classifier_network import (
    MomentumVectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.update_rules import Momentum, UpdateRule
from tests.array_network_contract import ArrayNetworkSpec, multiclass_network_tests
from tests.helpers import matching_momentum_array_backprop_networks


def _equivalent(sizes: list[int], output: int, momentum: float) -> tuple[list[LayerSpec], UpdateRule]:
    # this network as layer specs and an update rule
    return (
        [*(Dense(size) for size in sizes), Dense(output, output=True)],
        Momentum(momentum),
    )


SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": MomentumVectorizedMultiClassBackpropClassifierNetwork,
        "rust": MomentumRustArrayMultiClassBackpropClassifierNetwork,
    },
    matching=matching_momentum_array_backprop_networks,
    equivalent=_equivalent,
    hyperparameters={"momentum": 0.5},
    saved_hyperparameters_test="momentum_coefficient",
    saved_hyperparameters={"momentum": 0.7},
)

globals().update(multiclass_network_tests(SPEC))
