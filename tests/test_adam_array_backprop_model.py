from indrajala_ml.model.adam_array_backprop_classifier_network import AdamArrayBackpropClassifierNetwork
from indrajala_ml.model.adam_rust_array_backprop_classifier_network import AdamRustArrayBackpropClassifierNetwork
from indrajala_ml.model.layer_specs import Dense, LayerSpec
from indrajala_ml.model.update_rules import Adam, UpdateRule
from tests.array_network_contract import ArrayNetworkSpec, single_output_network_tests


def _equivalent(
    sizes: list[int], output: int, beta1: float, beta2: float, epsilon: float
) -> tuple[list[LayerSpec], UpdateRule]:
    # this network as layer specs and an update rule
    return (
        [*(Dense(size) for size in sizes), Dense(output, output=True)],
        Adam(beta1, beta2, epsilon),
    )


SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": AdamArrayBackpropClassifierNetwork,
        "rust": AdamRustArrayBackpropClassifierNetwork,
    },
    equivalent=_equivalent,
    hyperparameters={"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
    saved_hyperparameters_test="adam_hyperparameters",
    saved_hyperparameters={"beta1": 0.8, "beta2": 0.99, "epsilon": 1e-6},
)

globals().update(single_output_network_tests(SPEC))
