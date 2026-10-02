from indrajala_ml.model.l2_array_backprop_classifier_network import L2ArrayBackpropClassifierNetwork
from indrajala_ml.model.l2_rust_array_backprop_classifier_network import L2RustArrayBackpropClassifierNetwork
from indrajala_ml.model.specs.layer_specs import Dense, LayerSpec
from indrajala_ml.model.specs.update_rules import UpdateRule, WeightDecay
from tests.array_network_contract import ArrayNetworkSpec, single_output_network_tests


def _equivalent(sizes: list[int], output: int, l2_lambda: float) -> tuple[list[LayerSpec], UpdateRule]:
    # this network as layer specs and an update rule
    return (
        [*(Dense(size) for size in sizes), Dense(output, output=True)],
        WeightDecay(l2_lambda),
    )


SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": L2ArrayBackpropClassifierNetwork,
        "rust": L2RustArrayBackpropClassifierNetwork,
    },
    equivalent=_equivalent,
    hyperparameters={"l2_lambda": 0.05},
    saved_hyperparameters_test="l2_lambda",
    saved_hyperparameters={"l2_lambda": 0.02},
)

globals().update(single_output_network_tests(SPEC))
