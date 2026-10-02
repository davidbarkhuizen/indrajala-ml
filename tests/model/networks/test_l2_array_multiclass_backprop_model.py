from indrajala_ml.model.networks.numpy.l2_vectorized_multiclass_backprop_classifier_network import (
    L2VectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.networks.rust.l2_rust_array_multiclass_backprop_classifier_network import (
    L2RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.layer_specs import Dense, LayerSpec
from indrajala_ml.model.specs.update_rules import UpdateRule, WeightDecay
from tests.array_network_contract import ArrayNetworkSpec, multiclass_network_tests


def _equivalent(sizes: list[int], output: int, l2_lambda: float) -> tuple[list[LayerSpec], UpdateRule]:
    # this network as layer specs and an update rule
    return (
        [*(Dense(size) for size in sizes), Dense(output, output=True)],
        WeightDecay(l2_lambda),
    )


SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": L2VectorizedMultiClassBackpropClassifierNetwork,
        "rust": L2RustArrayMultiClassBackpropClassifierNetwork,
    },
    equivalent=_equivalent,
    hyperparameters={"l2_lambda": 0.05},
    saved_hyperparameters_test="l2_lambda",
    saved_hyperparameters={"l2_lambda": 0.02},
)

globals().update(multiclass_network_tests(SPEC))
