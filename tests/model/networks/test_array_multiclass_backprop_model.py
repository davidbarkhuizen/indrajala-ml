from indrajala_ml.model.networks.numpy.vectorized_multiclass_backprop_classifier_network import (
    VectorizedMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.networks.rust.rust_array_multiclass_backprop_classifier_network import (
    RustArrayMultiClassBackpropClassifierNetwork,
)
from indrajala_ml.model.specs.layer_specs import Dense, LayerSpec
from indrajala_ml.model.specs.update_rules import SGD, UpdateRule
from tests.array_network_contract import ArrayNetworkSpec, multiclass_network_tests


def _equivalent(sizes: list[int], output: int) -> tuple[list[LayerSpec], UpdateRule]:
    # this network as layer specs and an update rule
    return [*(Dense(size) for size in sizes), Dense(output, output=True)], SGD()


SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": VectorizedMultiClassBackpropClassifierNetwork,
        "rust": RustArrayMultiClassBackpropClassifierNetwork,
    },
    equivalent=_equivalent,
    learning_rate=0.3,
    learn_steps=100,
    learn_batches=20,
)

globals().update(multiclass_network_tests(SPEC))
