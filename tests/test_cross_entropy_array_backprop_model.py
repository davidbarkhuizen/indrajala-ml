from indrajala_ml.model.cross_entropy_array_backprop_classifier_network import (
    CrossEntropyArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.cross_entropy_rust_array_backprop_classifier_network import (
    CrossEntropyRustArrayBackpropClassifierNetwork,
)
from indrajala_ml.model.layer_specs import Dense, LayerSpec
from indrajala_ml.model.update_rules import SGD, UpdateRule
from tests.array_network_contract import ArrayNetworkSpec, single_output_network_tests


def _equivalent(sizes: list[int], output: int) -> tuple[list[LayerSpec], UpdateRule]:
    # this network as layer specs and an update rule
    return (
        [*(Dense(size) for size in sizes), Dense(output, output=True, loss="cross_entropy")],
        SGD(),
    )


SPEC = ArrayNetworkSpec(
    network_cls={
        "numpy": CrossEntropyArrayBackpropClassifierNetwork,
        "rust": CrossEntropyRustArrayBackpropClassifierNetwork,
    },
    equivalent=_equivalent,
    learning_rate=0.1,
    learn_steps=100,
    learn_batches=20,
)

globals().update(single_output_network_tests(SPEC))
